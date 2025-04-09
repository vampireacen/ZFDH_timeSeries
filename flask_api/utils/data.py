import pandas as pd
import numpy as np
import requests
import datetime
from math import exp
import time
import logging
import sys

class CustomFormatter(logging.Formatter):
    """自定义日志格式化器，使日志时间变为一个月前"""
    
    def formatTime(self, record, datefmt=None):
        # 获取原始日志时间（秒级时间戳）
        log_time = datetime.datetime.fromtimestamp(record.created)
        
        # 计算一个月前的时间
        one_month_ago = log_time - pd.DateOffset(months=1)
        
        # 按指定格式返回时间
        return one_month_ago.strftime(datefmt if datefmt else "%Y-%m-%d %H:%M:%S")

def setup_logger(log_file=None):
    log_format = "%(asctime)s [%(levelname)s]: %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    logger = logging.getLogger("CustomLogger")
    logger.setLevel(logging.DEBUG)

    formatter = CustomFormatter(log_format, datefmt=date_format)

    # 输出到控制台
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 输出到文件
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

logger = setup_logger("")

MJ_to_KJ = lambda x: x * 1e3
mm_to_cm = lambda x: x / 10.
tdew_to_kpa = lambda x: ea_from_tdew(x) / 10 * 10.
to_date = lambda d: d.date()
power_variables = ["TOA_SW_DWN", "ALLSKY_SFC_SW_DWN", "T2M", 
                   "T2MDEW", "WS2M", "PRECTOTCORR"]

def getnasadata(latitude, longitude, start_date, end_date):
    server = "https://power.larc.nasa.gov/api/temporal/hourly/point"
    # Variable names in POWER data
    payload = {"request": "execute",
               "parameters": ",".join(power_variables),
               "latitude": latitude,
               "longitude": longitude,
               "start": start_date.strftime("%Y%m%d"),
               "end": end_date.strftime("%Y%m%d"),
               "community": "AG",
               "format": "JSON",
               "user": "anonymous"
               }
    logger.info(f"正在发送请求: {server}, 参数为: {payload}")
    req = requests.get(server, params=payload)
    return req.json()


def _process_POWER_records(powerdata):
    """Process the meteorological records returned by NASA POWER
    """
    fill_value = float(powerdata["header"]["fill_value"])
    
    df_power = {}
    for varname in power_variables:
        s = pd.Series(powerdata["properties"]["parameter"][varname])
        s[s == fill_value] = np.NaN
        df_power[varname] = s
    df_power = pd.DataFrame(df_power)
    df_power["DAY"] = pd.to_datetime(df_power.index, format="%Y%m%d%H")

    # find all rows with one or more missing values (NaN)
    # ix = df_power.isnull().any(axis=1)
    # Get all rows without missing values
    # df_power = df_power[~ix]

    return df_power


def _estimate_AngstAB(df_power):
    """Determine Angstrom A/B parameters from Top-of-Atmosphere (ALLSKY_TOA_SW_DWN) and
    top-of-Canopy (ALLSKY_SFC_SW_DWN) radiation values.

    :param df_power: dataframe with POWER data
    :return: tuple of Angstrom A/B values

    The Angstrom A/B parameters are determined by dividing swv_dwn by toa_dwn
    and taking the 0.05 percentile for Angstrom A and the 0.98 percentile for
    Angstrom A+B: toa_dwn*(A+B) approaches the upper envelope while
    toa_dwn*A approaches the lower envelope of the records of swv_dwn
    values.
    """
    # check if sufficient data is available to make a reasonable estimate:
    # As a rule of thumb we want to have at least 200 days available
    angstA = 0.29
    angstB = 0.49
    if len(df_power) < 200:
        msg = ("Less then 200 days of data available. Reverting to " +
               "default Angstrom A/B coefficients (%f, %f)")
        print(msg)
        return angstA, angstB

    # calculate relative radiation (swv_dwn/toa_dwn) and percentiles
    relative_radiation = df_power.ALLSKY_SFC_SW_DWN / df_power.TOA_SW_DWN
    ix = relative_radiation.notnull()
    angstrom_a = float(np.percentile(relative_radiation[ix].values, 5))
    angstrom_ab = float(np.percentile(relative_radiation[ix].values, 98))
    angstrom_b = angstrom_ab - angstrom_a
    MIN_A = 0.1
    MAX_A = 0.4
    MIN_B = 0.3
    MAX_B = 0.7
    MIN_SUM_AB = 0.6
    MAX_SUM_AB = 0.9
    A = abs(angstrom_a)
    B = abs(angstrom_b)
    SUM_AB = A + B
    if A < MIN_A or A > MAX_A or B < MIN_B or B > MAX_B or SUM_AB < MIN_SUM_AB or SUM_AB > MAX_SUM_AB:
        msg = ("Angstrom A/B values (%f, %f) outside valid range " +
               "Reverting to default values.")
        msg = msg % (angstrom_a, angstrom_b)
        print(msg)
        return angstA, angstB

    return angstrom_a, angstrom_b


def _POWER_to_PCSE(df_power):
    # Convert POWER data to a dataframe with PCSE compatible inputs
    df_pcse = pd.DataFrame({"DAY": df_power.DAY.apply(to_date),
                            "IRRAD": df_power.ALLSKY_SFC_SW_DWN.apply(MJ_to_KJ),
                            "TMIN": df_power.T2M_MIN,
                            "TMAX": df_power.T2M_MAX,
                            "VAP": df_power.T2MDEW.apply(tdew_to_kpa),
                            "WIND": df_power.WS2M,
                            "RAIN": df_power.PRECTOTCORR})

    return df_pcse


def ea_from_tdew(tdew):
    # Raise exception:
    if (tdew < -95.0 or tdew > 65.0):
        # Are these reasonable bounds?
        msg = 'tdew=%g is not in range -95 to +60 deg C' % tdew
        raise ValueError(msg)

    tmp = (17.27 * tdew) / (tdew + 237.3)
    ea = 0.6108 * exp(tmp)
    return ea


# 单点获取
def get_data():
    # 获取当前时间戳（秒级）
    current_timestamp = int(time.time())

    # 转换为 pandas datetime
    dt = pd.to_datetime(current_timestamp, unit='s')

    # 判断是否是 1 号
    is_first_day = dt.day == 1
    if is_first_day:
        one_month_ago = dt - pd.DateOffset(months=2)
        dt = dt - pd.DateOffset(months=1)
        start_date = datetime.date(one_month_ago.year, one_month_ago.month, one_month_ago.day - 1)
        end_date = datetime.date(dt.year, dt.month, dt.day - 1)
        latitude = 8.1
        longitude = -80.98
        logger.info(f"正在爬取数据: 经纬度: ({latitude}, {longitude}), 开始时间为: {start_date}, 结束时间为: {end_date}")
        powerdata = getnasadata(latitude, longitude, start_date, end_date)
        if not powerdata:
            msg = "Failure retrieving POWER data from server. This can be a connection problem with " \
                "the NASA POWER server, retry again later."
            logger.info(msg)
        description = [powerdata["header"]["title"]]
        elevation = float(powerdata["geometry"]["coordinates"][2])

        df_power = _process_POWER_records(powerdata)
        logger.info(f'成功爬取到数据: 长度为: {len(df_power)}, 列名称为: {df_power.columns.to_list()}')
        # Determine Angstrom A/B parameters
        # angstA, angstB = _estimate_AngstAB(df_power)

        # # Convert power records to PCSE compatible structure
        # df_pcse = _POWER_to_PCSE(df_power)
        # # 去除空值，填充空值，加上降雪
        # SNOWDEPTH = [-999 for i in range((end_date - start_date).days + 1)]
        # std_datatime = pd.DataFrame(pd.date_range(start=start_date, end=end_date, freq='D'), columns=['DAY'])
        # data = df_pcse.dropna()
        # data['DAY'] = pd.to_datetime(data['DAY'])
        # result = pd.merge(std_datatime, data, how='left', on='DAY')
        # result['SNOWDEPTH'] = SNOWDEPTH
        # # 找出缺失值索引
        # nan_indexset = set(np.where(np.isnan(result.iloc[:, 1:]))[0])
        # nan_index = list(nan_indexset)
        # miss_value = len(nan_index)
        # nan_index.sort()

        # # 替换空值
        # for i in nan_index:
        #     result.loc[i, 'IRRAD'] = result.iloc[i - 5:i + 6, 1].mean()
        #     result.loc[i, 'TMIN'] = result.iloc[i - 5:i + 6, 2].mean()
        #     result.loc[i, 'TMAX'] = result.iloc[i - 5:i + 6, 3].mean()
        #     result.loc[i, 'VAP'] = result.iloc[i - 5:i + 6, 4].mean()
        #     result.loc[i, 'WIND'] = result.iloc[i - 5:i + 6, 5].mean()
        #     result.loc[i, 'RAIN'] = result.iloc[i - 5:i + 6, 6].mean()
        # # 写文件表头
        # excelhead = pd.DataFrame({'DAY': ['Site Characteristics', 'Country', 'Station', 'Description', 'Source', 'Contact',
        #                                   'Missing values', 'Longitude', longitude, 'Observed data', 'DAY', 'date'],
        #                           'IRRAD': [np.NaN, 'China', 'miss_value={}days'.format(miss_value), description,
        #                                     'Meteorology and Air Quality Group, Wageningen University', 'Peter Uithol',
        #                                     -999, 'Latitude', latitude, np.NaN, 'IRRAD', 'kJ/m2/day or hours'],
        #                           'TMIN': [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, 'Elevation', elevation,
        #                                    np.NaN, 'TMIN', 'Celsius'],
        #                           'TMAX': [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, 'AngstromA', angstA,
        #                                    np.NaN, 'TMAX', 'Celsius'],
        #                           'VAP': [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, 'AngstromB', angstB,
        #                                   np.NaN, 'VAP', 'kPa'],
        #                           'WIND': [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, 'HasSunshine', False,
        #                                    np.NaN, 'WIND', 'm/sec'],
        #                           'RAIN': [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN,
        #                                    'RAIN', 'mm'],
        #                           'SNOWDEPTH': [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN,
        #                                         np.NaN, 'SNOWDEPTH', 'cm']})
        # dataexcel = pd.concat([excelhead, result], axis=0, ignore_index=True)
        data_path = r'backend_data/data_{}_{}.csv'.format(start_date, end_date)
        df_power.to_csv(data_path, index=False, header=None)
        logger.info(f'数据已保存到：{data_path}')
    else:
        logger.info('当前时间无需获取上月数据.')


if __name__ == '__main__':
    get_data()
