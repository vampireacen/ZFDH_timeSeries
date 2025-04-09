import flask
from flask import request, jsonify
from typing import List, Tuple
import pandas as pd
import numpy as np
import datetime
import time
from utils.data import setup_logger, get_data
from sklearn.metrics import mean_absolute_error, mean_squared_error

logger = setup_logger("")

CURRENT_INDEX = 6844
THRESHOLD = 2000
D_OUTLINE_THRESHOLD = 1600
E_OUTLINE_THRESHOLD = 2850
Absent_start = 0
Absent_end = 0

LOCATION_DICT = {
    0: ['Laoshan',      (120.53, 36.14), (0.03, 0.03)],
    1: ['Jimo',         (120.55, 36.30), (0.05, 0.03)],
    2: ['Huangdao',     (120.175, 36.0), (0.05, 0.05)],
    3: ['Shinan',       (120.39, 36.075), (0.03, 0.01)],
    4: ['Licang',       (120.43, 36.15), (0.05, 0.03)],
    5: ['Chengyang',    (120.40, 36.30), (0.03, 0.03)],
}
np.random.seed(int(time.time()))

data = pd.read_csv('./backend_data/data.csv').values
true_data = np.load('./backend_data/true.npy', allow_pickle=True)
pred_data = np.load('./backend_data/pred.npy', allow_pickle=True)
data_length = data.shape[0]

app = flask.Flask(__name__, static_folder='../static')
app.config['JSON_AS_ASCII'] = False

aera_stations = []


class AeraStation:
    def __init__(self, as_id):
        self.as_id = as_id
        self.nodes: List[Node] = []
        self.name: str = LOCATION_DICT[as_id][0]
        self.location: Tuple = LOCATION_DICT[as_id][1]
        self.loc_range: Tuple = LOCATION_DICT[as_id][2]
        self.producer_num = 0
        self.consumer_num = 0
        self.producer_ratio = 0
        self.consumer_ratio = 0
        self.as_electricity = 0
        self.as_demand = 0
        self.disconnect_threshold = 500
        self.e_outline = 0
        self.d_outline = 0
        self.nodes_connection_status: List[int] = [] # 0: 发电连接, 1: 发电断开, 2: 用电连接
        self.nodes_electricity: List[float] = []
        self.disconnect_nodes = []

    # 获取所有 node
    def get_nodes(self):
        result = []
        for node in self.nodes:
            result.append({
                'n_id': node.n_id,
                'n_type': node.n_type,
                'n_loc': {
                    'lng': node.location[0],
                    'lat': node.location[1]
                },
                'n_status': node.disconnect_status
            })
        return result

    # 设置所有 node 的位置
    def set_nodes_loc(self):
        for node in self.nodes:
            x = np.random.uniform(
                self.location[0] - self.loc_range[0], self.location[0] + self.loc_range[0])
            y = np.random.uniform(
                self.location[1] - self.loc_range[1], self.location[1] + self.loc_range[1])
            node.set_loc(round(x, 6), round(y, 6))

    # 更新当前台区的总发电量和耗电量
    def update_electricity_and_demand(self, electricity, demand):
        self.as_electricity = electricity * self.producer_ratio
        self.as_demand = demand * self.consumer_ratio

    # 更新节点的连接状态
    def update_nodes_connection_status(self):
        store_sum = self.as_electricity - self.as_demand
        is_store = int(store_sum > 0)
        disconnect_nodes = [] # 断开的节点
        if store_sum > self.disconnect_threshold:
            # 断开节点
            # 按照 nodes_electricity 的大小获取排序结果
            sorted_indices = sorted(range(len(self.nodes)), key=lambda i: self.nodes_electricity[i])[::-1]
            as_electricity = self.as_electricity
            reset_idx = 0
            for i, node_idx in enumerate(sorted_indices):
                if self.nodes[node_idx].n_type == 1:
                    reset_idx = i + 1
                    break
                else:
                    # 逐个断开发电节点并判断
                    self.nodes_connection_status[node_idx] = 1
                    as_electricity -= self.nodes_electricity[node_idx] / self.producer_ratio * self.as_electricity
                    disconnect_nodes.append(node_idx)
                    if as_electricity - self.as_demand < self.disconnect_threshold:
                        reset_idx = i + 1
                        break
            while reset_idx < len(sorted_indices) and self.nodes_connection_status[sorted_indices[reset_idx]] != 2:
                if self.nodes[sorted_indices[reset_idx]].n_type == 1:
                    self.nodes_connection_status[sorted_indices[reset_idx]] = 0
                reset_idx += 1
        else:
            # 重新连接所有用电节点
            for node_idx in range(len(self.nodes)):
                if self.nodes[node_idx].n_type == 0:
                    self.nodes_connection_status[node_idx] = 0
        is_disconnect = len(disconnect_nodes) > 0
        # 更新每个节点
        for node_idx in range(len(self.nodes)):
            disconnect_status = node_idx not in disconnect_nodes
            self.nodes[node_idx].disconnect_status = int(disconnect_status)
        self.disconnect_nodes = disconnect_nodes
        
        return is_store, store_sum, is_disconnect
    
class Node:
    def __init__(self, n_id, n_type):
        self.n_id: int = n_id
        self.n_type: int = n_type  # 0:发电节点，1:用电节点
        self.location = None
        self.disconnect_status = 0 # 0: 连接, 1: 断开

    def set_loc(self, x, y):
        self.location = (x, y)

def calculate_accuracy(y_true, y_pred):
    # 计算误差指标
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    smape = 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))
    
    # 计算准确率
    mean_y = np.mean(y_true)
    accuracy_mae = (1 - mae / mean_y) * 100
    accuracy_rmse = (1 - rmse / mean_y) * 100
    accuracy_mape = 100 - mape
    accuracy_smape = 100 - smape
    
    return {
        "Accuracy_MAE": accuracy_mae,
        "Accuracy_RMSE": accuracy_rmse,
        "Accuracy_MAPE": accuracy_mape,
        "Accuracy_sMAPE": accuracy_smape,
    }

def softmax(x):
    # 为了避免溢出，通常会减去最大值进行归一化
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)


def init_aera_stations():
    as_num = np.random.randint(4, 7)
    global xn_list
    xn_list = Limited2DList(as_num, 10)
    total_producer_num = 0
    total_consumer_num = 0
    for i in range(as_num):
        as_id = i
        as_station = AeraStation(as_id)
        n_num = np.random.randint(4, 7)
        as_station.nodes_electricity = [0] * n_num
        for j in range(n_num):
            n_id = j
            n_type = int(np.random.randint(0, 9) > 6) # 60%概率为发电节点
            node = Node(n_id, n_type)
            as_station.nodes.append(node)
            as_station.producer_num += n_type == 0
            as_station.consumer_num += n_type == 1
            as_station.nodes_connection_status.append(0 if n_type == 0 else 2) # 所有节点默认都连接
        # 确保每个台区至少有一个发电节点和一个用电节点
        if as_station.producer_num == 0 or as_station.consumer_num == 0:
            # 随机选择一个节点改变类型
            idx = np.random.randint(0, n_num)
            if as_station.producer_num == 0:
                as_station.nodes[idx].n_type = 0  # 改变当前节点为发电节点
                as_station.producer_num += 1  # 发电节点数加一
                as_station.consumer_num -= 1  # 用电节点数减一
                as_station.nodes_connection_status[idx] = 0  # 修改为发电连接
            else:
                as_station.nodes[idx].n_type = 1  # 改变当前节点为用电节点
                as_station.producer_num -= 1
                as_station.consumer_num += 1
                as_station.nodes_connection_status[idx] = 2 # 修改为用电连接
        total_producer_num += as_station.producer_num
        total_consumer_num += as_station.consumer_num
        as_station.set_nodes_loc()
        aera_stations.append(as_station)
    init_area_electricity(total_producer_num, total_consumer_num)

class LimitedList:
    def __init__(self, max_length):
        self.max_length = max_length
        self.data = []

    def append(self, value):
        if len(self.data) >= self.max_length:
            self.data.pop(0)  # 移除第一个元素
        self.data.append(value)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return repr(self.data)


class Limited2DList:
    def __init__(self, max_rows, max_cols):
        self.max_rows = max_rows
        self.max_cols = max_cols
        self.data = LimitedList(max_rows)  # 限制行数的 LimitedList
        for _ in range(max_rows):
            self.data.append(LimitedList(max_cols))  # 每一行是一个 LimitedList，限制列数

    def append_to_row(self, row, value):
        """
        向指定行添加值。
        """
        if 0 <= row < self.max_rows:
            self.data[row].append(value)
        else:
            raise IndexError("Row index out of range.")

    def __getitem__(self, index):
        """
        实现支持单层和双层索引：
        - 使用 [] 单层索引返回一维数组。
        - 使用 [][] 双层索引返回具体元素。
        """
        return self.data[index]  # 返回一维数组（行）

    def __repr__(self):
        """
        用于打印二维数组的结构化表示。
        """
        return '\n'.join(repr(row) for row in self.data)

xn_list: Limited2DList

def init_area_electricity(producer_num, consumer_num):
    e_node_ratio = np.random.random(producer_num)
    e_node_ratio = softmax(e_node_ratio)
    c_node_ratio = np.random.random(consumer_num)
    c_node_ratio = softmax(c_node_ratio)

    # 初始化发电量和耗电量
    e_id = 0
    c_id = 0
    for as_station in aera_stations:
        producer_ratio = 0
        consumer_ratio = 0
        for node_idx in range(len(as_station.nodes)):
            if as_station.nodes[node_idx].n_type == 0:
                as_station.nodes_electricity[node_idx] = e_node_ratio[e_id]
                producer_ratio += e_node_ratio[e_id]
                e_id += 1
            else:
                as_station.nodes_electricity[node_idx] = -c_node_ratio[c_id]
                consumer_ratio += c_node_ratio[c_id]
                c_id += 1
        as_station.producer_ratio = producer_ratio
        as_station.consumer_ratio = consumer_ratio
        as_station.e_outline = E_OUTLINE_THRESHOLD * \
            producer_ratio * np.random.uniform(0.95, 1.05)  # 设定每个台区的上限
        as_station.d_outline = D_OUTLINE_THRESHOLD * \
            consumer_ratio * np.random.uniform(0.95, 1.05)


def wind_level(wind_speed):
    if wind_speed < 0.5:
        return 0
    elif wind_speed < 1.5:
        return 1
    elif wind_speed < 3.3:
        return 2
    elif wind_speed < 5.5:
        return 3
    elif wind_speed < 8.0:
        return 4
    elif wind_speed < 10.8:
        return 5
    elif wind_speed < 13.9:
        return 6
    elif wind_speed < 17.2:
        return 7
    elif wind_speed < 20.7:
        return 8
    elif wind_speed < 24.5:
        return 9
    elif wind_speed < 28.4:
        return 10
    elif wind_speed < 32.6:
        return 11
    elif wind_speed < 36.9:
        return 12
    elif wind_speed < 41.4:
        return 13
    elif wind_speed < 46.1:
        return 14
    elif wind_speed < 50.9:
        return 15
    elif wind_speed < 56.0:
        return 16
    else:
        return 17

def get_accuracy(pred, true):
    # mse
    mse = np.mean((pred - true) ** 2)

def last_month_same_or_last_day(currnent_datatime):
    currnent_datatime = datetime.datetime.strptime(
        currnent_datatime, '%Y-%m-%d %H:%M:%S')
    # 计算上个月的同一天
    try:
        # 尝试获取上个月的同一天
        last_month_same_day = currnent_datatime.replace(
            month=currnent_datatime.month-1)
    except ValueError:
        # 如果当前是1月，就计算去年的12月的日期
        if currnent_datatime.month == 1:
            last_month_same_day = currnent_datatime.replace(
                year=currnent_datatime.year-1, month=12)
        else:
            # 如果上个月没有今天这一天，则返回上个月的最后一天
            # 先获取这个月的第一天，然后减去一天得到上个月的最后一天
            first_day_of_this_month = currnent_datatime.replace(day=1)
            last_month_same_day = first_day_of_this_month - \
                datetime.timedelta(days=1)

    return str(last_month_same_day)


def return_data(status, data):
    msg = 'success' if status == 200 else 'fail'
    info = jsonify({
        'status': status,
        'msg': msg,
        'data': data
    })
    return info


def print_absent():
    global data
    if data[CURRENT_INDEX, -1] == 0:
        return
    global Absent_start
    global Absent_end
    if Absent_start <= CURRENT_INDEX <= Absent_end:
        return
    assert data[CURRENT_INDEX, -1] == 1
    Absent_start = CURRENT_INDEX
    for i in range(CURRENT_INDEX, data_length):
        if data[i, -1] == 0:
            Absent_end = i
            break
    absent_data = data[Absent_start: Absent_end]
    df = pd.DataFrame(absent_data[:,1:-1], columns=['demand', 'temperature', 'humidity', 'cloud', 'wind_speed', 'electricity'])
    df_bk = df.copy()
    df['demand'] = np.nan
    logger.info(f'检测到缺失值范围从 {Absent_start} 到 {Absent_end}, 数据为: \n{df}')
    logger.info(f'缺失值补全范围从 {Absent_start} 到 {Absent_end}, 补全后的数据为: \n{df_bk}')


@app.route('/weather', methods=['POST'])
def weather():
    api_name = 'weather'
    request_data = request.get_json()
    if not request_data:
        logger.info(f'API: {api_name}, 请求失败.')
        return return_data(0, None)

    global CURRENT_INDEX
    CURRENT_INDEX += 1
    print_absent()
    get_data()
    if CURRENT_INDEX >= data_length:
        CURRENT_INDEX = 745
    # 更新所有台区的节点状态
    update_all_area()

    time = data[CURRENT_INDEX, 0]
    weather_data = true_data[CURRENT_INDEX - 1][0]
    temperature = str(round(weather_data[1], 1))
    humidity = str(round(weather_data[2], 4))
    cloud = str(round(weather_data[3], 4))
    wind_speed = weather_data[4]

    logger.info(f'API: {api_name}, 请求成功.')
    return return_data(200, {
        'time': time,
        'temperature': temperature,
        'humidity': humidity,
        'cloud': cloud,
        'wind_level': wind_level(wind_speed)
    })

def update_all_area():

    time = data[CURRENT_INDEX, 0]

    # 四个小时算一组
    pred_demand_data = pred_data[CURRENT_INDEX, :4, 0] # 降低demand的值 用来显示断开节点
    pred_E_data = pred_data[CURRENT_INDEX, :4, -1]
    pred_E_data[pred_E_data < 0] = 0

    for as_station in aera_stations:
        as_station.update_electricity_and_demand( # TODO: 设置一个函数循环处理所有的台区的节点 每次请求一个天气数据就更新一次
            pred_E_data.sum(), pred_demand_data.sum())
        
        # 更新当前台区的节点状态 (断开/重新连接)
        is_store, store_sum, is_disconnect = as_station.update_nodes_connection_status()
        if is_store:
            if len(xn_list[as_station.as_id]) == 0:
                xn_list[as_station.as_id].append((time, round(store_sum, 2), is_disconnect))
            else:
                if xn_list[as_station.as_id][-1][0] != time:
                    xn_list[as_station.as_id].append((time, round(store_sum, 2), is_disconnect))


def find_farthest_points(as_id):
    current_as: AeraStation = aera_stations[as_id]
    points = [node.location for node in current_as.nodes]
    logger.info(f'当前台区内所有的设备的位置为: {points}')
    max_distance = 0
    farthest_pair = (None, None)

    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            p1 = points[i]
            p2 = points[j]
            # 计算 p1 和 p2 之间的欧几里得距离
            distance = np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
            if distance > max_distance:
                max_distance = distance
                farthest_pair = (p1, p2)

    return farthest_pair[0], farthest_pair[1], max_distance

def detect_outlier(outline_list, true_data, threshold):
    true_data = np.array(true_data, dtype=float)
    new_data = true_data.copy()
    if 1 in outline_list:
        outline_list = np.array(outline_list)
        outlier_list = np.where(outline_list == 1)[0]
        logger.info( f'检测到异常值. 异常值序列为: {true_data}, 异常值为: {true_data[outlier_list]}')
        
        # 平滑处理 
        epoch = 0
        while (new_data[outlier_list] > threshold).sum() != 0:
            if epoch > 10:
                still_list = np.where(new_data[outlier_list] > threshold)[0]
                for idx in still_list:
                    new_data[outlier_list[idx]] = threshold + np.random.randint(-20, 0)
                break
            for idx in range(len(new_data)):
                if idx in outlier_list:
                    if idx == 0:
                        new_data[idx] = (true_data[idx] + true_data[idx + 1]) / 2
                    elif idx == len(outline_list) - 1:
                        new_data[idx] = (true_data[idx - 1] + true_data[idx]) / 2
                    else:
                        new_data[idx] = (true_data[idx - 1] + true_data[idx] + true_data[idx + 1]) / 3
            epoch += 1
        logger.info(f'经过平滑处理为: {new_data[outlier_list].round(2)}')
    return new_data

@app.route('/demand', methods=['POST'])
def demand():
    api_name = 'demand'
    request_data = request.get_json()
    if not request_data:
        logger.info(f'API: {api_name}, 请求失败.')
        return return_data(0, None)

    as_idx = 0
    try:
        as_idx = int(request_data['as_id'])
    except (KeyError, ValueError):
        logger.info(f'API: {api_name}, as_id 不合法.')
        pass

    global CURRENT_INDEX
    global data
    global true_data
    global pred_data

    current_as: AeraStation = aera_stations[as_idx]
    consumer_ratio = current_as.consumer_ratio
    
    time = data[CURRENT_INDEX: CURRENT_INDEX + 24, 0]
    time_list = time.tolist()

    demand_list = (true_data[CURRENT_INDEX, :, 0] * consumer_ratio).round(
        2).astype('str').tolist()
    pred_demand_list = (pred_data[CURRENT_INDEX, :, 0]
                        * consumer_ratio).round(2).astype('str').tolist()
    absent_list = data[CURRENT_INDEX: CURRENT_INDEX + 24, -1].tolist()
    demand_outline_list = (
        true_data[CURRENT_INDEX, :, 0]  * consumer_ratio > current_as.d_outline).astype('int').tolist()
    pred_demand_outline_list = (
        pred_data[CURRENT_INDEX, :, 0]  * consumer_ratio > current_as.d_outline).astype('int').tolist()  # TODO：每个台区设定不同阈值
    logger.info(f'API: {api_name}, 请求成功.')
    detect_outlier(demand_outline_list, demand_list, current_as.d_outline)
    return return_data(200, {
        'time': time_list,
        'true_demand': demand_list,
        'pred_demand': pred_demand_list,
        'absent': absent_list,
        'true_outline': demand_outline_list,
        'pred_outline': pred_demand_outline_list
    })


@app.route('/electricity', methods=['POST'])
def electricity():
    api_name = 'electricity'
    request_data = request.get_json()
    if not request_data:
        logger.info(f'API: {api_name}, 请求失败.')
        return return_data(0, None)

    as_idx = 0
    try:
        as_idx = int(request_data['as_id'])
    except (KeyError, ValueError):
        logger.info(f'API: {api_name}, as_id 不合法.')
        pass

    global CURRENT_INDEX
    global data
    global true_data
    global pred_data

    current_as: AeraStation = aera_stations[as_idx]
    producer_ratio = current_as.producer_ratio

    p1, p2, distance = find_farthest_points(as_idx)
    logger.info(f"当前台区内最远的两个位置为： {p1}, {p2}. 符合 9*9 平方公里网格覆盖范围.")
    weather_data = true_data[CURRENT_INDEX - 1]
    logger.info(f"引入用于预测的天气数据: \n {pd.DataFrame(weather_data, columns=['0', 'temperature', 'humidity', 'cloud', 'wind_speed', 'electricity'])}")

    time = data[CURRENT_INDEX: CURRENT_INDEX + 24, 0]
    time_list = time.tolist()

    E_data = true_data[CURRENT_INDEX, :, -1] * producer_ratio
    E_list = [str(round(d, 2) if d > 0 else 0) for d in E_data]
    pred_E_data = pred_data[CURRENT_INDEX, :, -1] * producer_ratio
    pred_E_data[pred_E_data < 0] = 0
    pred_E_list = pred_E_data.round(2).astype('str').tolist()
    absent_list = data[CURRENT_INDEX: CURRENT_INDEX + 24, -1].tolist()
    E_outline_list = (true_data[CURRENT_INDEX, :, -1] * producer_ratio
                      > current_as.e_outline).astype('int').tolist()
    pred_E_outline_list = (
        pred_data[CURRENT_INDEX, :, -1]  * producer_ratio > current_as.e_outline).astype('int').tolist()

    logger.info(f'API: {api_name}, 请求成功.')
    return return_data(200, {
        'time': time_list,
        'true_demand': E_list,
        'pred_demand': pred_E_list,
        'absent': absent_list,
        'true_outline': E_outline_list,
        'pred_outline': pred_E_outline_list
    })


@app.route('/ratio', methods=['POST'])
def ratio():
    api_name = 'ratio'
    request_data = request.get_json()
    if not request_data:
        logger.info(f'API: {api_name}, 请求失败.')
        return return_data(0, None)

    global CURRENT_INDEX
    global data

    time = data[CURRENT_INDEX - 24: CURRENT_INDEX, 0]
    time_list = [t for t in time]
    now_time = data[CURRENT_INDEX, 0]
    last_month_time = last_month_same_or_last_day(now_time)

    now_demand_data = data[CURRENT_INDEX - 24: CURRENT_INDEX, 1]
    last_day_demand_data = data[CURRENT_INDEX - 48: CURRENT_INDEX - 24, 1]

    index = CURRENT_INDEX
    while data[index, 0] != last_month_time:
        index -= 24
    last_month_demand_data = data[index: index + 24, 1]

    ratio = now_demand_data / last_day_demand_data
    cycle_ratio = now_demand_data / last_month_demand_data
    time_0to24 = np.full(24, "00")
    ratio_0to24 = np.zeros((2, 24)) # 0: 时间, 1: 当前比上一天, 2: 当前比上个月

    for idx in range(24):
        hour_idx = int(time_list[idx][-8:-6])
        time_0to24[hour_idx] = time_list[idx][-8:-6]
        ratio_0to24[0, hour_idx] = ratio[idx]
        ratio_0to24[1, hour_idx] = cycle_ratio[idx]
    ratio_list = ratio_0to24.reshape(2, 6, 4).mean(-1).round(4).tolist() # 四小时一组取均值
    logger.info(f'API: {api_name}, 请求成功.')
    return return_data(200, {
        'time': ['00:00-04:00', '04:00-08:00', '08:00-12:00', '12:00-16:00', '16:00-20:00', '20:00-24:00'],
        'ratio': ratio_list[0],
        'cycle_ratio': ratio_list[1]
    })


@app.route('/decision', methods=['POST'])
def decision():
    api_name = 'decision'
    request_data = request.get_json()
    if not request_data:
        logger.info(f'API: {api_name}, 请求失败.')
        return return_data(0, None)

    as_idx = 0
    try:
        as_idx = int(request_data['as_id'])
    except (KeyError, ValueError):
        logger.info(f'API: {api_name}, as_id 不合法.')
        pass

    global CURRENT_INDEX

    current_as: AeraStation = aera_stations[as_idx]

    time = data[CURRENT_INDEX, 0]

    # # 四个小时算一组
    # pred_demand_data = pred_data[CURRENT_INDEX, :4, 0] # 降低demand的值 用来显示断开节点
    # pred_E_data = pred_data[CURRENT_INDEX, :4, -1]
    # pred_E_data[pred_E_data < 0] = 0

    # current_as.update_electricity_and_demand(
    #     pred_E_data.sum(), pred_demand_data.sum())

    # # 更新当前台区的节点状态 (断开/重新连接)
    # is_store, store_sum = current_as.update_nodes_connection_status()
    # if is_store:
    #     xn_list[as_idx].append(store_sum)
    current_xn = xn_list[as_idx]
    # 转换为普通的list
    time_history = [t[0] for t in current_xn]
    xn_history = [t[1] for t in current_xn]

    logger.info(f'API: {api_name}, 请求成功.')
    return return_data(200, {
        'current_time': time,
        'decision_list': {
            'time_list': time_history,
            'store_num_list': xn_history
        },
        'disconnect_nodes': current_as.disconnect_nodes
    })


@app.route('/aera', methods=['POST'])
def aera():
    api_name = 'aera'
    request_data = request.get_json()
    if not request_data:
        logger.info(f'API: {api_name}, 请求失败.')
        return return_data(0, None)

    global aera_stations
    global CURRENT_INDEX

    results = []
    for as_station in aera_stations:
        results.append({
            'as_id': as_station.as_id,
            'location': as_station.location,
            'as_name': as_station.name,
            'nodes': as_station.get_nodes()
        })

    logger.info(f'API: {api_name}, 请求成功.')
    return return_data(200, results)


def main():
    init_aera_stations()
    app.run(host='0.0.0.0', port=8888)


if __name__ == "__main__":
    main()
