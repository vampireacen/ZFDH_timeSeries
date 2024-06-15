import flask
from flask import request, jsonify
import pandas as pd
import numpy as np
import datetime

CURRENT_INDEX = 745
THRESHOLD = 1000

np.random.seed(0)

data = pd.read_csv('./backend_data/data.csv').values
true_data = np.load('./backend_data/true.npy', allow_pickle=True)
pred_data = np.load('./backend_data/pred.npy', allow_pickle=True)
data_length = data.shape[0]

app = flask.Flask(__name__, static_folder='../static')
app.config['JSON_AS_ASCII'] = False


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


@app.route('/weather', methods=['POST'])
def weather():
    api_name = 'weather'
    request_data = request.get_json()
    if not request_data:
        print(f'API: {api_name}, Request Failed.')
        return return_data(0, None)

    global CURRENT_INDEX
    CURRENT_INDEX += 1
    if CURRENT_INDEX >= data_length:
        CURRENT_INDEX = 745

    time = data[CURRENT_INDEX, 0]
    weather_data = true_data[CURRENT_INDEX - 1][0]
    temperature = str(round(weather_data[1], 1))
    humidity = str(round(weather_data[2], 4))
    cloud = str(round(weather_data[3], 4))
    wind_speed = weather_data[4]

    print(f'API: {api_name}, Request Successful.')
    return return_data(200, {
        'time': time,
        'temperature': temperature,
        'humidity': humidity,
        'cloud': cloud,
        'wind_level': wind_level(wind_speed)
    })


@app.route('/demand', methods=['POST'])
def demand():
    api_name = 'demand'
    request_data = request.get_json()
    if not request_data:
        print(f'API: {api_name}, Request Failed.')
        return return_data(0, None)

    global CURRENT_INDEX
    global data
    global true_data
    global pred_data

    time = data[CURRENT_INDEX: CURRENT_INDEX + 24, 0]
    time_list = [t for t in time]

    demand_data = true_data[CURRENT_INDEX, :, 0]
    demand_list = [str(round(d, 2)) for d in demand_data]
    pred_demand_data = pred_data[CURRENT_INDEX, :, 0]
    pred_demand_list = [str(round(d, 2)) for d in pred_demand_data]

    print(f'API: {api_name}, Request Successful.')
    return return_data(200, {
        'time': time_list,
        'true_demand': demand_list,
        'pred_demand': pred_demand_list
    })

# TODO: 缺失值补全（随机一个区间，表示该区间缺失），异常值(超过阈值，就认为是缺失值)处理
@app.route('/electricity', methods=['POST'])
def electricity():
    api_name = 'electricity'
    request_data = request.get_json()
    if not request_data:
        print(f'API: {api_name}, Request Failed.')
        return return_data(0, None)

    global CURRENT_INDEX
    global data
    global true_data
    global pred_data

    time = data[CURRENT_INDEX: CURRENT_INDEX + 24, 0]
    time_list = [t for t in time]

    E_data = true_data[CURRENT_INDEX, :, -1]
    E_list = [str(round(d, 2) if d > 0 else 0) for d in E_data]
    pred_E_data = pred_data[CURRENT_INDEX, :, -1]
    pred_E_list = [str(round(d, 2) if d > 0 else 0) for d in pred_E_data]

    print(f'API: {api_name}, Request Successful.')
    return return_data(200, {
        'time': time_list,
        'true_demand': E_list,
        'pred_demand': pred_E_list
    })


@app.route('/ratio', methods=['POST'])
def ratio():
    api_name = 'ratio'
    request_data = request.get_json()
    if not request_data:
        print(f'API: {api_name}, Request Failed.')
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

    print(f'API: {api_name}, Request Successful.')
    return return_data(200, {
        'time': time_list,
        'ratio': [str(round(r, 2)) for r in ratio],
        'cycle_ratio': [str(round(r, 2)) for r in cycle_ratio]
    })


@app.route('/decision', methods=['POST'])
def decision():
    api_name = 'decision'
    request_data = request.get_json()
    if not request_data:
        print(f'API: {api_name}, Request Failed.')
        return return_data(0, None)

    global CURRENT_INDEX

    time = data[CURRENT_INDEX, 0]

    pred_demand_data = pred_data[CURRENT_INDEX, :4, 0]
    pred_E_data = pred_data[CURRENT_INDEX, :4, -1]
    for i in range(4):
        if pred_E_data[i] < 0:
            pred_E_data[i] = 0

    store = pred_E_data - pred_demand_data
    store_sum = store.sum() + 3500

    is_store = int(store_sum > THRESHOLD)

    print(f'API: {api_name}, Request Successful.')
    return return_data(200, {
        'time': time,
        'is_store': is_store,
        'store_sum': str(store_sum)
    })

class AeraStation:
    def __init__(self, as_id):
        self.as_id = as_id
        self.nodes = []

    def get_nodes(self):
        result = []
        for node in self.nodes:
            result.append({
                'n_id': node.n_id,
                'n_type': node.n_type
            })
        return result

class Node:
    def __init__(self, n_id, n_type):
        self.n_id = n_id
        self.n_type = n_type # 0:发电节点，1:用电节点

@app.route('/aera', methods=['POST'])
def aera():
    api_name = 'aera'
    request_data = request.get_json()
    if not request_data:
        print(f'API: {api_name}, Request Failed.')
        return return_data(0, None)
    
    as_num = np.random.randint(4, 7)
    aera_stations = []
    for i in range(as_num):
        as_id = i
        as_station = AeraStation(as_id)
        n_num = np.random.randint(4, 7)
        for j in range(n_num):
            n_id = j
            n_type = np.random.randint(0, 2)
            node = Node(n_id, n_type)
            as_station.nodes.append(node)
        aera_stations.append(as_station)
    
    results = []
    for as_station in aera_stations:
        results.append({
            'as_id': as_station.as_id,
            'nodes': as_station.get_nodes()
        })
    
    print(f'API: {api_name}, Request Successful.')
    return return_data(200, results)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8888)
