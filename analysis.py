import math
import sys
from datetime import datetime
from PIL import Image
from datetime import timedelta


import numpy as np

# self-made
import anomaly
import new_functions

def generate_block_time_histogram_of_activities(AS, activities, step, start = None, end = None, target = ''):
    """
    This counts the statistics of activities in an activity sequence.
    
    Parameters
    ----------
    AS : list of ActivityDataPoint
        Time series data of activities.
    activities : list of activity_model.Activity
        Target activities to count.    
    step : datetime.timedelta
        Time step to count the number of times of the activities.
    start : datetime.timedelta, default None
        Start date and time to count.
        If this isn't specified then 'start' is set as AS[0].start.
    end : datetime.timedelta, default None
        End date and time to count.
        If this isn't specified then 'end' is set as AS[-1].end.
    target : str, default ''
        Target statistics to count.
        It is in ['frequency', 'duration'].
        'frequency': sum of the number of times to do the activity in the period.
        'duration': sum of the duration time [hour] to do the activity in the period.
    
    Returns
    -------
    counts : list of int
        counts[i] = sum of the number of times of all activities in i-th period.
        If the start time of the activity is included i-th period, then it is counted.
    
    Raises
    ------
        Input 'start' must be earlier than input 'end'.
        Period between 'start' and 'end' must be in the range of the activity sequence.
        If the target is not input.
    """ 
    
    if start == None:
        start = AS[0].start
    if end == None:
        end = AS[-1].end
        
    # Value error
    if target == '':
        raise ValueError('taget must be input!')
    if start > end:
        raise ValueError('Input start must be earlier than input end! If you don\'t specify start or end as input, start or end is defined by activity sequence.')
    if start < AS[0].start or AS[-1].end < end:
        print('Some period between start and end is out of range of activity_sequence. If you don\'t specify start or end as input, start or end is defined by activity sequence.')
        
    counts = []
    name_list = [x.name for x in activities]
    
    activity_index = 0
    for (i, act) in enumerate(AS):
        if act.start <= start <= act.end:
            activity_index = i
    act = AS[activity_index]  # first activity after 'start'
    
    if target == 'frequency':
        for t in new_functions.date_generator(start, end, step):
            temp_count = 0
            tt = t + step
            while True:
                if act.activity.name in name_list and t <= act.start <= tt:
                    temp_count += 1
                if act.end <= tt:
                    activity_index += 1
                    if activity_index >= len(AS):
                        break
                    act = AS[activity_index]
                else:
                    break
            counts.append(temp_count)
            
    elif target == 'duration':
        for t in new_functions.date_generator(start, end, step):
            tt = t + step
            temp_d = 0  # duration time in the period [t, t+1]
            while True:
                if act.end <= tt:
                    if act.activity.name in name_list:
                        temp_d += (act.end - max(act.start, t)) / timedelta(hours = 1)
                    activity_index += 1
                    if activity_index >= len(AS):
                        break
                    act = AS[activity_index]
                else:
                    if act.activity.name in name_list:
                        temp_d += (tt - max(act.start, t) )  / timedelta(hours = 1)
                    break
            counts.append(temp_d)
    return counts

def generate_on_seconds_histogram(data, ha_index, step, start = None, end = None):
    """
    This generates time (seconds) histogram that the home appliance turns on.
    
    Parameters
    ----------
    data : list of tuple
        data[i] = (time, sensor_index, sensor_state),
            time : datetime.timedelta
                Time the sensor changes its state.
            sensor_index : int
                Index of the sensor. This is the number that is written in sensor_model.Sensor.index, not the index of input 'sensors'.
            sensor_state : boolean
                Sensor state.
                For now, binary sensors are considered.
    ha_index : int
       Target index of sensor to calculate its cost.
    step : datetime.timedelta
        Time step to calculate cost.
    start : datetime.timedelta, default None
        Start date and time to count.
        If this isn't specified then 'start' is set as data[0][0].
    end : datetime.timedelta, default None
        End date and time to count.
        If this isn't specified then 'end' is set as data[-1][0].
    
    Returns
    -------
    seconds : list of tuple
        seconds[i] = (start, end, second)
        start : datetime.timedelta
            Start date of this bin.
        end : datetime.timedelta
            End date of this bin.
        second : float
            Seconds that the home applicne turns on in the time range of this bin.
        Duration between start and end equals to input's ``step``.
    
    Raises
    ------
        Input 'start' must be earlier than input 'end'.
        Period between 'start' and 'end' must be in the range of the activity sequence.
    """ 
    
    if start == None:
        start = data[0][0]
    if end == None:
        end = data[-1][0]
    
    # raise value errors
    if start > end:
        raise ValueError('Input start must be earlier than input end. If you don\'t specify start or end as input, start or end is defined by sensor data.')
    if start < data[0][0] or data[-1][0] < end:
        print('Some periods between start and end are out of the range of sensor data. If you don\'t specify start or end as input, start or end is defined by sensor data.')
        
    on_range = []  # (start date and time, end date and time)
    temp_start = None
    for d in data:
        if (d[1] == ha_index) and d[2]:
            temp_start = d[0]
        if (d[1] == ha_index) and (not d[2]):
            on_range.append((temp_start, d[0]))
    range_index = 0
    for (i, r) in enumerate(on_range):
        if start <= r[0]:
            range_index = i
            break
            
    seconds = []
    
    for t in new_functions.date_generator(start, end, step):
        start, end = t, t + step
        in_interval = True
        _sum = 0.0
        while in_interval == True:
            if range_index == len(on_range):
                break
            if start <= on_range[range_index][0]:
                if (on_range[range_index][0] <= end):
                    if on_range[range_index][1] < end:
                        _sum += (on_range[range_index][1] - on_range[range_index][0]).total_seconds()
                        range_index += 1
                    else:
                        _sum += (end - on_range[range_index][0]).total_seconds()
                        in_interval = False
                else:
                    in_interval = False
            else:
                if on_range[range_index][1] <= end:
                    _sum += (on_range[range_index][1] - start).total_seconds()
                    range_index += 1
                else:
                    _sum += (end - start).total_seconds()
                    in_interval = False
        seconds.append(_sum)
        
    return seconds


def top_K_cost(cost, K = 15, reverse = False):
    """
    This calculates the top high K indexes of the ``cost``.
    
    Parameters
    ----------
    cost : numpy.ndarray
        Cost sequence.
    K : int
    reverse : boolean, default False
        If this is True, then return top low K cost.
    
    Returns
    -------
    top_list : list of float
        The top K indexes of the ``cost``.
    """
    if K < 0 or len(cost) < K:
        raise ValueError("K = {} is not appropritate.".format(K))
    if reverse == False:
        return list(np.argsort(-cost)[:K])
    else:
        return list(np.argsort(cost)[:K])


def load_sensor_data(path):
    """
    This loads sensor data.
    
    Prameters
    ---------
    path : pathlib.Path
        Path to load the file.
        
    Returns
    -------
    sensor_data : list of tuple
    sensor_data[i] = (time, sensor_index, sensor_state) of i-th line,
        time : datetime.timedelta
            Time the sensor changes its state.
        sensor_index : int
            Index of the sensor.
        sensor_state : boolean
            Sensor state.
    """
    sensor_data = []
    flag = False
    with open(path, 'r') as f:
        for (i, line) in enumerate(f):
            if flag and (len(line) > 1):
                sensor_data.append(line2sensor_data(line))
            if 'day' in line:
                flag = True
    return sensor_data
   
    
def line2sensor_data(line):
    """
    This converts a line in sensor log to sensor data.
    
    Parameters
    ----------
    line : str
        For example, line = '0000 05 59 39 200000 #22 ON'.
        
    Returns
    -------
    sensor_data = (time, sensor_index, sensor_state),
        time : datetime.timedelta
            Time the sensor changes its state.
        sensor_index : int
            Index of the sensor.
        sensor_state : boolean
            Sensor state.
    """
    day = int(line[:4])
    h = int(line[5:7] )
    m = int(line[8:10])
    s = int(line[11:13])
    ms = int(line[14:20])
    state_pos = 0
    state = None
    ON_state_pos = line.find('ON')
    if ON_state_pos != -1:
        state_pos = ON_state_pos
        state = True
    else:
        state_pos = line.find('OFF')
        state = False
    sensor_num = int(line[line.find('#') + 1: state_pos])
    t = timedelta(days = day, hours = h, minutes = m, seconds = s, microseconds = ms)
    return (t, sensor_num, state)


def load_activity_sequence(path):
    """
    This loads activity sequence.
    
    Prameters
    ---------
    path : pathlib.Path
        Path to load the file.
        
    Returns
    -------
    activity_data : list of tuple
    activity_data[i] = (time, activity_name) of i-th line,
        time : datetime.timedelta
            Start time of the activity.
        activity name : str
            Name of the activity.
    """
    activity_data = []
    with open(path, 'r') as f:
        for (i, line) in enumerate(f):
            if (i != 0) and (len(line) > 1):
                activity_data.append(line2activity_data(line))
    return activity_data


def line2activity_data(line):
    """
    This converts a line in activity log to activity data.
    
    Parameters
    ----------
    line : str
        For example, line = '0000 00 00 00 000000 Sleep'.
        
    Returns
    -------
    activity_data = (time, activity_name),
        time : datetime.timedelta
            Start time of the activity.
        activity name : str
            Name of the activity.
    """
    day = int(line[:4])
    h = int(line[5:7] )
    m = int(line[8:10])
    s = int(line[11:13])
    ms = int(line[14:20])
    t = timedelta(days = day, hours = h, minutes = m, seconds = s, microseconds = ms)
    act_name = line[21:]
    return (t, act_name)


def generate_activation_list(sensor_data, sensor_indexes):
    """
    This generates activation_list that consists of time of activation of sensors in sensor_indexes.
    
    Parameters
    ----------
    sensor_data = (time, sensor_index, sensor_state),
        time : datetime.timedelta
            Time the sensor changes its state.
        sensor_index : int
            Index of the sensor.
        sensor_state : boolean
            Sensor state.
    sensor_indexes : list of int
        Indexes of sensors used to estimate the speed.

    Returns
    -------
    activation_list : list of float
        activation_list[i] = 'i-th activation of sensors that include in sensor_indexes'.
    """
    activation_list = []  # activation time
    activation_dict = {}
    for (i, s) in enumerate(sensor_data):
        if s[1] in sensor_indexes:
            if s[2] and (s[1] not in activation_dict.keys()):
                activation_dict[s[1]] = s[0]
            if (not s[2]) and (s[1] in activation_dict.keys()):
                temp_seconds = (s[0] - activation_dict[s[1]]).total_seconds()
                activation_list.append(temp_seconds)
                del activation_dict[s[1]]
    return activation_list


def estimate_walking_speed(sensor_data, sensor_indexes, radius):
    """
    This estimates a walking speed.
    Mean distance is calculated as mean line segment distance of circle whose radius is radius.
    Parameters
    ----------
    sensor_data = (time, sensor_index, sensor_state),
        time : datetime.timedelta
            Time the sensor changes its state.
        sensor_index : int
            Index of the sensor.
        sensor_state : boolean
            Sensor state.
    sensor_indexes : list of int
        Indexes of sensors used to estimate the speed.
    radius : float
        radius of PIR motion sensors. Radii of all sensors are equal.
    
    Returns
    -------
    (speed_mean, speed_sd)
        speed_mean : float
            Mean walking speed.
        speed_sd : float
            Standard deviation of walking speed.
    """
    mean_distance = 4 * radius / np.pi
    speed_list = []
    activation_dict = {}
    for (i, s) in enumerate(sensor_data):
        if s[1] in sensor_indexes:
            if s[2] and (s[1] not in activation_dict.keys()):
                activation_dict[s[1]] = s[0]
            if (not s[2]) and (s[1] in activation_dict.keys()):
                temp_seconds = (s[0] - activation_dict[s[1]]).total_seconds()
                speed_list.append(mean_distance / temp_seconds)
                del activation_dict[s[1]]
    speed_array = np.array(speed_list)
    return (speed_array.mean(), np.sqrt(speed_array.var()))


def window_matrix(SD, time, duration, rate, indexes):
    """
    This generates sensor data matrix with fixed window length before a end_time (= time as input).
    Zero padding is done if the time - duration is before the start time of SD.
    
    Parameters
    ----------
    SD : list of tuple
        Raw sensor data.
        (time, index, state).
    time : datetime.timedelta
        End time of sampling. End time is not included in matrix!
        If you set `time = t_0`, the time t < t_0 is target.   
    duration : datetime.timedelta
        Sampling duration.
    rate : datetime.timedelta
        Sampling rate.
    indexes : list of int
        Sensor indexes.
        
    Returns
    -------
    mat : numpy.ndarray
        Sensor data matrix. mat.shape = (len(indexes), sampling times).
    """
    initial_time = SD[0][0]
    start_time = time - duration
    if start_time < initial_time:
        raise ValueError('Start time of sampling is before the start time of sensor data.')

    ON_periods = {}  # record ON period of i-th sensor
    for ind in indexes:
        ON_periods[ind] = []
    sensor_states = {i: False for i in indexes}
    left_time = {i: None for i in indexes}
    in_window = False
    for (i, d) in enumerate(SD):
        if not(start_time <= d[0] <= time) and in_window:
            for x in indexes:
                if sensor_states[x]:
                    ON_periods[x].append((left_time[x], time))
            break
        if start_time <= d[0] <= time:
            in_window = True
            if d[2]:
                sensor_states[d[1]] = True
                left_time[d[1]] = d[0]
            else:
                if not(sensor_states[d[1]]):
                    ON_periods[d[1]].append((start_time, d[0]))
                else:
                    ON_periods[d[1]].append((left_time[d[1]], d[0]))
                    sensor_states[d[1]] = False
        
    mat = []
    for t in new_functions.date_generator(start_time, time, step = rate):
        vec = [any([start <= t <= end for (start, end) in ON_periods[x]]) for x in indexes]
        mat.append(vec)
    return np.array(mat).T


def label_vector(AL_periods, time):
    """
    This generates a matrix of anomaly labels at a period.
    
    Parameters
    ----------
    AL_periods : dict of list of tuple of datetime.timedelta
        Each periods of anomalies.
        For example, AL_period[anomaly.HOUSEBOUND] = [(timedelta(days = 10), timedelta(days = 20)),
                                                      (timedelta(days = 100), timedelta(days = 107))]
    time : datetime.timedelta
        End time of sampling. End time is not included in matrix!
        If you set `time = t_0`, the time t < t_0 is target.
        
    Returns
    -------
    vec : numpy.ndarray
        Label vector. mat.shape = (number of types of anomalies).
        label_vec = [semi-bedridden,
                     hosuebound,
                     forgetting,
                     wandering,
                     fall while walking,
                     fall while standing]
    """
    keys = [anomaly.BEING_SEMI_BEDRIDDEN,
            anomaly.BEING_HOUSEBOUND,
            anomaly.FORGETTING,
            anomaly.WANDERING,
            anomaly.FALL_WHILE_WALKING,
            anomaly.FALL_WHILE_STANDING]
    vec = np.array([any([s <= time <= e for (s, e) in AL_periods[k]]) for k in keys])
    return vec


def matrix2image(path, name, mat):
    """
    This saves the image of matrix whose elements take a value between [0, 1].
    The closer the number is to 1, the darker the pixel is. 
    
    Parameters
    ----------
    path : pathlib.Path
        Path to save the image file.
    name : str
        Filename of the image
    mat : numpy.ndarray
        Matrix.
    """
    mat = 1 - mat
    img = Image.fromarray(np.uint8(mat * 255), mode = 'L')
    img.save(path / (name + '.png'))


def memory_size(obj):
    size = sys.getsizeof(obj)
    units = ('B', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB')
    i = math.floor(math.log(size, 1024)) if size > 0 else 0
    size = round(size / 1024 ** i, 2)
    return f"{size} {units[i]}"


