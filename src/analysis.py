import copy
import math
import sys
from datetime import timedelta

import numpy as np
from hmmlearn import hmm
from PIL import Image

# self-made
from src import anomaly_model, sensor_model, utils


def generate_block_time_histogram_of_activities(
    AS, activities, step, start=None, end=None, target=""
):
    """
    This counts the statistics of activities in an activity sequence.

    Parameters
    ----------
    AS : list of ActivityDataPoint
        Time series data of activities.
    activities : list of str
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

    if start is None:
        start = AS[0].start
    if end is None:
        end = AS[-1].end

    # Value error
    if target not in ["frequency", "duration"]:
        raise ValueError("target is fault!")
    if start > end:
        raise ValueError(
            "Input start must be earlier than input end! If you don't specify start or end as input, start or end is defined by activity sequence."
        )
    if start < AS[0].start or AS[-1].end < end:
        print(
            "Some period between start and end is out of range of activity_sequence. If you don't specify start or end as input, start or end is defined by activity sequence."
        )

    counts = []

    activity_index = 0
    for i, act in enumerate(AS):
        if act.start <= start <= act.end:
            activity_index = i
    act = AS[activity_index]  # first activity after 'start'

    if target == "frequency":
        for t in utils.date_generator(start, end, step):
            temp_count = 0
            tt = t + step
            while True:
                if act.activity.name in activities and t <= act.start <= tt:
                    temp_count += 1
                if act.end <= tt:
                    activity_index += 1
                    if activity_index >= len(AS):
                        break
                    act = AS[activity_index]
                else:
                    break
            counts.append(temp_count)

    elif target == "duration":
        for t in utils.date_generator(start, end, step):
            tt = t + step
            temp_d = 0  # duration time in the period [t, t+1]
            while True:
                if act.end <= tt:
                    if act.activity.name in activities:
                        temp_d += (act.end - max(act.start, t)) / timedelta(hours=1)
                    activity_index += 1
                    if activity_index >= len(AS):
                        break
                    act = AS[activity_index]
                else:
                    if act.activity.name in activities:
                        temp_d += (tt - max(act.start, t)) / timedelta(hours=1)
                    break
            counts.append(temp_d)
    return counts


def generate_on_seconds_histogram(data, ha_index, step, start=None, end=None):
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
        seconds[i] = second
        second : float
            Seconds that the home applicne turns on in the time range of this bin.
        Duration between start and end equals to input's ``step``.

    Raises
    ------
        Input 'start' must be earlier than input 'end'.
        Period between 'start' and 'end' must be in the range of the activity sequence.
    """

    if start is None:
        start = data[0][0]
    if end is None:
        end = data[-1][0]

    # raise value errors
    if start > end:
        raise ValueError(
            "Input start must be earlier than input end. If you don't specify start or end as input, start or end is defined by sensor data."
        )
    if start < data[0][0] or data[-1][0] < end:
        print(
            "Some periods between start and end are out of the range of sensor data. If you don't specify start or end as input, start or end is defined by sensor data."
        )

    on_range = []  # (start date and time, end date and time)
    temp_start = None
    for d in data:
        if (d[1] == ha_index) and d[2]:
            temp_start = d[0]
        if (d[1] == ha_index) and (not d[2]):
            on_range.append((temp_start, d[0]))
    range_index = 0
    for i, r in enumerate(on_range):
        if start <= r[0]:
            range_index = i
            break

    seconds = []

    _max_num = (end - start) // step + 1
    diff = 1
    for i, t in enumerate(utils.date_generator(start, end, step)):
        if i % diff == 0:
            utils.print_progress_bar(_max_num, i, "Calculate histograms.")
        _start, _end = t, t + step
        in_interval = True
        _sum = 0.0
        while in_interval:
            if range_index == len(on_range):
                break
            if _start <= on_range[range_index][0]:
                if on_range[range_index][0] <= _end:
                    if on_range[range_index][1] < _end:
                        _sum += (
                            on_range[range_index][1] - on_range[range_index][0]
                        ).total_seconds()
                        range_index += 1
                    else:
                        _sum += (_end - on_range[range_index][0]).total_seconds()
                        in_interval = False
                else:
                    in_interval = False
            else:
                if on_range[range_index][1] <= _end:
                    _sum += (on_range[range_index][1] - _start).total_seconds()
                    range_index += 1
                else:
                    _sum += (_end - _start).total_seconds()
                    in_interval = False
        seconds.append(_sum)

    return seconds


def top_K_cost(cost, K=15, reverse=False):
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
    if not reverse:
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
    with open(path, "r") as f:
        for i, line in enumerate(f):
            if flag and (len(line) > 1):
                sensor_data.append(line2sensor_data(line))
            if "day" in line:
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
    h = int(line[5:7])
    m = int(line[8:10])
    s = int(line[11:13])
    ms = int(line[14:20])
    state_pos = 0
    state = None
    ON_state_pos = line.find("ON")
    if ON_state_pos != -1:
        state_pos = ON_state_pos
        state = True
    else:
        state_pos = line.find("OFF")
        state = False
    sensor_num = int(line[line.find("#") + 1 : state_pos])
    t = timedelta(days=day, hours=h, minutes=m, seconds=s, microseconds=ms)
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
    with open(path, "r") as f:
        for i, line in enumerate(f):
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
    h = int(line[5:7])
    m = int(line[8:10])
    s = int(line[11:13])
    ms = int(line[14:20])
    t = timedelta(days=day, hours=h, minutes=m, seconds=s, microseconds=ms)
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
    for i, s in enumerate(sensor_data):
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
    for i, s in enumerate(sensor_data):
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
        raise ValueError(
            "Start time of sampling is before the start time of sensor data."
        )

    ON_periods = {}  # record ON period of i-th sensor
    for ind in indexes:
        ON_periods[ind] = []
    sensor_states = {i: False for i in indexes}
    left_time = {i: None for i in indexes}
    in_window = False
    for i, d in enumerate(SD):
        if not (start_time <= d[0] <= time) and in_window:
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
                if not (sensor_states[d[1]]):
                    ON_periods[d[1]].append((start_time, d[0]))
                else:
                    ON_periods[d[1]].append((left_time[d[1]], d[0]))
                    sensor_states[d[1]] = False

    mat = []
    for t in utils.date_generator(start_time, time, step=rate):
        vec = [
            any([start <= t <= end for (start, end) in ON_periods[x]]) for x in indexes
        ]
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
    keys = [
        anomaly_model.BEING_SEMI_BEDRIDDEN,
        anomaly_model.BEING_HOUSEBOUND,
        anomaly_model.FORGETTING,
        anomaly_model.WANDERING,
        anomaly_model.FALL_WHILE_WALKING,
        anomaly_model.FALL_WHILE_STANDING,
    ]
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
    img = Image.fromarray(np.uint8(mat * 255), mode="L")
    img.save(path / (name + ".png"))


def memory_size(obj):
    size = sys.getsizeof(obj)
    units = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB")
    i = math.floor(math.log(size, 1024)) if size > 0 else 0
    size = round(size / 1024**i, 2)
    return f"{size} {units[i]}"


def matrix_with_discretized_time_interval(SD, AL, start, end, duration, _type="raw"):
    """
    This divided the time into fixed time intervals.
    The SD and AL_periods are discretized similary.
    Some feature representations of sensor data [1] can be used,
    such as 'raw', 'change point', 'last-fired'.
    [1] TLM van Kasteren, G. Englebienne, and B. J. Krose,
        "Human activity recognition from wireless sensor network data: Benchmark and software."
        Proc. of Activity recognition in pervasive intelligent environments, 2011, 165–186.

    Parameters
    ----------
    SD : list of tuple
        Raw sensor data.
        (time, index, state).
    AL : dict of list of tuple
        Anomaly labels.
        For example, AL[anomaly.HOUSEBOUND]
                     = [(timedelta(days = 10), timedelta(days = 20)),
                        (timedelta(days = 100), timedelta(days = 107))]
    start : datetime.timedelta
        Start time of discretization.
    end : datetime.timedelta
        End time of discretization.
    duration : datetime.timedelta
        Length of time interval.
    _type : str, default 'raw'
        Data representation as cited in [1].
        'raw': original, so take 1 if a sensor ON in the time interval
        'change point': take 1 if a sensor changes its value in the time interval, else 0
        'last-fired' : take 1 if the state of the sensor was changed lastly among all sensors, else 0

    Returns
    -------
    (SD_mat, SD_indexes, AL_mat, AL_indexes) : tuple of matrix
         SD_mat : numpy.ndarray
             SD_mat.shape = (number of time intervals, number of sensors)
             SD_mat[i][j] = True if the j-th sensor takes True in the i-th time intervals, else False.
         SD_names : list
             Raw header of SD_mat. sorted.
         AL_mat : numpy.ndarray
             AL_mat.shape = (number of time intervals, number of anomalies)
             AL_mat[i][j] = True if the j-th anomaly occurs in the i-th time intervals, else False.
         AL_names : list
             Raw header of AL_mat.

    Test code
    ---------
    # Test for analysis.matrix_with_discretized_time_interval
    test_SD = [(timedelta(seconds = 10), 0, True), (timedelta(seconds = 50), 0, False),
               (timedelta(seconds = 80), 0, True), (timedelta(seconds = 100), 0, False),
               (timedelta(seconds = 20), 1, True), (timedelta(seconds = 40), 1, False),
               (timedelta(seconds = 60), 1, True), (timedelta(seconds = 90), 1, False),
               (timedelta(seconds = 30), 2, True), (timedelta(seconds = 70), 2, False)]

    test_AL = {'A': [(timedelta(seconds = 20), timedelta(seconds = 60)), (timedelta(seconds = 80), timedelta(seconds = 90))],
               'B': [(timedelta(seconds = 40), timedelta(seconds = 50)), (timedelta(seconds = 70), timedelta(seconds = 100))]}

    test_SD.sort(key = lambda x: x[0])
    start = timedelta(seconds = -20)
    end = timedelta(seconds = 180)
    step = timedelta(seconds = 5)
    (SD_mat, SD_names, AL_mat, AL_names) = analysis.matrix_with_discretized_time_interval(test_SD, test_AL, start, end, step, _type = 'last-fired')
    print(SD_mat)
    print(SD_names)
    for (i, sd) in enumerate(SD_mat):
        print(f"{timedelta(seconds = i*step.seconds) + start} {sd}")
    for (i, x) in enumerate(AL_mat):
        print(f"{timedelta(seconds = i*step.seconds) + start} {x}")
    """

    def transformation4change_point(SD):
        """
        This transform the raw sensor data to be able to calculate
        change point features in matrix_with_discretized_time_interval.

        Parameters
        ----------
        SD : list of tuple
            Raw sensor data.
            (time, index, state).

        Returns
        -------
        CP : list of tuple
            (time, index, state).
        """
        CP = []
        epsilon = timedelta(
            milliseconds=1
        )  # small value to represent small time interval as a change point
        for x in SD:
            if x[2]:
                CP.append((x[0], x[1], True))
                CP.append((x[0] + epsilon, x[1], False))
            else:
                CP.append((x[0] - epsilon, x[1], True))
                CP.append((x[0], x[1], False))
        return CP

    def transformation4last_fired(SD, start, end, duration):
        """
        This transform the raw sensor data to be able to calculate
        last-fired features in matrix_with_discretized_time_interval.

        Parameters
        ----------
        SD : list of tuple
            Raw sensor data.
            (time, index, state).
        start : datetime.timedelta
            Start time of discretization.
        end : datetime.timedelta
            End time of discretization.
        duration : datetime.timedelta
            Length of time interval.

        Returns
        -------
        LF : list of tuple
            (time, index, state).
        """
        # LF = [(SD[0][0], SD[0][1], True)]
        # last_sensor = SD[0][1]
        # for x in SD[1:]:
        #     LF.append((x[0], last_sensor, False))
        #     LF.append((x[0], x[1], True))
        #     last_sensor = x[1]

        LF = []
        epsilon = timedelta(
            milliseconds=1
        )  # small value to represent small time interval
        SD_i = 0
        len_SD = len(SD)
        reach_last_SD = False
        last_sensor = None
        for t in utils.date_generator(start, end, duration):
            tt = t + duration
            index_list = []
            if not (reach_last_SD):
                while t <= SD[SD_i][0] < tt:
                    index_list.append(SD[SD_i][1])
                    SD_i += 1
                    if SD_i >= len_SD:
                        reach_last_SD = True
                        break
            if (last_sensor is None) and (len(index_list) == 0):
                continue
            if len(index_list) != 0:
                last_sensor = index_list[-1]
                if len(LF) == 0:
                    LF.append((t, last_sensor, True))
                else:
                    if LF[-1][1] == last_sensor:
                        continue
                    else:
                        LF.append((t - epsilon, LF[-1][1], False))
                        LF.append((t, last_sensor, True))

        return LF

    if _type not in ["raw", "change point", "last-fired"]:
        raise ValueError(
            "undefined type of _type option in matrix_with_discretized_time_interval"
        )
    if _type == "change point":
        SD = transformation4change_point(SD)
    if _type == "last-fired":
        SD = transformation4last_fired(SD, start, end, duration)

    # check the names of sensors
    SD_names = []
    for sd in SD:
        if sd[1] not in SD_names:
            SD_names.append(sd[1])
    SD_names.sort()
    AL_names = list(AL.keys())

    SD_name2index = {name: i for (i, name) in enumerate(SD_names)}
    AL_name2index = {name: i for (i, name) in enumerate(AL_names)}

    AD = []  # convert AL into sensor-like representation
    for a in AL_names:
        for x in AL[a]:
            AD.append((x[0], a, True))
            AD.append((x[1], a, False))
    AL = sorted(AD, key=lambda x: x[0])

    mat_len = (end - start) // duration + 1
    SD_mat = np.zeros((mat_len, len(SD_names)), dtype=bool)
    AL_mat = np.zeros((mat_len, len(AL_names)), dtype=bool)

    SD_states = {v: False for v in SD_names}
    AL_states = {v: False for v in AL_names}
    SD_i, AL_i = 0, 0

    while SD[SD_i][0] < start:
        SD_i += 1
    while AL[AL_i][0] < start:
        AL_i += 1

    len_SD = len(SD)
    len_AL = len(AL)
    reach_last_SD, reach_last_AL = False, False  # whether to reach the last indexes

    _max_num = (end - start) // duration - 1
    for i, t in enumerate(utils.date_generator(start, end, duration)):
        utils.print_progress_bar(_max_num, i, "Making the sensor matrix.", step = 100000)
        tt = t + duration
        if not (reach_last_SD):
            while t <= SD[SD_i][0] < tt:
                SD_states[SD[SD_i][1]] = SD[SD_i][2]
                SD_mat[i][SD_name2index[SD[SD_i][1]]] = True
                SD_i += 1
                if SD_i >= len_SD:
                    reach_last_SD = True
                    break
        if not (reach_last_AL):
            while t <= AL[AL_i][0] < tt:
                AL_states[AL[AL_i][1]] = AL[AL_i][2]
                AL_mat[i][AL_name2index[AL[AL_i][1]]] = True
                AL_i += 1
                if AL_i >= len_AL:
                    reach_last_AL = True
                    break
        for x in SD_names:
            if SD_states[x]:
                SD_mat[i][SD_name2index[x]] = True
        for x in AL_names:
            if AL_states[x]:
                AL_mat[i][AL_name2index[x]] = True
    print()

    return (SD_mat, SD_names, AL_mat, AL_names)


def make_AL_periods(path):
    """
    This generates dictionary AL_periods.

    Parameters
    ----------
    path : pathlib.Path
        Path that includes the folder.
    """
    AL = utils.pickle_load(path, "AL")
    AL_periods = {}
    AL_periods[anomaly_model.BEING_SEMI_BEDRIDDEN] = AL[
        anomaly_model.BEING_SEMI_BEDRIDDEN
    ]
    AL_periods[anomaly_model.BEING_HOUSEBOUND] = AL[anomaly_model.BEING_HOUSEBOUND]
    AL_periods[anomaly_model.FORGETTING] = [
        (x[4], x[5]) for x in AL[anomaly_model.FORGETTING]
    ]
    AL_periods[anomaly_model.WANDERING] = [
        (x[1], x[2]) for x in AL[anomaly_model.WANDERING]
    ]
    AL_periods[anomaly_model.FALL_WHILE_WALKING] = [
        (x[0], x[0] + timedelta(seconds=x[1]))
        for x in AL[anomaly_model.FALL_WHILE_WALKING]
    ]
    AL_periods[anomaly_model.FALL_WHILE_STANDING] = [
        (x[0], x[0] + timedelta(seconds=x[1]))
        for x in AL[anomaly_model.FALL_WHILE_STANDING]
    ]
    return AL_periods


def normal_anomaly_ranges(AL_mat):
    """
    This divides time into normal range and anomaly range.
    Normal ranges are consecutive 0 sequence in AL_mat.
    Anomaly ranges are consecutive 1 sequence in AL_mat.
    For example,
    AL_mat = [1, 1, 0, 1, 1, 0, 0, 0, 0, 0].
    Normal ranges are [(2, 3), (5, 10)].
    Anomaly ranges are [(0, 2), (3, 5)].

    Parameters
    ----------
    AL_mat : numpy.ndarray
         AL_mat.shape = (number of time intervals,)
         AL_mat[i] = 1 if the anomaly occurs in the i-th time intervals, else 0.

    Returns
    -------
    (normal_range, anomaly_range) : tuple of list
    """
    normal = []
    anomaly = []
    in_range = False
    start = None
    for i, x in enumerate(AL_mat):
        if not (in_range) and x:
            start = i
            in_range = True
        if in_range and not (x):
            in_range = False
            anomaly.append((start, i))
    if AL_mat[-1] == 1:
        anomaly.append((start, len(AL_mat)))

    if len(anomaly) == 0:
        return ([0, len(AL_mat)], [])
    if anomaly[0][0] != 0:
        normal.append((0, anomaly[0][0]))
    start = anomaly[0][1]
    for i in range(1, len(anomaly)):
        (s, e) = anomaly[i]
        normal.append((start, s))
        start = e
    if anomaly[-1][1] != len(AL_mat):
        normal.append((start, len(AL_mat)))

    return (normal, anomaly)


def LF_mat2vec(LF):
    """
    This converts a data matrix into a vector under the last-fired representation [1].
    For example or 4 sensors (number 0, 1 and 2) in 7 times,
    LF =
    [[0, 0, 0, 0],
     [0, 0, 0, 0],
     [0, 0, 0, 1],
     [0, 0, 0, 1],
     [0, 0, 0, 1],
     [1, 0, 0, 0],
     [0, 1, 0, 0]]
    This exapmle is converted into below ones;
    LF = [4, 4, 3, 3, 3, 0, 1], the index of activated sensors.
    Number 4 represents the state which no sensor is activated.

    Parameters
    ----------
    LF : numpy.ndarray
        Sensor activation matrix under the last-fired representation [1].
        LF.shape = (number of times, number of sensors)
        LF[t][k] can only be in {0, 1}

    References
    ----------
    [1] TLM van Kasteren, G. Englebienne, and B. J. Krose,
    "Human activity recognition from wireless sensor network data: Benchmark and software."
    Proc of Activity recognition in pervasive intelligent environments, 2011, 165–186.

    Returns
    -------
    LF_vec : numpy.ndarray
        LF_vec.shape = (number of times, )
        LF_vec[t] can only be in {0, 1, ..., number of sensors}
        LF_vec[t] represents the activated sensors.
        The number of sensors represents the state which no sensors is activated.
    """
    LF_vec = []
    no_activated = LF.shape[1]
    for sd in LF:
        True_pos = np.where(sd)[0]
        if len(True_pos) == 1:
            LF_vec.append(True_pos[0])
        elif len(True_pos) == 0:  # No sensor is activated
            LF_vec.append(no_activated)
        else:
            raise ValueError("More than two data are activated.")
    return np.array(LF_vec)


class NaiveBayes:
    use_log_prob = True
    """
    Naive Bayes especially for binary sensors.
    Sometimes, this probability model is called as dynamic naive Bayes.
    Time slices are independent with each other.
    Output of binary sensors are independent with each other.
    Log probabilities are used.

    Abbreviations
    n : int
        Number of the states (labels).
    m : int
        Number of the binary sensors.
    C : numpy.ndarray
        State's probabilities.
        C.shape = (n, ).
    P : numpy.ndarray
        Parameters of bernoulli distributions for each state.
        P.shape = (n, m).

    s_t : Random variables of the state at time t.
    c_i : State's probability, P(s_t = i) = c_i.
    p_i_j : Parameter of bernoulli dstribution of j-th value at state i.
    x_t : Output symbol vector (length m) at time t, for now,
          P(x_t = b | s_t = i) = (p_i_1^b * (1-p_i_1)^{1-b}) * ... * (p_i_m^b * (1-p_i_m)^{1-b}).

    References
    ----------
    [1] H. H. Avilés-Arriaga et al. "A comparison of dynamic naive bayesian classifiers and
        hidden markov models for gesture recognition."
        Journal of applied research and technology, 9-1(2011), 81-102.
    [2] TL van Kasteren, G. Englebienne, and B. J. Kröse, "Human activity recognition from wireless sensor
        network data: Benchmark and software." Proc. of the Activity recognition in pervasive
        intelligent environments, Paris, Atlantis Press, 2011, 165-186.
    """

    def fit(self, observations, states):
        """
        This learns parameters of naive Bayes by maximum likelihood estimation.

        Parameters
        ----------
        observations : list of numpy.ndarray
            observations[i].shape = (number of times, number of sensors)
            The values can only be False (0) or True (1).
            If len(observations) > 1, the output parameters are the average of the estimated parameters for each data.
        states : list of numpy.ndarray
            states[i].shape = (number of times,)
            states[i] are states (labels) of observations[i].
            The length of states[i] equals to the one of observations[i].
            The values can only be {0, 1, 2, ..., state_num - 1}.
        """
        if self.use_log_prob:
            self._fit_log(observations, states)
        else:
            self._fit(observations, states)

    def _fit(self, observations, states):
        if not isinstance(observations, list):
            observations = [observations]
            states = [states]
        state_set = set()
        for s in states:
            state_set |= set(np.unique(s))
        self.n = len(state_set)
        self.m = observations[0].shape[1]
        self.C = np.zeros(self.n)
        self.P = np.zeros((self.n, self.m))
        len_o = len(observations)
        for seq, s in zip(observations, states):
            # counts[i] = the total number of occurrences of state i in the sequence
            counts = np.bincount(s)
            self.C += counts / seq.shape[0]
            for i in range(self.n):
                for j in range(self.m):
                    # The value of P[i][j] is bounded up to 2^32 for now
                    self.P[i][j] += (s == i).astype(np.uint32) @ seq[:, j] / counts[i]
        self.C /= len_o
        self.P /= len_o
        return

    def _fit_log(self, observations, states):
        if not isinstance(observations, list):
            observations = [observations]
            states = [states]
        state_set = set()
        for s in states:
            state_set |= set(np.unique(s))
        self.n = len(state_set)
        self.m = observations[0].shape[1]
        self.logC = np.array([-float("inf") for _ in range(self.n)])
        self.logP = np.full((self.n, self.m), -float("inf"))
        len_o = len(observations)
        for seq, s in zip(observations, states):
            # counts[i] = the total number of occurrences of state i in the sequence
            counts = np.bincount(s)
            self.logC = np.logaddexp(self.logC, np.log(counts) - np.log(seq.shape[0]))
            for i in range(self.n):
                for j in range(self.m):
                    self.logP[i][j] = np.logaddexp(
                        self.logP[i][j],
                        np.log((s == i).astype(np.uint32) @ seq[:, j]) - np.log(counts[i]),
                    )
        self.logC -= np.log(len_o)
        self.logP -= np.log(len_o)
        return

    def predict_states(self, observation):
        """
        This predicts states from observed sequences.
        P(state | observations, parameters) are maximized.

        Parameters
        ----------
        observation : numpy.ndarray
            observation.shape = (number of times, number of sensors)
            The values can only be False (0) or True (1).

        Returns
        -------
        (state, prob) : tuple

        state : numpy.ndarray
            states.shape = (number of times,)
            The length equals to the one of observations.
        prob : float
            Probability of the 'state'. if self.use_log_prob is True, then return log probabilities.
            prob.shape = (number of times,), because the output is independent with time.
            max P(s_1, ..., s_T, O_1, ..., O_T | parameters) or
            max log P(s_1, ..., s_T, O_1, ..., O_T | parameters).
        """
        if self.use_log_prob:
            return self._predict_states_log(observation)
        else:
            return self._predict_states(observation)

    def _predict_states(self, observation):
        len_o = len(observation)
        ret = np.zeros(len_o)
        prob = np.zeros(len_o)

        # 1 - p
        oP = np.ones_like(self.P) - self.P

        diff = 100000
        for t in range(len_o):
            if t % diff == 0:
                utils.print_progress_bar(len_o, t, "prediction in dynamic naive Bayes.")
            o_t = observation[t].astype(np.uint32)
            # p = [self.C[i] * np.prod(self.P[i]*o_t + (ones_vec - self.P[i])*(ones_vec - o_t)) for i in range(self.n)]
            p = self.C * np.prod(
                np.where(np.repeat(o_t[np.newaxis, :], self.n, axis=0), self.P, oP),
                axis=1,
            )
            ret[t] = np.argmax(p)
            prob[t] = np.max(p)
        print("")
        return ret, prob

    def _predict_states_log(self, observation):
        len_o = len(observation)
        ret = np.zeros(len_o)
        prob = np.zeros(len_o)
        # ones_vec = np.ones(self.m)

        # log(1-p)
        log1P = np.array(
            [
                [self.logsub(0, self.logP[i][j]) for j in range(self.m)]
                for i in range(self.n)
            ]
        )

        diff = 100000
        for t in range(len_o):
            if t % diff == 0:
                utils.print_progress_bar(len_o, t, "prediction in dynamic naive Bayes.")
            o_t = observation[t].astype(np.uint32)
            # logp = [self.logC[i] + np.sum(self.logP[i]*o_t + log1P[i]*(ones_vec - o_t)) for i in range(self.n)]
            # logp = self.logC + np.sum(self.logP*o_t + log1P*(ones_vec - o_t), axis = 1)  # can't work if prob = 0
            # logp = [self.logC[i] + np.sum([self.logP[i][j] if o_t[j] else log1P[i][j] for j in range(self.m)]) for i in range(self.n)] # can't work if prob = 0
            logp = self.logC + np.sum(
                np.where(
                    np.repeat(o_t[np.newaxis, :], self.n, axis=0), self.logP, log1P
                ),
                axis=1,
            )
            ret[t] = np.argmax(logp)
            prob[t] = np.max(logp)
        print("")
        return ret, prob

    @staticmethod
    def logadd(logX, logY):
        """
        This calculates log(X + Y) from log(X) and log(Y).

        Parameters
        ----------
        logX : float
        logY : float

        Returns
        -------
        ret : float
        """
        # return log(X + Y)
        if logX < logY:
            logX, logY = logY, logX
        if logX == float("inf"):
            return logX
        diff = logY - logX
        if diff < -20:
            return logX
        return logX + np.log(1 + np.exp(diff))

    @staticmethod
    def logsub(logX, logY):
        """
        This calculates log(X - Y) from log(X) and log(Y).

        Parameters
        ----------
        logX : float
        logY : float

        Returns
        -------
        ret : float
        """
        if logX < logY:
            raise ValueError("logX must be larger than logY")
        if logX == float("inf"):
            return logX
        diff = logY - logX
        if diff < -20:
            return logX
        return logX + np.log(1 - np.exp(diff))


class HMM4binary_sensors:
    use_log_prob = True
    """
    Hidden Markov model (HMM) especially for binary sensors.
    Outputs of binary sensors are independent with each other.
    Outputs of binary sensors (0 or 1) are modeled as a bernoulli distribution.

    Abbreviations
    n : Number of the hidden states.
    m : Number of the binary sensors.
    s_t : Random variables of hidden states at time t.
    c_i : Initial probability at state i, P(s_1 = i) = c_i.
    a_i_j : Transition probability from state i to state j, P(s_t = j | s_{t-1} = i) = a_i_j.
    p_i_j : Parameter of bernoulli dstribution of j-th value at state i.
    x_t : Output symbol vector (length m) at time t, for now,
          P(x_t = b | s_t = i) = (p_i_1^b * (1-p_i_1)^{1-b}) * ... * (p_i_m^b * (1-p_i_m)^{1-b}).

    n : int
        Number of hidden states.
    C : numpy.ndarray
        Initial probabilities of states.
        C.shape = (n, ).
    A : numpy.ndarray
        Transition matrix.
        A.shape = (n, n).
    P : numpy.ndarray
        Parameters of bernoulli distributions.
        P.shape = (n, m).

    References
    ----------
    [1] L. R. Rabiner, "A tutorial on hidden Markov models and selected applications in speech recognition."
        Proc. of the IEEE, 77-2(1989), 257-286.
    [2] TL van Kasteren, G. Englebienne, and B. J. Kröse, "Human activity recognition from wireless sensor
        network data: Benchmark and software." Proc. of the Activity recognition in pervasive
        intelligent environments, Paris, Atlantis Press, 2011, 165-186.
    """

    def fit_with_true_hidden_states(self, observations, states):
        if self.use_log_prob:
            self._fit_with_true_hidden_states_log(observations, states)
        else:
            self._fit_with_true_hidden_states(observations, states)
        """
        This learn parameters of HMM from the observations with true hidden states.
        Maximum likelihood estimator.

        Parameters
        ----------
        observations : list of numpy.ndarray
            observations[i].shape = (number of times, number of sensors)
            The values can only be False (0) or True (1).
            If len(observations) > 1, the output parameters are the average of the estimated parameters for each data.
        states : list of numpy.ndarray
            states[i].shape = (number of times,)
            states[i] are hidden states of observations[i].
            The length of states[i] equals to the one of observations[i].
            The values can only be {0, 1, 2, ..., state_num - 1}.
        """

    def _fit_with_true_hidden_states(self, observations, states):
        """
        This learn parameters of HMM from the observations with true hidden states.
        Maximum likelihood estimator.

        Parameters
        ----------
        observations : list of numpy.ndarray
            observations[i].shape = (number of times, number of sensors)
            The values can only be False (0) or True (1).
            If len(observations) > 1, the output parameters are the average of the estimated parameters for each data.
        states : list of numpy.ndarray
            states[i].shape = (number of times,)
            states[i] are hidden states of observations[i].
            The length of states[i] equals to the one of observations[i].
            The values can only be {0, 1, 2, ..., state_num - 1}.
        """
        if not isinstance(observations, list):
            observations = [observations]
            states = [states]
        state_set = set()
        for s in states:
            state_set |= set(np.unique(s))
        state_num = len(state_set)
        sensor_num = observations[0].shape[1]
        self.n = state_num
        self.m = sensor_num
        self.C = np.zeros(self.n)
        self.A = np.zeros((self.n, self.n))
        self.P = np.zeros((self.n, self.m))

        for seq, s in zip(observations, states):
            self.C[int(s[0])] += 1
            # counts[i] = the total number of occurrences of state i in the sequence
            counts = np.bincount(s)
            # pairs[i] = [s_{t}, s_{t+1}], 1<= t <= T-1, T: total length of the observation
            pairs = np.column_stack((s[:-1], s[1:]))
            for i in range(self.n):
                for j in range(self.n):
                    self.A[i][j] += np.sum((pairs[:, 0] == i) & (pairs[:, 1] == j)) / (counts[i] - (int(s[-1]) == i))
                for j in range(self.m):
                    # The value of P[i][j] is bounded up to 2^32 for now
                    self.P[i][j] += (s == i).astype(np.uint32) @ seq[:, j] / counts[i]
        len_o = len(observations)
        self.C /= len_o
        self.A /= len_o
        self.P /= len_o
        return

    def _fit_with_true_hidden_states_log(self, observations, states):
        """
        This learn parameters of HMM from the observations with true hidden states.
        Maximum likelihood estimator.
        This uses log probabilities.

        Parameters
        ----------
        observations : list of numpy.ndarray
            observations[i].shape = (number of times, number of sensors)
            The values can only be False (0) or True (1).
            If len(observations) > 1, the output parameters are the average of the estimated parameters for each data.
        states : list of numpy.ndarray
            states[i].shape = (number of times,)
            states[i] are hidden states of observations[i].
            The length of states[i] equals to the one of observations[i].
            The values can only be {0, 1, 2, ..., state_num - 1}.
        """
        if not isinstance(observations, list):
            observations = [observations]
            states = [states]
        state_set = set()
        for s in states:
            state_set |= set(np.unique(s))
        state_num = len(state_set)
        sensor_num = observations[0].shape[1]
        self.n = state_num
        self.m = sensor_num
        self.logC = np.array([-float("inf") for _ in range(self.n)])
        self.logA = np.full((self.n, self.n), -float("inf"))
        self.logP = np.full((self.n, self.m), -float("inf"))

        for seq, s in zip(observations, states):
            self.logC[int(s[0])] = np.logaddexp(self.logC[int(s[0])], 0)
            # counts[i] = the total number of occurrences of state i in the sequence
            counts = np.bincount(s)
            # pairs[i] = [s_{t}, s_{t+1}], 1<= t <= T-1, T: total length of the observation
            pairs = np.column_stack((s[:-1], s[1:]))
            for i in range(self.n):
                for j in range(self.n):
                    self.logA[i][j] = np.logaddexp(
                        self.logA[i][j],
                        np.log(np.sum((pairs[:, 0] == i) & (pairs[:, 1] == j)))
                        - np.log((counts[i] - (s[-1] == i))),
                    )
                for j in range(self.m):
                    self.logP[i][j] = np.logaddexp(
                        self.logP[i][j],
                        np.log((s == i).astype(np.uint32) @ seq[:, j]) - np.log(counts[i]),
                    )
        len_o = len(observations)
        self.logC -= np.log(len_o)
        self.logA -= np.log(len_o)
        self.logP -= np.log(len_o)
        return

    def predict_states(self, observation):
        """
        This predicts states from observed sequences.
        P(state | observations, parameters) are maximized by Viterbi algorithm.

        Parameters
        ----------
        observation : numpy.ndarray
            observation.shape = (number of times, number of sensors)
            The values can only be False (0) or True (1).

        Returns
        -------
        (state, prob) : tuple

        state : numpy.ndarray
            states.shape = (number of times,)
            The length equals to one of observations.
        prob : float
            Probability of the 'state'.
            If use_log_prob is True, then returns log probabilities.
            max P(s_0, ..., s_T, O_0, ..., O_T | parameters) or
            max log P(s_0, ..., s_T, O_0, ..., O_T | parameters).
        """

        if self.use_log_prob:
            return self._predict_states_log(observation)
        else:
            return self._predict_states(observation)

    def _predict_states(self, observation):
        """
        This predicts states from observed sequences.
        P(state | observations, parameters) are maximized by Viterbi algorithm.

        Parameters
        ----------
        observation : numpy.ndarray
            observation.shape = (number of times, number of sensors)
            The values can only be False (0) or True (1).

        Returns
        -------
        (state, prob) : tuple

        state : numpy.ndarray
            states.shape = (number of times,)
            The length equals to one of observations.
        prob : float
            Probability of the 'state'.
            max P(s_0, ..., s_T, O_0, ..., O_T | parameters).
        """

        len_o = len(observation)

        # delta records the quantity.
        # Let O_t be the observed value at time t.
        # delta[t][i] = max_{s_0, ... s_{t-1}}{P(s_0, ..., s_{t-1}, s_t = i, O_0, ..., O_t | parameters)}.
        # delta[t+1][j] = max_{i}{delta[t][i] * a_i_j} * P(x_{t+1} = O_{t+1} | s_t = j)
        delta = np.zeros((len_o, self.n))

        # psi records the argument which maximized delta[t+1][j]
        psi = np.zeros((len_o, self.n))

        # Viterbi algorithm

        # Initialization
        ones_vec = np.ones(self.m).astype(np.uint32)
        for i in range(self.n):
            delta[0][i] = self.C[i] * np.prod(
                np.abs((ones_vec - observation[0]) - self.P[i])
            )
            psi[0][i] = 0

        # Recursion
        diff = 100000
        for t in range(1, len_o):
            if t % diff == 0:
                utils.print_progress_bar(len_o, t, "Recursion prediction in HMM.")
            for j in range(self.n):
                score_vec = delta[t - 1] * self.A[:, j]
                delta[t][j] = np.max(score_vec) * np.prod(
                    np.abs((ones_vec - observation[t]) - self.P[j])
                )
                psi[t][j] = np.argmax(score_vec)

        # Termination
        prob = np.max(delta[-1])
        state = np.zeros(len_o, dtype=int)
        state[-1] = np.argmax(delta[-1])

        # Path backtracking
        for t in reversed(range(len_o - 1)):
            state[t] = psi[t + 1][state[t + 1]]
        print("")

        return (state, prob)

    def _predict_states_log(self, observation):
        """
        This predicts states from observed sequences using log probabilities.
        P(state | observations, parameters) are maximized by Viterbi algorithm.

        Parameters
        ----------
        observation : numpy.ndarray
            observation.shape = (number of times, number of sensors)
            The values can only be False (0) or True (1).

        Returns
        -------
        (state, prob) : tuple

        state : numpy.ndarray
            states.shape = (number of times,)
            The length equals to one of observations.
        prob : float
            Log Probability of the 'state'.
            max log P(s_0, ..., s_T, O_1, ..., O_T | parameters).
        """

        len_o = len(observation)

        # delta records the quantity.
        # Let O_t be the observed value at time t.
        # delta[t][i] = max_{s_0, ... s_{t-1}}{P(s_0, ..., s_{t-1}, s_t = i, O_1, ..., O_t | parameters)}.
        # delta[t+1][j] = max_{i}{delta[t][i] * a_i_j} * P(s_{t+1} = O_{t+1} | s_t = j)
        logdelta = np.full((len_o, self.n), -float("inf"))

        # psi records the argument which maximized delta[t+1][j]
        psi = np.zeros((len_o, self.n))

        # log(1-p)
        log1P = np.array(
            [
                [self.logsub(0, self.logP[i][j]) for j in range(self.m)]
                for i in range(self.n)
            ]
        )

        # Viterbi algorithm

        # Initialization

        # ones_vec = np.ones(self.m)
        # for i in range(self.n):
        #     _temp = np.where(observation[0], self.logP[i], log1P[i])
        #     logdelta[0][i] = self.logC[i] + np.sum(_temp)
        #     psi[0][i] = 0
        logdelta[0] = self.logC + np.sum(
            np.where(
                np.repeat(observation[0][np.newaxis, :], self.n, axis=0),
                self.logP,
                log1P,
            ),
            axis=1,
        )
        psi[0] = np.zeros(self.n)

        # Recursion
        diff = 100000
        for t in range(1, len_o):
            if t % diff == 0:
                utils.print_progress_bar(len_o, t, "Recursion prediction in HMM.")
            # for j in range(self.n):
            #     score_vec = logdelta[t-1] + self.logA[:, j]
            #     _temp = np.where(observation[t], self.logP[j], log1P[j])
            #     logdelta[t][j] = np.max(score_vec) + np.sum(_temp)
            #     psi[t][j] = np.argmax(score_vec)
            score_mat = logdelta[t - 1].reshape((self.n, 1)) + self.logA
            logdelta[t] = np.max(score_mat, axis=0) + np.sum(
                np.where(
                    np.repeat(observation[t][np.newaxis, :], self.n, axis=0),
                    self.logP,
                    log1P,
                ),
                axis=1,
            )
            psi[t] = np.argmax(score_mat, axis=0)

        # Termination
        prob = np.max(logdelta[-1])
        state = np.zeros(len_o, dtype=int)
        state[-1] = np.argmax(logdelta[-1])

        # Path backtracking
        for t in reversed(range(len_o - 1)):
            state[t] = psi[t + 1][state[t + 1]]
        print("")

        return (state, prob)

    @staticmethod
    def logadd(logX, logY):
        """
        This calculates log(X + Y) from log(X) and log(Y).

        Parameters
        ----------
        logX : float
        logY : float

        Returns
        -------
        ret : float
        """
        # return log(X + Y)
        if logX < logY:
            logX, logY = logY, logX
        if logX == float("inf"):
            return logX
        diff = logY - logX
        if diff < -20:
            return logX
        return logX + np.log(1 + np.exp(diff))

    @staticmethod
    def logsub(logX, logY):
        """
        This calculates log(X - Y) from log(X) and log(Y).

        Parameters
        ----------
        logX : float
        logY : float

        Returns
        -------
        ret : float
        """
        if logX < logY:
            raise ValueError("logX must be larger than logY")
        if logX == float("inf"):
            return logX
        diff = logY - logX
        if diff < -20:
            return logX
        return logX + np.log(1 - np.exp(diff))


class HMM4categorical:
    """
    Hidden Markov model (HMM) especially for binary sensors.
    Outputs are categorical distribution.

    Abbreviations
    n : Number of the hidden states.
    M : Number of the outputs.
    s_t : Random variables of hidden states at time t.
    c_i : Initial probability at state i, P(s_1 = i) = c_i.
    a_i_j : Transition probability from state i to state j, P(s_t = j | s_{t-1} = i) = a_i_j.
    p_i_j : Parameter of a categorical distribution of j-th value at state i.
    x_t : Output symbol at time t, for now,
          P(x_t = k | s_t = i) = p_i_k.

    n : int
        Number of hidden states.
    C : numpy.ndarray
        Initial probabilities of states.
        C.shape = (n, ).
    A : numpy.ndarray
        Transition matrix.
        A.shape = (n, n).
    P : numpy.ndarray
        Parameters of a categorical dstributions.
        P.shape = (n, M).

    References
    ----------
    [1] L. R. Rabiner, "A tutorial on hidden Markov models and selected applications in speech recognition."
        Proc. of the IEEE, 77-2(1989), 257-286.
    """

    def fit_with_true_hidden_states(self, observations, states):
        """
        This learn parameters of HMM from the observations with true hidden states.
        Maximum likelihood estimator.

        Parameters
        ----------
        observations : list of numpy.ndarray
            observations[i].shape = (number of times, )
            The values can only be {0, 1, 2, ..., M - 1}.
            If len(observations) > 1, the output parameters are the average of the estimated parameters for each data.
        states : list of numpy.ndarray
            states[i].shape = (number of times,)
            states[i] are hidden states of observations[i].
            The length of states[i] equals to the one of observations[i].
            The values can only be {0, 1, 2, ..., n - 1}.
        """
        if not isinstance(observations, list):
            observations = [observations]
            states = [states]
        state_set = set()
        for s in states:
            state_set |= set(np.unique(s))
        state_num = len(state_set)
        output_kinds = len(np.unique(observations[0]))
        self.n = state_num
        self.M = output_kinds
        self.C = np.zeros(self.n)
        self.A = np.zeros((self.n, self.n))
        self.P = np.zeros((self.n, self.M))

        for seq, s in zip(observations, states):
            seq = seq.astype(int)
            s = s.astype(int)
            self.C[s[0]] += 1
            # counts[i] = the total number of occurrences of state i in the sequence
            counts = np.bincount(s)
            # pairs[i] = (s_{t}, s_{t+1}), 1<= t <= T-1, T: total length of the obsevation
            pairs = [(s[t - 1], s[t]) for t in range(1, len(s))]
            for i in range(self.n):
                for j in range(self.n):
                    self.A[i][j] = pairs.count((i, j)) / (counts[i] - (s[-1] == i))
                for k in range(self.M):
                    self.P[i][k] = np.sum((s == i) * (seq == k)) / counts[i]
        len_o = len(observations)
        self.C /= len_o
        self.A /= len_o
        self.P /= len_o
        return

    def predict_states(self, observation):
        """
        This predicts states from observed sequences.
        P(state | observations, parameters) are maximized by Viterbi algorithm.

        Parameters
        ----------
        observation : numpy.ndarray
            observation.shape = (number of times,)
            The values can only be {0, 1, 2, ..., M - 1}.

        Returns
        -------
        (state, prob) : tuple

        state : numpy.ndarray
            states.shape = (number of times,)
            The length equals to one of observations.
        prob : float
            Probability of the 'state'.
            max log P(s_1, ..., s_T, O_1, ..., O_T | parameters).
        """

        len_o = len(observation)
        observation = observation.astype(int)

        # delta records the quantity.
        # Let O_t be the observed value at time t.
        # delta[t][i] = max_{s_0, ... s_{t-1}}{P(s_1, ..., s_{t-1}, s_t = i, O_1, ..., O_t | parameters)}.
        # delta[t+1][j] = max_{i}{delta[t][i] * a_i_j} * P(x_{t+1} = O_{t+1} | s_t = j)
        delta = np.zeros((len_o, self.n))

        # psi records the argument which maximized delta[t+1][j]
        psi = np.zeros((len_o, self.n))

        # Viterbi algorithm

        # Initialization
        for i in range(self.n):
            delta[0][i] = self.C[i] * self.P[i][observation[0]]
            psi[0][i] = 0

        # Recursion
        diff = 100000
        for t in range(1, len_o):
            if t % diff == 0:
                utils.print_progress_bar(len_o, t, "Recursion prediction in HMM.")
            for j in range(self.n):
                score_vec = delta[t - 1] * self.A[:, j]
                delta[t][j] = np.max(score_vec) * self.P[i][observation[t]]
                psi[t][j] = np.argmax(score_vec)

        # Termination
        prob = np.max(delta[-1])
        state = np.zeros(len_o, dtype=int)
        state[-1] = np.argmax(delta[-1])

        # Path backtracking
        for t in reversed(range(len_o - 1)):
            state[t] = psi[t + 1][state[t + 1]]

        return (state, prob)


class HMM_likelihood_classifier:
    """
    This classifies data by comparing likelihood between two Hidden Markov models (HMM): normal and anomaly.
    If P(observation | anomaly) > P(observation), then classifies anomaly.

    Memo
    ----
    logprob, states = HMM.decode(data)
    logprob = HMM.decode(data)
    print(HMM.transmat_)
    print(HMM.emissionprob_)
    """

    def fit(self, n_components, SD, AL):
        self.n_components = n_components
        self.n_features = SD.shape[1] + 1
        (normal_ranges, anomaly_ranges) = normal_anomaly_ranges(AL)
        self.normal_model = hmm.CategoricalHMM(
            n_components=self.n_components, n_features=self.n_features
        )
        self.anomaly_model = hmm.CategoricalHMM(
            n_components=self.n_components, n_features=self.n_features
        )
        LF_vec = LF_mat2vec(SD)

        normal_data = [LF_vec[s:e] for (s, e) in normal_ranges]
        normal_train = np.concatenate(normal_data, axis=0)
        normal_train = normal_train.reshape((len(normal_train), 1))
        normal_lengths = np.array([len(d) for d in normal_data])
        self.normal_model.fit(normal_train, normal_lengths)

        anomaly_data = [LF_vec[s:e] for (s, e) in anomaly_ranges]
        anomaly_train = np.concatenate(anomaly_data, axis=0)
        anomaly_train = anomaly_train.reshape((len(anomaly_train), 1))
        anomaly_lengths = np.array([len(d) for d in anomaly_data])
        self.anomaly_model.fit(anomaly_train, anomaly_lengths)

    def convergence_info(self):
        print("normal model")
        print(self.normal_model.monitor_)
        print(self.normal_model.monitor_.converged)
        print("anomaly model")
        print(self.anomaly_model.monitor_)
        print(self.anomaly_model.monitor_.converged)

    def predict(self, data):
        """
        This classifies a sequence into one label.

        Parameters
        ----------
        data : numpy.ndarray
            data.shape = (number of times, )

        Returns
        -------
        label : bool
            Whether the data is "anomaly".
        """
        return self.anomaly_model.score(data) > self.normal_model.score(data)

    def sequential_labeling(self, SD, w):
        """
        This predicts a sequence into a label sequence.
        The classification at time t is done by a window that is centered at t.

                       window length
                  |<------------------>|
        --------------------t--------------------------

        Parameters
        ----------
        SD : numpy.ndarray
            SD.shape = (number of times, )
        w : int
            Length of windows.
            This can only be an odd number.

        Returns
        -------
        labels : numpy.ndarray
            labels.shape = (n_t - w + 1, ), n_t = (number of times in data)
        """
        data = LF_mat2vec(SD)
        half_w = int((w - 1) / 2)
        labels = []
        total_len = int(data.shape[0] - w + 1)
        diff = 100000
        for i in range(half_w, int(data.shape[0] - half_w)):
            if i % diff == 0:
                utils.print_progress_bar(total_len, i, "Sequential labeling.")
            d = data[i - half_w : i + half_w + 1]
            d = d.reshape((len(d), 1))
            labels.append(self.anomaly_model.score(d) > self.normal_model.score(d))
        return np.array(labels)


def linear_chain_CRF():
    """
    Linear chain conditional random field (CRF).

    https://towardsdatascience.com/implementing-a-linear-chain-conditional-random-field-crf-in-pytorch-16b0b9c4b4ea
    https://qiita.com/research-PORT-INC/items/518b960e6d0d0475e1df
    https://pytorch-crf.readthedocs.io/en/stable/
    https://github.com/severinsimmler/chaine/tree/master
    https://pytorch-crf.readthedocs.io/en/stable/_modules/torchcrf.html#CRF.forward
    """

    def fit(
        self,
        data,
        labels,
    ):
        """
        This learns parameters.

        Parameters
        ----------
        data : numpy.ndarray
            data.shape = (number of times, number of features).
        labels : numpy.ndarray
            labels.shape = (number of times, ).
        """
        pass

def find_true_regions_in_ndarray(arr):
    """
    Find continuous True regions.

    Parameters
    ----------
    arr : numpy.ndarray of bool
        arr.shape = (n, ).

    Returns
    -------
    start_end_indices : list of tuple of int
        start_end_indices[i] = (index of start, index of end) of ith regions.

    Examples
    --------
    >>> find_true_regions(np.array([False, True, True, False, False, True, True, True, False, True]))
    >>> [(1, 3), (5, 8), (9, 10)]
    """
    indices = np.where(arr)[0]
    split_points = np.where(np.diff(indices) > 1)[0] + 1
    ranges = np.split(indices, split_points)
    start_end_indices = [(r[0], r[-1] + 1) for r in ranges]
    return start_end_indices


def seq2interval(seq):
    """
    Parameters
    ----------
    seq : numpy.ndarray of bool
        Labels.

    Returns
    -------
    intervals : list of tuple of int
        Intervals of consecutive True regions.
    """
    # Take the difference, indicating places of True as 1 and places of False as 0
    diff = np.diff(seq.astype(int))
    # Detect where True begins (where it becomes 1)
    starts = np.where(diff == 1)[0] + 1
    # Detect where True ends (where it becomes -1)
    ends = np.where(diff == -1)[0] + 1
    
    # If the sequence starts with True
    if seq[0]:
        starts = np.insert(starts, 0, 0)
    # If the sequence ends with True
    if seq[-1]:
        ends = np.append(ends, len(seq))
    
    # Pair the start and end points to form a list
    intervals = list(zip(starts, ends))
    
    return intervals


def nonresponse_duration(mat, pressure_sensor_indexes, time_step = 1, _type = "present", window_len = None, print_progress_bar = True):
    """
    Calculate nonresponse duration in time windows.
    A day is exclusively divided into time windows with a window's length.
    Last fired sensors at time t are last activated sensors of all sensors before time t.
    Time since the last activation (last fired elapsed time) of a sensor s at time t is the time [seconds] during which sensor s continues to be the last fired sensor.
    If the sensor s is not last fired at time t, then the elapsed time is 0.
    Based on the last fired elapsed time, nonresponse duration of sensor s in each time window is defined in the following two ways;
    sum: the sum of the last fired elapsed time of s which elapsed within that time window. So the value cannot exceed the window's length, and
    max: maximum last fired elapsed time at any point within that time window.

    Parameters
    ----------
    mat : numpy.ndarray of bool
        Raw sensor data matrix.
        mat[i][j] = j-th sensor state at i-th time.
    pressure_indexes : list of int
        Indexes of pressure sensors.
    time_step : float, default 1
        Time step length [seconds] of mat (= mat[1][0] - mat[0][0]).
    _type : str, default "max"
        Type of nonresponse duration.
        "present" : present nonresponse duration without window. 
        "sum" : total duration the sensor is last fired sensor within the time window. 
        "max" : maximum nonresponse duration at any time point within the time window.
    window_len : int
        Length of each window.
        window_len is the number of columns in mat.
        If time_step = 1 [sec.] and window_len = 60 [sec.] = 1[min.], a day is divided into 60*24 time windows.
    print_progress_bar : bool
    
    Returns
    -------
    nrd
    nrd : numpy.ndarray
        "present": nrd[i][s] = nonresponse duration [seconds] of the s-th sensor of the i-th index.
        "max" or "sum": nrd[s][i] = nonresponse duration [seconds] of the s-th sensor of the i-th time window.

    Examples
    --------
    * : fired
    window_length = 30[s]
    sensor1             *  *   *       *
    sensor2     *   *          *     *   *
    time(w=30s) |-----|-----|-----|-----|-----|-----|

    + : Last fired state
    sensor1              +++++++++++++  ++   
    sensor2      ++++++++       ++++++++  +++++++++++

    _type = "sum"
    sensor1        0    20    30    20     5     0
    sensor2       30    10    15    25    25    30  

    _type = "max"
    sensor1        0    20    50    65    10     0
    sensor2       30    40    15    40    25    55

    Check examples
    >>> test_SD = np.zeros((180, 2), dtype=bool)
    >>> sensor1_true = [40, 55, 75, 115]
    >>> sensor2_true = [0, 20, 75, 105, 125]
    >>> for ind in sensor1_true:
    >>>     test_SD[ind][0] = True
    >>> for ind in sensor2_true:
    >>>     test_SD[ind][1] = True
    >>> nrd1 = nonresponse_duration(test_SD, [], 1, "sum", 30)
    >>> nrd2 = nonresponse_duration(test_SD, [], 1, "max", 30)
    >>> print(nrd1)
    >>> print(nrd2)

    [[ 0 20 30 20  5  0]
     [30 10 15 25 25 30]]
    [[ 0 20 50 65 10  0]
     [30 40 15 40 25 55]]
    """
    if _type not in ["present", "max", "sum"]:
        raise ValueError("_type error!")
    if _type in ["max", "sum"]:
        if window_len is None:
            raise ValueError("This needs the parameter: window_len.")
        
        num_time_points = mat.shape[0]
        num_windows = num_time_points // window_len

        # Initialize the output array
        sensor_num = mat.shape[1]
        nrd = np.zeros((sensor_num, num_windows), dtype = np.uint16)  # uint16
        last_fired_time = np.full((sensor_num,), -1)
        last_fired_sensors = np.zeros(sensor_num, dtype = bool)

        for w in range(num_windows):
            window_start = w * window_len
            window_end = (w + 1) * window_len
            window_data = mat[window_start:window_end, :]

            for t in range(window_len):
                # Update last fired sensors
                if np.any(window_data[t]):
                    for s, _id in enumerate(range(sensor_num)):
                        if not last_fired_sensors[s] and window_data[t][s]:
                            last_fired_time[s] = window_start + t
                        if last_fired_sensors[s] and not window_data[t][s]:
                            last_fired_time[s] = -1
                    last_fired_sensors = window_data[t]

                # Calculate elapsed time for each sensor
                for s, _id in enumerate(range(sensor_num)):
                    if _type == "sum":
                        nrd[s][w] += (last_fired_time[s] != -1)
                    elif _type == "max":
                        elapsed = (window_start + t - last_fired_time[s] + 1) if last_fired_time[s] != -1 else 0
                        nrd[s][w] = max(nrd[s][w], elapsed)
                        
        nrd *= time_step
        return nrd
    elif _type == "present":
        nrd = np.zeros((mat.shape[0], mat.shape[1]), dtype = np.uint16)  # uint16
        # Initialize the output array
        sensor_num = mat.shape[1]
        last_fired_time = -1
        last_fired_sensors = np.ones(sensor_num, dtype = bool)
        pressure_v = np.zeros(len(pressure_sensor_indexes), dtype = np.uint16)
        ret = np.zeros(mat.shape[1], dtype = np.uint16)

        for (i, sd) in enumerate(mat):
            if print_progress_bar:
                utils.print_progress_bar(mat.shape[0] - 1, i, "Extract fall features.", step = 100000)
            
            for j, si in enumerate(pressure_sensor_indexes):
                if sd[si]:
                    pressure_v[j] += 1
                else:
                    pressure_v[j] = 0
            if np.any(sd):
                last_fired_sensors = sd
                last_fired_time = i
                ret *= 0
                for j, si in enumerate(pressure_sensor_indexes):
                    ret[si] = pressure_v[j]
                nrd[i] = ret
            else:
                ret =  (i - last_fired_time) * time_step * last_fired_sensors
                for j, si in enumerate(pressure_sensor_indexes):
                    ret[si] = pressure_v[j]
                nrd[i] = ret
        if print_progress_bar:
            print("")
        return nrd


def plot_nonresponse_duration(ax, nrt, window_duration, extra_x = None, extra_y = None):
    """
    Plot scatter plot of nonresponse time.
    x-axis : time of day
    y-axis : nonresponse time

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The Matplotlib Axes object where the plot will be drawn.
    nrt : numpy.ndarray
        nrt[i] = nonresponse time [seconds] of the i-th time window.
    window_duration : float
        Duration time [seconds] of each window.
        Assume that 24*60*60 % window_duration == 0. 
    extra_x : list of int
        Extra data point. List of numbers of window.
    extra_y : list of float
        Extra data point. List of nrt values.
        
    Examples
    --------
    >>> fig, ax = plt.subplots()
    >>> plot_nonresponse_duration(ax, nrt, 60)
    """
    sec_of_day = 86400
    if sec_of_day % window_duration !=0:
        raise ValueError(f"window_duration {window_duration} [seconds] cannot divide 24*60*60[seconds].")
    num_windows_per_day = int(sec_of_day / window_duration)
    window_centers = np.arange(window_duration / 2, sec_of_day, window_duration)
    num_days = int(len(nrt) // num_windows_per_day)

    x_data = []
    y_data = []
    for day in range(num_days):
        for window in range(num_windows_per_day):
            x_data.append(window_centers[window])
            y_data.append(nrt[day * num_windows_per_day + window])
    ax.scatter(x_data, y_data, alpha=0.2, c = 'black')
    if (extra_x is not None) and (extra_y is not None):
        ax.scatter(window_centers[extra_x], extra_y, c = "tab:orange")

    ax.set_xticks(np.arange(0, 86400, 3600), 
               [(f"{int(t/3600)%24:02d}:00") for t in np.arange(0, 86400, 3600)], 
               rotation=45)
    ax.set_xlabel("Time of day.")
    ax.set_ylabel("Nonresponse Time [seconds].")
    ax.set_ylim(bottom=0)


def nonresponse_duration_sliding(mat, time_step, window_len, _type = "max"):
    """
    Calculate nonresponse time in time windows.
    A day is divided into time windows with a sliding window.
    Last fired sensors at time t are last activated sensors of all sensors before time t.
    Time since last activation (last fired elapsed time) of sensor s at time t is the time [sec.]
    during which sensor s continues to be the last fired sensor.
    If the sensor s is not last fired at time t, then the elapsed time is 0.
    Based on the last fired elapsed time, nonresponse time of sensor s in each time window of the day
    is defined in the following two ways;
    (NRT1) The sum of the last fired elapsed time of s which elapsed within that time window.
    So the NRT1 of each time window cannot exceed windows length.
    (NRT2) Maximum last fired elapsed time at any point within that time window.

    Parameters
    ----------
    mat : nummpy.ndarray of bool
        Raw sensor data matrix.
        mat[i][j] = j-th sensor state at i-th time.
    time_step : float
        Time step length [seconds] of mat = mat[1][0] - mat[0][0].
    window_len : int
        Length of window that divide time exclusively.
        window_len is the number of columns in mat.
        window_len must be an odd number.
    _type : str, default "max"
        Type of nonresponse time.
        "sum" : total nonresponse time within the time window. Max = window_len.
        "max"      : maximum nonresponse time at any time point within the time window.
    
    Returns
    -------
    nrt
    nrt : numpy.ndarray
        nrt[s][i] = nonresponse time [seconds] of the s-th sensor of the i-th time window.
        nrt.shape[0] == mat.shape[0].
        Let w be a (window_len - 1)/2.
        nrt.shape[i] is a vector made from mat[i-w/2:i+w/2+1] for any w/2 <= i <mat.shape[0]-w/2
        nrt.shape[i] is a vector made from mat[0:i+w/2+1] for any i < w/2.
        nrt.shape[i] is a vector made from mat[i-w/2:mat.shape[0]] for any i >= mat.shape[0]-w/2.
        nrt.shape[0] does not equal to mat.shape[1] because cost sensors are removed.
    """

    def summarize_sliding_window(window_data, nrt_type):
        if nrt_type == "sum":
            return np.count_nonzero(window_data, axis=1)
        elif nrt_type == "max":
            return np.max(window_data, axis = 1)
        
    if _type not in ["max", "sum"]:
        raise ValueError("_type error!")
    if window_len % 2 == 0:
        raise ValueError("window_len must be an odd number!")

    # Initialize the output array
    sensor_num = mat.shape[1]
    nrt = np.zeros((sensor_num, mat.shape[0]), dtype = np.int32)
    last_fired_time = np.full((sensor_num,), -1)
    last_fired_sensors = np.zeros(sensor_num, dtype = bool)
    
    w = int((window_len - 1)/2)
    window_data = np.zeros((sensor_num, window_len))

    for i in range(w):
        sd = mat[i]
        if np.any(sd):
            for s, _id in enumerate(range(sensor_num)):
                if not last_fired_sensors[s] and sd[s]:
                    last_fired_time[s] = i
                if last_fired_sensors[s] and not sd[s]:
                    last_fired_time[s] = -1
            last_fired_sensors = sd

        instantaneous_nrt = np.empty(sensor_num)
        for s, _id in enumerate(range(sensor_num)):
            instantaneous_nrt[s] = ((i - last_fired_time[s] + 1) if last_fired_time[s] != -1 else 0)
        window_data[:, i+w+1] = instantaneous_nrt

    # [0, 0, 0, a, b] for example of w = 2.

    for i in range(w, mat.shape[0]):
        center = i - w  # index of window center
        sd = mat[i]
        if np.any(sd):
            for s, _id in enumerate(range(sensor_num)):
                if not last_fired_sensors[s] and sd[s]:
                    last_fired_time[s] = i
                if last_fired_sensors[s] and not sd[s]:
                    last_fired_time[s] = -1
            last_fired_sensors = sd

        instantaneous_nrt = np.empty(sensor_num)
        for s, _id in enumerate(range(sensor_num)):
            instantaneous_nrt[s] = ((i - last_fired_time[s] + 1) if last_fired_time[s] != -1 else 0)

        # shift window
        # For example of w = 2;
        # from [a, b, c, d, e] to [b, c, d, e, f]
        window_data = np.roll(window_data, shift=-1, axis=1)
        window_data[:, -1] = instantaneous_nrt

        # Calculate elapsed time of the sliding window
        nrt[:, center] = summarize_sliding_window(window_data, _type)

    for center in range(mat.shape[0]-w, mat.shape[0]):
         # shift window
        window_data = np.roll(window_data, shift=-1, axis=1)
        window_data[:, -1] = np.zeros(sensor_num)

        # Calculate elapsed time of the sliding window
        nrt[:, center] = summarize_sliding_window(window_data, _type)

                    
    nrt *= time_step
    return nrt

def original_metrics(y_true, y_pred, threshold = 5, use_fixed_alarm = False, alarm_length = 5 * 60):
    """
    Parameters
    ----------
    y_true : numpy.ndarray
        True labels
    y_pred : numpy.ndarray
        Predicted labels.
    threshold : int, default 5
        Predicted label intervals whose length is less than the threshold will be deleted.
    use_fixed_alarm : bool, default False
        Whether to use fixed alarm.
    alarm_length : int, default 5 * 60
        Length of the alarm.

    Returns
    -------
    (sensitivity, false_alarm, mean_alarm_length)
    sensitivity : float
        The number of true label interval that is overlapped with a predicted label interval.
    false_alarm : float
        The number of predicted labels that is not overlapped with any true label intervals.
    mean_alarm_length : float
        The mean length of predicted label interval.
        
    Notes
    -----
    Label interval is a consecutive labels which continues to be True.
    Label intervals whose length is less than threshold are replaced 0s.
    """

    true_intervals = seq2interval(y_true)
    pred_intervals = seq2interval(y_pred)
    pred_intervals = [x for x in pred_intervals if x[1] - x[0] >= threshold]
    if use_fixed_alarm:
        filtered_y_pred = np.zeros(y_pred.shape)
        for start, end in pred_intervals:
            filtered_y_pred[start:end] = True
        i = 0
        _max = filtered_y_pred.shape[0]
        while i < _max:
            if filtered_y_pred[i]:
                filtered_y_pred[i:min(i+alarm_length, _max)] = True
                i += alarm_length
            else:
                i += 1
        pred_intervals = seq2interval(filtered_y_pred)
 
    overlapping_count = 0
    for true_start, true_end in true_intervals:
        for pred_start, pred_end in pred_intervals:
            if true_start <= pred_end and pred_start <= true_end:
                overlapping_count += 1
                break
    sensitivity = overlapping_count / len(true_intervals)

    false_alarm = 0
    for pred_start, pred_end in pred_intervals:
        overlap = False
        for true_start, true_end in true_intervals:
            if pred_start <= true_end and true_start <= pred_end:
                overlap = True
                break
        if not overlap:
            false_alarm += 1
 
    mean_alarm_length = sum([x[1] - x[0] for x in pred_intervals]) / len(pred_intervals)
    return (sensitivity, false_alarm, mean_alarm_length)


def print_decision_path(dt, data, marge_rules=True):
    """
    Print the decision path for given data using a trained DecisionTreeClassifier.

    Parameters
    ----------
    dt : DecisionTreeClassifier
        A trained DecisionTreeClassifier.
    data : numpy.ndarray
        Data to determine the decision path for.
    marge_rules : bool, default True
        Whether to merge rules for readability.

    Examples
    --------
    from sklearn.datasets import load_iris
    from sklearn.tree import DecisionTreeClassifier
    iris = load_iris()
    X, y = iris.data, iris.target
    dt = DecisionTreeClassifier()
    dt.fit(X, y)
    print_decision_path(dt, X, marge_rules=False)
    print("")
    print_decision_path(dt, X)
    """
    feature = dt.tree_.feature
    threshold = dt.tree_.threshold
    node_indicator = dt.decision_path(data)
    leaf_id = dt.apply(data)
    label = dt.predict(data)

    for sample_id in range(data.shape[0]):
        # Get node index path for the sample
        node_index = node_indicator.indices[node_indicator.indptr[sample_id]:node_indicator.indptr[sample_id + 1]]
        rules = {}
        
        for node_id in node_index:
            if leaf_id[sample_id] == node_id:
                break

            # Feature and threshold for the current node
            f = feature[node_id]
            t = threshold[node_id]
            condition = data[sample_id, f]

            if f not in rules:
                rules[f] = []
            
            # Update rule based on the condition
            if condition <= t:
                rules[f].append(f"(X[{f}] <= {t:.2f})")
            else:
                rules[f].append(f"({t:.2f} < X[{f}])")

        # Simplify and print rules
        if marge_rules:
            simplified_rules = {}
            for f, conditions in rules.items():
                v = data[sample_id, f]
                # Find range for conditions if mergeable

                if len(conditions) == 1:
                    c = conditions[0]
                    if "<=" in c:
                        th = float(c.split(" ")[-1][:-1])
                        simplified_rules[f] = f"(X[{f}]({v:.2f}) <= {th:.2f})"
                    elif ("<" in c) and ("=" not in c):
                        th = float(c.split(" ")[0][1:])
                        simplified_rules[f] = f"({th:.2f} < X[{f}]({v:.2f}))"
                else:
                    # Find min and max threshold for the feature
                    upper_bounds = [float(c.split(" ")[-1][:-1]) for c in conditions if "<=" in c]
                    lower_bounds = [float(c.split(" ")[0][1:]) for c in conditions if ("<" in c) and ("=" not in c)]
                    if lower_bounds:
                        min_threshold = max(lower_bounds)
                    if upper_bounds:
                        max_threshold = min(upper_bounds)
                    if lower_bounds and upper_bounds:
                        simplified_rules[f] = f"({min_threshold:.2f} < X[{f}]({v:.2f}) <= {max_threshold:.2f})"
                    elif lower_bounds:
                        simplified_rules[f] = f"({min_threshold:.2f} < X[{f}]({v:.2f}))"
                    elif upper_bounds:
                        simplified_rules[f] = f"(X[{f}]({v:.2f}) <= {max_threshold:.2f})"
            simplified_rules = {k: simplified_rules[k] for k in sorted(simplified_rules)}
            rule_str = ' and '.join(simplified_rules.values())
        else:
            # Print without merging
            rule_str = " and ".join([cond for rule in rules.values() for cond in rule])
        print(f"Sample {sample_id}: {rule_str}, predict {label[sample_id]}")


def print_positive_path(dt, marge_rules = True, name_dict = None, target_class = 1):
    """
    Print the decision rules to classify data into positive class for a binary classification decision tree.

    Parameters
    ----------
    dt : DecisionTreeClassifier
        A trained DecisionTreeClassifier.
    marge_rules : bool, default True
        Whether to marge rules.
        For example, 
        False: (X[19] > 10.50) and (X[29] > 11.50) and (X[19] > 13.50) and (X[29] > 14.50) and (X[19] > 16.50) and (X[29] <= 17.50)
        True: (13.50 < X[19]) and (16.50 < X[29] <= 17.50)
    name_dict : dict, default None
        Convert the name of the features.
        keys: the index of the feature
        values: name str
    target_class : int
        

    Examples
    --------
    from sklearn.datasets import load_iris
    from sklearn.tree import DecisionTreeClassifier

    iris = load_iris()
    For now, class 1 is regarded as a positive class
    X, y = iris.data, iris.target == 1
    dt = DecisionTreeClassifier()
    dt.fit(X, y)
    print_positive_path(dt, marge_rules=False)
    print("")
    print_positive_path(dt)
    """
    children_left = dt.tree_.children_left
    children_right = dt.tree_.children_right
    features = dt.tree_.feature
    thresholds = dt.tree_.threshold
    
    if name_dict is None:
        name_dict = dict()
        for i in range(max(features) + 1):
            name_dict[i] = f"X[{i}]"
    # key: node ID, value: all path to the node
    
    if not marge_rules:
        def recurse(node, rules):
            """
            Search nodes to leaf nodes recursively
            """
            if children_left[node] == children_right[node]:  # leaf node
                if dt.tree_.value[node].argmax():  # When the positive class
                    print(" and ".join(rules))
            else:
                # add a rule to the left node
                left_rule = f"({name_dict[features[node]]} <= {thresholds[node]:.2f})"
                recurse(children_left[node], rules + [left_rule])
                
                # add a rule to the right node
                right_rule = f"({thresholds[node]:.2f} < {name_dict[features[node]]})"
                recurse(children_right[node], rules + [right_rule])
        
        recurse(0, [])

    else:
        def recurse(node, rules):
            """
            Search nodes to leaf nodes recursively and simplify rules.
            """
            if children_left[node] == children_right[node]:  # leaf node
                # Simplify rules
                simplified_rules = {}
                for feature, rule in rules.items():
                    if len(rule) == 1:
                        simplified_rules[feature] = rule[0]
                    else:
                        # Find min and max threshold for the feature
                        upper_bounds = [float(r.split(" ")[-1][:-1]) for r in rule if "<=" in r]
                        lower_bounds = [float(r.split(" ")[0][1:]) for r in rule if ("<" in r) and ("=" not in r)]
                        if lower_bounds:
                            min_threshold = max(lower_bounds)
                        if upper_bounds:
                            max_threshold = min(upper_bounds)
                        if lower_bounds and upper_bounds:
                            simplified_rules[feature] = f"({min_threshold:.2f} < {name_dict[feature]} <= {max_threshold:.2f})"
                        elif lower_bounds:
                            simplified_rules[feature] = f"({min_threshold:.2f} < {name_dict[feature]})"
                        elif upper_bounds:
                            simplified_rules[feature] = f"({name_dict[feature]} <= {max_threshold:.2f})"
                
                simplified_rules = {k: simplified_rules[k] for k in sorted(simplified_rules)}
                # Print simplified rules for positive class leaves
                if dt.tree_.value[node].argmax() == target_class:  # positive class
                    print(" and ".join(simplified_rules.values()) + "\n")
                return
            
            # Rule for left child
            feature = features[node]
            threshold = f"{thresholds[node]:.2f}"
            if feature not in rules:
                rules[feature] = []
            rules[feature].append(f"({name_dict[feature]} <= {threshold})")
            recurse(children_left[node], copy.deepcopy(rules))  # Use copy of rules for branching
            
            # Rule for right child
            rules[feature][-1] = f"({threshold} < {name_dict[feature]})"
            recurse(children_right[node], copy.deepcopy(rules))

        recurse(0, {})


def estimate_sleep_duration(data, bed_sensor_id_list, sensors, step=timedelta(seconds=1)):
    """
    Parameters
    ----------
    data : numpy.ndarray
    bed_sensor_id_list : list of int
    sensors : list of Sensor
    step : datetime.timedelta
        Length of time interval in data.

    Returns
    -------
    counts : list of float
        counts[i] is the duration time [hour] of sleep in the ith day.
    """

    def ind2day(i, step):
        return timedelta(seconds=i * (step / timedelta(seconds=1))).days

    cost_sensor_id = []
    for s in sensors:
        if isinstance(s, sensor_model.CostSensor):
            cost_sensor_id.append(s.index)

    _day = 0
    max_day = ind2day(data.shape[0], step)
    counts = []
    past_time = None
    _time = timedelta(days=0)
    for i, x in enumerate(data):
        detected = False
        for _id in bed_sensor_id_list:
            detected = detected or x[_id]
        if detected:
            present_time = timedelta(seconds=i * (step / timedelta(seconds=1)))
            if past_time is not None:
                diff = present_time - past_time
                if diff > timedelta(minutes=1):
                    _time += diff
            past_time = present_time
        else:
            other_sensors = x.copy()
            for _id in bed_sensor_id_list:
                other_sensors[_id] = False
            for _id in cost_sensor_id:
                other_sensors[_id] = False
            if np.any(other_sensors):
                past_time = None
        new_day = ind2day(i, step)

        if new_day != _day:
            utils.print_progress_bar(max_day, new_day, "estimate sleep duration a day")
            if past_time is not None:
                present_time = timedelta(seconds=i * (step / timedelta(seconds=1)))
                _time += present_time - past_time
                past_time = present_time
            counts.append(_time)
            _time = timedelta(days=0)
            for _ in range(_day, new_day - 1):
                counts.append(_time)
            _day = new_day

    if _day != max_day:
        counts.append(_time)

    counts = [d / timedelta(hours=1) for d in counts]
    return counts


def estimate_go_out_freq(data, door_sensor_id, sensors, step=timedelta(seconds=1)):
    """
    Parameters
    ----------
    data : numpy.ndarray
    door_sensor_id : int
    sensors : list of Sensor
    step : datetime.timedelta
        Length of time interval in data.

    Returns
    -------
    counts : list of int
        Counts[i] is the number of go out in ith day.
    """

    def ind2day(i, step):
        return timedelta(seconds=i * (step / timedelta(seconds=1))).days

    _day = 0
    max_day = ind2day(data.shape[0], step)
    counts = []
    past_time = None
    num = 0

    cost_sensor_id = []
    for s in sensors:
        if isinstance(s, sensor_model.CostSensor):
            cost_sensor_id.append(s.index)

    for i, x in enumerate(data):
        if x[door_sensor_id]:
            present_time = timedelta(seconds=i * (step / timedelta(seconds=1)))
            if past_time is not None:
                if present_time - past_time > timedelta(minutes=1):
                    num += 1
            past_time = present_time
        else:
            other_sensors = x.copy()
            other_sensors[door_sensor_id] = False
            for _id in cost_sensor_id:
                other_sensors[_id] = False
            if np.any(other_sensors):
                past_time = None
        new_day = ind2day(i, step)
        if new_day != _day:
            utils.print_progress_bar(
                max_day, new_day, "estimate go out frequency a day"
            )
            counts.append(num)
            num = 0
            for _ in range(_day, new_day - 1):
                counts.append(num)
            _day = new_day

    if _day != max_day:
        counts.append(num)

    return counts


def detect_subregions_via_threshold(data, threshold, _len, _type="below"):
    """
    Detect subregions in a list of numerical data where values are continuously below
    the threshold 'threhold' for at least '_len' consecutive times.

    Parameters
    ----------
    data : list
        List of numerical data.
    threshold : float
        Threshold value for comparison.
    _len : int
        Minimum consecutive times below threshold to consider a subregion.
    _type : str
        'below' :
        'above' :

    Returns
    -------
    ret : list
        A list of lists containing the indices of detected subregions.

    Examples
    --------
    >>> data = [2, 3, 1, 0, 4, 5, 0, 0, 0, 6, 7, 8, 0, 9, 0, 1, 2, 0]
    >>> threshold = 3
    >>> min_length = 3
    >>> result = detect_subregions_below_threshold(data, threshold, min_length)
    >>> for region in result:
    >>>     print(region)  # Print indices of detected subregions

    [6, 7, 8]
    [14, 15, 16, 17]
    """
    subregions = []
    current_subregion = []

    if _type == "below":
        for i, value in enumerate(data):
            if value < threshold:
                current_subregion.append(i)
            else:
                if len(current_subregion) >= _len:
                    subregions.append(current_subregion)
                current_subregion = []

        if len(current_subregion) >= _len:
            subregions.append(current_subregion)

    if _type == "above":
        for i, value in enumerate(data):
            if value > threshold:
                current_subregion.append(i)
            else:
                if len(current_subregion) >= _len:
                    subregions.append(current_subregion)
                current_subregion = []

        if len(current_subregion) >= _len:
            subregions.append(current_subregion)

    return subregions


def extract_semibedridden_and_housebound(days, AL_periods):
    # daily labels of semi bedridden
    true_housebound_labels = [False for _ in range(days)]
    true_semi_bedridden_labels = [False for _ in range(days)]
    true_housebound_days = []
    true_semi_bedridden_days = []
    for period in AL_periods[anomaly_model.BEING_HOUSEBOUND]:
        true_housebound_days += list(range(period[0].days, period[1].days + 1))
    for period in AL_periods[anomaly_model.BEING_SEMI_BEDRIDDEN]:
        true_semi_bedridden_days += list(range(period[0].days, period[1].days + 1))
    for i in range(days):
        if i in true_housebound_days:
            true_housebound_labels[i] = True
        if i in true_semi_bedridden_days:
            true_semi_bedridden_labels[i] = True
    return true_semi_bedridden_labels, true_housebound_labels


def labeling_semi_bedridden(sleep_duration, threshold):
    days = len(sleep_duration)
    estimated_semi_bedridden_labels = [False for _ in range(days)]
    estimated_semi_bedridden_days = []
    # estimate semi bedridden
    regions = detect_subregions_via_threshold(
        sleep_duration, threshold, 7, _type="above"
    )
    for region in regions:
        estimated_semi_bedridden_days += region
    for i in range(days):
        if i in estimated_semi_bedridden_days:
            estimated_semi_bedridden_labels[i] = True
    return estimated_semi_bedridden_labels


def labeling_housebound(go_out_counts, semi_bedridden_labels, threshold):
    days = len(go_out_counts)
    estimated_housebound_labels = [False for _ in range(days)]
    estimated_housebound_days = []
    regions = detect_subregions_via_threshold(
        go_out_counts, threshold, 7, _type="below"
    )
    for region in regions:
        for r in region:
            if not semi_bedridden_labels[r]:
                estimated_housebound_days.append(r)
    for i in range(days):
        if i in estimated_housebound_days:
            estimated_housebound_labels[i] = True
    return estimated_housebound_labels


def extract_forgetting_features_of_a_cost_sensor(path, sensor_index, seconds_of_step = 60 * 60 * 2):
    """
    Parameters
    ----------
    path: str
        Path to data
    sensor_index: int
        Index of the cost sensor
    seconds_of_step: int
        Time interval for measuring appliance sensor usage.

    Returns
    -------
    (X_normal, X_anomaly): tuple of numpy.ndarray
    X_normal: numpy.ndarray
        Normal data of the sensor activation.
        The matrix size is (number of time intervals, number of features).
        There are three features for now;
        (1) Total duration time [minutes] of the sensor activation in the time interval.
        (2) Maximum distance between the sensor and another sensor activated during the time interval.
        (3) Midtime of the time interval.
    X_anomaly: numpy.ndarray
        Anomaly data of the sensor activation during anomaly 'forgetting'.
    """

    def find_true_sequences(arr):
        start = None
        intervals = []

        for i, val in enumerate(arr):
            if val and start is None:
                start = i
            elif not val and start is not None:
                intervals.append((start, i))
                start = None

        if start is not None:
            intervals.append((start, len(arr)))

        return intervals

    SD = utils.pickle_load(path / "experiment", "SD_mat_raw_1")
    cost_sensor_data = SD[:, sensor_index]
    SD_model = utils.pickle_load(path, "SD_model")
    AL = utils.pickle_load(path, "AL")
    sensor_name = SD_model[sensor_index].name
    sensor_x, sensor_y = SD_model[sensor_index].x, SD_model[sensor_index].y
    

    distance_dict = {}  # distance from other sensors
    for s in SD_model:
        distance_dict[s.index] = np.sqrt((s.x - sensor_x) ** 2 + (s.y - sensor_y) ** 2)
    forgetting = []
    for f in AL[anomaly_model.FORGETTING]:
        if f[2] == sensor_name:
            forgetting.append(utils.TimeInterval(f[4], f[5]))

    
    seconds_of_day = 24 * 60 * 60
    end_seconds = cost_sensor_data.shape[0]

    normal_data_minutes = []
    normal_data_distance = []
    normal_data_mid_time = []
    anomaly_data_minutes = []
    anomaly_data_distance = []
    anomaly_data_mid_time = []
    for i in range(0, end_seconds, seconds_of_step):
        j = i + seconds_of_step
        hour_interval = cost_sensor_data[i:j]
        minutes = np.sum(hour_interval) / 60
        first_true_index = np.argmax(hour_interval)
        last_true_index = len(hour_interval) - np.argmax(hour_interval[::-1]) - 1
        mid_index = int((first_true_index + last_true_index) / 2)
        mid_time = (i + mid_index) % seconds_of_day
        present_interval = utils.TimeInterval(
            timedelta(seconds=i), timedelta(seconds=j)
        )

        true_intervals = find_true_sequences(hour_interval)
        max_distance = 0
        for interval in true_intervals:
            sensor_mat = SD[i + interval[0] : i + interval[1]]
            sensor_has_true = np.any(sensor_mat, axis=0)
            sensor_indices_with_true = np.where(sensor_has_true)[0]
            for k in sensor_indices_with_true:
                if distance_dict[k] > max_distance:
                    max_distance = distance_dict[k]
        anomaly_flag = False
        for f in forgetting:
            if f.overlap(present_interval):
                anomaly_flag = True
                break
        if anomaly_flag:
            anomaly_data_minutes.append(minutes)
            anomaly_data_mid_time.append(mid_time)
            anomaly_data_distance.append(max_distance)
        else:
            normal_data_minutes.append(minutes)
            normal_data_mid_time.append(mid_time)
            normal_data_distance.append(max_distance)

    X_normal = np.column_stack(
        (normal_data_minutes, normal_data_distance, normal_data_mid_time)
    )
    X_anomaly = np.column_stack(
        (anomaly_data_minutes, anomaly_data_distance, anomaly_data_mid_time)
    )
    return X_normal, X_anomaly
