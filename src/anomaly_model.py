import os
import random
from datetime import timedelta

import numpy as np
import scipy.stats as stats

import src.activity_model as activity_model
import src.utils as utils

MMSE = "MMSE"
BEING_SEMI_BEDRIDDEN = "being semi-bedridden"
BEING_HOUSEBOUND = "being housebound"
FORGETTING = "forgetting"
WANDERING = "wandering"
FALL_WHILE_WALKING = "fall while walking"
FALL_WHILE_STANDING = "fall while standing"


def simulate_MMSE(
    start, end, step, init_v=29.5, g=-((29.0 - 19.5) / 9.0), error_s=None, error_e=0
):
    """
    This simulates MMSE as the criterion about dementia from a simple state space model.
    Let x_t as true MMSE value in time t,
        x_0 as initial MMSE value,
        y_t as MMSE value with noise in time t,
        s as interval times [year] between t and t+1, e.g. s = 1/12 means 1 month equals to duration between x_t and x_{t+1}, or x increments its index if 1 month pass,
        g as a gradient of anually change of MMSE.
    Then calculate x_t and y_t as
        x_0 = initial_value,
        x_{t+1} = x_t + c + m_t, m_t ~ Normal(mean = 0, standard validation =    state error scale),
        y_t = x_t + e_t,         e_t ~ Normal(mean = 0, standard validation = emission error scale),
        c/s = g.

    Parameters
    ----------
    start : datetime.timedelta
        Start date.
    end : datetime.timedelta
        End date.
    step : datetime.timedelta
        Time step of MMSE.
    init_v : float
        Initial value of MMSE.
    g : float
        Gradient of anually change of MMSE.
    error_s : float
        Standard deviation of state errors.
    error_e : float
        Standard deviation of emission errors.

    Returns
    -------
    MMSE : list of tuple
        MMSE[i] = (s, e, mmse)
        s : datetime.timedelta
            Start date and time.
        e : datetime.timedelta
            End date and time.
        mmse : float
            MMSE value between s and e.
        MMSE[0] may become (start, start + step, initial_value).
        Each duration between s and e equals to step.

    Notes
    -----
    For now, suppose 1 month as 30 days and 1 year as 30 * 12 days.
    This supposes a patient who become (mild) dementia and reduce MMSE score monotonically and linearly.

    Raises
    ------
    Input 'start' must be earlier than input 'end'.

    Examples
    --------
    If you want MMSE values per month during 9 year then,
    >>> start, end, step = timedelta(days = 0), timedelta(days = 9*360), timedelta(days = 30)
    >>> MMSE = anomaly.simulate_MMSE(start, end, step)

    If you want MMSE values without emission error then,
    >>> start, end, step = timedelta(days = 0), timedelta(days = 9*360), timedelta(days = 30)
    >>> MMSE = anomaly.simulate_MMSE(start, end, step, error_e = 0)

    """

    # Value error
    if start > end:
        raise ValueError("Input start must be earlier than input end!")

    mmse = init_v
    MMSE = [(start, start + step, init_v)]
    c = g * (step / timedelta(days=360))

    if error_s is None:
        error_s = abs(c)

    for t in utils.date_generator(start + step, end, step):
        m, e = np.random.normal(loc=0, scale=error_s), np.random.normal(
            loc=0, scale=error_e
        )
        mmse += c + m
        MMSE.append((t, t + step, min(30, max(0, mmse + e))))  # 0 <= MMSE <= 30

    return MMSE


def simulate_values_from_MMSE(MMSE, start, end, step, func):
    """
    This simulates the value that is made by function using MMSE.

    Parameters
    ----------
    MMSE : list of tuple
        MMSE[i] = (start, end, mmse)
        s : datetime.timedelta
            Start date and time.
        e : datetime.timedelta
            End date and time.
        mmse : float
            MMSE value between s and e.
        MMSE[0] may become (start, start + step, initial_value).
        Each duration between s and e equals to step.
    start : datetime.timedelta
        Start date.
    end : datetime.timedelta
        End date.
    step : datetime.timedelta
        Time step.
    func : function
        Function that converts MMSE into the target value.

    Returns
    -------
    values : list of tuple
        values[i] = (s, e, v)
        s : datetime.timedelta
            Start date and time.
        e : datetime.timedelta
            End date and time.
        v : float
            Target value between s and e.
        Each duration between s and e equals to 'step'.

    Notes
    -----
    For now, suppose 1 month as 30 days and 1 year as 30 * 12 days.

    Raises
    ------
    Input 'start' must be earlier than input 'end' and must be MMSE[0][0] <= start < end <= MMSE[-1][1].
    """

    # Value error
    if start > end:
        raise ValueError("Input start must be earlier than input end!")
    if start < MMSE[0][0] or MMSE[-1][1] < end:
        print(
            "start:{}\nMMSE[0][0]:{}\nMMSE[-1][1]:{}\nend:{}\n".format(
                start, MMSE[0][0], MMSE[-1][0], end
            )
        )
        raise ValueError(
            "The input range (from start to end) is out of range of MMSE! Those must be MMSE[0][0] <= start < end <= MMSE[-1][1]."
        )

    values = []
    for t in utils.date_generator(start, end, step):
        mmse = [x[2] for x in MMSE if x[0] <= t][-1]
        values.append((t, t + step, func(mmse)))

    return values


def simulate_state_anomaly_periods(MMSE, mean_freq, mean_d, sd_d):
    """
    This decides the periods of state anomalies, e.g., being housebound and being semi-bedridden.

    Parameters
    ----------
    MMSE : list of tuple
        MMSE[i] = (start, end, mmse)
        s : datetime.timedelta
            Start date and time.
        e : datetime.timedelta
            End date and time.
        mmse : float
            MMSE value between s and e.
        MMSE[0] may become (start, start + step, initial_value).
        Each duration between s and e equals to step.
    mean_freq : float
        Mean frequency of each period.
    mean_d : float
        Mean of duration time [day] of a state anomaly.
    mean_d : float
        Standard deviation of duration time [day] of a state anomaly.

    Returns
    -------
    periods : list of tuple
        Periods of state anomalies.
        periods[i] = (s, e)
        s : datetime.timedelta
            Start date and time.
        e : datetime.timedelta
            End date and time.
        Each duration between s and e equals to 'step'.

    Notes
    -----
    For now, suppose 1 month as 30 days and 1 year as 30 * 12 days.

    Raises
    ------
    Input 'start' must be earlier than input 'end' and must be MMSE[0][0] <= start < end <= MMSE[-1][1].
    """
    ret = []
    for interval in MMSE:
        s, e = interval[0].days, interval[1].days
        days_list = list(range(s, e))
        n = stats.poisson.rvs(mean_freq)
        for _ in range(n):
            s_day = random.choice(days_list)
            d = np.random.normal(mean_d, sd_d)
            ret.append(
                (timedelta(days=s_day), timedelta(days=s_day) + timedelta(days=d))
            )
    return ret


def forgetting_labels(AS, WT, forgetting_num):
    """
    This generates labels of forgetting.
    Forgetting activities are selected randomly from forgetting_activities.

    Parameters
    ----------
    AS : list of ActivityDataPoint
        Time series data of activities.
    WT : list of WalkingTrajectory
        Walking trajectories.
        len(WT) = len(AS) - 1.
    forgetting_num : list of tuple
        forgetting_num[i] = (s, e, v)
        s : datetime.timedelta
            Start date and time.
        e : datetime.timedelta
            End date and time.
        v : float
            The number of forgetting occurs between s and e.

    Returns
    -------
    labels : list of tuple
        Labels of forgetting.
        labels[i] = (forgetting_ind, notice_ind, name, cost, start_time, end_time)
        forgetting_ind : int
            Index of an activity after which a resident will be forgetting.
        notice_ind : int
            Index of an activity that a resident notice forgetting and turn off the home appliance.
        name : str
            Name of a home appliance the resident forget to turn off.
        cost : float
            Cost of the home appliance per hour.
        start_time : datetime.timedelta
            Start time of forgetting, so the home appliance is left on.
        end_time : datetime.timedelta
            End time of forgetting, so the home appliance is turned off.

    Notes
    -----
    For now, suppose 1 month as 30 days and 1 year as 30 * 12 days.

    Raises
    ------
    If the range of forgetting_num differs from AS.
    If the number of candidates at each period is smaller than the number of forgetting at the period, stop to assign forgetting during the period and print it's warning.
    """

    if forgetting_num[-1][1] < AS[0].start or AS[-1].end < forgetting_num[0][0]:
        print(
            "Some periods of forgetting_num are out of range of the activity sequence."
        )
    forgetting_activities = [
        act.activity.name for act in AS if act.activity.home_equipment != {}
    ]

    # determination of labels[i][0]s
    error_periods = (
        []
    )  # the ranges will be added into this if the number of candidates at each period is smaller than the number of forgetting at the period.
    indexes = (
        []
    )  # indexes of activities that occurs forgetting after the activity finish
    act_start, act_end = AS[0].start, AS[-1].end

    # forgetting_index = 0
    # x = forgetting_num[forgetting_index]
    # candidates = []
    # for (i, act) in enumerate(AS):
    #     if x[0] <= act.start <= x[1]:
    #         if act.activity.name in forgetting_activities:
    #             candidates.append(i)
    #     else:
    #         if len(candidates) < x[2]:
    #             error_periods.append((x[0], x[1]))
    #         else:
    #             indexes += random.sample(candidates, x[2])
    #             candidates = []
    #             forgetting_index += 1
    #             x = forgetting_num[forgetting_index]

    for j, x in enumerate(forgetting_num):
        utils.print_progress_bar(len(forgetting_num), j, "Making forgetting labels")
        if act_start <= x[1] and x[0] <= act_end:
            candidates = []
            already_checked = False
            for i, act in enumerate(AS):
                if (x[0] <= act.start) and (act.end <= x[1]):
                    already_checked = True
                    if act.activity.name in forgetting_activities:
                        candidates.append(i)
                else:
                    if already_checked:
                        break
            # If the number of candidates is smaller than the number of forgetting at the period, stop to assign forgetting during the period and print it's warning.
            if len(candidates) < x[2]:
                error_periods.append((x[0], x[1]))
            else:
                indexes += random.sample(candidates, x[2])

    if len(error_periods) != 0:
        print(
            "Warning: The number of candidates is smaller than the number of forgetting in some period. So, the real number of forgetting is less than the input. You should reduce the number of forgetting of input. Error periods are:"
        )
        for x in error_periods:
            print("{} - {}, ".format(x[0], x[1]), end="")
        print("")
    if len(indexes) == 0:
        print("Warning: No forgetting was made.")

    indexes.sort()

    # determination of labels[i][2]s and labels[i][3]s
    names, costs = [], []
    for i in indexes:
        home_equipment = random.choice(
            list(AS[i].activity.home_equipment.keys())
        )  # one of (not all) the home appliances is forgetting to turn off
        names.append(home_equipment)
        costs.append(AS[i].activity.home_equipment[home_equipment])

    # determination of labels[i][1]s
    # notice the forgetting when a resident will reach the home appliance again, or do activity whose place contains the home appliance
    indexes_2 = []
    end_index = len(AS) - 1
    for i in indexes:
        if i == end_index:
            indexes_2.append(end_index)
            continue
        else:
            end_flag = False
            for j in range(i + 1, end_index):
                if AS[j].place == AS[i].place:
                    indexes_2.append(j)
                    end_flag = True
                    break
            if not end_flag:
                indexes_2.append(end_index)

    # determination of labels[i][4]s and labels[i][5]s
    start_times, end_times = [], []
    for i in indexes:
        start_times.append(WT[i].start_time)
    for j in indexes_2:
        end_times.append(AS[j].start)

    # make ranges
    ret = []
    for i in range(len(indexes)):
        ret.append(
            (indexes[i], indexes_2[i], names[i], costs[i], start_times[i], end_times[i])
        )
    return ret


def determine_fall_indexes(AS, fall_parameters, fall_type):
    """
    This determines the fall indexes.
    If this returns i, then the resident falls down in a walking trajectory between AS[i] and AS[i+1], so W[i].

    Parameters
    ----------
    AS : list of ActivityDataPoint
        Time series data of activities.
    fall_parameters : dict
        Parameters of fall.
        For example, fall_w_parameters = {'num': fall_w_num, 'mean_lie_down_seconds': 30},
                     fall_s_parameters = {'num': fall_s_num, 'mean_lie_down_seconds': 30, 'place': fall_s_place_bed}.
    fall_type : str
        Type of falls. 'w' means falls while walking, 's' means falls while standing.

    Returns
    -------
    indexes : list of int
        Indexes of a walking trajectory where the resident falls down.

    Notes
    -----
    Indexes are determined to avoid to fill up the time of former activity, e.g., index i is not be a candidate if duration time of an activity of AS[i] is too short.
    If the number of candidates at each period is smaller than the number of falls at the period, stop to assign falls during the period and print it's warning.
    Falls while walking (falls while standing) are not occurred multiple times in one walking trajectory.
    But a fall while walking and a fall while standing may be occurred if those indexes are overlapped.
    """

    # check values
    keys = fall_parameters.keys()
    if fall_type == "w":
        if ("num" not in keys) or ("mean_lie_down_seconds" not in keys):
            raise ValueError(
                "parameters['num'] and parameters['mean_lie_down_seconds'] are needed."
            )
    if fall_type == "s":
        if (
            ("num" not in keys)
            or ("mean_lie_down_seconds" not in keys)
            or ("place" not in keys)
        ):
            raise ValueError(
                "parameters['num'], parameters['mean_lie_down_seconds'] and parameters['place'] are needed."
            )
    if (
        fall_parameters["num"][-1][1] < AS[0].start
        or AS[-1].end < fall_parameters["num"][0][0]
    ):
        print("Some periods of falls are out of range of the activity sequence.")

    # duration time of an activity before fall must be longer than duration_limit
    duration_limit = timedelta(seconds=4 * fall_parameters["mean_lie_down_seconds"])

    # determination of indexes
    error_periods = (
        []
    )  # the ranges will be added into this if the number of candidates at each period is smaller than the number of falls at the period.
    indexes = []  # indexes of falls
    act_start, act_end = AS[0].start, AS[-1].end
    maximum_ind = len(AS) - 2
    for x in fall_parameters["num"]:
        if act_start <= x[1] and x[0] <= act_end:
            candidates = []
            for i, act in enumerate(AS):
                if fall_type == "w":
                    if (
                        i <= maximum_ind
                        and act.duration > duration_limit
                        and act.place != AS[i + 1].place
                    ):
                        candidates.append(i)
                elif fall_type == "s":
                    if (
                        i <= maximum_ind
                        and act.duration > duration_limit
                        and act.place in fall_parameters["place"]
                        and act.place != AS[i + 1].place
                    ):
                        candidates.append(i)
            # If the number of candidates at each period is smaller than the number of falls at the period, stop to assign falls during the period and print it's warning.
            if len(candidates) < x[2]:
                error_periods.append((x[0], x[1]))
            else:
                indexes += random.sample(candidates, x[2])

    if len(error_periods) != 0:
        print(
            "Warning: The number of candidates is smaller than the number of falls in some period. So, the real number of falls is less than the input. You should reduce the number of falls of input. Error periods are:"
        )
        for x in error_periods:
            print("{} - {}, ".format(x[0], x[1]), end="")
        print("")
    if len(indexes) == 0:
        if fall_type == "w":
            print("Warning: No falls while walking was made.")
        if fall_type == "s":
            print("Warning: No falls while standing was made.")

    return indexes


def basic_activity_model_with_anomaly(
    act_model, day, state_anomaly_labels, anomaly_parameters
):
    """
    This generates a basic activity model with anomalies.

    Parameters
    ----------
    act_model : dict of list of activity_model.Activity
        For instance, see activity_model.basic_activity_model.
    day : int
        Index of the date.
    state_anomaly_labels : dict of list of tuple, dict()
        Periods of state anomalies.
        periods[anomaly][i] = (s, e)
        s : datetime.timedelta
            Start date and time.
        e : datetime.timedelta
            End date and time.
        For example, state_anomaly_labels = {'being housebound': [(d_1, d_2), ...], 'being semi-bedridden': [(d_a, d_b), ...]}
    anomaly_parameters : dict of float, default None
        Parameters that is changed during state anomalies.

    Returns
    -------
    temp_model : dict
        For instance, see activity_model.basic_activity_model.
    """
    # take anomalies into considertaion
    temp_model = {}
    for k, v in act_model.items():
        temp_model[k] = []
        for i in v:
            temp_model[k].append(i)

    # wandering
    # - add 'wandering' as necessary_activities
    if (
        "wandering_minute" in anomaly_parameters.keys()
        and "wandering_num" in anomaly_parameters.keys()
    ):
        num = decide_which_parameter(anomaly_parameters["wandering_num"], day)
        num = num / (
            (
                anomaly_parameters["wandering_num"][0][1]
                - anomaly_parameters["wandering_num"][0][0]
            )
            / timedelta(days=1)
        )
        duration = decide_which_parameter(anomaly_parameters["wandering_minute"], day)
        temp_model["necessary_activities"].append(
            activity_model.wandering.modified_new_activity(
                num_mean=num, duration_mean=duration, duration_sd=duration / 5
            )
        )

    # housebound
    # - change 'go out' into necessary activities, reduce the duration time of 'go out' and reduce the number of times of 'go out'
    # - change 'use a phone' into necessary activities, reduce the duration time of 'talk on the phone' and reduce the number of times of 'use a phone'
    if BEING_HOUSEBOUND in state_anomaly_labels.keys():
        is_housebound = False
        for i in state_anomaly_labels[BEING_HOUSEBOUND]:
            if i[0] <= timedelta(days=day) <= i[1]:
                is_housebound = True
        if is_housebound:
            # delete 'go out' and 'use a phone' from random activity, and add those in necessary activity for determining the frequency of those
            exist_go_out = activity_model.go_out in temp_model["random_activities"]
            exist_use_the_phone = (
                activity_model.use_the_phone in temp_model["random_activities"]
            )
            temp_model["random_activities"] = [
                act
                for act in temp_model["random_activities"]
                if act.name != activity_model.go_out.name
                and act.name != activity_model.use_the_phone.name
            ]
            if exist_go_out:
                temp_model["necessary_activities"].append(
                    activity_model.go_out_necessary.modified_new_activity(
                        num_mean=anomaly_parameters["housebound_go_out_num"],
                        duration_mean=anomaly_parameters["housebound_go_out_duration"],
                        duration_sd=anomaly_parameters["housebound_go_out_duration"]
                        / 5,
                    )
                )
            if exist_use_the_phone:
                temp_model["necessary_activities"].append(
                    activity_model.use_the_phone_necessary.modified_new_activity(
                        num_mean=anomaly_parameters["housebound_use_the_phone_num"],
                        duration_mean=anomaly_parameters[
                            "housebound_use_the_phone_duration"
                        ],
                        duration_sd=anomaly_parameters[
                            "housebound_use_the_phone_duration"
                        ]
                        / 5,
                    )
                )

    # semi-bedridden
    # - add 'nap' as random activity
    # - change 'go out' into necessary activities, reduce the duration time of 'go out' and reduce the number of times of 'go out'
    # - increase the duration time of 'rest'
    if BEING_SEMI_BEDRIDDEN in state_anomaly_labels.keys():
        is_semi_bedridden = False
        for i in state_anomaly_labels[BEING_SEMI_BEDRIDDEN]:
            if i[0] <= timedelta(days=day) <= i[1]:
                is_semi_bedridden = True
        if is_semi_bedridden:
            # add nap to random activity and make duraion of rest longer
            temp_model["random_activities"].append(
                activity_model.nap.modified_new_activity(
                    duration_mean=anomaly_parameters["semi_bedridden_nap_duration"],
                    duration_sd=anomaly_parameters["semi_bedridden_nap_duration"] / 5,
                )
            )
            exist_go_out = activity_model.go_out in temp_model["random_activities"]
            exist_rest = activity_model.rest in temp_model["random_activities"]
            temp_model["random_activities"] = [
                act
                for act in temp_model["random_activities"]
                if act.name != activity_model.rest.name
                and act.name != activity_model.go_out.name
            ]
            if exist_go_out:
                temp_model["necessary_activities"].append(
                    activity_model.go_out_necessary.modified_new_activity(
                        num_mean=anomaly_parameters["semi_bedridden_go_out_num"],
                        duration_mean=anomaly_parameters[
                            "semi_bedridden_go_out_duration"
                        ],
                        duration_sd=anomaly_parameters["semi_bedridden_go_out_duration"]
                        / 5,
                    )
                )
            if exist_rest:
                temp_model["random_activities"].append(
                    activity_model.rest.modified_new_activity(
                        duration_mean=anomaly_parameters[
                            "semi_bedridden_rest_duration"
                        ],
                        duration_sd=anomaly_parameters["semi_bedridden_rest_duration"]
                        / 5,
                    )
                )
    return temp_model


def decide_which_parameter(time_param_list, day):
    """
    This decides which parameter are chosen.

    Parameters
    ----------
    time_param_list : list of tuple
        time_param_list[i] = (start time(timedelta), end time(timedelta), parameter).
    day : int
        Index of the dates.

    Returns
    -------
    parameter : type of time_param_list[2]
        Target parameter.
    """
    for x in time_param_list:
        if x[0] <= timedelta(days=day) <= x[1]:
            return x[2]
    raise ValueError(
        "This cannot find appropriate time range on {} day.\nThe range of time_param_list is from {} to {}.\nThe day must be contained in this range.".format(
            day, time_param_list[0][0], time_param_list[-1][1]
        )
    )


def save_MMSE(path, values, file_name="MMSE"):
    """
    This saves the MMSE parameters as text format.

    Parameters
    ----------
    path : pathlib.Path
        Path to save the text file.
    values : list of tuple
        values[i] = (s, e, v)
            s : datetime.timedelta
                Start date and time.
            e : datetime.timedelta
                End date and time.
            v : float
                Target value between s and e.
            Each duration between s and e equals to 'step'.
    file_name : str
        File name of this file.

    Returns
    -------
    None
    """
    text = ""
    for x in values:
        text += "{:04} [day] {:02} [h] {:02} [m] {:02} [s] {:06} [ms] - {:04} [day] {:02} [h] {:02} [m] {:02} [s] {:06} [ms], value: {}\n".format(
            *utils.get_d_h_m_s_ms(x[0]), *utils.get_d_h_m_s_ms(x[1]), str(x[2])
        )
    text += "\n"
    if not os.path.exists(path):
        os.makedirs(path)
    file_path = path / "{}.txt".format(file_name)
    with open(file_path, "w") as f:
        f.write(text)


def save_housebound_labels(path, housebound_labels, file_name="housebound_labels"):
    """
    This saves the housebound lables as text format.

    Parameters
    ----------
    path : pathlib.Path
        Path to save the text file.
    housebound_lables : list of tuple
        (start_date, end_date)
        start_date : datetime.timedelta
            Start date to be housebound.
        end_date : datetime.timedelta
            End date ot be housebound.
    file_name : str
        File name of this file.

    Returns
    -------
    None
    """
    text = ""
    for interval in housebound_labels:
        text += "{:04} [day] {:02} [h] {:02} [m] {:02} [s] {:06} [ms] - {:04} [day] {:02} [h] {:02} [m] {:02} [s] {:06} [ms]\n".format(
            *utils.get_d_h_m_s_ms(interval[0]), *utils.get_d_h_m_s_ms(interval[1])
        )
    if not os.path.exists(path):
        os.makedirs(path)
    file_path = path / "{}.txt".format(file_name)
    with open(file_path, "w") as f:
        f.write(text)


def save_anomaly_parameters(path, anomaly_paraters, file_name="anomaly_parameters"):
    """
    This saves the anomaly parameters as text format.

    Parameters
    ----------
    path : pathlib.Path
        Path to save the text file.
    anomaly_paraters : dict
        Parameters of anomalies.
    file_name : str
        File name of this file.

    Returns
    -------
    None
    """
    text = ""
    for k, v in anomaly_paraters.items():
        text += "{}:\n".format(k)
        if isinstance(v, int) or isinstance(v, float):
            text += "{}\n".format(str(v))
        elif k == "place":  # v is a list of str
            for x in v:
                text += "{}\n".format(x)
        elif isinstance(v, list):
            # suppose that the list has following shape;
            # values : list of tuple
            # values[i] = (s, e, v)
            # s : datetime.timedelta
            #     Start date and time.
            # e : datetime.timedelta
            #     End date and time.
            # v : float
            #     Target value between s and e.
            # Each duration between s and e equals to 'step'.
            for x in v:
                text += "{:04} [day] {:02} [h] {:02} [m] {:02} [s] {:06} [ms] - {:04} [day] {:02} [h] {:02} [m] {:02} [s] {:06} [ms], value: {}\n".format(
                    *utils.get_d_h_m_s_ms(x[0]), *utils.get_d_h_m_s_ms(x[1]), str(x[2])
                )
        text += "\n"
    if not os.path.exists(path):
        os.makedirs(path)
    file_path = path / "{}.txt".format(file_name)
    with open(file_path, "w") as f:
        f.write(text)


def save_wandering_labels(path, WT, file_name="wandering_labels"):
    """
    This saves the anomaly parameters as text format.

    Parameters
    ----------
    path : pathlib.Path
        Path to save the text file.
    WT : list of WalkingTrajectory
        Walking trajectories.
    file_name : str
        File name of this file.

    Returns
    -------
    None
    """
    text = ""
    count = 0
    for i, wt in enumerate(WT):
        if wt.walking_type == utils.WANDERING_WALKING:
            count += 1
            text += "Activity index: {:04}, Start: {:04} [day] {:02} [h] {:02} [m] {:02} [s] {:06} [ms], End: {:04} [day] {:02} [h] {:02} [m] {:02} [s] {:06} [ms], Duration: {:.2f}[m].\n".format(
                i,
                *utils.get_d_h_m_s_ms(wt.start_time),
                *utils.get_d_h_m_s_ms(wt.end_time),
                (wt.end_time - wt.start_time) / timedelta(minutes=1)
            )
    text = "Total number of wanderings: {})\n".format(count) + text + "\n"
    if not os.path.exists(path):
        os.makedirs(path)
    file_path = path / "{}.txt".format(file_name)
    with open(file_path, "w") as f:
        f.write(text)


def save_fall_labels(path, WT, file_name="fall_labels"):
    """
    This saves the anomaly parameters as text format.

    Parameters
    ----------
    path : pathlib.Path
        Path to save the text file.
    WT : list of WalkingTrajectory
        Walking trajectories.
    file_name : str
        File name of this file.

    Returns
    -------
    None
    """
    text = ""
    count_w, count_s = 0, 0
    for i, wt in enumerate(WT):
        if wt.fall_w or wt.fall_s:
            text += "{}-th trajectory ({} - {})\n".format(
                i, wt.start_place, wt.end_place
            )
            if wt.fall_w:
                text += "fall while walking:\n"
                text += "{:04} [day] {:02} [h] {:02} [m] {:02} [s] {:06} [ms]\n".format(
                    *utils.get_d_h_m_s_ms(wt.timestamp[wt.fall_w_index])
                )
                text += "lie down: {} [sec.]\n".format(wt.lie_down_seconds_w)
                count_w += 1
            if wt.fall_s:
                text += "fall while standing:\n"
                text += "{:04} [day] {:02} [h] {:02} [m] {:02} [s] {:06} [ms]\n".format(
                    *utils.get_d_h_m_s_ms(wt.timestamp[wt.fall_s_index])
                )
                text += "lie down: {} [sec.]\n".format(wt.lie_down_seconds_s)
                count_s += 1
    text = (
        "Total number of fall while walking: {}, Total number of fall while walking: {}\n".format(
            count_w, count_s
        )
        + text
        + "\n"
    )
    if not os.path.exists(path):
        os.makedirs(path)
    file_path = path / "{}.txt".format(file_name)
    with open(file_path, "w") as f:
        f.write(text)


def save_forgetting_labels(path, forgetting_labels, file_name="forgetting_labels"):
    """
    This saves the anomaly parameters as text format.

    Parameters
    ----------
    path : pathlib.Path
        Path to save the text file.
    forgetting_labels : list of tuple
        Labels of forgetting.
        labels[i] = (forgetting_ind, notice_ind, name, cost, start_time, end_time)
        forgetting_ind : int
            Index of an activity after which a resident will be forgetting.
        notice_ind : int
            Index of an activity that a resident notice forgetting and turn off the home appliance.
        name : str
            Name of a home appliance the resident forget to turn off.
        cost : float
            Cost of the home appliance per hour.
        start_time : datetime.timedelta
            Start time of forgetting, so the home appliance is left on.
        end_time : datetime.timedelta
            End time of forgetting, so the home appliance is turned off.
    file_name : str, default 'forgetting_labels'
        File name of this file.

    Returns
    -------
    None
    """
    text = ""
    count = 0
    for i, x in enumerate(forgetting_labels):
        text += "{}\n".format(i)
        text += "start: {:04} [day] {:02} [h] {:02} [m] {:02} [s] {:06} [ms]\n".format(
            *utils.get_d_h_m_s_ms(x[4])
        )
        text += "end  : {:04} [day] {:02} [h] {:02} [m] {:02} [s] {:06} [ms]\n".format(
            *utils.get_d_h_m_s_ms(x[5])
        )
        text += "name: {}, cost: {}\n".format(x[2], x[3])
        count += 1
    text = "Total number of forgetting: {}\n".format(count) + text + "\n"
    if not os.path.exists(path):
        os.makedirs(path)
    file_path = path / "{}.txt".format(file_name)
    with open(file_path, "w") as f:
        f.write(text)
