# Programs for other methods or existing researches to compare the performance.
import itertools
from datetime import datetime, timedelta
from pathlib import Path

import Levenshtein
import numpy as np

# self-made
import src.activity_model as activity_model
import src.utils as utils


def load_CASAS_Aruba(
    path,
    is_filled=True,
    go_out_exists=True,
    delete_short_other=True,
    rename_eating=True,
    rename_nap=True,
    delete_enter_leave=True,
    delete_toilet=True,
    delete_respirate=True,
):
    """
    This loads CASAS aruba dataset [1][2] as an activity sequence.

    [1] D. Cook et al., "CASAS: A Smart Home in a Box." IEEE Computer, 46-7(2013), 62-69.
    [2] http://casas.wsu.edu/datasets/

    Original data contains following activities: Meal Preparation, Relax, Eating, Work, Sleeping, Wash Dishes, Bed to Toilet, Enter Home, Leave Home, Housekeeping and Resperate.

    Parameters
    ----------
    path : pathlib.Path
        Path of the Aruba dataset.
    is_filled : boolean
        Whether the dataset is filled up with new activity 'Other' when any activity label is not assigned.
        If this is False, then output containes undefined time (not labeled time).
    go_out_exists : boolean
        Whether a new activity 'Go out' is assigned from an end time of 'Leave_Home' to a start time of 'Enter_Home'.
    delete_short_other : boolean
        Whether to delete short 'Other' activities.
    rename_eating  : boolean
        Whether to divide 'Eating' label into several labels: 'Take_Breakfast', 'Take_Lunch' and 'Take_Dinner'.
    rename_nap : boolean
        Whether to assign new activity 'Nap' for the sleep activities that are done except for at night.
    delete_enter_leave : boolean
        Whether to delete 'Enter_Home' and 'Leave_Home', and to joint the activity with its former activity.
    delete_toilet : boolean
        Whether to delete 'Bed_to_Toilet', and to joint the activity with its former activity.
    delete_respirate : boolean
        Whether to delete 'Respirate', and to joint the activity with its former activity.

    Returns
    -------
    Aruba : list of tuple
        Aruba[i] = (act_name, time_interval)
        act_name : str
            Activity name.
        time_interval : new_functions.TimeInterval
            Time interval of this activity.

    See Also
    --------
    new_functions.TimeInterval

    Notes
    -----
    1. We regard the start time of sampling as datetime(year = 2010, month = 11, day = 4)
       and regard the end time of sampling as datetime(year = 2011, month = 6, day = 12).
       We extend the 'Sleeping' near datetime(year = 2010, month = 11, day = 4) and datetime(year = 2011, month = 6, day = 12) appropriately.
    2. The overlap of two activities occur several times in Aruba dataset, so we treat this as follows.
       overlap type 1:
           original
           a1  ----------------
           a2          -----------------
           a3                       ---------------
           modified
           a1  --------
           a2          -------------
           a3                       ---------------
               -------------------------------------> time
       overlap type 2:
           original
           a1  ------------------------------
           a2          -------------
           modified
           a1  --------             ---------
           a2          -------------
               -------------------------------------> time
    3. The original Aruba dataset has overlapped data (error?) from '2011-05-23 02:43:44.902903 M003 OFF Sleeping end' to '2011-05-23 21:44:59.759837 M003 ON Sleeping begin' twice. So, skip these data once.
    """

    def connect_same_activities(act_list):
        """
        This connects consecutive same activities.

        Parameters
        ----------
        act_list : list of tuple
            act_list[i] = (act_name, time_interval)
            act_name : str
                Activity name.
            time_interval : new_functions.TimeInterval
                Time interval of this activity.

        Returns
        -------
        ret : list of tuple
            ret[i] = (act_name, time_interval)
            act_name : str
                Activity name.
            time_interval : new_functions.TimeInterval
                Time interval of this activity.
        """
        intervals = []
        start, end, last_index = 0, 0, len(act_list) - 1
        while end <= last_index:
            start = end
            name = act_list[end][0]
            while True:
                end += 1
                if end > last_index:
                    break
                if act_list[end][0] != name:
                    break
            intervals.append((start, end - 1))
        ret = []
        for s, e in intervals:
            ret.append(
                (
                    act_list[s][0],
                    utils.TimeInterval(act_list[s][1].start, act_list[e][1].end),
                )
            )
        return ret

    other = "Other"

    start_point = datetime(year=2010, month=11, day=4)
    end_point = datetime(year=2011, month=6, day=12)
    skip_s = datetime(
        year=2011, month=5, day=23, hour=2, minute=43, second=44, microsecond=902903
    )
    skip_e = datetime(
        year=2011, month=5, day=23, hour=21, minute=44, second=59, microsecond=759837
    )
    begin_count, end_count = 0, 0
    Aruba = []
    is_contained = False  # Whether the data between skip_s and skip_e was skipped.
    with open(path, mode="r") as f:
        activity_name, start_time, end_time = None, None, None
        other_start_time = None
        is_overlapped = False
        intermitted_activity_name = None
        for (
            line
        ) in f:  # e.g. line = '2010-11-04 00:03:50.209589 M003 ON Sleeping begin'
            splitted_line = line.split()
            if len(splitted_line) != 6:
                continue
            date = splitted_line[0].split("-")
            time = splitted_line[1].split(":")
            seconds = time[2].split(".")
            this_time = None
            if len(seconds) == 1:
                this_time = datetime(
                    year=int(date[0]),
                    month=int(date[1]),
                    day=int(date[2]),
                    hour=int(time[0]),
                    minute=int(time[1]),
                    second=int(seconds[0]),
                )
            elif len(seconds) == 2:
                while (
                    len(seconds[1]) < 6
                ):  # '00:03:50.0209' -> seconds = ['50', '0209000']
                    seconds[1] += "0"
                this_time = datetime(
                    year=int(date[0]),
                    month=int(date[1]),
                    day=int(date[2]),
                    hour=int(time[0]),
                    minute=int(time[1]),
                    second=int(seconds[0]),
                    microsecond=int(seconds[1]),
                )

            if is_contained and (skip_s <= this_time <= skip_e):
                continue
            if not (is_contained) and (this_time == skip_e):
                is_contained = True

            if "begin" in line:
                begin_count += 1
            if "end" in line:
                end_count += 1
            if begin_count - end_count > 1:
                is_overlapped = True

            if not is_overlapped:  # Activities are not overlapped now.
                if "begin" in line:
                    if is_filled and other_start_time is not None:
                        Aruba.append(
                            (
                                other,
                                utils.TimeInterval(
                                    start=other_start_time, end=this_time - start_point
                                ),
                            )
                        )
                    activity_name = splitted_line[4]
                    start_time = this_time - start_point
                if "end" in line:
                    end_time = this_time - start_point
                    Aruba.append(
                        (
                            activity_name,
                            utils.TimeInterval(start=start_time, end=end_time),
                        )
                    )
                    other_start_time = this_time - start_point
            else:  # Activities are overlapped now.
                if "begin" in line:
                    intermitted_activity_name = activity_name
                    end_time = this_time - start_point
                    Aruba.append(
                        (
                            activity_name,
                            utils.TimeInterval(start=start_time, end=end_time),
                        )
                    )
                    activity_name = splitted_line[4]
                    start_time = this_time - start_point
                if "end" in line:
                    if splitted_line[4] == intermitted_activity_name:
                        is_overlapped = False
                        continue
                    else:
                        end_time = this_time - start_point
                        Aruba.append(
                            (
                                activity_name,
                                utils.TimeInterval(start=start_time, end=end_time),
                            )
                        )
                        start_time = this_time - start_point
                        activity_name = intermitted_activity_name
                        is_overlapped = False

    # Extend time of activities near edge of data.
    Aruba[0] = (
        Aruba[0][0],
        utils.TimeInterval(start=start_point - start_point, end=Aruba[0][1].end),
    )
    Aruba[-1] = (
        Aruba[-1][0],
        utils.TimeInterval(start=Aruba[-1][1].start, end=end_point - start_point),
    )

    # Delete short 'Other' activity.
    # example 1:
    # ..., ('Rest', 1[d]13[h]05[m], 1[d]13[h]34[m]), ('Other', 1[d]13[h]34[m], 1[d]13[h]36[m]) , ('Work', 1[d]13[h]36[m], 1[d]14[h]15[m]), ...
    # -> ..., ('Rest', 1[d]13[h]05[m], 1[d]13[h]36[m]), ('Work', 1[d]13[h]36[m], 1[d]14[h]15[m]), ...
    # example 2:
    # ..., ('Rest', 1[d]13[h]05[m], 1[d]13[h]34[m]), ('Other', 1[d]13[h]34[m], 1[d]13[h]35[m]) , ('Rest', 1[d]13[h]35[m], 1[d]14[h]22[m]), ...
    # -> ..., ('Rest', 1[d]13[h]05[m], 1[d]14[h]22[m]), ...
    threshold_time = timedelta(
        minutes=10
    )  # baseline duration time to delete short activity
    if delete_short_other:
        ret = []
        for i in range(0, len(Aruba) - 2):
            if Aruba[i][0] == other:
                if Aruba[i][1].duration > threshold_time:
                    ret.append(Aruba[i])
            else:
                if Aruba[i + 1][0] == other:
                    if Aruba[i + 1][1].duration <= threshold_time:
                        ret.append(
                            (
                                Aruba[i][0],
                                utils.TimeInterval(
                                    start=Aruba[i][1].start, end=Aruba[i + 1][1].end
                                ),
                            )
                        )
                    else:
                        ret.append(Aruba[i])
                else:
                    ret.append(Aruba[i])
        ret.append(Aruba[-1])
        Aruba = connect_same_activities(ret)

    # Generate new activities go_out between 'Leave_Home' and 'Enter_Home'.
    go_out = "Go_out"
    if go_out_exists:
        ret = []
        ret.append(Aruba[0])
        for i in range(1, len(Aruba) - 1):
            if (
                Aruba[i][0] == other
                and Aruba[i - 1][0] == "Leave_Home"
                and Aruba[i + 1][0] == "Enter_Home"
            ):
                ret.append(
                    (
                        go_out,
                        utils.TimeInterval(
                            start=Aruba[i][1].start, end=Aruba[i][1].end
                        ),
                    )
                )
            else:
                ret.append(Aruba[i])
        ret.append(Aruba[-1])
        Aruba = ret

    # Delete Enter_Home and Leave_Home, and to joint the activity with its former activity.
    # example #1,
    #     ..., ('Work', 1[d]12[h]56[m], 1[d]13[h]05[m]), ('Leave_Home', 1[d]13[h]05[m], 1[d]13[h]34[m]), ('Go_out', 1[d]13[h]34[m], 1[d]14[h]36[m]) ,
    #          ('Other', 1[d]14[h]36[m], 1[d]14[h]58[m]), ('Enter_Home', 1[d]14[h]58[m], 1[d]15[h]20[m]), ...
    # ->  ..., ('Work', 1[d]12[h]56[m], 1[d]13[h]34[m]), ('Go_out', 1[d]13[h]34[m], 1[d]14[h]36[m]), ('Other', 1[d]14[h]36[m], 1[d]15[h]20[m]), ...
    # example #2,
    # Work 8 days, 11:20:00.003069
    # Other 8 days, 11:23:46.759039
    # Leave_Home 8 days, 11:46:02.357588
    # Enter_Home 8 days, 11:46:12.380361
    # Leave_Home 8 days, 11:46:31.671941
    # Go_out 8 days, 11:46:36.034292
    # Enter_Home 8 days, 13:43:49.151516
    # Leave_Home 8 days, 13:48:28.867420
    # Enter_Home 8 days, 13:48:47.364360
    # Meal_Preparation 8 days, 13:50:35.700848
    # ->
    # Work 8 days, 11:20:00.003069
    # Other 8 days, 11:23:46.759039
    # Go_out 8 days, 11:46:36.034292
    # Meal_Preparation 8 days, 13:50:35.700848
    if delete_enter_leave:
        intervals = []
        start, end, last_index = 0, 0, len(Aruba) - 1
        while end <= last_index:
            start = end
            while True:
                end += 1
                if end > last_index:
                    break
                if Aruba[end][0] != "Enter_Home" and Aruba[end][0] != "Leave_Home":
                    break
            intervals.append((start, end - 1))
        ret = []
        for s, e in intervals:
            ret.append(
                (Aruba[s][0], utils.TimeInterval(Aruba[s][1].start, Aruba[e][1].end))
            )
        Aruba = connect_same_activities(ret)

    # Rename take breakfast, take lunch, take dinner
    take_breakfast = "Take_Breakfast"
    take_lunch = "Take_Lunch"
    take_dinner = "Take_Dinner"
    if rename_eating:
        breakfast = (4, 11)
        lunch = (11, 16)
        dinner1 = (16, 24)
        dinner2 = (0, 1)
        ret = []
        for a in Aruba:
            n = a[0]
            if n == "Eating":
                hour = utils.get_d_h_m_s_ms(a[1].start)[1]
                if breakfast[0] <= hour < breakfast[1]:
                    n = take_breakfast
                if lunch[0] <= hour < lunch[1]:
                    n = take_lunch
                if dinner1[0] <= hour < dinner1[1] or dinner2[0] <= hour < dinner2[1]:
                    n = take_dinner
            ret.append((n, a[1]))
        Aruba = ret

    # Rename nap if the resident sleep except for at night.
    nap = "Nap"
    if rename_nap:
        ret = []
        normal_sleep = (6, 17)
        for a in Aruba:
            n = a[0]
            if n == "Sleeping":
                hour = utils.get_d_h_m_s_ms(a[1].start)[1]
                if normal_sleep[0] <= hour < normal_sleep[1]:
                    n = nap
            ret.append((n, a[1]))
        Aruba = ret

    # Delete Bed_to_Toilet, and to joint the activity with its former activity.
    if delete_toilet:
        intervals = []
        start, end, last_index = 0, 0, len(Aruba) - 1
        while end <= last_index:
            start = end
            while True:
                end += 1
                if end > last_index:
                    break
                if Aruba[end][0] != "Bed_to_Toilet":
                    break
            intervals.append((start, end - 1))
        ret = []
        for s, e in intervals:
            ret.append(
                (Aruba[s][0], utils.TimeInterval(Aruba[s][1].start, Aruba[e][1].end))
            )
        Aruba = connect_same_activities(ret)

    # Delete Respirate, and to joint the activity with its former activity
    if delete_respirate:
        intervals = []
        start, end, last_index = 0, 0, len(Aruba) - 1
        while end <= last_index:
            start = end
            while True:
                end += 1
                if end > last_index:
                    break
                if Aruba[end][0] != "Respirate":
                    break
            intervals.append((start, end - 1))
        ret = []
        for s, e in intervals:
            ret.append(
                (Aruba[s][0], utils.TimeInterval(Aruba[s][1].start, Aruba[e][1].end))
            )
        Aruba = connect_same_activities(ret)

    return Aruba


def convert_data_into_days(data):
    """
    This converts a flatten activity sequence into the divided activity sequences for each day.
    Even if the data contains overlapped activites, e.g., ('Take Lunch', 12[d]12[h]43[m], 12[d]13[h]15[m]), ('TV', 12[d]13[h]02[m], 12[d]13[h]30[m]), this function can work.
    However, any time must be filled by some activity, so the data do not have undefined activities at any time.

    Parameters
    ----------
    data : list of tuple
        data[i] = (activity_name, time_interval)
        activity_name : str
        time_interval : new_functions.TimeInterval

    Returns
    -------
        days : list of list of tuple
        days[i] : list
            Activity sequence of i-th day.
        days[i][j] = (activity_name, time_interval)
            days[i][j][0] : str
                Activity name of j-th activity in i-th day.
            days[i][j][1] : TimeInterval
                Time interval of j-th activity in i-th day.
                This time extracts only time of day, does not contain days information (so days = 0).
        days[i][0].start =  timedelta(hours = 0)
        days[i][-1][1].end = timedelta(hours = 24)

    Example
    -------
    input =
    [ ...,
        ('TV',       121[d]19[h]43[m], 121[d]22[h]50[m]),
        ('Sleeping', 121[d]22[h]50[m], 122[d]06[h]13[m]),
        ('Rest',     122[d]06[h]13[m], 122[d]07[h]55[m]),
    ... ]

    - >
    ret =
    [ [...],
      ...,
      [..., ('TV', 19[h]43[m], 22[h]50[m]), ('Sleeping', 22[h]50[m], 24[h]00[m]) ],
      [ ('Sleeping', 00[h]00[m], 06[h]13[m]), ('Rest', 06[h]13[m], 07[h]55[m]), ...],
      ...,
      [...]]

    ret[121] = [..., ('TV', 19[h]43[m], 22[h]50[m]), ('Sleeping', 22[h]50[m], 24[h]00[m]) ]
    ret[122] = [ ('Sleeping', 00[h]00[m], 06[h]13[m]), ('Rest', 06[h]13[m], 07[h]55[m]), ...]
    """
    if data[0][1].start != timedelta(days=0) or data[-1][1].end != timedelta(
        days=data[-1][1].end.days
    ):
        raise ValueError(
            "data must be just from timedelta(days = 0) to timedelta(days = ?)."
        )
    ret = [[] for _ in range(data[-1][1].end.days)]
    start_of_a_day, end_of_a_day = timedelta(hours=0), timedelta(hours=24)
    for a in data:
        temp_day = a[1].start.days
        temp_start = utils.extract_time_of_day(a[1].start)
        temp_end = utils.extract_time_of_day(a[1].end)
        if a[1].start.days != a[1].end.days:
            ret[temp_day].append((a[0], utils.TimeInterval(temp_start, end_of_a_day)))
            if a[1].end.days - a[1].start.days > 1:
                for i in range(a[1].start.days + 1, a[1].end.days):
                    ret[i].append(
                        (a[0], utils.TimeInterval(start_of_a_day, end_of_a_day))
                    )
            if temp_end != start_of_a_day:
                ret[a[1].end.days].append(
                    (a[0], utils.TimeInterval(start_of_a_day, temp_end))
                )
        else:
            ret[temp_day].append((a[0], utils.TimeInterval(temp_start, temp_end)))
    return ret


def estimate_daily_frequency(data, act_names=None):
    """
    Parameter estimation of daily frequency (occurrence numbers per day) of activities from data.
    Even if the data contains overlapped activites, e.g., ('Take Lunch', 12[d]12[h]43[m], 12[d]13[h]15[m]), ('TV', 12[d]13[h]02[m], 12[d]13[h]30[m]), this function can work.

    Parameters
    ----------
    data : list of tuple
        data[i] = (activity_name, time_interval)
        activity_name : str
        time_interval : new_functions.TimeInterval
        data[0][1].start must be timedelta(hours = 0)
    act_names : list of str
        Names of target activities.

    Returns
    -------
    (daily_freq_mean, daily_freq_sd)
    daily_freq_mean : dict of float
        daily_freq_mean[a] = mean frequency (occurrence numbers per day) of an activity a.
        For example, {'Sleeping': 1.5681818181818181, 'Meal_Preparation': 4.945454545454545}.
    daily_freq_sd : dict of float
        daily_freq_sd[a] = standard deviations of frequency (occurrence numbers per day) of an activity a.
        For example, {'Sleeping': 0.5639844965924988, 'Meal_Preparation': 2.1964283554582784}.
    """
    if act_names is None:
        act_names = []
        for a in data:
            if a[0] not in act_names:
                act_names.append(a[0])
    if data[0][1].start != timedelta(hours=0):
        raise ValueError("Data must starts from timedelta(hours = 0)!")
    daily_freq_mean, daily_freq_sd = {name: None for name in act_names}, {
        name: None for name in act_names
    }
    num_dates = data[-1][1].end.days
    freq_dic = {x: [0 for _ in range(num_dates)] for x in act_names}
    divided_data = convert_data_into_days(data)
    for i, day in enumerate(divided_data):
        for a in day:
            freq_dic[a[0]][i] += 1
    for act_name in act_names:
        daily_freq_mean[act_name] = np.array(freq_dic[act_name]).mean()
        daily_freq_sd[act_name] = np.sqrt(np.array(freq_dic[act_name]).var())
    return (daily_freq_mean, daily_freq_sd)


def make_act2char_dic(act_list):
    """
    This return a dictionary that maps an activity to 1 character.

    Parameters
    ----------
    act_list : list of str
        Activity names.

    Returns
    -------
    ret : dict
        A dictionary that maps an activity to 1 character.

    Raises
    ------
    If the number of activities is greater than the length of candidates of characters.
    """
    candidates = "abcdefghijklmnopqrstuvwxyz0123456789!?+-"
    ret = {a: None for a in act_list}
    i = 0
    for a in act_list:
        ret[a] = candidates[i]
        i += 1
        if i >= len(candidates):
            raise ValueError(
                "Number of activities is greater than the length of candidates of characters!"
            )
    return ret


def convert_day_into_str(day, act2char_map, T_d):
    """
    This converts one day data into a string.

    Parameters
    ----------
    day : list of tuple
        day[i] = (activity_name, TimeInterval).
        day[0][1].start must be timedelta(hours = 0), and day[-1][1].end must be timedelta(hours = 24).
    act2char_map : dic
        dic[activity name] = the corresponded character.
    T_d : datetime.timedelta
        Unit time interval to make bins.
        24 [hour] must be divisible by T_d.

    Returns
    -------
    string : str
        Converted string whose length is 24/T_d.

    Example
    -------
    ('Sleep', timedelta(hours = 0), timedelta(hours = 2, minutes = 20)), ('Work', timedelta(hours = 2, minutes = 20), timedelta(hours = 4)), ...
    T_d = 0.5[h],
    act2char_map['Sleep'] = s, act2char_map['Work'] = w,
    Then, ret = ssssswww...

    Notes
    -----
    Even if the data contains overlapped activites, e.g., ('Take Lunch', 12[d]12[h]43[m], 12[d]13[h]15[m]), ('TV', 12[d]13[h]02[m], 12[d]13[h]30[m]), this function can work.
    """
    if timedelta(hours=24) % T_d != timedelta(hours=0):
        raise ValueError("24 [h] must be divisible by T_d!")
    occupancy_acts = []  # activity names that mainly occupy the time interval
    for t in utils.date_generator(timedelta(hours=0), timedelta(hours=24), step=T_d):
        interval = utils.TimeInterval(t, t + T_d)
        max_act, max_overlap = None, 0
        for a in day:
            overlap = interval.overlap_length_with(a[1]) / timedelta(hours=1)
            if overlap > max_overlap:
                max_act, max_overlap = a[0], overlap
        occupancy_acts.append(max_act)
    ret = ""
    if None in occupancy_acts:
        print("---------------------------------------------------------")
        for a in day:
            print("{} {} {}".format(a[0], a[1].start, a[1].end))
        print(act2char_map)
        print(occupancy_acts)
    for a in occupancy_acts:
        ret += act2char_map[a]
    return ret


def same_days_distance(days, T_d, print_progress=True):
    """
    This calculates mean distance between different days in 'days'.
    If N days are included, 2-combination of {1, 2, ..., N} are compared.

    Parameters
    ----------
    days : list of tuple
        days[i] : list
            Activity sequence of i-th day.
        days[i][j][0] : str
            Activity name of j-th activity in i-th day.
        days[i][j][1] : TimeInterval
            TimeInterval of j-th activity in i-th day.
    T_d : datetime.timedelta
        Unit time interval that is converted into one character.
        24 [hour] must be divisible by T_d.
    print_progress : boolean, default True
        Whether to print a progress bar.

    Returns
    -------
    (mean, sd) : tuple of float
        mean : float
            Mean distance of 2-combination days of N days.
        sd : float
            Standard deviation.

    Notes
    -----
    Even if the data contains overlapped activites, e.g., ('Take Lunch', 12[d]12[h]43[m], 12[d]13[h]15[m]), ('TV', 12[d]13[h]02[m], 12[d]13[h]30[m]), this function can work.
    """
    activity_names = []
    for day in days:
        for a in day:
            if a[0] not in activity_names:
                activity_names.append(a[0])
    act2char_map = make_act2char_dic(activity_names)
    distances = []
    n_d = len(days)
    n = int(n_d * (n_d - 1) / 2)
    day_strings = []
    for i, day in enumerate(days):
        day_strings.append(convert_day_into_str(day, act2char_map, T_d))
        if print_progress:
            simple_progress_bar(i, n_d, "converting days into strings", step=10)
    for i, (day1_str, day2_str) in enumerate(itertools.combinations(day_strings, 2)):
        distances.append(Levenshtein.distance(day1_str, day2_str))
        if print_progress:
            simple_progress_bar(i, n, "calculate distances", step=100)
    mean = np.array(distances).mean()
    sd = np.sqrt(np.array(distances).var())
    return (mean, sd)


def mean_same_days_distance(days_list, T_d, print_progress=False):
    """
    This calculates a mean and standard deviation of same days distance.

    Parameters
    ----------
    days_list : list of list of list of tuple
        days_list[k] : list of list of tuple
            Activity sequences of k-th dataset.
        days_list[k][i][j] = (activity_name, time_interval)
            days_list[k][i][j][0] : str
                Activity name of j-th activity in i-th day of k-th dataset.
            days_list[k][i][j][1] : TimeInterval
                Time interval of j-th activity in i-th day of k-th dataset.
                This time extracts only time of day, does not contain days information (so days = 0).
        days_list[k][i][0].start =  timedelta(hours = 0)
        days_list[k][i][-1][1].end = timedelta(hours = 24)
    T_d : datetime.timedelta
        Unit time interval that is converted into one character.
        24 [hour] must be divisible by T_d.
    print_progress : boolean, default False
        Whether to print a progress bar.

    Returns
    -------
    (mean_list, sd_list)
        mean_list : list of float
            mean_list[k] : float
                Mean same days distance of k-th dataset.
        sd_list : list of float
            sd_list[k] : float
                Standard deviation of same days distance of k-th dataset.
    """
    mean_list, sd_list = [], []
    for i, _days in enumerate(days_list):
        temp = same_days_distance(_days, T_d, print_progress=print_progress)
        mean_list.append(temp[0])
        sd_list.append(temp[1])
        print("Finish the calculation of {}-th dataset.\r".format(i), end="")
    return (mean_list, sd_list)


def different_days_distance(days_X, days_Y, T_d, activity=None, print_progress=True):
    """
    This calculates mean distance between days in 'days_X' and days in 'days_Y'.
    'days_X' and 'days_Y' are different.
    If 'days_X' contains M days and 'days_Y' contains N days, M * N pairs are compared.

    Parameters
    ----------
    days_X : list of tuple
        days_X [i] : list
            Activity sequence of i-th day.
        days_X [i][j][0] : str
            Activity name of j-th activity in i-th day.
        days_X [i][j][1] : TimeInterval
            TimeInterval of j-th activity in i-th day.
    days_Y : list of tuple
        Same as 'days_X'.
    T_d : datetime.timedelta
        Unit time interval that is converted into one character.
        24 [hour] must be divisible by T_d.
    print_progress : boolean, default True
        Whether to print a progress bar.
    activity : str
        If this parameter is inputted, then consider only this activity.
        So, edit distance among activities becomes hamming ditane of this activity.
    print_progress : boolean, default True
        whether to print a progress bar

    Returns
    -------
    (mean, sd) : tuple of float
        mean : float
            Mean distance between days in 'days_X' and days in 'days_Y'.
        sd : float
            Standard deviation.

    Notes
    -----
    Even if the data contains overlapped activites, e.g., ('Take Lunch', 12[d]12[h]43[m], 12[d]13[h]15[m]), ('TV', 12[d]13[h]02[m], 12[d]13[h]30[m]), this function can work.
    """
    activity_names = []
    for day in days_X:
        for a in day:
            if a[0] not in activity_names:
                activity_names.append(a[0])
    for day in days_Y:
        for a in day:
            if a[0] not in activity_names:
                activity_names.append(a[0])
    act2char_map = None
    if activity is None:
        act2char_map = make_act2char_dic(activity_names)
    else:
        if activity not in activity_names:
            raise ValueError("parameter 'activity' does not appear in the data")
        act2char_map = {}
        for a in activity_names:
            act2char_map[a] = "0"
        act2char_map[activity] = "1"
    distances = []
    n_x, n_y = len(days_X), len(days_Y)
    n = n_x * n_y
    Xday_strings, Yday_strings = [], []
    for i, day in enumerate(days_X):
        Xday_strings.append(convert_day_into_str(day, act2char_map, T_d))
        if print_progress:
            simple_progress_bar(i, n_x, "converting days of Xday into strings", step=10)
    for i, day in enumerate(days_Y):
        Yday_strings.append(convert_day_into_str(day, act2char_map, T_d))
        if print_progress:
            simple_progress_bar(i, n_y, "converting days of Yday into strings", step=10)
    for i, (Xday_str, Yday_str) in enumerate(
        itertools.product(Xday_strings, Yday_strings)
    ):
        distances.append(Levenshtein.distance(Xday_str, Yday_str))
        if print_progress:
            simple_progress_bar(i, n, "calculate distances", step=100)
    mean = np.array(distances).mean()
    sd = np.sqrt(np.array(distances).var())
    return (mean, sd)


def mean_different_days_distance(days_list_1, days_list_2, T_d, print_progress=False):
    """
    This calculates a mean and standard deviation of different days distances.
    The different days distances between days_list_1[i] and days_list_2[i] for any i are calculated.

    Parameters
    ----------
    days_list_1 : list of list of list of tuple
        days_list_1[k] : list of list of tuple
            Activity sequences of k-th dataset.
        days_list_1[k][i][j] = (activity_name, time_interval)
            days_list_1[k][i][j][0] : str
                Activity name of j-th activity in i-th day of k-th dataset.
            days_list_1[k][i][j][1] : TimeInterval
                Time interval of j-th activity in i-th day of k-th dataset.
                This time extracts only time of day, does not contain days information (so days = 0).
        days_list_1[k][i][0].start =  timedelta(hours = 0)
        days_list_1[k][i][-1][1].end = timedelta(hours = 24)
    days_list_2 : list of list of list of tuple
        The shape is same as days_list_1
    T_d : datetime.timedelta
        Unit time interval that is converted into one character.
        24 [hour] must be divisible by T_d.
    print_progress : boolean, default False
        Whether to print a progress bar.

    Returns
    -------
    (mean_list, sd_list)
        mean_list : list of float
            mean_list[k] : float
                Mean different days distance between days_list_1[k] dataset and days_list_2[k] dataset.
        sd_list : list of float
            sd_list[k] : float
                Standard deviation of different days distance between days_list_1[k] dataset and days_list_2[k] dataset.
    """
    mean_list, sd_list = [], []
    for i, (_days_1, _days_2) in enumerate(zip(days_list_1, days_list_2)):
        temp = different_days_distance(
            _days_1, _days_2, T_d, print_progress=print_progress
        )
        mean_list.append(temp[0])
        sd_list.append(temp[1])
        print("Finish the calculation of {}-th datasets.\r".format(i), end="")
    return (mean_list, sd_list)


def simple_progress_bar(i, n, title="", step=1):
    """
    This prints a progress bar.

    Parameters
    ----------
    i : int
        Present index.
    n : int
        Total number of indexes.
    title: str
        Title of this progress bar.
    step : int
        This prints text if i is a multiple of 'step'.
    """
    if i == n - 1:
        print("\r{}, 100% completed!".format(title))
    else:
        if i % step == 0:
            print("\r{}, {}% doing.".format(title, int(100 * (i + 1) / n)), end="")


def estimate_start_time(data, act_names=None):
    """
    Parameter estimation of start time of activities from data.
    In particular, this estimates the start time of fundamental activities using polar coordinates system.
    Even if the data contains overlapped activites, e.g., ('Take Lunch', 12[d]12[h]43[m], 12[d]13[h]15[m]), ('TV', 12[d]13[h]02[m], 12[d]13[h]30[m]), this function can work.

    Parameters
    ----------
    data : list of tuple
        data[i] = (activity_name, time_interval)
        activity_name : str
        time_interval : new_functions.TimeInterval
    act_names : list of str
        Names of target activities.

    Returns
    -------
    (start_hour_mean, start_hour_sd)
    start_hour_mean : dict of float
        start_hour_mean[a] = mean start time (hours from 0:00) of an activity a.
        For example, {'Sleeping': 23.792797357000797, 'Take_Breakfast': 9.280758422589825}.
    start_hour_sd : dict of float
        start_hour_sd[a] = standard deviations of start time (hours from 0:00) of an activity a.
        For example, {'Sleeping': 1.3825254504899922, 'Take_Breakfast': 1.1465514153031076}.
    """
    if act_names is None:
        act_names = []
        for a in data:
            if a[0] not in act_names:
                act_names.append(a[0])
    start_hour_mean, start_hour_sd = {x: None for x in act_names}, {
        x: None for x in act_names
    }
    for act_name in act_names:
        start_hours = [
            a[1].start for a in data[1:] if a[0] == act_name
        ]  # exclude first activity because it was cutted to start from 0:00
        polar_x, polar_y = [], []  # polar coordinates
        for start in start_hours:
            temp_hour = utils.extract_time_of_day(start) / timedelta(
                hours=1
            )  # [hour from the start of the day]
            temp_theta = 2 * np.pi * temp_hour / 24
            polar_x.append(np.cos(temp_theta))
            polar_y.append(np.sin(temp_theta))
        mu_x, mu_y = sum(polar_x) / len(polar_x), sum(polar_y) / len(polar_y)
        temp_pi = np.arctan2(mu_y, mu_x)
        mu_theta = temp_pi if temp_pi > 0 else 2 * np.pi + temp_pi
        mu_start = 12 * mu_theta / np.pi
        start_hour_mean[act_name] = mu_start
        polar_theta = [np.arctan2(polar_y[i], polar_x[i]) for i in range(len(polar_x))]
        temp_sum = 0
        for theta in polar_theta:
            delta_theta = abs(mu_theta - theta)
            temp_sum += min(delta_theta, 2 * np.pi - delta_theta) ** 2
        sd_theta = np.sqrt(temp_sum / len(polar_theta))
        start_hour_sd[act_name] = 12 * sd_theta / np.pi
    return (start_hour_mean, start_hour_sd)


def estimate_duration_time(data, act_names=None):
    """
    Parameter estimation of duration time of activities from data.
    Even if the data contains overlapped activites, e.g., ('Take Lunch', 12[d]12[h]43[m], 12[d]13[h]15[m]), ('TV', 12[d]13[h]02[m], 12[d]13[h]30[m]), this function can work.

    Parameters
    ----------
    data : list of tuple
        data[i] = (activity_name, time_interval)
        activity_name : str
        time_interval : new_functions.TimeInterval
    act_names : list of str
        Names of target activities.

    Returns
    -------
    (duration_hour_mean, duration_hour_sd)
    duration_hour_mean : dict of float
        duration_hour_mean[a] = mean duration time of an activity a.
        For example, {'Sleeping': 6.93457506270895, 'Meal_Preparation': 0.23188109775888477}.
    duration_hour_sd : dict of float
        start_hour_sd[a] = standard deviations of duration time of an activity a.
        For example, {'Sleeping': 1.780218623323245, 'Meal_Preparation': 0.22684304752596204}.
    """
    if act_names is None:
        act_names = []
        for a in data:
            if a[0] not in act_names:
                act_names.append(a[0])
    duration_hour_mean, duration_hour_sd = {name: None for name in act_names}, {
        name: None for name in act_names
    }
    for act_name in act_names:
        # exclude first activity and end activity because it was cutted to start from 0:00 or finish until 24:00
        duration = [
            a[1].duration / timedelta(hours=1)
            for a in data[1 : len(data) - 1]
            if a[0] == act_name
        ]
        duration_hour_mean[act_name] = sum(duration) / len(duration)
        duration_hour_sd[act_name] = np.sqrt(np.array(duration).var())
    return (duration_hour_mean, duration_hour_sd)


def Aruba_activity_model(
    n_day=220, Aruba_path=Path("./aruba/data"), modify_sleep_param=False
):
    """
    This generates an activity model of Aruba dataset.

    Parameters
    ----------
    n_day : int
        Parameters of activity model are estimated by Aruba[:n_day].
        If n_day = 220, then contain 0day, 1day, ..., 219 day.
    Aruba_path : pathlib.Path, default Path('./aruba/data')
        Path to the file of Aruba dataset.
    modify_sleep_param : boolean, default False
        Whether to modify sleep parameters to avoid to overlap between 'Sleeping' and 'Take_Breakfast'.
    Returns
    -------
    Aruba_act_model : dict of list
        This shape is similar to activity_model.basic_acitivty_model.
    """
    Aruba = load_CASAS_Aruba(Aruba_path)

    Aruba_n_day = []
    for a in Aruba:
        if a[1].end.days < n_day:
            Aruba_n_day.append(a)
        else:
            last_act = (a[0], utils.TimeInterval(a[1].start, timedelta(days=n_day)))
            Aruba_n_day.append(last_act)
            break

    activity_names = []
    for a in Aruba_n_day:
        if a[0] not in activity_names:
            activity_names.append(a[0])
    fundamental_act_names = ["Sleeping", "Take_Breakfast", "Take_Lunch", "Take_Dinner"]
    for a in fundamental_act_names:
        if a not in activity_names:
            raise ValueError(
                "Parameters of fundamental activitiy {} cannot be learned.".format(a)
            )
    start_hour_mean, start_hour_sd = estimate_start_time(
        Aruba_n_day, fundamental_act_names
    )
    duration_hour_mean, duration_hour_sd = estimate_duration_time(
        Aruba_n_day, activity_names
    )
    frequency_mean, _ = estimate_daily_frequency(Aruba_n_day, activity_names)

    # fundamental acitivities
    sleep_start_sd, sleep_duration_sd = None, None
    if modify_sleep_param:
        sleep_start_sd = timedelta(
            hours=start_hour_sd["Take_Breakfast"] / 2
        ) / timedelta(minutes=1)
        sleep_duration_sd = timedelta(
            hours=duration_hour_sd["Take_Breakfast"] / 2
        ) / timedelta(minutes=1)
    else:
        sleep_start_sd = timedelta(hours=start_hour_sd["Take_Breakfast"]) / timedelta(
            minutes=1
        )
        sleep_duration_sd = timedelta(
            hours=duration_hour_sd["Take_Breakfast"]
        ) / timedelta(minutes=1)

    take_breakfast = activity_model.FundamentalActivity(
        "Take_Breakfast",
        timedelta(hours=start_hour_mean["Take_Breakfast"]) / timedelta(minutes=1),
        timedelta(hours=start_hour_sd["Take_Breakfast"]) / timedelta(minutes=1),
        timedelta(hours=duration_hour_mean["Take_Breakfast"]) / timedelta(minutes=1),
        timedelta(hours=duration_hour_sd["Take_Breakfast"]) / timedelta(minutes=1),
        ["Dinner_Table_Chair", "Entrance"],
        "#ffd700",
        {},
    )
    take_lunch = activity_model.FundamentalActivity(
        "Take_Lunch",
        timedelta(hours=start_hour_mean["Take_Lunch"]) / timedelta(minutes=1),
        timedelta(hours=start_hour_sd["Take_Lunch"]) / timedelta(minutes=1),
        timedelta(hours=duration_hour_mean["Take_Lunch"]) / timedelta(minutes=1),
        timedelta(hours=duration_hour_sd["Take_Lunch"]) / timedelta(minutes=1),
        ["Dinner_Table_Chair", "Entrance"],
        "#ffa500",
        {},
    )
    take_dinner = activity_model.FundamentalActivity(
        "Take_Dinner",
        timedelta(hours=start_hour_mean["Take_Dinner"]) / timedelta(minutes=1),
        timedelta(hours=start_hour_sd["Take_Dinner"]) / timedelta(minutes=1),
        timedelta(hours=duration_hour_mean["Take_Dinner"]) / timedelta(minutes=1),
        timedelta(hours=duration_hour_sd["Take_Dinner"]) / timedelta(minutes=1),
        ["Dinner_Table_Chair", "Entrance"],
        "#b22222",
        {},
    )
    sleeping = activity_model.FundamentalActivity(
        "Sleeping",
        timedelta(hours=start_hour_mean["Sleeping"]) / timedelta(minutes=1),
        sleep_start_sd,
        timedelta(hours=duration_hour_mean["Sleeping"]) / timedelta(minutes=1),
        sleep_duration_sd,
        ["Bed"],
        "#696969",
        {},
    )
    # necessary activities.
    nap = activity_model.NecessaryActivity(
        "Nap",
        frequency_mean["Nap"],
        timedelta(hours=duration_hour_mean["Nap"]) / timedelta(minutes=1),
        timedelta(hours=duration_hour_sd["Nap"]) / timedelta(minutes=1),
        ["Bed", "Sofa"],
        "#7cfc00",
        {},
    )
    wash_dishes = activity_model.NecessaryActivity(
        "Wash_Dishes",
        frequency_mean["Wash_Dishes"],
        timedelta(hours=duration_hour_mean["Wash_Dishes"]) / timedelta(minutes=1),
        timedelta(hours=duration_hour_sd["Wash_Dishes"]) / timedelta(minutes=1),
        ["Kitchen_Stove"],
        "#008080",
        {"faucet in the kitchen": 1},
    )
    meal_preparation = activity_model.NecessaryActivity(
        "Meal_Preparation",
        frequency_mean["Meal_Preparation"],
        timedelta(hours=duration_hour_mean["Meal_Preparation"]) / timedelta(minutes=1),
        timedelta(hours=duration_hour_sd["Meal_Preparation"]) / timedelta(minutes=1),
        ["Kitchen_Stove"],
        "#ff0000",
        {"stove in the kitchen": 10, "faucet in the kitchen": 1},
    )
    housekeeping = activity_model.NecessaryActivity(
        "Housekeeping",
        frequency_mean["Housekeeping"],
        timedelta(hours=duration_hour_mean["Housekeeping"]) / timedelta(minutes=1),
        timedelta(hours=duration_hour_sd["Housekeeping"]) / timedelta(minutes=1),
        [],
        "#bc8f8f",
        {},
    )
    # random activities
    relax = activity_model.RandomActivity(
        "Relax",
        timedelta(hours=duration_hour_mean["Relax"]) / timedelta(minutes=1),
        timedelta(hours=duration_hour_sd["Relax"]) / timedelta(minutes=1),
        ["Sofa", "Bed", "Desk_Chair"],
        "#d3d3d3",
        {},
        weight=frequency_mean["Relax"],
    )
    go_out = activity_model.RandomActivity(
        "Go_out",
        timedelta(hours=duration_hour_mean["Go_out"]) / timedelta(minutes=1),
        timedelta(hours=duration_hour_sd["Go_out"]) / timedelta(minutes=1),
        ["Entrance"],
        "#ffe4c4",
        {},
        weight=frequency_mean["Go_out"],
    )
    work = activity_model.RandomActivity(
        "Work",
        timedelta(hours=duration_hour_mean["Work"]) / timedelta(minutes=1),
        timedelta(hours=duration_hour_sd["Work"]) / timedelta(minutes=1),
        ["Desk_Chair", "Desk"],
        "#808000",
        {},
        weight=frequency_mean["Work"],
    )
    other = activity_model.RandomActivity(
        "Other",
        timedelta(hours=duration_hour_mean["Other"]) / timedelta(minutes=1),
        timedelta(hours=duration_hour_sd["Other"]) / timedelta(minutes=1),
        [],
        "#800000",
        {},
        weight=frequency_mean["Other"],
    )
    # activitiy model
    Aruba_act_model = {
        "fundamental_activities": [take_breakfast, take_lunch, take_dinner, sleeping],
        "necessary_activities": [nap, wash_dishes, meal_preparation, housekeeping],
        "random_activities": [relax, go_out, work, other],
        "fundamental_activity_order": [
            take_breakfast,
            take_lunch,
            take_dinner,
            sleeping,
        ],
        # The order of the fundamental activities: the smaller activity's index is, the earlier the activity is done.
        "sleep": [sleeping],
    }

    return Aruba_act_model


def generate_random_activity_sequence(
    start_day, end_day, name_list, mean_list, sd_list
):
    """
    This generates an activity sequence at random.
    But the duration time of each activity is estimated by a normal distribution.

    Parameters
    ----------
    start_day : datetime.timedelta
        Start day of the sequence.
    end_day : datetime.timedelta
        End day of the sequence.
    name_list : list of str
        Activity names.
    mean_list : list of float
        Mean duration time [minutes] of activities, the order is same as 'name_list'.
    sd_list : list of float
        Standard deviations of duration time [minutes] of activities, the order is same as 'name_list'.

    Returns
    -------
    ret : list of tuple
        ret[i] = (act_name, time_interval)
        act_name : str
            Activity name.
        time_interval : new_functions.TimeInterval
            Time interval of this activity.
    """
    ret = []
    start_time, end_time = timedelta(days=start_day), timedelta(days=end_day + 1)
    t = start_time
    index_list = [i for i in range(len(name_list))]
    while t < end_time:
        index = np.random.choice(index_list)
        lower_limit = mean_list[index] / 10
        upper_limit = mean_list[index] * 19 / 10
        is_valid = False
        try_count = 0
        duration = None
        max_try = 100
        while not is_valid:
            duration = np.random.normal(mean_list[index], sd_list[index])
            if lower_limit <= duration <= upper_limit:
                is_valid = True
            try_count += 1
            if try_count >= max_try:
                raise ValueError(
                    "This failed to sample duration time of {} times.".format(try_count)
                )
        duration_time = timedelta(minutes=duration)
        ret.append((name_list[index], utils.TimeInterval(t, t + duration_time)))
        t += duration_time
    ret[-1] = (ret[-1][0], utils.TimeInterval(ret[-1][1].start, timedelta(end_day + 1)))
    return ret


def Aruba_activity_model_sleep_only(n_day=220, Aruba_path=Path("./aruba/data")):
    """
    This generates an activity model of Aruba dataset.
    Sleep is only estimated as fundamental activity.
    All other activties are classified into random activity.

    Parameters
    ----------
    n_day : int
        Parameters of activity model are estimated by Aruba[:n_day].
        If n_day = 220, then contain 0day, 1day, ..., 219 day.
    Aruba_path : pathlib.Path, default Path('./aruba/data')
        Path to the file of Aruba dataset.

    Returns
    -------
    Aruba_act_model : dict of list
        This shape is similar to activity_model.basic_acitivty_model
    """
    Aruba = load_CASAS_Aruba(Aruba_path)
    Aruba_n_day = []
    for a in Aruba:
        if a[1].end.days < n_day:
            Aruba_n_day.append(a)
        else:
            last_act = (a[0], utils.TimeInterval(a[1].start, timedelta(days=n_day)))
            Aruba_n_day.append(last_act)
            break

    activity_names = []
    for a in Aruba_n_day:
        if a[0] not in activity_names:
            activity_names.append(a[0])
    fundamental_act_names = ["Sleeping"]
    for a in fundamental_act_names:
        if a not in activity_names:
            raise ValueError(
                "Parameters of fundamental activitiy {} cannot be learned.".format(a)
            )
    start_hour_mean, start_hour_sd = estimate_start_time(
        Aruba_n_day, fundamental_act_names
    )
    duration_hour_mean, duration_hour_sd = estimate_duration_time(
        Aruba_n_day, activity_names
    )
    frequency_mean, _ = estimate_daily_frequency(Aruba_n_day, activity_names)

    # fundamental acitivites
    sleeping = activity_model.FundamentalActivity(
        "Sleeping",
        timedelta(hours=start_hour_mean["Sleeping"]) / timedelta(minutes=1),
        timedelta(hours=start_hour_sd["Sleeping"]) / timedelta(minutes=1),
        timedelta(hours=duration_hour_mean["Sleeping"]) / timedelta(minutes=1),
        timedelta(hours=duration_hour_sd["Sleeping"]) / timedelta(minutes=1),
        ["Bed"],
        "#696969",
        {},
    )
    # random activities
    take_breakfast = activity_model.RandomActivity(
        "Take_Breakfast",
        timedelta(hours=duration_hour_mean["Take_Breakfast"]) / timedelta(minutes=1),
        timedelta(hours=duration_hour_sd["Take_Breakfast"]) / timedelta(minutes=1),
        ["Dinner_Table_Chair", "Entrance"],
        "#ffd700",
        {},
    )
    take_lunch = activity_model.RandomActivity(
        "Take_Lunch",
        timedelta(hours=duration_hour_mean["Take_Lunch"]) / timedelta(minutes=1),
        timedelta(hours=duration_hour_sd["Take_Lunch"]) / timedelta(minutes=1),
        ["Dinner_Table_Chair", "Entrance"],
        "#ffa500",
        {},
    )
    take_dinner = activity_model.RandomActivity(
        "Take_Dinner",
        timedelta(hours=duration_hour_mean["Take_Dinner"]) / timedelta(minutes=1),
        timedelta(hours=duration_hour_sd["Take_Dinner"]) / timedelta(minutes=1),
        ["Dinner_Table_Chair", "Entrance"],
        "#b22222",
        {},
    )
    nap = activity_model.RandomActivity(
        "Nap",
        timedelta(hours=duration_hour_mean["Nap"]) / timedelta(minutes=1),
        timedelta(hours=duration_hour_sd["Nap"]) / timedelta(minutes=1),
        ["Bed", "Sofa"],
        "#7cfc00",
        {},
    )
    wash_dishes = activity_model.RandomActivity(
        "Wash_Dishes",
        timedelta(hours=duration_hour_mean["Wash_Dishes"]) / timedelta(minutes=1),
        timedelta(hours=duration_hour_sd["Wash_Dishes"]) / timedelta(minutes=1),
        ["Kitchen_Stove"],
        "#008080",
        {"faucet in the kitchen": 1},
    )
    meal_preparation = activity_model.RandomActivity(
        "Meal_Preparation",
        timedelta(hours=duration_hour_mean["Meal_Preparation"]) / timedelta(minutes=1),
        timedelta(hours=duration_hour_sd["Meal_Preparation"]) / timedelta(minutes=1),
        ["Kitchen_Stove"],
        "#ff0000",
        {"stove in the kitchen": 10, "faucet in the kitchen": 1},
    )
    housekeeping = activity_model.RandomActivity(
        "Housekeeping",
        timedelta(hours=duration_hour_mean["Housekeeping"]) / timedelta(minutes=1),
        timedelta(hours=duration_hour_sd["Housekeeping"]) / timedelta(minutes=1),
        [],
        "#bc8f8f",
        {},
    )
    relax = activity_model.RandomActivity(
        "Relax",
        timedelta(hours=duration_hour_mean["Relax"]) / timedelta(minutes=1),
        timedelta(hours=duration_hour_sd["Relax"]) / timedelta(minutes=1),
        ["Sofa", "Bed", "Desk_Chair"],
        "#d3d3d3",
        {},
    )
    go_out = activity_model.RandomActivity(
        "Go_out",
        timedelta(hours=duration_hour_mean["Go_out"]) / timedelta(minutes=1),
        timedelta(hours=duration_hour_sd["Go_out"]) / timedelta(minutes=1),
        ["Entrance"],
        "#ffe4c4",
        {},
    )
    work = activity_model.RandomActivity(
        "Work",
        timedelta(hours=duration_hour_mean["Work"]) / timedelta(minutes=1),
        timedelta(hours=duration_hour_sd["Work"]) / timedelta(minutes=1),
        ["Desk_Chair", "Desk"],
        "#808000",
        {},
    )
    other = activity_model.RandomActivity(
        "Other",
        timedelta(hours=duration_hour_mean["Other"]) / timedelta(minutes=1),
        timedelta(hours=duration_hour_sd["Other"]) / timedelta(minutes=1),
        [],
        "#800000",
        {},
    )
    # activity model
    Aruba_act_model = {
        "fundamental_activities": [sleeping],
        "necessary_activities": [],
        "random_activities": [
            take_breakfast,
            take_lunch,
            take_dinner,
            nap,
            wash_dishes,
            meal_preparation,
            housekeeping,
            relax,
            go_out,
            work,
            other,
        ],
        "fundamental_activity_order": [sleeping],
        # The order of the fundamental activities: the smaller activity's index is, the earlier the activity is done.
        "sleep": [sleeping],
    }

    return Aruba_act_model


def load_kasterenActData(
    path,
    first_date=datetime(year=2008, month=2, day=26),
    last_date=datetime(year=2008, month=3, day=20),
    add_other=True,
    remove_exception=True,
):
    """
    This loads kasterenActData dataset [1] as an activity sequence.
    [1] T. Van Kasteren et al., "Accurate activity recognition in a home setting." Proc. of the 10th international conference on Ubiquitous computing (UbiComp 2008), Sep. 2008, 1–9.
    Original data contains the following activities: prepare breakfast, take shower, prepare dinner, go to bed, use toilet, leave house, and get drink.

    Parameters
    ----------
    path : pathlib.Path
        Path of the kasterenActData.txt.
    first_date : datetime.datetime, default datetime(year = 2008, month = 2, day = 26)
        This function cuts the dates before the start of first_date.
    last_date : datetime.datetime, default datetime(year = 2008, month = 3, day = 20)
        This function cuts the dates after the end of last_date.
    add_other : boolean, True
        Whether to assign the label 'Other' to an undefined activity.
    remove_exception : boolean, True
        Whether to replace exception activities with new activity 'exception'.
        Exception activities are:
        (1) three long leave house longer than 23 hours,
        (2) four short go to bed less than 6 hours, and
        (3) four early get drink before 12 am.
        Removed activities are replaced as 'Other' activity.

    Returns
    -------
    ret : list of tuple
        ret[i] = (act_name, time_interval)
            act_name : str
                Activity name.
            time_interval : new_functions.TimeInterval
                Time interval of this activity.
    """

    ret = None
    act_map = {}  # id: name of the activity
    month_map = {"Feb": 2, "Mar": 3}
    raw_data = []
    with open(path, mode="r") as f:
        for i, line in enumerate(f):
            if 5 <= i <= 11:  # id and activity
                act_map[line[: line.find(":")]] = line[
                    line.find("'") + 1 : line.rfind("'")
                ]
            if 15 <= i <= 259:  # data
                splitted_line = line.split()
                start_time = datetime(
                    year=int(splitted_line[0].split("-")[2]),
                    month=month_map[splitted_line[0].split("-")[1]],
                    day=int(splitted_line[0].split("-")[0]),
                    hour=int(splitted_line[1].split(":")[0]),
                    minute=int(splitted_line[1].split(":")[1]),
                    second=int(splitted_line[1].split(":")[2]),
                )
                end_time = datetime(
                    year=int(splitted_line[2].split("-")[2]),
                    month=month_map[splitted_line[2].split("-")[1]],
                    day=int(splitted_line[2].split("-")[0]),
                    hour=int(splitted_line[3].split(":")[0]),
                    minute=int(splitted_line[3].split(":")[1]),
                    second=int(splitted_line[3].split(":")[2]),
                )
                act_name = act_map[splitted_line[4]]
                raw_data.append(
                    (act_name, utils.TimeInterval(start=start_time, end=end_time))
                )

    # extract between first_date and last_date
    end_date = last_date + timedelta(days=1)
    data_24 = []
    is_in = False
    for a, interval in raw_data:
        if first_date < interval.end and not is_in:
            is_in = True
            data_24.append(
                (
                    a,
                    utils.TimeInterval(
                        first_date - first_date, interval.end - first_date
                    ),
                )
            )
        elif end_date < interval.end and is_in:
            data_24.append(
                (
                    a,
                    utils.TimeInterval(
                        interval.start - first_date, end_date - first_date
                    ),
                )
            )
            break
        elif is_in and first_date < interval.start and interval.end < end_date:
            data_24.append(
                (
                    a,
                    utils.TimeInterval(
                        interval.start - first_date, interval.end - first_date
                    ),
                )
            )
    ret = data_24

    if add_other:
        data_with_other = []
        other_data = []
        other = "Other"
        if ret[0][1].start != first_date - first_date:
            other_data.append(
                (
                    other,
                    utils.TimeInterval(
                        first_date - first_date, ret[0][1].start - first_date
                    ),
                )
            )
        end_index = len(ret) - 1
        _temp_end = ret[0][
            1
        ].end  # end time of past one activity or past multiple activities that overlap
        _other_start, _other_end = (
            ret[0][1].end,
            None,
        )  # start time and end time of other activity
        i = 1
        while i <= end_index:
            now_start, now_end = (
                ret[i][1].start,
                ret[i][1].end,
            )  # start time and end time of present acitivity
            if now_start < _temp_end:
                if _temp_end < now_end:
                    _temp_end = now_end
            else:
                _other_start = _temp_end
                _other_end = now_start
                other_data.append((other, utils.TimeInterval(_other_start, _other_end)))
                _temp_end = now_end
            i += 1
        if data_24[-1][1].end != end_date - first_date:
            other_data.append(
                (
                    other,
                    utils.TimeInterval(
                        ret[-1][1].end - first_date, end_date - first_date
                    ),
                )
            )
        data_with_other = sorted(ret + other_data, key=lambda x: x[1].start)
        ret = data_with_other

    exception = "exception"
    if remove_exception:
        for i, a in enumerate(ret):
            if (
                (a[0] == "leave house" and a[1].duration / timedelta(hours=1) > 23)
                or (
                    a[0] == "get drink"
                    and a[1].start < timedelta(days=a[1].start.days, hours=12)
                )
                or (a[0] == "go to bed" and a[1].duration / timedelta(hours=1) < 6)
            ):
                ret[i] = (exception, utils.TimeInterval(start=a[1].start, end=a[1].end))

    return ret


def kasteren_activity_model(kasteren, n_day=None, modify_sleep_param=True):
    """
    This generates an activity model of kasteren dataset.

    Parameters
    ----------
    kasteren : list of tuple
         kasteren[i] = (act_name, time_interval)
            act_name : str
                Activity name.
            time_interval : new_functions.TimeInterval
                Time interval of this activity.
    n_day : int, default None
        Parameters of an activity model are estimated by using kasteren[:n_day] data.
        If n_day = 220, then data of 0day, 1day, ..., and 219 day is used.
    modify_sleep_param : boolean
        Whether to shrink sleep parameters not to overlap with the 'prepare_Breakfast' activity.

    Returns
    -------
    kasteren_act_model : dict of list
        This shape is similar to activity_model.basic_acitivty_model
    """
    if n_day is None:
        n_day = kasteren[-1][1].start.days
    kasteren_n_day = []
    for a in kasteren:
        if a[1].end.days < n_day:
            kasteren_n_day.append(a)
        else:
            last_act = (a[0], utils.TimeInterval(a[1].start, timedelta(days=n_day)))
            kasteren_n_day.append(last_act)
            break

    activity_names = []
    for a in kasteren_n_day:
        if a[0] not in activity_names:
            activity_names.append(a[0])
    fundamental_act_names = [
        "go to bed",
        "prepare Breakfast",
        "take shower",
        "prepare Dinner",
    ]
    for a in fundamental_act_names:
        if a not in activity_names:
            raise ValueError(
                "Parameters of fundamental activitiy {} cannot be learned.".format(a)
            )
    start_hour_mean, start_hour_sd = estimate_start_time(
        kasteren_n_day, fundamental_act_names
    )
    duration_hour_mean, duration_hour_sd = estimate_duration_time(
        kasteren_n_day, activity_names
    )
    frequency_mean, _ = estimate_daily_frequency(kasteren_n_day, activity_names)

    take_breakfast = activity_model.FundamentalActivity(
        "prepare Breakfast",
        timedelta(hours=start_hour_mean["prepare Breakfast"]) / timedelta(minutes=1),
        timedelta(hours=start_hour_sd["prepare Breakfast"]) / timedelta(minutes=1),
        timedelta(hours=duration_hour_mean["prepare Breakfast"]) / timedelta(minutes=1),
        timedelta(hours=duration_hour_sd["prepare Breakfast"]) / timedelta(minutes=1),
        ["Dinner_Table_Chair", "Entrance"],
        "#ffd700",
        {},
    )
    take_shower = activity_model.FundamentalActivity(
        "take shower",
        timedelta(hours=start_hour_mean["take shower"]) / timedelta(minutes=1),
        timedelta(hours=start_hour_sd["take shower"]) / timedelta(minutes=1),
        timedelta(hours=duration_hour_mean["take shower"]) / timedelta(minutes=1),
        timedelta(hours=duration_hour_sd["take shower"]) / timedelta(minutes=1),
        ["Bathroom_Door"],
        "#00ffff",
        {"faucet in the bathroom": 1},
    )
    take_dinner = activity_model.FundamentalActivity(
        "prepare Dinner",
        timedelta(hours=start_hour_mean["prepare Dinner"]) / timedelta(minutes=1),
        timedelta(hours=start_hour_sd["prepare Dinner"]) / timedelta(minutes=1),
        timedelta(hours=duration_hour_mean["prepare Dinner"]) / timedelta(minutes=1),
        timedelta(hours=duration_hour_sd["prepare Dinner"]) / timedelta(minutes=1),
        ["Dinner_Table_Chair", "Entrance"],
        "#b22222",
        {},
    )
    sleep_start_sd = timedelta(hours=start_hour_sd["go to bed"]) / timedelta(minutes=1)
    sleep_duration_sd = timedelta(hours=duration_hour_sd["go to bed"]) / timedelta(
        minutes=1
    )
    if modify_sleep_param:
        sleep_start_sd /= 3
        sleep_duration_sd /= 3
    sleeping = activity_model.FundamentalActivity(
        "go to bed",
        timedelta(hours=start_hour_mean["go to bed"]) / timedelta(minutes=1),
        sleep_start_sd,
        timedelta(hours=duration_hour_mean["go to bed"]) / timedelta(minutes=1),
        sleep_duration_sd,
        ["Bed"],
        "#696969",
        {},
    )
    use_toilet = activity_model.NecessaryActivity(
        "use toilet",
        frequency_mean["use toilet"],
        timedelta(hours=duration_hour_mean["use toilet"]) / timedelta(minutes=1),
        timedelta(hours=duration_hour_sd["use toilet"]) / timedelta(minutes=1),
        ["Toilet_Door"],
        "#4b0082",
        {},
    )
    # go_out = activity_model.NecessaryActivity('leave house', frequency_mean['leave house'],
    #                         timedelta(hours = duration_hour_mean['leave house']) / timedelta(minutes = 1),
    #                         timedelta(hours = duration_hour_sd['leave house']) / timedelta(minutes = 1), ['Entrance'], '#ffe4c4', {})
    go_out = activity_model.RandomActivity(
        "leave house",
        timedelta(hours=duration_hour_mean["leave house"]) / timedelta(minutes=1),
        timedelta(hours=duration_hour_sd["leave house"]) / timedelta(minutes=1),
        ["Entrance"],
        "#ffe4c4",
        {},
        weight=frequency_mean["leave house"],
    )
    get_drink = activity_model.RandomActivity(
        "get drink",
        timedelta(hours=duration_hour_mean["get drink"]) / timedelta(minutes=1),
        timedelta(hours=duration_hour_sd["get drink"]) / timedelta(minutes=1),
        ["Kitchen_Stove"],
        "#ff0000",
        {},
        weight=frequency_mean["get drink"],
    )
    other = activity_model.RandomActivity(
        "Other",
        timedelta(hours=duration_hour_mean["Other"]) / timedelta(minutes=1),
        timedelta(hours=duration_hour_sd["Other"]) / timedelta(minutes=1),
        [],
        "#800000",
        {},
        weight=frequency_mean["Other"],
    )
    kasteren_act_model = {
        "fundamental_activities": [take_breakfast, take_shower, take_dinner, sleeping],
        "necessary_activities": [use_toilet],
        "random_activities": [go_out, get_drink, other],
        "fundamental_activity_order": [
            take_breakfast,
            take_shower,
            take_dinner,
            sleeping,
        ],
        # The order of the fundamental activities: the smaller activity's index is, the earlier the activity is done.
        "sleep": [sleeping],
    }

    return kasteren_act_model


def kasteren_activity_model_sleep_only(kasteren, n_day=None, modify_sleep_param=True):
    """
    This generates an activity model of kasteren dataset.
    Sleep is only estimated as a fundamental activity.
    All other activties are classified into random activity.

    Parameters
    ----------
    kasteren : list of tuple
         kasteren[i] = (act_name, time_interval)
            act_name : str
                Activity name.
            time_interval : new_functions.TimeInterval
                Time interval of this activity.
    n_day : int, default None
        Parameters of an activity model are estimated by using kasteren[:n_day] data.
        If n_day = 220, then data of 0day, 1day, ..., and 219 day is used.
    modify_sleep_param : boolean
        Whether to shrink sleep parameters not to overlap with the 'prepare_Breakfast' activity.

    Returns
    -------
    kasteren_act_model : dict of list
        This shape is similar to activity_model.basic_acitivty_model.
    """
    if n_day is None:
        n_day = kasteren[-1][1].start.days
    kasteren_n_day = []
    for a in kasteren:
        if a[1].end.days < n_day:
            kasteren_n_day.append(a)
        else:
            last_act = (a[0], utils.TimeInterval(a[1].start, timedelta(days=n_day)))
            kasteren_n_day.append(last_act)
            break

    activity_names = []
    for a in kasteren_n_day:
        if a[0] not in activity_names:
            activity_names.append(a[0])
    fundamental_act_names = ["go to bed"]
    for a in fundamental_act_names:
        if a not in activity_names:
            raise ValueError(
                "Parameters of fundamental activitiy {} cannot be learned.".format(a)
            )

    start_hour_mean, start_hour_sd = estimate_start_time(
        kasteren_n_day, fundamental_act_names
    )
    duration_hour_mean, duration_hour_sd = estimate_duration_time(
        kasteren_n_day, activity_names
    )
    frequency_mean, _ = estimate_daily_frequency(kasteren_n_day, activity_names)

    sleep_start_sd = timedelta(hours=start_hour_sd["go to bed"]) / timedelta(minutes=1)
    sleep_duration_sd = timedelta(hours=duration_hour_sd["go to bed"]) / timedelta(
        minutes=1
    )
    if modify_sleep_param:
        sleep_start_sd /= 3
        sleep_duration_sd /= 3
    sleeping = activity_model.FundamentalActivity(
        "go to bed",
        timedelta(hours=start_hour_mean["go to bed"]) / timedelta(minutes=1),
        sleep_start_sd,
        timedelta(hours=duration_hour_mean["go to bed"]) / timedelta(minutes=1),
        sleep_duration_sd,
        ["Bed"],
        "#696969",
        {},
    )
    take_breakfast = activity_model.RandomActivity(
        "prepare Breakfast",
        timedelta(hours=duration_hour_mean["prepare Breakfast"]) / timedelta(minutes=1),
        timedelta(hours=duration_hour_sd["prepare Breakfast"]) / timedelta(minutes=1),
        ["Dinner_Table_Chair", "Entrance"],
        "#ffd700",
        {},
        weight=frequency_mean["prepare Breakfast"],
    )
    take_shower = activity_model.RandomActivity(
        "take shower",
        timedelta(hours=duration_hour_mean["take shower"]) / timedelta(minutes=1),
        timedelta(hours=duration_hour_sd["take shower"]) / timedelta(minutes=1),
        ["Bathroom_Door"],
        "#00ffff",
        {"faucet in the bathroom": 1},
        weight=frequency_mean["take shower"],
    )
    take_dinner = activity_model.RandomActivity(
        "prepare Dinner",
        timedelta(hours=duration_hour_mean["prepare Dinner"]) / timedelta(minutes=1),
        timedelta(hours=duration_hour_sd["prepare Dinner"]) / timedelta(minutes=1),
        ["Dinner_Table_Chair", "Entrance"],
        "#ffd700",
        {},
        weight=frequency_mean["prepare Dinner"],
    )
    use_toilet = activity_model.RandomActivity(
        "use toilet",
        timedelta(hours=duration_hour_mean["use toilet"]) / timedelta(minutes=1),
        timedelta(hours=duration_hour_sd["use toilet"]) / timedelta(minutes=1),
        ["Toilet_Door"],
        "#4b0082",
        {},
        weight=frequency_mean["use toilet"],
    )
    # Other','leave house', 'get drink',
    go_out = activity_model.RandomActivity(
        "leave house",
        timedelta(hours=duration_hour_mean["leave house"]) / timedelta(minutes=1),
        timedelta(hours=duration_hour_sd["leave house"]) / timedelta(minutes=1),
        ["Entrance"],
        "#ffe4c4",
        {},
        weight=frequency_mean["leave house"],
    )
    get_drink = activity_model.RandomActivity(
        "get drink",
        timedelta(hours=duration_hour_mean["get drink"]) / timedelta(minutes=1),
        timedelta(hours=duration_hour_sd["get drink"]) / timedelta(minutes=1),
        ["Kitchen_Stove"],
        "#ff0000",
        {},
        weight=frequency_mean["get drink"],
    )
    other = activity_model.RandomActivity(
        "Other",
        timedelta(hours=duration_hour_mean["Other"]) / timedelta(minutes=1),
        timedelta(hours=duration_hour_sd["Other"]) / timedelta(minutes=1),
        [],
        "#800000",
        {},
        weight=frequency_mean["Other"],
    )
    kasteren_act_model = {
        "fundamental_activities": [sleeping],
        "necessary_activities": [],
        "random_activities": [
            take_breakfast,
            take_shower,
            take_dinner,
            use_toilet,
            go_out,
            get_drink,
            other,
        ],
        "fundamental_activity_order": [sleeping],
        # The order of the fundamental activities: the smaller activity's index is, the earlier the activity is done.
        "sleep": [sleeping],
    }

    return kasteren_act_model
