# new functions
# some functions refers to the program of SISG4HEIAlpha from, [1] C. Jiang and A. Mita, "SISG4HEI_Alpha: Alpha version of simulated indoor scenario generator for houses with elderly individuals." Journal of Building Engineering, 35-101963(2021), and [2] https://github.com/Idontwan/SISG4HEI_Alpha.
import itertools
import json
import math
import os
import random
import sys
from mpl_toolkits.mplot3d import Axes3D
from copy import deepcopy
from pathlib import Path
from datetime import timedelta
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import scipy.stats as stats

# self-made
import activity_model
import anomaly
import floor_plan
import sensor_model

DIRECT_WALKING = 'Direct'
WANDERING_WALKING = 'Wander'
NOT_WALKING = 'stop'

def generate_data_folder(path, folder_name):
    """
    This class generates a path of a 'folder_name' folder in 'path'.
    If the folder does not exist, this generates the folder.
    
    Parameters
    ----------
    path : pathlib.Path
        Path that includes the folder.
    folder_name : str
        Name of the folder.
        
    Returns
    -------
    folder_path : pathlib.Path
        Path to the folder.
    """
    folder_path = path / Path(folder_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return folder_path

def present_date_and_time():
    """
    This class returns present date and time as str.
    
    Returns
    -------
    date_and_time : str
        Date and time, e.g., 2022_10_23_22_04_08.
    """
    now = datetime.datetime.now()
    return(now.strftime('%Y_%m_%d_%H_%M_%S'))
    

class ActivityDataPoint:
    """
    This class represets a data point in a time series data of activities.
    An activity sequence is a list of this class.
    
    Attributes
    ----------
    activity : activity_model.Activity class or activity_model.MetaActivity class
        Activity information at this data point.
    start : datetime.timedeta
        Start time of this activity.
    end : datetime.timedelta
        End time of this activity.
    duration : datetime.timedelta
        Duration time of this activity.
    place : str
        Place that the resident does this activity at this data point.
        
    Example
    -------
    activity = activity_model.sleep
    start = timedelta(days = 3, hours = 10)
    end = timedelta(days = 4, hours = 5)
    duration = end - start
    """

    def __init__(self, activity, start, end, place):
        """
        Initialization of ActivityDataPoint class.

        Parameters
        ----------
        Same as Attributes of this class.
        """
        self.__activity = activity
        self.__start = start
        self.__end = end
        self.__place = place
        self.__duration = end - start
        
    def __str__(self):
        return "<ActivityDataPoint> {} from {} to {} at {}".format(self.activity, self.start, self.end, self.place)
    
    # getter
    @property
    def activity(self):
        return self.__activity
    @property
    def start(self):
        return self.__start
    @property
    def end(self):
        return self.__end
    @property
    def duration(self):
        return self.__duration
    @property
    def place(self):
        return self.__place


def generate_activity_sequence(start_day, end_day, path, original_act_model = activity_model.basic_activity_model, 
                               state_anomaly_labels = dict(), anomaly_parameters = dict()):
    """
    This generates an activity sequence.

    Parameters
    ----------
    start_day : int
        Day when this function starts to sample.
    end_day : int
        The day when this function finishes sampling. This simulator generates from `0:00 in start_day day` to `0:00 in end_day day (= 24:00 in (end_day - 1) day)`.
    path : pathlib.Path
        Path to the layout data.
    original_act_model : dict
        For example, see activity_model.basic_activity_model.
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
    activity_sequence : list of ActivityDataPoint
        Time series data of activities.
        
    Notes
    -----
    original_act_model may be modified by the condition of the layout, e.g., activity 'Watch TV' at TV is excluded if the layout does not have any TVs in it.
    """ 
    act_model = activity_model.determine_activity_set(path, original_act_model)
    
    is_changed = False  # the flag of difference between act_model and original_act_model
    for c in ['fundamental_activities', 'necessary_activities', 'random_activities']:
        if len(act_model[c]) != len(original_act_model[c]):
            is_changed = True
    if is_changed == True:
        print('The original_act_model was modified to activity_model.determine_activity_set(path, original_act_model), so that it can be applied in the layout.')
        
    temp_sequence = list(activity_generator(start_day, end_day - 1, act_model, 
                                            state_anomaly_labels = state_anomaly_labels, anomaly_parameters = anomaly_parameters))
    
    # connect adjacent same activities in the schedule, in particular for sleep activity over 2 days
    # e.g.,
    # [('Sleep', s0, e0), (Sleep', s1, e1), ('Toilet', s2, e2), ('Work', s3, e3), ('TV', s4, e4), ('TV', s5, e5), 
    #  ('TV', s6, e6), ('Sleep', s7, e7), ('Sleep', s8, e8)]
    # intervals = [(0, 1), (2, 2), (3, 3), (4, 6), (7, 8)]
    # -> [('Sleep', s0, e1), ('Toilet', s2, e2), ('Work', s3, e3), ('TV', s4, e6), ('Sleep', s7, e8)]
    intervals = []
    start, end, last_index = 0, 0, len(temp_sequence) - 1
    while end <= last_index:
        start = end
        name = temp_sequence[end][0].name
        while True:
            end += 1
            if end > last_index:
                break
            if temp_sequence[end][0].name != name:
                break
        intervals.append((start, end - 1))
    connected_sequence = []
    for (s, e) in intervals:
        connected_sequence.append((temp_sequence[s][0], temp_sequence[s][1], temp_sequence[e][2]))
    
    fp = floor_plan.FloorPlan()
    fp.load_layout(path)
    existing_places = fp.existing_places()
    
    # change from list of tuple to list of ActivityDataPoint
    activity_sequence = []
    for x in connected_sequence:
        a = ActivityDataPoint(x[0], x[1], x[2], x[0].determine_place(existing_places))
        activity_sequence.append(a)
    return activity_sequence
    
    
def activity_generator(start_day, end_day, act_model, state_anomaly_labels = dict(), anomaly_parameters = dict(), max_try = 1000):
    """
    generator for activity sequence
    make activity schedule between start_day day and end_day day
    This function is generator, so this function become empty iterator once you yield each element in this object. If you use each element repeatedly, use as list by list(activity_generator(start_day, end_day)) or please modify this code by yourself.
    
    Parameters
    ----------
    start_day : int
        sample start day
    end_day : int
        sample end day
    act_model : dict of list of activity_model.Activity
        For instance, see activity_model.basic_activity_model
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
    max_try: int, default 100
        maximum number of time to reject abnormal sample
    
    Yields
    -------
    (activity, activity_start_time, activity_end_time)
        activity : activity_model.Activity
            activity
        activity_start_time : datetime.timedelta
            start time of an activity
        activity_end_time : datetime.timedelta
            end time of an activity

    See Also
    --------
    
    Notes
    -----
    First date is a dammy date and 0 days of start_time corresponding second date right after first date.
    Be careful that day1 sleep and day2 sleep are connected as activity but they are separated in activity generator.

    Raises
    ------
         
    Examples
    --------

    """
    
    first_date = day_schedule(num_days = start_day - 1)
    first_date.fill_up_schedule_with_basic_activity_model(act_model, act_model['sleep'][0], timedelta(hours = 6, minutes = 30))
    last_activity = first_date.last_activity
    extra_time_of_last_activity = first_date.extra_time_of_last_activity
    sleep_after_24 = first_date.sleep_after_24
    day = start_day
    next_date = None
    for day in range(start_day, end_day + 1):
        try_count = 0
        is_valid = False
        while is_valid == False:
            next_date = day_schedule(num_days = day)
            temp_act_model = None
            if anomaly_parameters == dict():
                temp_act_model = act_model
            else:
                temp_act_model = anomaly.basic_activity_model_with_anomaly(act_model, day, state_anomaly_labels, anomaly_parameters)
            try:
                next_date.fill_up_schedule_with_basic_activity_model(temp_act_model, last_activity, extra_time_of_last_activity, sleep_after_24 = sleep_after_24)
            except:
                try_count += 1
                if try_count >= max_try:
                    raise ValueError("It fails to sample {} day {} times. Please try again. One of the reason cause from over sleeping so that taking breakfast at next date can't do. One solution of this is reduce sleep time duration or move sleep time forward.".format(day, try_count))
            else:
                is_valid = True
        schedule = next_date.copy_schedule()
        for x in schedule:
            yield x
        last_activity = next_date.last_activity
        extra_time_of_last_activity = next_date.extra_time_of_last_activity
        sleep_after_24 = next_date.sleep_after_24

    return


class day_schedule:
    """
    make an activity schedule in one whole day
    
    Instance variables
    ------------------
    schedule : list of tuple
        label sequence whose activity was already defined
        shcedule[i] = (class, start_time, end_time) or i-th activity in this day
            class : activity_model.Activity
            start_time : datetime.timedelta
            end_time : datetime.timedelta
    
    remainder : list of tuple
        time range whose activity is not defined yet
        remainder[i] = (start_time(datetime.timedelta object), end_time(datetime.timedelta object))
    
    last_activity : str
        last activity name in a day
    
    extra_time_of_last_activity : datetime.timedelta
        extra time from self.__num_days 24:00 to end time of last activity
        
    sleep_after_24 : tuple, default None
        if the start time of sleep is over 24:00, then;
        sleep_after_24[0] = activity_model.sleep,
        sleep_after_24[1] = the extra time from self.__num_days 24:00 to the start time of sleep
        sleep_after_24[2] = the extra time from self.__num_days 24:00 to the end time of sleep
    """
    
    def __init__(self, num_days):
        """
        initialize class

        Parameters
        ----------
        num_days : int
            number of days
        """
        
        self.__num_days = num_days
        self.__schedule = []
        self.__remainder = [(timedelta(days = self.__num_days, hours = 0), timedelta(days = self.__num_days, hours = 24))]
        self.__last_activity = None
        self.__extra_time_of_last_activity = None
        self.__sleep_after_24 = None
           
    # getter, property except for list objects
    @property
    def num_days(self):
        return self.__num_days
    @property
    def last_activity(self):
        return self.__last_activity
    @property
    def extra_time_of_last_activity(self):
        return self.__extra_time_of_last_activity
    @property
    def sleep_after_24(self):
        return self.__sleep_after_24
    
    # methods to return copy for list objects
    def copy_schedule(self):
        return deepcopy(self.__schedule)
    def copy_remainder(self):
        return deepcopy(self.__remainder)
    
    def update_schedule(self, activity, start_time, end_time, is_test = False):
        """
        add the activity in the schedule

        Parameters
        ----------
        activity : acitivity_model.Activity
            target activity
        start_time : datetime.timedelta
            start time of the activity
        end_time : datetime.timedelta
            end time of the activity
        is_test : boolean, default False
            If this is True, then any instance valuables will not be updated.
            Use this as True, when you want to check if the activity of the input is not overlap with the present schedule. 

        Returns
        -------
        is_OK : boolean
        if this updates success, then return True

        """
        
        # if the start time and the end time are same, then delete the activity
        if start_time == end_time:
            return False
        
        end_of_today = timedelta(days = self.__num_days, hours = 24)
        # if the end time of the activity succeeds to next day 
        if end_of_today <= end_time:
            if is_test == False:
                self.__last_activity = activity
                self.__extra_time_of_last_activity = end_time - end_of_today
            end_time = end_of_today
        
        focused_index = None
        focused_sub_range = None
        for (i, sub_range) in enumerate(self.__remainder):
            if sub_range[0] <= start_time <= sub_range[1]:
                focused_index = i
                focused_sub_range = sub_range
                break
        # appropriate range doesn't exist
        if focused_index == None:
            return False
        # activity time spend over the range
        if focused_sub_range[1] < end_time:
            return False
        
        if is_test == True:
            return True
        
        self.__schedule.append((activity, start_time, end_time))
        
        if focused_sub_range[0] == start_time and end_time == focused_sub_range[1]:
            self.__remainder.pop(focused_index)
        elif focused_sub_range[0] == start_time and end_time != focused_sub_range[1]:
            self.__remainder[focused_index] = (end_time, focused_sub_range[1])
        elif focused_sub_range[0] != start_time and end_time == focused_sub_range[1]:
            self.__remainder[focused_index] = (focused_sub_range[0], start_time)
        else:
            self.__remainder[focused_index] = (focused_sub_range[0], start_time)
            self.__remainder.insert(focused_index + 1, (end_time, focused_sub_range[1]))
        return True
    
    
    def update_schedule_with_multi(self, act_list, is_test = False, is_fundamental = False):
        """
        add the multiple activities in the schedule

        Parameters
        ----------
        act_list : list of acitivity_model.Activity
            list of target activities
            act_list[i] = (activity, start_time, end_time) of i-th activity
                activity : acitivity_model.Activity
                start_time : datetime.timedelta
                end_time : datetime.timedelta

        is_test : boolean, default False
            If this is True, then any instance valuables will not be updated.
            Use this as True, when you want to check if the activities of the input is not overlap with the present schedule. 

        Returns
        -------
        is_OK : boolean
        if this updates success, then return True

        """
        if is_test == False:
            for act in act_list:
                self.update_schedule(*act)
        else:
            if is_fundamental == False:
                # check 1: each activities do not overlap with the present schedule
                for act in act_list:
                    if self.update_schedule(*act, is_test = True) == False:
                        return False
            # check 2: each activities do not overlap with each other
            sorted_act_list = sorted(act_list, key = lambda x: x[1])
            for i in range(len(sorted_act_list) - 1):
                if sorted_act_list[i][2] > sorted_act_list[i+1][1]:
                    return False
        return True
    
    
    def fill_up_schedule_with_basic_activity_model(self, basic_activity_model, first_activity, extra_time, max_try = 100, sleep_after_24 = None):
        """
        generate a schedule in a whole day with a basic activity model

        Parameters
        ----------
        basic_activity_model : dict of list of activity_model.Activity
            For instance, see activity_model.basic_activity_model
            informations about activities
        
        first_activity : activity_model.Activity
            the activity which is taken in 0:00 in this day
        
        extra_time : datetime.timedelta
            the extra time of the last activity yesterday
            
        max_try : int, default 100
            the maximum number of time to reject an abnormal sample
            
        sleep_after_24 : tuple, default None
            if the start time of last sleep is over 0:00, then;
            sleep_after_24[0] = activity_model.sleep,
            sleep_after_24[1] = the extra time from self.__num_days 0:00 to the start time of sleep
            sleep_after_24[2] = the extra time from self.__num_days 0:00 to the end time of sleep

        Returns
        -------
        None

        See Also
        --------

        Notes
        -----

        Raises
        ------

        Examples
        --------

        """
        
        start_of_today = timedelta(days = self.__num_days)
        end_of_today = timedelta(days = self.__num_days, hours = 24)
        
        # update last activity yesterday
        self.update_schedule(first_activity, start_of_today, start_of_today + extra_time)
        
        # update if the start time of last sleep is over 0:00
        if sleep_after_24 != None:
            self.update_schedule(sleep_after_24[0], start_of_today + sleep_after_24[1], start_of_today + sleep_after_24[2])

        # step1: fill up all fundametal activities
        fundamental_act_info = []
        is_valid = False
        try_count = 0
        while is_valid == False:
            update_activities = []  # contains all activities (or sub activities of MetaActivity) to update schedule
            fundamental_act_info = []  # contains information about fundamental activities (not contain sub activities)
            is_not_overlap = True
            for act in basic_activity_model['fundamental_activities']:
                start_time = act.sampling_start_time(self.__num_days)
                duration_time = 0
                if isinstance(act, activity_model.Activity):
                    duration_time = act.sampling_duration_time()
                    update_activities.append((act, start_time, start_time + duration_time))
                elif isinstance(act, activity_model.MetaActivity):
                    sub_activities = act.sampling_sub_activities(start_time)
                    duration_time = sub_activities[-1][2] - start_time
                    update_activities += sub_activities
                else:
                    raise ValueError('act in basic_activity_model must be instance of Activity class or MetaActivity class.')
                fundamental_act_info.append((act, start_time, start_time + duration_time))
                
            # check if the activities are not overlap with the last sleep
            is_valid = self.update_schedule_with_multi(fundamental_act_info, is_test = True, is_fundamental = True)
            # check the order of the fundamental activities
            is_valid *= activity_model.FundamentalActivity.check_order(fundamental_act_info, basic_activity_model['fundamental_activity_order'])
            # This checks whether the overlap of the fundamental activities exists.
            # (R): The interval between fundamental activities must be more than at least threshold_min minutes.
            # The requirement (R) was prepared to avoid making too short interval time in the daily schedule.
            threshold_min = 5
            if sleep_after_24 != None:
                if start_of_today + sleep_after_24[2] + timedelta(minutes = threshold_min) > fundamental_act_info[0][1]:
                    is_valid = False
            else:
                if start_of_today + extra_time + timedelta(minutes = threshold_min) > fundamental_act_info[0][1]:
                    is_valid = False
            for i in range(len(fundamental_act_info) - 1):
                if fundamental_act_info[i][2] + timedelta(minutes = threshold_min) > fundamental_act_info[i + 1][1]:
                    is_valid = False
                    
            try_count += 1
            if try_count > max_try:
                # print("-----------------------------------")
                # print("end time of last sleep (yesterday): {}".format(start_of_today + sleep_after_24[2]))
                # for a in update_activities:
                #     print("{} {} {}".format(a[0], a[1], a[2]))
                raise ValueError("fail to sample fundamental activities {} times".format(max_try))
        
        for x in update_activities:
            # if the start time of sleep is over 24:00
            if x[0].name == basic_activity_model['sleep'][0].name and end_of_today <= x[1]:
                self.__sleep_after_24 = (x[0], x[1] - end_of_today, x[2] - end_of_today)
            else:
                self.update_schedule(*x)
            
        # step2: sample the number of each necessary activity and fill up all necessary activities
        temp_list = basic_activity_model['necessary_activities']
        for act in random.sample(temp_list, len(temp_list)):  # shuffled order
            act_num = act.sampling_number_of_activity()
            if isinstance(act, activity_model.Activity):
                for n in range(act_num):
                    duration_time = act.sampling_duration_time()
                    start_time = None
                    if act.exist_repulsive_constraints == False and act.exist_existence_constraints == False:
                        start_time = act.sampling_start_time(duration_time, self.__remainder)
                    else:
                        start_time = act.sampling_start_time_with_constraints(duration_time, self.__remainder, self.__schedule)
                    self.update_schedule(act, start_time, start_time + duration_time)
            elif isinstance(act, activity_model.MetaActivity):
                for n in range(act_num):
                    temp_start_time = timedelta(days = 0)
                    sub_activities = act.sampling_sub_activities(temp_start_time)
                    duration_time = sub_activities[-1][2] - sub_activities[0][1]
                    start_time = act.sampling_start_time(duration_time, self.__remainder)
                    sub_activities = [(x[0], x[1] + start_time, x[2] + start_time) for x in sub_activities]
                    self.update_schedule_with_multi(sub_activities)
                    # for x in sub_activities:
                    #     self.update_schedule(*x)
            else:
                raise ValueError('act in basic_activity_model must be instance of Activity class or MetaActivity class.')
                
        # step3: fill up remainder time by sampling random activities
        len_remainder = len(self.__remainder)
        while len_remainder != 0:
            sub_range = self.__remainder[0]
            act = activity_model.RandomActivity.sampling_random_activity(basic_activity_model['random_activities'])
            duration_time = None
            sub_activities = []
            if isinstance(act, activity_model.Activity):
                duration_time = act.sampling_duration_time()
            elif isinstance(act, activity_model.MetaActivity):
                sub_activities = act.sampling_sub_activities(sub_range[0])
                duration_time = sub_activities[-1][2] - sub_activities[0][1]
            else:
                raise ValueError('act in basic_activity_model must be instance of Activity class or MetaActivity class.')
            # To prevent to make short remainder (time range whose activity is not defined yet), reject to sample if the sample arise remainder that is shorter than rejection_range as a result
            # rejection_range = timedelta(minutes = act.duration_mean) / 10
            rejection_range = timedelta(minutes = 1)
            # if the start time of sleep is over 24:00
            if self.__sleep_after_24 != None and end_of_today <= sub_range[0] + duration_time:
                if sub_range[0] + duration_time + rejection_range > end_of_today + self.__sleep_after_24[1]:
                    if isinstance(act, activity_model.Activity):
                        self.update_schedule(act, sub_range[0], end_of_today + self.__sleep_after_24[1])
                    elif isinstance(act, activity_model.MetaActivity):
                        continue
                    else:
                        raise ValueError('act in basic_activity_model must be instance of Activity class or MetaActivity class.')
                else:
                    if isinstance(act, activity_model.Activity):
                        self.update_schedule(act, sub_range[0], sub_range[0] + duration_time)
                    elif isinstance(act, activity_model.MetaActivity):
                        self.update_schedule_with_multi(sub_activities)
                    else:
                        raise ValueError('act in basic_activity_model must be instance of Activity class or MetaActivity class.')
                    
            else:
                if sub_range[0] + duration_time + rejection_range > sub_range[1]:
                    if isinstance(act, activity_model.Activity):
                        self.update_schedule(act, sub_range[0], sub_range[1])
                    elif isinstance(act, activity_model.MetaActivity):
                        continue
                    else:
                        raise ValueError('act in basic_activity_model must be instance of Activity class or MetaActivity class.')
                else:
                    if isinstance(act, activity_model.Activity):
                        self.update_schedule(act, sub_range[0], sub_range[0] + duration_time)
                    elif isinstance(act, activity_model.MetaActivity):
                        self.update_schedule_with_multi(sub_activities)
                    else:
                        raise ValueError('act in basic_activity_model must be instance of Activity class or MetaActivity class.')
            len_remainder = len(self.__remainder)
        self.sort_schedule()
        self.connect_same_activities()

        
    def connect_same_activities(self):
        """
        connect adjacent same activity in schedule
        
        See also
        --------
        activity_model.Activity.__eq__ and Activity.__lt__ for comparision of activity_model.Activity class
        """
        complete_scan = False
        not_break = True
        while complete_scan == False:
            for i in range(len(self.__schedule) - 1):
                if self.__schedule[i][0] == self.__schedule[i+1][0]:
                    self.__schedule[i] = (self.__schedule[i][0], self.__schedule[i][1], self.__schedule[i+1][2])
                    self.__schedule.pop(i + 1)
                    not_break = False
                    break
            complete_scan = not_break
            not_break = True
            
                
    def sort_schedule(self):
        """
        sort schedule by start_time in ascending order
        """
        self.__schedule.sort(key = lambda x:x[1])
        
        
    def delete_0_range(self):
        """
        delete 0 time range from remainder
        """
        unnecessary_indexes = []
        for (i, sub_range) in enumerate(self.__remainder):
            if sub_range[0] == subrange[1]:
                unnecessary_indexes.append(i)
        for i in sorted(unnecessary_indexes, reverse=True):
            self.__remainder.pop(i)
    
    
def save_activity_sequence(path, AS, file_name = 'activity_sequence'):
    """
    This saves an activity sequence as a text.
    
    Parameters
    ----------
    path : pathlib.Path
        Path to save the activity sequence.   
    AS : list of ActivityDataPoint
        Time series data of activities.
    file_name : str, default 'activity_sequence'
        File name of this file.
        
    Returns
    -------
    None

    See Also
    --------
    SISG4HEI_Alpha-main/Human_Activities_Schedule_Generation/HASave/Mark2Txt in [1, 2] as a reference.
    [1] C. Jiang and A. Mita, "SISG4HEI_Alpha: Alpha version of simulated indoor scenario generator for houses with elderly individuals." Journal of Building Engineering, 35-101963(2021)
    [2] https://github.com/Idontwan/SISG4HEI_Alpha.
         
    Examples
    --------
    0000 [day] 00 [h] 00 [m] 00 [s] 000000 [ms] Sleep
    0000 [day] 06 [h] 34 [m] 10 [s] 812474 [ms] Go out
    0000 [day] 06 [h] 41 [m] 33 [s] 026441 [ms] Urination
    ...
    0007 [day] 21 [h] 23 [m] 05 [s] 211270 [ms] Urination
    0007 [day] 21 [h] 26 [m] 14 [s] 307355 [ms] Clean
    0007 [day] 21 [h] 32 [m] 07 [s] 931410 [ms] Sleep
    """
    
    text = ''
    for act in AS:
        text += "{:04} [day] {:02} [h] {:02} [m] {:02} [s] {:06} [ms] {}\n".format(*get_d_h_m_s_ms(act.start), act.activity.name)
    if not os.path.exists(path):
        os.makedirs(path)
    file_path = path / "{}.txt".format(file_name)
    with open(file_path, "w") as f:
        f.write(text)
        
        
def save_activity_sequence_as_image(path, AS, file_name = 'activity_sequence', mask_activities = None,
                                    aspect_ratio = 1, ncol = 1, show = False, swap_axies = False, dpi = 400):
    """
    This saves an activity sequence as a image.
    
    Parameters
    ----------
    path : pathlib.Path
        Path to save the activity sequence.   
    AS : list of ActivityDataPoint
        Time series data of activities.
    file_name : str, default 'activity_sequence'
        File name of this file.
    mask_activities : list, default None
        List of names of activities.
        The activities in ``mask_activities`` are not shown.
    aspect_ratio : float, default 1
        Aspect ratio (y/x-scale).
    ncol : int, default 1
        Number of columns in the legend.
    show : boolean, default False
        Whether to show the image.
    swap_axies : boolean, default False
        Whether to swap axies.
        If this is False, then horizontal axis is Day and vertical axis is Hour.
    dpi : int
        dpi for matplot.pyplot
        
    Returns
    -------
    None
    """
    if mask_activities == None:
        mask_activities = []
        
    AS_days = []  # divide an activity sequence into each day
    for day in range(AS[0].start.days, AS[-1].start.days + 1):
        days = [(x.activity, x.start, x.end) for x in AS if x.start.days == day or x.end.days == day]
        days[0] = (days[0][0], timedelta(days = day), days[0][2])
        days[-1] = (days[-1][0], days[-1][1], timedelta(days = day + 1))
        AS_days.append(days)
        
    fig, ax = plt.figure(), plt.axes()

    labels = dict()
    for a in AS:
        if (a.activity.name not in labels.keys()) and (a.activity.name not in mask_activities):
            labels[a.activity.name] = a.activity.color      
    # print(set(labels.keys()) - set(mask_activities))
    
    y_len = 24
    handles = []  # for plt.legend()
    already_contained_activities = []
    one_day_seconds = timedelta(days = 1).total_seconds()
    for (i, acts) in enumerate(AS_days):
        for a in acts:
            if a[0].name in mask_activities:
                continue
            start = y_len * (extract_time_of_day(a[1]).total_seconds() / one_day_seconds)
            if a[2] == timedelta(days = i + 1):
                end = y_len
            else:
                end = y_len * (extract_time_of_day(a[2]).total_seconds() / one_day_seconds)
            label_name = labels[a[0].name]
            rectangle = None
            if swap_axies:
                rectangle = patches.Rectangle(xy = (start, i), width = end - start, height = 1, fc = a[0].color, label = label_name)
            else:
                rectangle = patches.Rectangle(xy = (i, start), width = 1, height = end - start, fc = a[0].color, label = label_name)
            temp = ax.add_patch(rectangle)
            if a[0].name not in already_contained_activities:
                handles.append(temp)
                already_contained_activities.append(a[0].name)

    if not os.path.exists(path):
        os.makedirs(path)
    file_path = path / "{}.png".format(file_name)
    
    plt.axis('scaled')
    ax.set_aspect(aspect_ratio)
    if swap_axies:
        ax.set_xlabel('Hour.')
        ax.set_ylabel('Day.')
        ax.set_xlim(0, 24)
        ax.set_ylim(0, len(AS_days))
    else:
        ax.set_xlabel('Day.')
        ax.set_ylabel('Hour.')
        ax.set_xlim(0, len(AS_days))
        ax.set_ylim(0, 24)
    plt.legend(handles, labels, fontsize = 7, bbox_to_anchor = (1, 1), loc = 'upper left', borderaxespad = 0, ncol = ncol)
    plt.savefig(file_path, dpi = dpi, bbox_inches = 'tight')
    if show == True:
        plt.show()
    plt.close()
    
    
def get_d_h_m_s_ms(td):
    """
    This calculates date information of the input date and time.

    Parameters
    ----------
    td : datetime.timedelta
        Target date and time.
    
    Returns
    -------
    (days, hours, minutes, seconds, microseconds) : tuple of int
         
    Examples
    --------
    td = datetime.timedelta(days=23, hours=3, minutes=44, seconds=55, microseconds=1023).
    return (23, 3, 44, 55, 1023)

    """ 
    m, s = divmod(td.seconds, 60)
    h, m = divmod(m, 60)
    return td.days, h, m, s, td.microseconds
        
        
def save_activity_pie_chart(path, AS, dpi = 400):
    """
    This saves an activity sequence as a pie chart per day.
    
    Parameters
    ----------
    path : pathlib.Path
        Path to save the pie chart.   
    AS : list of ActivityDataPoint
        Time series data of activities.
    dpi : int
        dpi for matplot.pyplot
    """ 

    for day in range(AS[0].start.days, AS[-1].start.days + 1):
        days = [(x.activity, x.start, x.end) for x in AS if x.start.days == day or x.end.days == day]
        days[0] = (days[0][0], timedelta(days = day), days[0][2])
        days[-1] = (days[-1][0], days[-1][1], timedelta(days = day + 1))
        days_AM = [x for x in days if get_d_h_m_s_ms(x[1])[1] < 12]
        days_AM[-1] = (days_AM[-1][0], days_AM[-1][1], timedelta(days = day, hours = 12))
        days_PM = [x for x in days if 12 <= get_d_h_m_s_ms(x[2])[1] or x[2].days == day + 1]
        days_PM[0] = (days_PM[0][0], timedelta(days = day, hours = 12), days_PM[0][2])

        for (j,focused) in enumerate([days_AM, days_PM]):
            pie_data = [(x[2] - x[1]).total_seconds() for x in focused]
            pie_label = [x[0].name + " ({:.1f} [min])".format((x[2] - x[1]) / timedelta(minutes = 1)) for x in focused]
            colors = [x[0].color for x in focused]
            plt.figure()
            plt.pie(pie_data, colors = colors, startangle = 90, counterclock = False)
            plt.legend(pie_label, fontsize = 10, bbox_to_anchor = (1, 1), loc = 'upper left', borderaxespad = 0, ncol = 2)
            if j == 0:
                plt.savefig(str(path) + "/day{}_AM.png".format(day), bbox_inches = 'tight', dpi = dpi)
            else:
                plt.savefig(str(path) + "/day{}_PM.png".format(day), bbox_inches = 'tight', dpi = dpi)
            plt.close()
            
            # augmented_colors = [color_translator.convert_into_color(key) for key in translator_keys] + colors
            # augmented_data = [0 for i in range(len(translator_keys))] + pie_data
            # augmented_label = translator_keys + pie_label
            # plt.figure()
            # plt.pie(augmented_data, labels = augmented_label, colors = augmented_colors, startangle=90, counterclock=False)
            # plt.legend(augmented_label[:len(translator_keys)], fontsize=10, bbox_to_anchor=(0.9, 0.7), loc='upper left', borderaxespad=0, ncol=2)
            # if j == 0:
            #     plt.savefig(str(path) + "/day{}_AM_augmented.png".format(i) , bbox_inches='tight')
            # else:
            #     plt.savefig(str(path) + "/day{}_PM_augmented.png".format(i) , bbox_inches='tight')
            # plt.close()


class WalkingTrajectory:
    """
    This class represents a data of a walking trajectory.
    Walking trajectoires consist of this class.
    
    Attributes
    ----------
    walking_type : str
        Walking pattern, e.g., 'Direct' or 'Wander'.
    start_place : str
        Name of the start place.
    end_place : str
        Name of the end place.
    start_time : datetime.timedelta
        Start time of the walking trajectory.
    end_time : datetime.timedelta
        End time of the walking trajectory.
    centers : list of tuple of float
        2d coordinates of the body centers of the resident in the walking trajectory. The unit is [cm].
    angles : list of float
        Front direction[radian] of the resident.
        angles[i] is the direction at centers[i].
    timestamp : list of datetime.timedelta
        Time of each body_centers.
        The resident exist at time[i] at centers[i].
    left_steps : list of tuple of float
        2D coordinates of the left foot of the resident in the walking trajectory. The unit is [cm]. This length may be differed from the length of 'centers'.
    right_steps : list of tuple of float
        2D coordinates of the right foot of the resident in the walking trajectory. The unit is [cm]. This length may be differed from the length of 'centers'.
    intermediate_places : list of str, default []
        If this trajectory represents a wandering, this records the intermediate places between start_place and end_place.
        The start_place and end_place are excluded in this list, so that the resident walks in the order of start_place, intermediate_places[0], ..., intermediate_places[-1], end_place.0
    fall_w : boolean, default False
        Whether the resident fall while walking in the walking trajectory. If this is 1 then a fall was occurred.
    fall_s : boolean, default False
        Whether the resident fall while standing in the walking trajectory.
    fall_w_index : int, default None
        Index of the 'centers', centers[fall_w_index] means the point of a fall while walking.
    fall_s_index : int, default None
        Index of the 'centers', centers[fall_s_index] means the point of a fall while standing.
    lie_down_seconds_w : float, default 0
        Duration time [sec.] the resident lies down after a fall while walking.
    lie_down_seconds_s : float, default 0
        Duration time [sec.] the resident lies down after a fall while standing.
    fall_w_body_range : float, default 40
        When the resident lies down before fall while walking, their body range is represented as the circle whose center is centers[fall_w_index] and radius is fall_w_body_range [cm].
     fall_s_body_range : float, default 40
        When the resident lies down before fall while standing, their body range is represented as the circle whose center is centers[fall_s_index] and radius is fall_s_body_range [cm].
    
        
    Example
    -------
    walking_type = 'Direct'
    start_place = 'Bed'
    end_place   = 'Sofa'
    start_time = timedelta(days = 7, hours = 9, minutes = 0, seconds = 13)
    end_time   = timedelta(days = 7, hours = 9, minutes = 2, seconds = 28)
    centers = [(0.00, 0.00), (1.03, 1.21), (2.44, 3.34), ..., (100.23, 43.78)]
    angles = [-0.05, 0.03, 0.12, 0.17, 0.23, ..., 1.3]
    timestamp = [timedelta(days = 7, hours = 9), timedelta(days = 7, hours = 9, seconds = 0.8), ..., timedelta(days = 7, hours = 9, minutes = 2)]
    left_steps  = [(-0.10, -0.20), (0.83, 1.01), (2.24, 3.14), ..., (100.04, 43.54)]
    right_steps = [(0.13,0.19), (1.33, 1.29), (2.64, 3.47), ..., (100.33, 43.90)]
    
    When the resident is wandering;
    intermediate_places = ['KitchenStove', 'Sofa', 'EntranceDoor']
    
    When the resident was fallen.
    fall_w = True
    fall_w_index = 3
    lie_down_seconds_w = 30
    fall_w_body_range = 40
    """

    def __init__(self, walking_type, start_place, end_place, start_time, end_time, centers, angles, timestamp, left_steps, right_steps,
                 intermediate_places = [],
                 fall_w = False, fall_s = False, fall_w_index = None, fall_s_index = None,
                 lie_down_seconds_w = 0, lie_down_seconds_s = 0, fall_w_body_range = 40, fall_s_body_range = 40):
        """
        Initialization of an object of ActivityDataPoint class.

        Parameters
        ----------
        Same with Attributes of this class.
        """
        self.__walking_type = walking_type
        self.__start_place = start_place
        self.__end_place = end_place
        self.__start_time = start_time
        self.__end_time = end_time
        self.centers = centers
        self.angles = angles
        self.timestamp = timestamp
        self.left_steps = left_steps
        self.right_steps = right_steps
        self.intermediate_places = intermediate_places
        self.__fall_w = fall_w
        self.__fall_s = fall_s
        self.__fall_w_index = fall_w_index
        self.__fall_s_index = fall_s_index
        self.__lie_down_seconds_w = lie_down_seconds_w
        self.__lie_down_seconds_s = lie_down_seconds_s
        self.__fall_w_body_range = fall_w_body_range
        self.__fall_s_body_range = fall_s_body_range
        
    def __str__(self):
        if self.intermediate_places == []:
            return "<WalkingTrajectory> type: {} from {} ({}) to {} ({})".format(self.walking_type, self.start_place, self.start_time, self.end_place, self.end_time)
        else:
            return "<WalkingTrajectory> type: {} from {} ({}) (, {}, ) to {} ({})".format(self.walking_type, self.start_place, self.start_time, ', '.join(self.intermediate_places), self.end_place, self.end_time)
    
    # getter
    @property
    def walking_type(self):
        return self.__walking_type
    @property
    def start_place(self):
        return self.__start_place
    @property
    def end_place(self):
        return self.__end_place
    @property
    def start_time(self):
        return self.__start_time
    @property
    def end_time(self):
        return self.__end_time
    @property
    def fall_w(self):
        return self.__fall_w
    @property
    def fall_s(self):
        return self.__fall_s
    @property
    def fall_w_index(self):
        return self.__fall_w_index
    @property
    def fall_s_index(self):
        return self.__fall_s_index
    @property
    def lie_down_seconds_w(self):
        return self.__lie_down_seconds_w
    @property
    def lie_down_seconds_s(self):
        return self.__lie_down_seconds_s
    @property
    def fall_w_body_range(self):
        return self.__fall_w_body_range
    @property
    def fall_s_body_range(self):
        return self.__fall_s_body_range
    
    def save_text(self):
        """
        This returns the text to represent this path.
        
        Returns
        -------
        text : str
            Text that represents this path.
        """
        text  = ''
        text += "start: {:04} [day] {:02} [h] {:02} [m] {:02} [s] {:06} [ms]\n".format(*get_d_h_m_s_ms(self.start_time))
        text += "end: {:04} [day] {:02} [h] {:02} [m] {:02} [s] {:06} [ms]\n".format(*get_d_h_m_s_ms(self.end_time))
        text += "place: {} - {} ({}), ".format(self.start_place, self.end_place, self.walking_type)
        text += "intermediate_places:"
        for p in self.intermediate_places:
            text += "{}, ".format(p)
        text += '\n'
        if self.fall_w:
            text += "Fall while walking: {}, {} [sec.] from {:04} [day] {:02} [h] {:02} [m] {:02} [s] {:06} [ms] with body range: {} \n".format(self.fall_w, self.lie_down_seconds_w, *get_d_h_m_s_ms(self.timestamp[self.fall_w_index]), self.fall_w_body_range)
        else:
            text += "Fall while walking: {} {} {} {}\n".format(self.fall_w, self.fall_w_index, self.lie_down_seconds_w, self.fall_w_body_range)
        if self.fall_s:
            text += "Fall while standing: {}, {} [sec.] from {:04} [day] {:02} [h] {:02} [m] {:02} [s] {:06} [ms] with body range: {} \n".format(self.fall_s, self.lie_down_seconds_s, *get_d_h_m_s_ms(self.timestamp[self.fall_s_index]), self.fall_s_body_range)
        else:
            text += "Fall while standing: {} {} {} {}\n".format(self.fall_s, self.fall_s_index, self.lie_down_seconds_s, self.fall_s_body_range)
        return text
    
    def save_as_3D_image(self, path, file_name, dpi = 120):
        """
        This saves walking trajectories as a 3D image.

        Parameters
        ----------
        path : pathlib.Path
            Path to save.
        file_name : str
            File name of this file.
        dpi : int, default 120
            dpi.
        """
        fig = plt.figure()
        ax = fig.gca(projection = '3d')
        x, y, z = [], [], []
        for c, t in zip(self.centers, self.timestamp):
            x.append(c[0])
            y.append(c[1]) 
            z.append(t.seconds - self.timestamp[0].seconds)
        ax.set_xlabel('x.')
        ax.set_ylabel('y.')
        ax.set_zlabel('Seconds in the walking.')
        ax.plot(x, y, z, label = 'Walking path.')
        ax.legend()
        plt.savefig(path / file_name, dpi = dpi, bbox_inches = 'tight')
        # plt.show()
        plt.close()
        
    def save_with_sensor_activations(self, path, file_name, sensor_data, dpi = 120):
        """
        This plots sensor activations while walking.

        Parameters
        ----------
        path : pathlib.Path
            Path to save.
        file_name : str
            File name of this file.
        sensor_data : list of tuple
            sensor_data[i] = (time, sensor_index, sensor_state),
                time : datetime.timedelta
                    Time the sensor changes its state.
                sensor_index : int
                    Index of the sensor. This is the number that is written in sensor_model.Sensor.index, not the index of input 'sensors'.
                sensor_state : boolean
                    Sensor state.
                    For now, binary sensors are considered.
        dpi : int, default 120
            dpi.
        """
        activations = []
        for d in sensor_data:
            if self.start_time <= d[0] <= self.end_time:
                s_time = d[0] - self.start_time
                activations.append((s_time, d[1], d[2]))
        # print(activations)
        data = {}
        ON_start = {}
        for d in activations:
            if d[2] and (d[1] not in ON_start.keys()):
                ON_start[d[1]] = d[0]
            if (not d[2]) and d[1] in ON_start.keys():
                if d[1] not in data.keys():
                    data[d[1]] = []
                data[d[1]].append((ON_start[d[1]].total_seconds(), d[0].total_seconds()))
                del ON_start[d[1]]
        # print(data)
        y_label = sorted(data.keys())
        y_pos = [i for (i, k) in enumerate(y_label)]
        y_dict = {k: i for (i, k) in enumerate(y_label)}
        for n, t_list in data.items():
            for t in t_list:
                plt.hlines(y_dict[n], t[0], t[1], color = 'black')
        plt.yticks(y_pos, y_label)
        plt.ylabel('Sensor number.')
        plt.xlabel('Elapsed seconds in the walking trajectory.')
        plt.savefig(path / Path(file_name), dpi = 120, bbox_inches = 'tight')
        # plt.show()
        plt.close()
    

def generate_walking_trajectories(layout_path, AS, stride, step_speed, prefer_foot, fall_w_parameters = dict(), fall_s_parameters = dict()):
    """
    This generates walking trajectories between activities in the layout.
    
    Parameters
    ----------
    layout_path : pathlib.Path
        Path to the layout files.
    AS : list of ActivityDataPoint
        Time series data of activities.
    stride : float
        Stride length of 1 step, [cm].
    step_speed : float
        Walking speed, [seconds/step].
    prefer_foot : string
        Preferable foot the resident push out when to start to walk.
    fall_w_parameters : dict, default dict()
        Parameters of falls while walking.
    fall_s_parameters : dict, default dict()
        Parameters of falls while standing.
    
    Returns
    -------
    WT : list of WalkingTrajectory
        Walking trajectories.
        len(WT) = len(AS) - 1.
        WT[i] is the walking trajecory from the start time of AS[i] to the start time of AS[i+1].
        If AS[i] is not a wandering, WT[i] may only be a move from AS[i].place at near AS[i].end to the AS[i+1].place at AS[i+1].start.
        If AS[i] is a wandering, WT[i] may contain whole wandering path, including wandering from AS[i].start to AS[i].end, and a move from AS[i].place to AS[i+1].place.

    See Also
    --------
    SISG4HEI_Alpha-main/Human_Path_Generation/HP_main/path_generate in SISG4HEIAlpha in [2] as a reference.
    """ 

    fp = floor_plan.FloorPlan()
    fp.load_layout(layout_path)
    existing_places = fp.existing_places()
    lims = fp.Boundary
    
    discom = fp.load_file(layout_path, 'Discomfortable_value', 'csv')
    destinations = fp.load_file(layout_path, 'Destinations', 'json')
    max_diss = fp.load_file(layout_path, 'Max_Distances', 'json')
    
    distances = {}
    for place in existing_places:
        distances[place] = fp.load_file(layout_path, place + '_distance', 'csv')
    edge, g_L, g_W = fp.edge, fp.g_L, fp.g_W
    WT = sample_walking_trajectories(AS, distances, discom, destinations, max_diss, lims, stride, step_speed, prefer_foot, edge, g_L, g_W, fall_w_parameters = fall_w_parameters, fall_s_parameters = fall_s_parameters)
    return WT

    
def sample_walking_trajectories(AS, distances, discom, destinations, max_diss, lims, stride, 
                                step_speed, prefer_foot, edge, g_L, g_W, fall_w_parameters = dict(), fall_s_parameters = dict(), distance_threshold = 30, max_try = 1000):
    """
    This generates walking trajectories between adjacent activites.
    For now, the resident walks by a constant speed except for anomalous falls.
    
    Parameters
    ----------
    AS : list of ActivityDataPoint
        Time series data of activities.
    distances : dict of numpy.ndarray
        distance[place] means the distance between from the position to the 'place'.
        If the resident cannot reach to the 'place' from the postion, the value is assigned to large value e.g., 99999.0.
    discom : numpy.ndarray
        Discomfortable value at each postions.
    destinations : dict of list of list of int
        destinations[place] represents possible place related with the place.
        For example, destinations['Bed'] includes the 2D coordinates e.g., [0, 0], [0, 1], [0, 2], ...
    max_diss : dict of float
        max_diss[place] means a maximum distance to reach the place.
    lims : list of list of int
        lims[0][0] means the minimum value of x-axis of the floor plan.
        lims[0][1] means the maximum value of x-axis of the floor plan.
        lims[1][0] means the minimum value of y-axis of the floor plan.
        lims[1][1] means the maximum value of y-axis of the floor plan.
    stride : float
        Stride length of 1 step, [cm].
    step_speed : float
        Walking speed, [seconds/step].
    prefer_foot : string
        Preferable foot the resident push out when to start to walk.
    edge : int
    g_L : int
    g_W : int
    fall_w_parameters : dict, default dict()
        Parameters of falls while walking.
    fall_s_parameters : dict, default dict()
        Parameters of falls while standing.
    distance_threshold : float, default 30
        Threshold distance [cm].
        If the distance between present end point and target point is longer than distance_threshold, then generate more steps to get closer to the target point.
    max_try: int, default 100
        Maximum number to resample.
    
    Returns
    -------
    WT : list of WalkingTrajectory
        Walking trajectories.
        len(WT) = len(AS) - 1.
        WT[i] is the walking trajecory from the start time of AS[i] to the start time of AS[i+1].
        If AS[i] is not a wandering, WT[i] may only be a move from AS[i].place at near AS[i].end to the AS[i+1].place at AS[i+1].start.
        If AS[i] is a wandering, WT[i] may contain whole wandering path, including wandering from AS[i].start to AS[i].end, and a move from AS[i].place to AS[i+1].place.

    See Also
    --------
    SISG4HEI_Alpha-main/Human_Path_Generation/HSave/sample_save_path in SISG4HEIAlpha in [1, 2] as a reference.
    [1] C. Jiang and A. Mita, "SISG4HEI_Alpha: Alpha version of simulated indoor scenario generator for houses with elderly individuals." Journal of Building Engineering, 35-101963(2021), and 
    [2] https://github.com/Idontwan/SISG4HEI_Alpha.

    Notes
    -----
    This function omits to generate a walking trajectory if the nexy activity does not exist, so that the last activity of the activity sequence does not to be a start point of a walking trajectory.
    Falls are represented as consecutive points with stop, e.g., if WT.centers[i] = WT.centers[i+1] then the residents stop at WT.centers[i] for timestamp[i+1] - timestamp[i] seconds.
    """ 
    
    place_list = list(distances.keys())
    WT = []
    i = 0
    
    fall_w_indexes, fall_s_indexes = [], []
    if fall_w_parameters != dict():
        fall_w_indexes = anomaly.determine_fall_indexes(AS, fall_w_parameters, 'w')
    if fall_s_parameters != dict():
        fall_s_indexes = anomaly.determine_fall_indexes(AS, fall_s_parameters, 's')
    
    last_position = None
    limit = len(AS) - 1
    while i < limit:
        print_progress_bar(limit - 1, i, 'Making walking trajectories.')
        start_place = AS[i].place
        start_position_indexes = None
        if last_position == None:
            start_position_indexes = destinations[start_place]
        else:
            start_position_indexes = [coordinate2index(*last_position, edge, g_L, g_W, lims)]
            # start_position_indexes = destinations[start_place]  # to generate start position randomly
        end_place = AS[i+1].place
        end_time = AS[i+1].start
        walking_type = ''
        start_time = None
        centers = []
        angles = []
        left_steps = []
        right_steps = []
        intermediate_places = []
            
        if (AS[i].activity.name != activity_model.WANDERING_NAME) and (start_place == end_place):  # not walking
            if i in fall_w_indexes or i in fall_s_indexes or AS[i].activity.name == activity_model.WANDERING_NAME:
                raise ValueError('The resident is not walking now. Falls or a wandering cannot be happend in 0 steps.')
            walking_type = NOT_WALKING
            start_time = end_time
            timestamp = []
            WT.append(WalkingTrajectory(walking_type, start_place, end_place, start_time, end_time, centers, angles, timestamp, left_steps, right_steps))
            i += 1
            continue
            
        if AS[i].activity.name != activity_model.WANDERING_NAME:  # walking
            walking_type = DIRECT_WALKING
            is_valid = False
            count = 0
            while is_valid == False:
                try:
                    (temp_centers, left_steps, right_steps) = sample_path(start_position_indexes, end_place, distances, discom, max_diss, lims, stride, prefer_foot, distance_threshold, edge, g_L, g_W)
                except:
                    count += 1
                    if count > max_try:
                        raise ValueError("sample_walking_trajectories() failed to generate walking trajectories {} times.".format(max_try))
                    continue
                else:
                    is_valid = True
                    centers = [(x[0], x[1]) for x in temp_centers]
                    angles = [x[2] for x in temp_centers] 
                    start_time = end_time - timedelta(seconds = (len(centers) - 1) * step_speed)
                    if start_time <= AS[i].start:
                        raise ValueError("The duration time of the {}-th activity is less than the duration time of {}-th walking to move between places. Try to increase duration time of short activities.".format(i, i))
                         
        if AS[i].activity.name == activity_model.WANDERING_NAME:  # wandering
            # be careful that the path between former activity and wandering is not contained in wandering path.
            # So, if (i-1)th activity a -> wandering -> (i+1)th activity b,
            # there are a path (from a to wandering) and a path (from wandering to b) that is the path calculated here.
            
            walking_type = WANDERING_WALKING
            is_valid = False
            count = 0
            candidates = [c for c in place_list if (c != start_place) and (c != end_place)]
            # ``candidates`` are the candidates of places where the resident can reach in a wandering.
            max_seconds = AS[i].duration / timedelta(seconds = 1)  # maximum seconds of a wandering
            
            while is_valid == False:
                walking_seconds = 0  # total seconds of walking
                locations = [start_place]
                temp_centers_list, temp_left_steps_list, temp_right_steps_list = [], [], []
                
                try:    # generate a wandering pattern during the duration time of the wandering
                    while True:
                        if temp_centers_list != []:
                            temp_lp = temp_centers_list[-1][-1]
                            start_position_indexes = [coordinate2index(temp_lp[0], temp_lp[1], edge, g_L, g_W, lims)]
                        temp_end_place = random.choice([c for c in candidates if c != locations[-1]])
                        (temp_centers, temp_left_steps, temp_right_steps) = sample_path(start_position_indexes, temp_end_place, distances, discom, max_diss, lims, stride, prefer_foot, distance_threshold, edge, g_L, g_W)
                        temp_seconds = (len(temp_centers) - 1) * step_speed
                        if len(locations) > 1:
                            temp_seconds += step_speed  # connect other centers
                        if walking_seconds + temp_seconds > max_seconds:
                            break
                        walking_seconds += temp_seconds
                        locations.append(temp_end_place)
                        temp_centers_list.append(temp_centers)
                        temp_left_steps_list.append(temp_left_steps)
                        temp_right_steps_list.append(temp_right_steps)
                                                
                    # adjust walking trajectories to reach the last place
                    while True:
                        if temp_centers_list != []:
                            temp_lp = temp_centers_list[-1][-1]
                            start_position_indexes = [coordinate2index(temp_lp[0], temp_lp[1], edge, g_L, g_W, lims)]
                        (temp_centers, temp_left_steps, temp_right_steps) = sample_path(start_position_indexes, end_place, distances, discom, max_diss, lims, stride, prefer_foot, distance_threshold, edge, g_L, g_W)
                        temp_seconds = (len(temp_centers) - 1) * step_speed
                        if len(locations) > 1:
                            temp_seconds += step_speed  # connect other centers
                            
                        if walking_seconds + temp_seconds > max_seconds:
                            if len(locations) == 1:
                                raise ValueError('The duration time of the wandering is less than the duration time of walking to move between places.')
                            locations.pop()
                            temp2_centers = temp_centers_list.pop()
                            temp_left_steps_list.pop()
                            temp_right_steps_list.pop()
                            walking_seconds -= (len(temp2_centers) - 1) * step_speed
                            if len(locations) > 1:
                                walking_seconds -= step_speed
                            continue
                        else:
                            walking_seconds += temp_seconds
                            locations.append(end_place)
                            temp_centers_list.append(temp_centers)
                            temp_left_steps_list.append(left_steps)
                            temp_right_steps_list.append(right_steps)
                            break
                except:
                    count += 1
                    if count > max_try:
                        raise ValueError("sample_walking_trajectories() failed to generate walking trajectories {} times.".format(max_try))
                    continue
                else:
                    is_valid = True
                    temp_centers = list(itertools.chain.from_iterable(temp_centers_list))
                    left_steps = list(itertools.chain.from_iterable(temp_left_steps_list))
                    right_steps = list(itertools.chain.from_iterable(temp_right_steps_list))
                    intermediate_places = list(locations[1:len(locations) - 1])
                    centers = [(x[0], x[1]) for x in temp_centers]
                    angles = [x[2] for x in temp_centers]
                    start_time = end_time - timedelta(seconds = (len(centers) - 1) * step_speed)
                    if start_time <= AS[i].start:
                        raise ValueError("The duration time of the {}-th activity is less than the duration time of {}-th wandering. Try to increase duration time of short activities.".format(i))
                
        # assign time to the walking trajectory
        fall_w, fall_s = False, False
        fall_w_sec, fall_s_sec = 0, 0
        fall_point_w, fall_point_s = 0, 0
        timestamp = [end_time - timedelta(seconds = i * step_speed) for i in range(len(centers))]  # normal walk by a constant speed without falls
        timestamp.reverse()
        start_time = end_time - timedelta(seconds = (len(temp_centers) - 1) * step_speed)

        if (i in fall_w_indexes) and (len(centers) != 0):  # when falls occur
            fall_w = True
            fall_w_sec = np.random.normal(loc = fall_w_parameters['mean_lie_down_seconds'], scale = fall_w_parameters['mean_lie_down_seconds'] / 5)
            lie_down_duration_w = timedelta(seconds = fall_w_sec)
            start_time -= lie_down_duration_w
            if start_time <= AS[i].start:
                raise ValueError("The duration time of the {}-th activity is less than the duration time of {}-th walking to move between places and fall for {} seconds. Try to increase duration time of short activities.".format(fall_w_sec))
            if len(centers) < 2:
                raise ValueError("{}- walking trajectory is too short to fall down.".format(i))
            fall_point_w = random.choice(range(1, len(centers)))

            # Insertion of the fall point in the walking trajectory.
            centers.insert(fall_point_w, centers[fall_point_w])
            angles.insert(fall_point_w, angles[fall_point_w])
            timestamp = [x if fall_point_w <= i else x - lie_down_duration_w for i, x in enumerate(timestamp)]
            timestamp.insert(fall_point_w, timestamp[fall_point_w] - lie_down_duration_w)

        if (i in fall_s_indexes) and (len(centers) != 0):  # sometimes, a fall while walking and a fall while standing are both happended in one walking trajectory. 
            fall_s = True
            fall_s_sec = np.random.normal(loc = fall_s_parameters['mean_lie_down_seconds'], scale = fall_s_parameters['mean_lie_down_seconds'] / 5)
            lie_down_duration_s = timedelta(seconds = fall_s_sec)
            start_time -= lie_down_duration_s
            if start_time <= AS[i].start:
                raise ValueError("The duration time of the {}-th activity is less than the duration time of {}-th walking to move between places and fall for {} seconds. Try to increase duration time of short activities.".format(fall_s_sec))
            fall_point_s = 0
            if fall_w == True:
                if fall_point_w == fall_point_s:  # The index of fall_point_w is shifted when the fall while standing occurs.
                    raise ValueError('Fall while walking and fall while standing are occurred at the same time!')
                if fall_point_s < fall_point_w:
                    fall_point_w += 1

            # Insertion of the fall point in the walking trajectory.
            centers.insert(fall_point_s, centers[fall_point_s])
            angles.insert(fall_point_s, angles[fall_point_s])
            timestamp = [x if fall_point_s <= i else x - lie_down_duration_s for i, x in enumerate(timestamp)]
            timestamp.insert(fall_point_s, timestamp[fall_point_s] - lie_down_duration_s)

        if len(centers) == 0:
            walking_type = NOT_WALKING
        WT.append(WalkingTrajectory(walking_type, start_place, end_place, start_time, end_time, centers, angles, timestamp, left_steps, right_steps,
                                    intermediate_places = intermediate_places,
                                    fall_w = fall_w, fall_w_index = fall_point_w, lie_down_seconds_w = fall_w_sec,
                                    fall_s = fall_s, fall_s_index = fall_point_s, lie_down_seconds_s = fall_s_sec))
        last_position = WT[-1].centers[-1]
            
        i += 1
    return WT


def print_progress_bar(n, i, text):
    """
    This prints a message text for progress bar.
    
    Parameters
    ----------
    n : int
        Max index.
    i : int
        Current index.
    text : str
        Header text.
    """
    if i == n:
        return print("{} {} / {}.".format(text, i, n) + 'Completed!')
    else:
        return print("{} {} / {}.\r".format(text, i, n), end = '')

def sample_path(start_position_indexes, end_place, distances, discom, max_diss, lims, stride, prefer_foot, distance_threshold, edge, g_L, g_W):
    """
    This samples path between two places.
    
    Parameters
    ----------
    start_position_indexes : list of tuple of int, default None
        The candidates of indexes of start postions in the path.
        Be careful that this is not coordinate on x-y axis, but indexes in csv files of floor plan.
        If the length of the list is one, then the postion is always selected.
    end_place : str
        End place of this path.
    distances : dict of numpy.ndarray
        distance[place] means the distance between from the position to the 'place'.
        If the resident cannot reach to the 'place' from the postion, the value is assigned to a large value e.g., 99999.0.
    discom : numpy.ndarray
        Discomfortable value at each postions.
    max_diss : dict of float
        max_diss[place] means a maximum distance to reach the place.
    lims : list of list of int
        lims[0][0] means the minimum value of x-axis of the floor plan.
        lims[0][1] means the maximum value of x-axis of the floor plan.
        lims[1][0] means the minimum value of y-axis of the floor plan.
        lims[1][1] means the maximum value of y-axis of the floor plan.
    stride : float
        Stride length of 1 step, [cm].    
    prefer_foot : string
        Preferable foot the resident push out when to start to walk. 'right' or 'left'.
    distance_threshold : float, default 30
        Threshold distance [cm].
        If the distance between the present end point and the target point is longer than distance_threshold, then this function generates more steps to get closer to the target point.
    edge : float
    g_L : float
    g_W : float
        
    Returns
    -------
    (centers, left_steps, right_steps) : tuple
    centers : list of tuple of float
        2D coordinates of the body centers of the resident in the walking trajectory.
    left_steps : list of tuple of float
        2D coordinates of the left foot of the resident in the walking trajectory.
    right_steps : list of tuple of float
        2D coordinates of the right foot of the resident in the walking trajectory.
        
    If the start_position is contained in the target area, return ([], [], []) (not walking).
    """
    HPath, Angles = direct_path(start_position_indexes, stride, discom, distances[end_place], max_diss[end_place], edge, g_L, g_W, distance_threshold = distance_threshold)
    if (len(HPath) == 1) and (len(Angles) == 0):
        return ([], [], [])
    else:
        centers, left_steps, right_steps = normal_bcfp(HPath, Angles, stride, lims, prefer_foot, edge, g_L, g_W)
        return (centers, left_steps, right_steps)

def index2coordinate(I, J, edge, g_L, g_W, lims):
    """
    This returns the coodinate in real x-y axis from path index (i, j) used in csv files.
    
    Parameters
    ----------
    I : int
    J : int
    edge : float
    g_L : float
    g_W : float
    lims : list of list of int
        lims[0][0] means the minimum value of x-axis of the floor plan.
        lims[0][1] means the maximum value of x-axis of the floor plan.
        lims[1][0] means the minimum value of y-axis of the floor plan.
        lims[1][1] means the maximum value of y-axis of the floor plan.
        
    Returns
    -------
    (x, y) : tuple of float
    """
    X0, Y0 = lims[0][0] - edge, lims[1][0] - edge
    return (X0 + g_L/2 + I*g_L, Y0 + g_W/2 + J*g_W)

    
def coordinate2index(x, y, edge, g_L, g_W, lims):
    """
    This returns the index (i, j) used in csv files from a coodinate in real x-y axis.
    
    Parameters
    ----------
    x : int
    y : int
    edge : float
    g_L : float
    g_W : float
    lims : list of list of int
        lims[0][0] means the minimum value of x-axis of the floor plan.
        lims[0][1] means the maximum value of x-axis of the floor plan.
        lims[1][0] means the minimum value of y-axis of the floor plan.
        lims[1][1] means the maximum value of y-axis of the floor plan.
        
    Returns
    -------
    (I, J) : tuple of int
    """
    X0, Y0 = lims[0][0] - edge, lims[1][0] - edge
    return (int((x - X0 - g_L / 2) / g_L), int((y - Y0 - g_W / 2) / g_W))


def direct_path(candidates, stride, Discom, Distance_E, max_dis, edge, g_L, g_W, distance_threshold = 30):
    """
    This samples a path between two places.
    
    Parameters
    ----------
    candidates: list of list of int
        Possible place related with the place.
        These coordinates are candidates of start point and one of them is selected randomly.
    stride : float
        Stride length of 1 step, [cm].
    Discom : numpy.ndarray
        Discomfortable value at each postions.
    Distance_E: numpy.ndarray
        This means the distance between from the position to the end place.
        If the resident cannot reach to the 'place' from the postion, the value is assigned to a large value e.g., 99999.0.
    max_dis: float
        A maximum distance to reach the place.
    edge : float
    g_L : float
    g_W : float
    distance_threshold : float, default 30
        Threshold distance [cm].
        If the distance between the present end point and the target point is longer than distance_threshold, then this function generates more steps to get closer to the target point.

    Returns
    -------
    (HPath, Angles) : tuple
    HPath : list of list of int
        HPath means the list of path.
        [i] means an i-th coordinate in the path, e.g., resident_path[i] = [100, 23].
    Angles : list of float
        Angles means the list of angles of the resident while walking, e.g., angles[i] = 0.0996.
        
    See Also
    --------
    SISG4HEI_Alpha-main/Human_Path_Generation/HumanPath/dire_pacing_path in SISG4HEIAlpha in [1, 2] as a reference.
    [1] C. Jiang and A. Mita, "SISG4HEI_Alpha: Alpha version of simulated indoor scenario generator for houses with elderly individuals." Journal of Building Engineering, 35-101963(2021), and
    [2] https://github.com/Idontwan/SISG4HEI_Alpha.
    """
    
    def O_dist(I, J):
        return math.sqrt((I*g_L)**2+(J*g_W)**2)

    def cal_an_dif(n_ang, o_ang):
        [n_I, n_J], [o_I, o_J] = n_ang, o_ang
        angl1 = math.atan2(n_J, n_I)
        angl0 = math.atan2(o_J, o_I)
        return min(abs(angl1-angl0), 2*math.pi-abs(angl1-angl0))


    def cal_an_val(an_diff):
        if an_diff > 3/4*math.pi: return 200
        if an_diff > math.pi/4: return 75
        if an_diff > math.pi/12: return (180*an_diff/math.pi)**2/45
        return (180*an_diff/math.pi)/3

    def od_add_ang_len(Od, Slen, angle):
        for key in Od:
            (ii, jj) = key
            sslen = O_dist(ii, jj)
            len_value = (sslen-Slen)**2/4
            Od[key] += len_value
            if angle != None:
                an_dif = cal_an_dif([ii, jj], angle)
                an_value = cal_an_val(an_dif)
                Od[key] += an_value

    def add_noise(Od):
        for key in Od:
            noise = 2*random.random()
            Od[key] += noise
            
    def orig_dist(Distances, Discom, O_node):
        # weighted-distance with discomfort values from O_node
        Wd = {}
        [I, J] = O_node
        O_discom = Discom[I][J]
        M, N = Distances.shape
        max_ii, max_jj = int(75 / g_L), int(75 / g_W)
        min_ii, min_jj = int(30 / g_L), int(30 / g_W)
        for (ii, jj) in itertools.product(range(- max_ii + 1, max_ii), range(- max_jj + 1, max_jj)):
            if (abs(ii) > min_ii or abs(jj) > min_jj) and (-1 < I+ii < M and -1 < J+jj < N):
                ii_h, jj_h = ii//2, jj//2
                disc_mean = (O_discom + Discom[I+ii_h][J+jj_h] + Discom[I+ii][J+jj])/3
                Wd[(ii, jj)] = Distances[I+ii][J+jj] + disc_mean * O_dist(ii, jj)
        return Wd
    
    def next_step(distance, discomf, location, stride, angle, max_dis):
        MWd = orig_dist(distance, discomf, location)
        od_add_ang_len(MWd, stride, angle)
        add_noise(MWd)
        n_key = min(MWd, key = MWd.get)
        (II, JJ) = n_key
        [I, J] = location
        I, J = I + II, J + JJ
        return (II, JJ), I, J

    first_point = random.choice(candidates)
    HPath, Angles = [first_point], []
    I, J = first_point[0], first_point[1]
    angle = None
    while Distance_E[I][J] > distance_threshold:
        angle, I, J = next_step(Distance_E, Discom, [I, J], stride, angle, max_dis)
        if len(HPath) > max_dis / stride:
            raise ValueError('The resident may not reach a target end point.')
        Angles.append(math.atan2(angle[1], angle[0]))
        HPath.append([I, J])
    return HPath, Angles


def normal_bcfp(Hpath, Angles, stride, lims, Preferfoot, edge, g_L, g_W):
    """
    This returns coordinates of center of body and footprints.
    
    Parameters
    ----------
    HPath : list of list of int
        HPath means the list of path.
        [i] means an i-th coordinate in the path, e.g., resident_path[i] = [100, 23].
    Angles : list of float
        Angles means the list of angles of the resident while walking, e.g., angles[i] = 0.0996.
    stride : float
        Stride length of 1 step, [cm]. 
    lims : list of list of int
        lims[0][0] means the minimum value of x-axis of the floor plan.
        lims[0][1] means the maximum value of x-axis of the floor plan.
        lims[1][0] means the minimum value of y-axis of the floor plan.
        lims[1][1] means the maximum value of y-axis of the floor plan.
    prefer_foot : string
        Preferable foot the resident push out when to start to walk. 'right' or 'left'.
    edge : float
    g_L : float
    g_W : float

    Returns
    -------
    (centers, left_steps, right_steps) : tuple
    centers : list of tuple of float
        2D coordinates of the body centers of the resident in the walking trajectory.
    left_steps : list of tuple of float
        2D coordinates of the left foot of the resident in the walking trajectory.
    right_steps : list of tuple of float
        2D coordinates of the right foot of the resident in the walking trajectory.
        
    See Also
    --------
    SISG4HEI_Alpha-main/Human_Path_Generation/HumanPath/dire_pacing_path in SISG4HEIAlpha in [1, 2] as a reference.
    [1] C. Jiang and A. Mita, "SISG4HEI_Alpha: Alpha version of simulated indoor scenario generator for houses with elderly individuals." Journal of Building Engineering, 35-101963(2021), and
    [2] https://github.com/Idontwan/SISG4HEI_Alpha.
    """
    
    def footprint(Hpath, Angles, stride, PreferFoot):
        L_I, W_J = stride/6/g_L, stride/6/g_W
        Real_angs, Real_Bcs, footprints, An_diff = [], [], [[], []], []

        def calinstep(Body_c, angle_, flags):
            [I0, J0] = Body_c
            ang0_l, ang0_r = angle_+math.pi/2, angle_-math.pi/2
            I0_l, I0_r = I0+L_I*math.cos(ang0_l), I0+L_I*math.cos(ang0_r)
            J0_l, J0_r = J0+W_J*math.sin(ang0_l), J0+W_J*math.sin(ang0_r)
            Real_Bcs.append([I0, J0])
            Real_angs.append(angle_)
            if flags[0]: footprints[0].append([I0_l, J0_l])
            if flags[1]: footprints[1].append([I0_r, J0_r])

        N = len(Angles)
        for i in range(N-1):
            temp = abs(Angles[i+1]-Angles[i])
            ang_dif = min(temp, 2*math.pi-temp)
            # by this way, ang_dif belongs to [0, 2*pi], but real delta angle may be ang_dif of -ang_dif
            An_diff.append(ang_dif)
        calinstep(Hpath[0], Angles[0], [1, 1])
        count = 1 if PreferFoot == 'right' else 0
        for i in range(N-1):
            if An_diff[i] < math.pi/4:
                # the angel of direction of human body is the middle angle between Angles[i] and Angles[i+1]
                # But you can not let ang = (Angles[i]+Angles[i+1])/2 directly, because in this way, the result
                # may be the inverse direction of you wanted ang. (may be ang+pi or ang-pi)
                ang = Angles[i] + An_diff[i]/2 # for the case when real delta angle is ang_dif
                if abs(min(abs(ang-Angles[i+1]), 2*math.pi-abs(ang-Angles[i+1]))-An_diff[i]/2) > 0.001:
                    # min(abs(ang-Angles[i+1]), 2*ma.pi-abs(ang-Angles[i+1]) is the delta angle between (Angles[i]+ang_dif/2)
                    # and the (Angles[i+1]), it may be ang_dif/2 of 3*ang_dif/2. It depends on real delta angle equal to
                    # an_dif of -an_dif
                    ang = Angles[i] - An_diff[i]/2 # for the case when real delta angle is -ang_dif
                m_c = count%2
                calinstep(Hpath[i+1], ang, [1-m_c, m_c])
                count += 1
            else:
                calinstep(Hpath[i+1], Angles[i], [1, 1])
                calinstep(Hpath[i+1], Angles[i+1], [1, 1])
        calinstep(Hpath[N], Angles[N-1], [1, 1])
        return Real_Bcs, footprints, Real_angs

    real_XYs = [[], [], []]  # centers, left_steps, right_steps
    X0, Y0 = lims[0][0] - edge, lims[1][0] - edge
    realbcs, footprints, realangs = footprint(Hpath, Angles, stride, Preferfoot)
    real_IJ = [realbcs, footprints[0], footprints[1]]
    for i in range(3):
        for [I, J] in real_IJ[i]:
            x, y = index2coordinate(I, J, edge, g_L, g_W, lims)
            real_XYs[i].append([round(x, 2), round(y, 2)])
    for i in range(len(realangs)):
        real_XYs[0][i].append(round(realangs[i], 3))  # real_XYs[0] include the angles
            
    return real_XYs[0], real_XYs[1], real_XYs[2]


def save_walking_trajectoires(path, WT, file_name = 'walking_trajectories'):
    """
    This saves walking trajectories as a text.
    
    Parameters
    ----------
    path : pathlib.Path
        Path to save the activity sequence.   
    WT : list of WalkingTrajectory
        Walking trajectories.
    file_name : str, default 'walking_trajectories'
        File name of this file.
    """
    
    text = ''
    for (i, wt) in enumerate(WT):
        text += "index: {}\n".format(i)
        text += wt.save_text()
        text += '\n'
    if not os.path.exists(path):
        os.makedirs(path)
    file_path = path / "{}.txt".format(file_name)
    with open(file_path, "w") as f:
        f.write(text)
        

def generate_motion_sensor_data(sensors, WT, sampling_seconds = 0.1, sync_reference_point = timedelta(days = 0), body_radius = 10):
    """
    This simulates sensor data with or withour anomalies, that is related with resident's motion.
    For now, the all sensors share the same sampling rate.
    
    Parameters
    ----------
    sensors : list of sensor_model.Sensor
        Target sensors.
    WT : list of WalkingTrajectory
        Walking trajectories.
    sampling_seconds : float
        Sampling duration[s] of sensors.
    sync_reference_point : datetime.timedelta, default timedelta(days = 0)
        Start time of the all activities.
        This is used to adjust the sampling time.
    body_radius : float
        Radius[cm] of resident's body on floor.
    
    Returns
    -------
    sensor_data : list of tuple
        sensor_data[i] = (time, sensor_index, sensor_state),
            time : datetime.timedelta
                Time the sensor changes its state.
            sensor_index : int
                Index of the sensor. This is the number that is written in sensor_model.Sensor.index, not the index of input 'sensors'.
            sensor_state : boolean
                Sensor state.
                For now, binary sensors are considered.
                
    Notes
    -----
    All binary sensor states are initialized as False. 
    """
    sensor_data = []
    # If the sensor is sensor_model.CircularPIRSensor or sensor_model.SquarePressureSensor, first sensor state is initialized as False.
    # States of other kinds of sensors are initialized as None (that is not updated in this function).
    sensor_states = {}
    for s in sensors:
        if isinstance(s, sensor_model.CircularPIRSensor) or isinstance(s, sensor_model.SquarePressureSensor):
            sensor_states[s.index] = False
        else:
            sensor_states[s.index] = None
        
    # update states of motion sensors for each walking trajectory
    for (i, wt) in enumerate(WT):
        print_progress_bar(len(WT), i, 'Making motion sensor data')
        if wt.centers != []:
            update_states_of_motion_sensors(sensors, sensor_data, sensor_states, wt, sampling_seconds, sync_reference_point, body_radius)
    return sensor_data


def update_states_of_motion_sensors(sensors, sensor_data, sensor_states, wt, sampling_seconds, sync_reference_point, body_radius):
    """
    Update states of sensors related with resident's walking.
    For now, we treat with PIR motion sensors and pressure sensors.
    For now, the all sensors share the same sampling rate.
    If the resident does not move in PIR motion sensors for sampling_seconds, seconds, then PIR sensor does not be activated.
    Pressure sensors can be activated while the resident does not move (like falling).
    For now, the pressure sensor does not be activated after the resident reaches the goal point.
    
    Parameters
    ----------
    sensors : list of sensor_model.Sensor
        Target sensors.
    sensor_data : list of tuple
        sensor_data[i] = (time, sensor_index, sensor_state),
            time : datetime.timedelta
                Time the sensor changes its state.
            sensor_index : int
                Index of the sensor. This is the number of Sensor.index, not the index in 'sensors'.
            sensor_state : boolean
                Sensor state.
                For now, binary sensors are considered.
    sensor_states : dict
        Present states of sensors.
        Keys are indexes of sensors that are recorded in sensor_model.Sensor.index, not indexes in input's 'sensors'.
        Value are states of sensors.
        If the corresponded sensor of the index is sensor_model.CircularPIRSensor or sensor_model.SquarePressureSensor, the value is boolean, else None.
    wt : list of WalkingTrajectory
        Walking trajectories.
    sampling_seconds : float
        Sampling duration [s] of sensors.
    sync_reference_point : datetime.timedelta, default timedelta(days = 0)
        Start time of the all activities.
        This is used to adjust the sampling time.
    body_radius : float
        Radius [cm] of resident's body on floor.
        
    Notes
    -----
    Suppose that the resident moves straight on a line (= between centers[i] and centers[i+1]).
    Calculate the 2d coordinate ot the resident using the interpolation between wt.centers[i] and wt.centers[i+1], e.g.,
    
    -------.------------------------------.------
           |                  |           |      
           c_i                p           c_{i+1}    2d coordinate,
           t_i               t_p          t_{i+1}    time,
           
    p = c_i + (c_{i+1} - c_i) * (t_p - t_i) /(t_{i+1} - t_i).
    """
    sampling_start = define_synchronous_sampling_point(sync_reference_point, wt.start_time, sampling_seconds)
    past_p = wt.centers[0]  # past 2d coordinate of the resident
    # calculate the 2d coordinate ot the resident using the interpolation between wt.centers[i] and wt.centers[i+1]
    temp_t = None
    for t in date_generator(sampling_start, wt.end_time, step = timedelta(seconds = sampling_seconds)):
        p = None
        if t == wt.end_time:
            p = wt.centers[-1]
        else:
            i, t_i = [(j, t_j) for (j, t_j) in enumerate(wt.timestamp) if t_j <= t][-1]
            t_ii = wt.timestamp[i + 1]
            c_i, c_ii = wt.centers[i], wt.centers[i + 1]
            p = tuple(np.array(c_i) + (np.array(c_ii) - np.array(c_i)) * (t - t_i) / (t_ii - t_i) )
            
        temp_body_radius = body_radius
        if wt.fall_w == True and i == wt.fall_w_index:
            temp_body_raius = wt.fall_w_body_range
        if wt.fall_s == True and i == wt.fall_s_index:
            temp_body_raius = wt.fall_s_body_range
        
        for s in sensors:
            # for PIR sensor
            if isinstance(s, sensor_model.CircularPIRSensor):
                min_d = 1  # min_d [cm] is a minimum distance between p and past_p to be activated.
                state = s.is_activated(p, past_p, temp_body_radius, min_d)
                update_state_of_binary_sensor(sensor_data, sensor_states, s, state, t)
            # for pressure sensor
            elif isinstance(s, sensor_model.SquarePressureSensor):
                state = s.collide(p, temp_body_radius)
                update_state_of_binary_sensor(sensor_data, sensor_states, s, state, t)
            else:
                continue
                
        past_p = p
        temp_t = t
        
    # When the resident reaches a goal point after wt.end.
    # Suppose next path does not start immidiately.
    
    t += timedelta(seconds = sampling_seconds)
    for s in sensors:
        # States of PIR sensors are False
        if isinstance(s, sensor_model.CircularPIRSensor):
            state = False
            update_state_of_binary_sensor(sensor_data, sensor_states, s, state, t)
        # States of pressure sensors are False
        elif isinstance(s, sensor_model.SquarePressureSensor):
            state = False
            update_state_of_binary_sensor(sensor_data, sensor_states, s, state, t)
        else:
            continue
            
             
def date_generator(start, stop, step = timedelta(days = 1)):
    """
    Generator for specific intervals of dates.
    
    Parameters
    ----------
    start : datetime.timedelta
        Start time.
    stop : datetime.timedelta
        End time.
    step : datetime.timedelta, default 1 day
        Interval time between outputs.
    Yields
    -------
    date : datetime.timedelta
        Dates
         
    Examples
    --------
    start = timedelta(days = 24)
    end = timedelta(days = 25, minutes = 30)
    step = timedelta(hours = 1)
    yield : timedelta(days = 24) -> timedelta(days = 24, hours = 1) -> timedelta(days = 24, hours = 2)
            ... -> timedelta(days = 24, hours = 23) -> timedelta(days = 25) -> stop

    """
    current = start
    while current < stop:
        yield current
        current += step
        
        
def define_synchronous_sampling_point(reference_point, start_time, sampling_seconds):
    """
    This calculates the start point for recording sensor data in one path.  
    |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |  ... sampling points (interval lengths eqaul to sampling_seconds)
    r                         s----------walking---------e
                                x
    r : reference_point
    s : start_time
    e : end time of this walking
    x : return of this function
    A sampling point x satisfies s <= x and there exists n such that (x = r + n * sampling_seconds).
    
    Parameters
    ----------
    reference_point : datetime.timedelta
        Start time of all sampling.
    start_time : datetime.timedelta
        Start time of this walking path.
    sampling_seconds : float
        Sampling duration[sec.] of sensors.
        
    Returns
    -------
    start_point : datetime.timedelta
        Start time for recording sensor data in this path.
    """
    ret = timedelta(seconds = ((start_time - reference_point).total_seconds() // sampling_seconds) * sampling_seconds + reference_point.total_seconds())
    while ret < start_time:
        ret += timedelta(seconds = sampling_seconds)
    return ret

    
def update_state_of_binary_sensor(sensor_data, sensor_states, sensor, state, t):
    """
    Update a state of binary sensor.
    
    Parameters
    ----------
    sensor_data : list of tuple
        sensor_data[i] = (time, sensor_index, sensor_state),
            time : datetime.timedelta
                Time the sensor changes its state.
            sensor_index : int
                Index of the sensor. This is the number of Sensor.index, not the index of sensors.
            sensor_state : boolean
                Sensor state.
                For now, binary sensors are considered.
    sensor_states : dict
        Present states of sensors.
        Keys are indexes of sensors that are recorded in sensor_model.Sensor.index, not indexes in input's 'sensors'.
        Value are states of sensors.
        If the corresponded sensor of the index is sensor_model.CircularPIRSensor or sensor_model.SquarePressureSensor, the value is boolean, else None.
    sensor : sensor_model.Sensor
        Index of the sensor.
    state : boolean
        Present state of the sensor.
    t : datetime.timedelta
        Time.
    """ 
    past_state = sensor_states[sensor.index]
    if past_state != state:
        sensor_states[sensor.index] = state
        sensor_data.append((t, sensor.index, state))
        
        
        
def save_binary_sensor_data(path, sensors, sensor_data, filename = ''):
    """
    This saves binary sensor data as a text file.
    
    Parameters
    ----------
    path : pathlib.Path
        Path to the folder to save this data.
    sensors : list of sensor_model.Sensor
        Target sensors.
    sensor_data : list of tuple
        sensor_data[i] = (time, sensor_index, sensor_state),
            time : datetime.timedelta
                Time the sensor changes its state.
            sensor_index : int
                Index of the sensor. This is the number that is written in sensor_model.Sensor.index, not the index of sensors.
            sensor_state : boolean
                Sensor state.
                For now, binary sensors are considered.
    filename : str, default ''
        Filename of the text file except for the file extension.
        If this is None, then this function adaptively generates filenames.
        
    See Also
    --------
    SISG4HEI_Alpha-main/Interface/PIRRecord in SISG4HEIAlpha in [1, 2] as a reference.
    """
    if filename == '':
        sensor_types = [s.type_name for s in sensors]
        sensor_types = list(dict.fromkeys(sensor_types))  # delete duplicated type_name
        filename += "{}".format('_'.join(sensor_types))
    index2sensor = {}
    for s in sensors:
        index2sensor[s.index] = s
    data = sorted(sensor_data, key = lambda x: x[0])
    Text = ''
    for i in sorted(index2sensor.keys()):
        Text += "Sensor #{:02}: {}\n".format(i, index2sensor[i].save_text())
    Text += '\n'
    for d in data:
        if d[2] == True:
            on_off = 'ON'
        else:
            on_off = 'OFF'
        Text += "{:04} [day] {:02} [h] {:02} [m] {:02} [s] {:06} [ms] Sensor #{:02} {}\n".format(*get_d_h_m_s_ms(d[0]), d[1], on_off)
    Text += '\n'
    file_path = path / "{}.txt".format(filename)
    with open(file_path, "w") as f:
        f.write(Text)  


def generate_cost_sensor_data(sensors, AS, WT, sampling_seconds = 1, sync_reference_point = timedelta(days = 0), forgetting_labels  = []):
    """
    This simulates sensor data that is related with home appliances with or without anomalies.
    For now, the all sensors share the same sampling rate.
    
    Parameters
    ----------
    sensors : list of sensor_model.Sensor
        Target sensors. Cost sensor in this input is only considered.
    AS : list of ActivityDataPoints
        An activity sequence.
    WT : list of WalkingTrajectory
        Walking trajectories.
    sampling_seconds : float
        Sampling duration [seconds] of sensors.
    sync_reference_point : datetime.timedelta, default timedelta(days = 0)
        Start time of the all activities.
        This is used to adjust the sampling time.
    forgetting_labels : list of tuple
        Labels of forgetting.
        forgetting_labels[i] = (forgetting_ind, notice_ind, name, cost, start_time, end_time)
        forgetting_ind : int
            Index of an activity after which a resident will be forgetting.
        notice_ind : int
            Index of an activity that a resident notice forgetting and turn off the home appliance.
        name : str
            Name of a home appliance that the resident forget to turn off.
        cost : float
            Cost of the home appliance per hour.
        start_time : datetime.timedelta
            Start time of forgetting, so the home appliance is left on.
        end_time : datetime.timedelta
            End time of forgetting, so the home appliance is turned off.
    
    Returns
    -------
    sensor_data : list of tuple
        sensor_data[i] = (time, sensor_index, sensor_state),
            time : datetime.timedelta
                Time the sensor changes its state.
            sensor_index : int
                Index of the sensor. This is the number that is written in sensor_model.Sensor.index, not the index of input 'sensors'.
            sensor_state : boolean
                Sensor state.
                For now, binary sensors are considered.
                
    Notes
    -----
    All binary sensor states are initialized as False. 
    """
    sensor_data = []
    # If the sensor is sensor_model.CostSensor, the first sensor state is initialized as False.
    # States of other kinds of sensors are initialized as None (that is not updated in this function).
    sensor_states = {}
    for s in sensors:
        if isinstance(s, sensor_model.CostSensor):
            sensor_states[s.index] = False
        else:
            sensor_states[s.index] = None
    
    # update states of cost sensors for each activity and walking trajectory
    sampling_start = define_synchronous_sampling_point(sync_reference_point, AS[0].start, sampling_seconds)
    sampling_end = AS[-1].end
    sampling_step = timedelta(seconds = sampling_seconds)
    act_index = 0
    act = AS[act_index]
    wt = WT[act_index]
    
    forgetting_act = {i: [] for i in range(0, len(AS))}
    for x in forgetting_labels:
        for i in range(x[0], x[1]):
            forgetting_act[i].append(x[2])
    
    for t in date_generator(sampling_start, sampling_end, sampling_step):
        print_progress_bar(sampling_end, t, 'Making cost sensor data')
        while act.end < t:
            act_index += 1
            act = AS[act_index]
            if act_index == len(WT):
                wt = None
            else:
                wt = WT[act_index]
        home_equipments = act.activity.home_equipment.keys()
        for s in sensors:
            if isinstance(s, sensor_model.CostSensor):
                temp_state = False
                if act_index == len(WT):
                    temp_state = s.name in home_equipments
                else:
                    temp_state = (s.name in home_equipments) and (t < wt.start_time)
                if s.name in forgetting_act[act_index]:
                    temp_state = True
                update_state_of_binary_sensor(sensor_data, sensor_states, s, temp_state, t)

    return sensor_data


def save_layout(output_path, layout_path, sensors = [], WT = [], show = False, color_map_name = 'Blues', filename = '', dpi = 400, mark_point = None):
    """
    This saves a layout data as a figure with walking trajectories and sensors.
    
    Parameters
    ----------
    output_path : pathlib.Path
        Path to save the output figure.
    layout_path : pathlib.Path
        Path which have the layout json file.
        Output figure is saved in this folder.
    sensors : dict, default []
        Dictionary of sensors' information.
        If this is [], this function does not draw any sensors.
    WT : list of WalkingTrajectory, default []
        Resident's walking trajectories.
        If this is [], this function does not draw any walking trajectories.
    show : boolean
        Whether to use plt.show().        
    color_map_name : str
        Colormaps in Matplotlib.
    filename : str, default ''
        Filename of this figure.
        If this is None, then this function adaptively generates filenames.
    dpi : int
        Resolution of the figure.
    mark_point : list of tuple of float
        2D coordinate to mark.
        For example, [(100.0, 51.2), (210, 13.9)].
    
    See Also
    --------
    SISG4HEI_Alpha-main/Plot/layout_plot in SISG4HEIAlpha in [1, 2] as a reference.
    [1] C. Jiang and A. Mita, "SISG4HEI_Alpha: Alpha version of simulated indoor scenario generator for houses with elderly individuals." Journal of Building Engineering, 35-101963(2021), and
    [2] https://github.com/Idontwan/SISG4HEI_Alpha.
    """
    if filename == '':
        filename += 'Layout'
        if sensors != []:
            sensor_types = [s.type_name for s in sensors]
            sensor_types = list(dict.fromkeys(sensor_types))  # delete duplicated type_name
            filename += "_{}".format('_'.join(sensor_types))
        if WT != []:
            filename += "_WT"
        filename += '.png'
    else:
        if filename[-4:] != '.png':
            filename += '.png'
    
    # draw layout
    fp = floor_plan.FloorPlan()
    fp.load_layout(layout_path)
    ax = fp.save_layout_figure('', '', show = False, save = False, close = False)
    
    # draw sensors
    for s in sensors:
        s.draw(ax)
        
    # draw walking trajectories
    if WT != []:
        if len(WT) != 1:
            # color gradient
            cm = plt.get_cmap(color_map_name)
            cm_interval = [ i / (len(WT) - 1) for i in range(len(WT)) ]
            cm = cm(cm_interval)
        else:
            cm = ['Blue']
        for (i, wt) in enumerate(WT):
            x, y = [], []
            for c in wt.centers:
                x.append(c[0])
                y.append(c[1])
            plt.plot(x, y, '--', c = cm[i])
    
    if mark_point != None:
        for p in mark_point:
            plt.scatter(p[0], p[1], marker = 'v', color = 'black')
    
    # save
    plt.savefig(str(output_path) + '/' + filename, dpi = dpi, bbox_inches = 'tight')
    if show: plt.show()
    plt.close()
    
        
def generate_six_anomalies(path, save_path, days, show = True):
    """
    This generates anomaly labels under the test settings.
    
    Parameters
    ----------
    path : pathlib.Path
        Path which have the layout json file.
        Output figure is saved in this folder.
        
    save_path : pathlib.Path
        Path to save the images.
        
    days : int
        Period to simulate anomalies.
        
    show : bool
        Whether to show an image.
        
    Returns
    -------
    ret_dict : dict
        Dictionary of anomaly labels.
    """

    # MMSE score
    start, end, step = timedelta(days = 0), timedelta(days = days), timedelta(days = 30)
    MMSE = anomaly.simulate_MMSE(start, end, step, error_e = 0)

    # Housebound
    # housebound_labels[i] = (start_time, end_time) of i-th being housebound
    housebound_labels = anomaly.simulate_state_anomaly_periods(MMSE, 1/10, 14, 14/5)

    # Semi-bedridden
    # semi_bedridden_labels[i] = (start_time, end_time) of i-th being semi-bedridden
    semi_bedridden_labels = anomaly.simulate_state_anomaly_periods(MMSE, 1/20, 30, 30/5)

    # Wandering
    def calculate_wandering_mean_num(mmse):
        # decide the mean frequency of wandering [times/month]
        if mmse < 0 or 30 < mmse:
            raise ValueError('mmse must be 0 <= mmse <= 30')
        return - 1.86 * mmse + 56
    def calculate_wandering_mean_minutes(mmse):
        # decide the mean duration time [minutes] of wandering
        if mmse < 0 or 30 < mmse:
            raise ValueError('mmse must be 0 <= mmse <= 30')
        return - 0.31 * mmse + 9.8
    wandering_mean_num = anomaly.simulate_values_from_MMSE(MMSE, start, end, step, calculate_wandering_mean_num)
    wandering_num = [(x[0], x[1], stats.poisson.rvs(x[2])) for x in wandering_mean_num]
    wandering_mean_minutes = anomaly.simulate_values_from_MMSE(MMSE, start, end, step, calculate_wandering_mean_minutes)

    # Falls
    def calculate_falling_mean_num(mmse):
        # decide the mean number [times/month] of falling
        if mmse < 0 or 30 < mmse:
            raise ValueError('mmse must be 0 <= mmse <= 30')
        return - mmse / 15 + 2
    # falling while walking (fall_w)
    fall_w_mean_num = anomaly.simulate_values_from_MMSE(MMSE, start, end, step, calculate_falling_mean_num)
    fall_w_num = [(x[0], x[1], stats.poisson.rvs(x[2])) for x in fall_w_mean_num]
    # fall while standing (fall_s)
    fall_s_mean_num = anomaly.simulate_values_from_MMSE(MMSE, start, end, step, calculate_falling_mean_num)
    fall_s_num = [(x[0], x[1], stats.poisson.rvs(x[2])) for x in fall_s_mean_num]

    # Forgetting
    def calculate_forgetting_mean_num(mmse):
        # This returns the mean number of forgetting [times/month] by MMSE
        return - mmse + 30

    forgetting_mean_num = anomaly.simulate_values_from_MMSE(MMSE, start, end, step, calculate_forgetting_mean_num)
    forgetting_num = [(x[0], x[1], stats.poisson.rvs(x[2])) for x in forgetting_mean_num]


    # Figures of anomalies
    fig, ax = plt.subplots(6, 1, sharex = 'all', facecolor = 'w',  figsize = (6, 12))
    graph_list = [MMSE, fall_w_mean_num, wandering_mean_num, wandering_mean_minutes, forgetting_mean_num]
    samples_list = [[], fall_w_num, wandering_num, [], forgetting_num]
    label_ylabel_list = ['MMSE.', 'Frequency of fall\n[times/month].', 'Frequency of wandering\n[times/month].',
                         'Mean duration [min.]\nof wandering\n.', 'Frequency of forgetting\n[times/month].']
    graph_time_step = timedelta(days = 360)
    for (i, data) in enumerate(zip(graph_list, samples_list, label_ylabel_list)):
        x, samples, label = data[0], data[1], data[2]
        ax[i].plot([xx[0] / graph_time_step for xx in x], [xx[2] for xx in x], 'k-', label = 'expected')
        if i == 1 or i == 2 or i == 4:
            ax[i].plot([xx[0] / graph_time_step for xx in samples], [xx[2] for xx in samples], 'k--', label = 'sample')
        ax[i].set_ylabel(label)
    ax[1].legend(loc = 'best', frameon = False)
    ax[2].legend(loc = 'best', frameon = False)
    ax[4].legend(loc = 'best', frameon = False)
    # ax[4].set_xlabel('Year.')

    point_0, point_last = (timedelta(days = 0), timedelta(days = 0)), (timedelta(days = MMSE[-1][1].days), timedelta(days =  MMSE[-1][1].days))
    graph_housebound = deepcopy(housebound_labels)
    graph_semi_bedridden = deepcopy(semi_bedridden_labels)
    for p in [point_0, point_last]:
        graph_housebound.append(p)
        graph_semi_bedridden.append(p)
    ax[5].set_xlabel('Year.')
    ax[5].set_ylabel('States of housebound ($y = 1$)\nand semi-bedridden ($y = 2$).')
    ax[5].set_ylim(0, 3)
    for interval in graph_housebound:
        ax[5].hlines(1, interval[0]/graph_time_step, interval[1]/graph_time_step, color = 'black', linewidth = 10.0)
    for interval in graph_semi_bedridden:
        ax[5].hlines(2, interval[0]/graph_time_step, interval[1]/graph_time_step, color = 'black', linewidth = 10.0)

    plt.tight_layout()
    plt.savefig(str(save_path) + '/anomaly_parameter_transitions.png', dpi = 500)
    if show:
        plt.show()
    plt.close()

    anomaly.save_MMSE(save_path, MMSE)
    anomaly.save_MMSE(save_path, wandering_num, 'wandering_num')
    anomaly.save_MMSE(save_path, wandering_mean_minutes, 'wandering_mean_minutes')
    anomaly.save_MMSE(save_path, forgetting_num, 'forgetting_num')
    anomaly.save_housebound_labels(save_path, housebound_labels)
    anomaly.save_housebound_labels(save_path, semi_bedridden_labels, 'semi_bedridden_labels')
    ret_dict = {'MMSE': MMSE, 
                'housebound_labels': housebound_labels,
                 'semi_bedridden_labels': semi_bedridden_labels, 
                 'wandering_num': wandering_num,
                 'wandering_mean_minutes': wandering_mean_minutes,
                 'fall_w_num': fall_w_num,
                 'fall_s_num': fall_s_num,
                 'forgetting_num': forgetting_num}
        
    return ret_dict
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        




        

    
class TimeInterval:

    def __init__(self, start, end):
        """
        initialize of TimeInterval class

        Parameters
        ----------
        start : datetime.timedelta
            start time of this time interval
        end : datetime.timedelta
            end time of this time interval
        """ 
        if end < start:
            raise ValueError("The end time is before the start time!\nstart:{}, end:{}".format(start, end))
        self.__start = start
        self.__end = end
        self.__duration = self.__end - self.__start
        
    def __str__(self):
        return "<TimeInterval>[{}, {}]".format(self.start, self.end)
    
    @property
    def start(self):
        return self.__start
    @property
    def end(self):
        return self.__end
    @property
    def duration(self):
        return self.__duration
    
    def include(self, t):
        """
        return True if time t is in this time interval, else return False

        Parameters
        ----------
        t : datetime.timedelta
            target time

        Returns
        -------
        is_included : boolean
            whether time t is included in this time interval
        """
        if not isinstance(target_interval, timedelta):
            raise ValueError('t must be a instance of timedelta.')
        return self.start <= t <= self.end
    
    def include_TimdeInterval(self, target_interval):
        """
        return True if all of target_interval is in this time interval, else return False

        Parameters
        ----------
        target_interval : TimeInterval class

        Returns
        -------
        is_included : boolean
            whether time t is included in this time interval
        """
        if not isinstance(target_interval, TimeInterval):
            raise ValueError('target_interval must be a instance of TimeInterval.')
        return self.start <= target_interval.start and target_interval.end <= self.end
    
    def is_included_in(self, target_interval):
        """
        return True if this time interval is included in target_interval, else return False

        Parameters
        ----------
        target_interval : TimeInterval class

        Returns
        -------
        ret : boolean
            whether this time interval is included in target_interval
        """
        if not isinstance(target_interval, TimeInterval):
            raise ValueError('target_interval must be a instance of TimeInterval.')
        return target_interval.start <= self.start and self.end <= target_interval.end
    
    def overlap(self, target_interval, exclude_only_1_point = False):
        """
        return True if this time interval is overlapped with target_interval, else return False

        Parameters
        ----------
        target_interval : TimeInterval class
        exclude_only_1_point : If self interval overlap with target_interval at only 1 point, then return False

        Returns
        -------
        ret : boolean
            whether this time interval is included in target_interval
        """ 
        if not isinstance(target_interval, TimeInterval):
            raise ValueError('target_interval must be a instance of TimeInterval.')
        if exclude_only_1_point == True:
            return target_interval.end > self.start and self.end > target_interval.start
        else:
            return target_interval.end >= self.start and self.end >= target_interval.start
    
    def overlap_length_with(self, target_interval):
        """
        return overlap length between self and target_interval

        Parameters
        ----------
        target_interval : TimeInterval class

        Returns
        -------
        ret : datetime.timedelta
            overlap length between self and target_interval
            
        Example
        -------
        example 1;
        self = (timedelta(days = 2, hours = 20), timedelta(days = 3, hours = 4))
        target_interval = (timedelta(days = 2, hours = 23), timedelta(days = 2, hours = 24))
        - > return timedelta(hours = 1)
        """
        if not isinstance(target_interval, TimeInterval):
            raise ValueError('target_interval must be a instance of TimeInterval.')
        if self.overlap(target_interval) == False:
            return timedelta(hours = 0)
        elif self.include_TimdeInterval(target_interval) == True:
            return target_interval.duration
        elif self.is_included_in(target_interval) == True:
            return self.duration
        elif (target_interval.end < self.end) == True:
            return target_interval.end - self.start
        elif (self.start < target_interval.start) == True:
            return self.end - target_interval.start
        else:
            print("self = ({}, {})\ntarget_interval = ({}, {})".format(self.start, self.end, target_interval.start, target_interval.end))
            raise ValueError('There are some situations that TimeInterval.overlap_length_with cannot consider.')
            
    def set_difference_of_this_and(self, interval):
        """
        Let A be [this.start, this.end] and B be [interval.start, interval.end].
        This function returns A \ B (set difference).

        Parameters
        ----------
        interval : TimeInterval class
            subtract the interval from this class

        Returns
        -------
        interval : TimeInterval class
            results of the set difference of this and an input interval
        """ 
        if not isinstance(interval, TimeInterval):
            raise ValueError('interval must be a instance of TimeInterval.')
        
        pass
    
def extract_time_of_day(td):
    """
    This extracts only time of day from the timedelta object, and make timedelta object less than 1 day.

    Parameters
    ----------
    td : datetime.timedelta
        For example, td = datetime.timedelta(weeks=1, days=1, hours=1, minutes=1, seconds=1, milliseconds=1, microseconds=1).
    
    Returns
    -------
    ret : datetime.timedelta
        Time of day.

    Examples
    --------
    >>> td = datetime.timedelta(days=23, hours=13, minutes=56, seconds=11)
    >>> extract_time_of_day(td)
    >>> datetime.timedelta(hours=13, minutes=56, seconds=11)

    """ 
    _, h, m, s, ms = get_d_h_m_s_ms(td)
    return timedelta(hours = h, minutes = m, seconds = s, microseconds = ms)
    
