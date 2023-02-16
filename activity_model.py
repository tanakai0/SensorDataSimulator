# programs of an activity sequence generator
import random
from copy import deepcopy
from collections import namedtuple
from datetime import timedelta
from functools import total_ordering

from numpy.random import choice
from numpy.random import normal
from numpy.random import poisson
from numpy.random import rand

# self-made
import floor_plan
import sensor_model

WANDERING_NAME = 'wandering'

@total_ordering
class Activity:

    def __init__(self, name, duration_mean, duration_sd, place, color, home_equipment):
        """
        Initialization of Activity class.

        Parameters
        ----------
        name : string
            Name of the activity.
        duration_mean : float
            Mean of the duration time of the activity, [minute].
        duration_sd : float
            Standard deviation of the duration time of the activity, [minute].
        place : list of string
            Places to do the activity.
            The earlier the index of the list is, the higher the priority is.
            If this list is empty, then the place is randomly chosen.
        color : string
            Color code that is used in an activity graph.
        home_equipment : dict, default None
            The key is a home appliances, home devices or home equipment that is used during this activity.
            The value is a cost of the home equipment per hour.
            For example, {'stove in the kitchen': 10, 'faucet in the kitchen': 1}.
        """ 
        for (k, v)in home_equipment.items():
            if v < 0:
                raise ValueError("cost of {} must be positive value".format(k))
        self.__name = name
        self.__duration_mean = duration_mean
        self.__duration_sd = duration_sd
        self.place = place
        self.__color = color
        self.home_equipment = dict()
        for (k, v) in home_equipment.items():
            self.home_equipment[k] = v
        
    def __str__(self):
        return "<Activity>" + self.name
    
    # when to calculate equation of Activity, Activity.name is only considered
    def __eq__(self, other):
        if not isinstance(other, Activity):
            return NotImplemented
        return self.name == other.name
    
    def __lt__(self, other):
        if not isinstance(other, Activity):
            return NotImplemented
        return False
    
    # getter
    @property
    def name(self):
        return self.__name
    @property
    def duration_mean(self):
        return self.__duration_mean
    @property
    def duration_sd(self):
        return self.__duration_sd
    @property
    def color(self):
        return self.__color
    
    def sampling_duration_time(self, lower_limit = None, upper_limit = None, max_try = 100):
        """
        This samples duration time of the activity from a normal distribution with a rejection range

        Parameters
        ----------
        lower_limit : datetime.timedelta, default None
            Lower limit of the output.
            If this is None then this function uses min_ = self.__duration_mean / 10.
        upper_limit: datetime.timedelta, default None
            Upper limit of the output.
            If this is None then this function uses max_ = self.__duration_mean * 19 / 10.  
        max_try: int
            Maximum number of time to resample.
            
        Returns
        -------
        duration_time : datetime.timedelta
            Duration time of the activity.

        Raises
        ------
        1. lower_limit > 0
        2. lower_limit <= upper_limit
        3. fail to sampling max_try times
        """ 
        
        mean = self.__duration_mean
        sd = self.__duration_sd
        if lower_limit == None:
            lower_limit = mean / 10
        if upper_limit == None:
            upper_limit = mean * 19 / 10
        
        if not(lower_limit > 0):
            raise ValueError('The lower_limit must be a positive value.')
        if upper_limit < lower_limit:
            raise ValueError('The lower_limit must be lower than the upper_limit. If you don\'t specify each of them, lower_limit or upper_limit are defined as another way. See detail in the decostring of this functions.')
            
        is_valid = False
        try_count = 0
        duration_time = None
        while is_valid == False:
            duration_time = normal(mean, sd)
            if lower_limit <= duration_time <= upper_limit:
                is_valid = True
            try_count += 1
            if try_count >= max_try:
                raise ValueError("Activity.sampling_duration_time() failed to sample duration time of {} {} times.".format(self.__name, try_count))

        return timedelta(minutes = duration_time)
    
    def determine_place(self, existing_places):
        """
        This determines the place to do this activity from the existing_places and self.place.
        
        Parameters
        ----------
        existing_places : list of str
            Names of places that exists in a layout.
        
        Returns
        -------
        place : str
            Name of the place to do this activity.
            If there are not any appropritate place, then this returns None.
            
        Notes
        -----
            For now, this function is expected to be the same combination.
            If you want to introduce randomness, please modify mainly anomaly.make_forgetting_ranges() function and so on.
        """
        # the earlier the index of the self.place is, the higher the priority is 
        for c in self.place:
            if c in existing_places:
                return c
        if self.place == []:
            return random.choices(existing_places, k = 1)[0]
        else:
            return None
        
class FundamentalActivity(Activity):
    
    def __init__(self, name, time_mean, time_sd, duration_mean, duration_sd, place, color, home_equipment):
        """
        Initialization of FundamentalActivity class.

        Parameters
        ----------
        name : string
            Name of the activity.
        time_mean : float
            Mean of the start time of the activity, [minutes from 0:00].
        time_sd : float
            Standard deviation of the start time of the activity, [minute].
        duration_mean : float
            Mean of the duration time of the activity, [minute].
        duration_sd : float
            Standard deviation of the duration time of the activity, [minute].
        place : list of string
            Places to do the activity.
            The earlier the index of the list is, the higher the priority is. 
            If this list is empty, then the place is randomly chosen.
        color : string
            Color code that is used in an activity graph.
        home_equipment : dict, default None
            The key is a home appliances, home devices or home equipment that is used during this activity.
            The value is a cost of the home equipment per hour.
            For example, {'stove in the kitchen': 10, 'faucet in the kitchen': 1}.
        """ 
        super().__init__(name, duration_mean, duration_sd, place, color, home_equipment)
        self.__time_mean = time_mean
        self.__time_sd = time_sd
    
    # getter
    @property
    def time_mean(self):
        return self.__time_mean
    @property
    def time_sd(self):
        return self.__time_sd
    
    def sampling_start_time(self, num_days, lower_limit = None, upper_limit = None, max_try = 100):
        """
        This samples start time from a normal distribution with a rejection range.

        Parameters
        ----------
        num_days : int
            number of days
        lower_limit : datetime.timedelta, default None
            Lower limit of the output.
            If this is None then this function uses self.__time_mean - 4 * self.__time_sd.           
        upper_limit: datetime.timedelta, default None
            Upper limit of the output.
            If this is None then this function uses self.__time_mean + 4 * self.__time_sd.            
        max_try: int, default 100
            Maximum number of time to resample.

        Returns
        -------         
        start_time : datetime.timedelta
            start time of the activity

        Raises
        ------
        1. lower_limit > 0
        2. upper_limit < 2 * 24 * 60
        3. lower_limit <= upper_limit
        4. fail to sampling max_try times
        """ 

        mean = self.__time_mean
        sd = self.__time_sd
        if lower_limit == None:
            lower_limit = self.__time_mean - 4 * self.__time_sd
        if upper_limit == None:
            upper_limit = self.__time_mean + 4 * self.__time_sd
        
        # Especially for sleep activity, the start time is shifted if the lower_limit is less than 0.
        # So, start time 01:00 may be converted into start time 25:00.
        # To avoid this exception clearly, please specify time_mean over 24:00, e.g., 01:34 -> 25:34.
        if lower_limit < 0:
            minutes_of_day = 24 * 60
            lower_limit += minutes_of_day
            upper_limit += minutes_of_day
            mean += minutes_of_day
        if (lower_limit <= 0) or (upper_limit < lower_limit) or (2 * 24 * 60 <= upper_limit):
            raise ValueError('The lower_limit and upper_limit must satisfy the following conditions. lower_limit > 0, upper_limit < 2 * 24 * 60, and lower_limit <= upper_limit. If you don\'t specify each of them, lower_limit or upper_limit are defined as another way. See detail in the decostring of this functions.')
            
        is_valid = False
        try_count = 0
        while is_valid == False:
            start_time = normal(mean, sd)
            if lower_limit <= start_time <= upper_limit:
                is_valid = True
            try_count += 1
            if try_count >= max_try:
                raise ValueError("FundamentalActivity.sampling_start_time() failed to sample start time of {} {} times.".format(self.name, try_count))

        return timedelta(days = num_days, minutes = start_time)
    
    @staticmethod
    def check_order(fundamental_activities, target_order):
        """
        This checks the order of fundamental activities.

        Parameters
        ----------
        fundametal_activities : list of tuple
            For example, fundamental_activities = [(take_breakfast, start_time, end_time), ..., (sleep, start_time, end_time)].
            fundamental_activities[i] = (act, start_time, end_time)
                act : activity_model.Activity
                start_time : datetime.timedelta
                end_time : datetime.timedelta
        target_order : list of Activity
            True order of fundamental activities.

        Returns
        -------
        is_OK : boolean
            Whether the fundamental activities satisfy the order condition.
        """ 
        fundamental_activities.sort(key = lambda x: x[1])
        order = [x[0].name for x in fundamental_activities]
        if order != [x.name for x in target_order]:
            return False
            # raise ValueError('The order of the sorted fundamental_activities does not eaul to target_order.')  
        return True
    
    
    def modified_new_activity(self, *, name = None, time_mean = None, time_sd = None, duration_mean = None, duration_sd = None, place = None, color = None, home_equipment = None):
        """
        This generates a new fundamental activity that is modified some parameters of original fundamental activity.
        The parameters of None input inherit original parameters. 

        Parameters
        ----------
        name : string
            Name of the activity.
        time_mean : float
            Mean of the start time of the activity, [minutes from 0:00].
        time_sd : float
            Standard deviation of the start time of the activity, [minute].
        duration_mean : float
            Mean of the duration time of the activity, [minute].
        duration_sd : float
            Standard deviation of the duration time of the activity, [minute].
        place : list of string
            Places to do the activity.
            The earlier the index of the list is, the higher the priority is.
            If this list is empty, then the place is randomly chosen.
        color : string
            Color code that is used in an activity graph.
        home_equipment : dict, default None
            The key is a home appliances, home devices or home equipment that is used during this activity.
            The value is a cost of the home equipment per hour.
            For example, {'stove in the kitchen': 10, 'faucet in the kitchen': 1}.
        
        Returns
        -------
        new_activity : activity_model.FundamentalActivity
            Modified fundamental activity
        """
        if name == None:
            name = self.name
        if time_mean == None:
            time_mean = self.time_mean
        if time_sd == None:
            time_sd = self.time_sd
        if duration_mean == None:
            duration_mean = self.duration_mean
        if duration_sd == None:
            duration_sd = self.duration_sd
        if place == None:
            place = self.place
        if color == None:
            color = self.color
        if home_equipment == None:
            home_equipment = self.home_equipment
        return FundamentalActivity(name, time_mean, time_sd, duration_mean, duration_sd, place, color, home_equipment)
    

class NecessaryActivity(Activity):
    
    repulsive_constraints = []
    exist_repulsive_constraints = False
    existence_constraints = []
    exist_existence_constraints = False
    
    def __init__(self, name, num_mean, duration_mean, duration_sd, place, color, home_equipment):
        """
        initialization of NecessaryActivity class

        Parameters
        ----------
        name : string
            Name of the activity.
        num_mean : int
            Mean number of daily frequnecy of the activity.
        duration_mean : float
            Mean of the duration time of the activity, [minute].
        duration_sd : float
            Standard deviation of the duration time of the activity, [minute].
        place : list of string
            Places to do the activity.
            The earlier the index of the list is, the higher the priority is.
            If this list is empty, then the place is randomly chosen.
        color : string
            Color code that is used in an activity graph.
        home_equipment : dict, default None
            The key is a home appliances, home devices or home equipment that is used during this activity.
            The value is a cost of the home equipment per hour.
            For example, {'stove in the kitchen': 10, 'faucet in the kitchen': 1}.
        """ 
        super().__init__(name, duration_mean, duration_sd, place, color, home_equipment)
        self.__num_mean = num_mean
    
    # getter
    @property
    def num_mean(self):
        return self.__num_mean
    
    def sampling_number_of_activity(self):
        """
        This samples the daily frequency of the necessary activity from a Poisson distribution.

        Returns
        -------
        act_num : int
            the number of this activity in a 1 day
        """ 
        return poisson(lam = self.__num_mean)
    
    def sampling_start_time(self, duration_time, remainder, rejection_range = None, max_try = 100):
        """
        This samples start time of the necessary activity from an uniform distribution between blank times with rejection conditions.

        Parameters
        ----------
        duration_time : datetime.timedelta
            Duration time of the activity.
        remainder : list of tuple of datetime.timedelta
            Time ranges in which any activity are not done yet.
            remainder[i] = (start_time, end_time)
                start_time : datetime.timedelta 
                end_time : datetime.timedelta 
        rejection_range : datetime.timedelta, default None
            To prevent to generate a short time range, this rejects a sample if the sample finally makes time range that is shorter than rejection_range.
            If this is None then this uses rejection_range = timedelta(minutes = 1).
        max_try: int, default 100
            Maximum number of time to resample.
            
        Returns
        -------
        start_time : datetime.timedelta
            Start time of the activity.

        Raises
        ------
        1. fail to sample max_try times
        """ 
        is_valid = False
        try_count = 0
        
        # rejection_range = timedelta(minutes = self.duration_mean / 10)
        rejection_range = timedelta(minutes = 1)
        rejection_seconds = rejection_range.total_seconds()
        
        while is_valid == False:
            if try_count >= max_try:
                raise ValueError("NecessaryActivity.sampling_start_time() failed to sample start time of {} {} times.".format(self.name, try_count))
            try_count += 1
            sub_range_index = choice(len(remainder))
            sub_range = remainder[sub_range_index]
            sub_start_seconds, sub_end_seconds = sub_range[0].total_seconds(), sub_range[1].total_seconds()
            start_time_seconds = rand() * (sub_end_seconds - sub_start_seconds) + sub_start_seconds
            # prevent to generate a short time range
            if start_time_seconds - sub_start_seconds < rejection_seconds:
                continue
            if start_time_seconds + duration_time.total_seconds() + rejection_seconds >= sub_end_seconds:
                continue
            is_valid = True
            
        return timedelta(seconds = start_time_seconds)
    
 
    def set_repulsive_constraints(self, constraints):
        """
        This prepares a repulsive constrains between activities.
        This activity apart from the specified activities from the viewpoint of time.

        Parameters
        ----------
        constraints : list of tuple
            constraints[i] = (act_name, act_interval)
                act_name     : str
                act_interval : datetime.timedelta
                The activity of this class must be act_interval or more from act_name.
        """
        self.repulsive_constraints = deepcopy(constraints)
        if self.repulsive_constraints != []:
            self.exist_repulsive_constraints = True
        
    def set_existence_constraints(self, constraints):
        """
        This prepares an existence constrains between activities.
        This activity must exist between specified activiites.
        If the specified activities do not exist, then the constraints are skipped (not thought).
        The faster the element appear, the higher the priority is,

        Parameters
        ----------
        constraints : tuple
            (pre_act, post_act)
            pre_act  : activity_model.Activity
            post_act : activity_model.Activity
            At least one activity of this class must exist in between pre_act and post_act.
        """
        self.existence_constraints = deepcopy(constraints)
        if self.existence_constraints != []:
            self.exist_existence_constraints = True
        
        
    def sampling_start_time_with_constraints(self, duration_time, remainder, schedule, rejection_range = None, max_try = 100):
        """
        This samples start time from an uniform distribution between blank times with rejection conditions and constraints.

        Parameters
        ----------
        duration_time : datetime.timedelta
            Duration time of the activity
        remainder : list of tuple of datetime.timedelta
            Time ranges in which any activity are not done yet.
            remainder[i] = (start_time, end_time)
                start_time : datetime.timedelta 
                end_time : datetime.timedelta 
        schedule : list of tuple
            Sequence data of already planned activities.
            shcedule[i] = (class, start_time, end_time)
                class : activity_model.Activity
                start_time : datetime.timedelta
                end_time : datetime.timedelta
        rejection_range : datetime.timedelta, default None
            To prevent to generate a short time range, this rejects a sample if the sample finally makes time range that is shorter than rejection_range.
            If this is None then this uses rejection_range = timedelta(minutes = 1).
        max_try: int, default 100
            Maximum number of time to resample.
            
        Returns
        -------
        start_time : datetime.timedelta
            start time of the activity

        Notes
        -----
        To prevent to make short activities, 

        Raises
        ------
        1. fail to sample max_try times.
        """ 
        is_valid = False
        try_count = 0
        
        rejection_range = timedelta(minutes = 1)
        rejection_seconds = rejection_range.total_seconds()
        
        ### consider the existence constraints: this activity must be in between the pre activity and the post activity.
        satisfy_existence = True  # whether it satisfies the existence constraint
        restricted_remainder = []
        if self.exist_existence_constraints == True:
            sorted_schedule = sorted(schedule, key = lambda x: x[1])
            pre_time, post_time = None, None  # end time of the pre activity, start time of the post activity
            # check whether the constraints are held.
            for c in self.existence_constraints:
                satisfy_existence = False
                pre_time, post_time = None, None
                is_pre = False  # whether already pass the pre activity
                for act in sorted_schedule:
                    if act[0].name == c[0].name:
                        pre_time = act[2]
                        is_pre = True
                    elif act[0].name == c[1].name and is_pre == True:
                        post_time = act[1]
                        break
                    elif act[0].name == self.name and is_pre == True:
                        satisfy_existence = True
                if pre_time == None or post_time == None:
                    satisfy_existence = True
                if satisfy_existence == False:
                    break
            if satisfy_existence == False:
                for r in remainder:
                    if pre_time <= r[0] and r[1] <= post_time:
                        restricted_remainder.append(r)
        
        while is_valid == False:
            if try_count >= max_try:
                raise ValueError("NecessaryActivity.sampling_start_time_with_constraints() failed to sample start time of {} {} times.".format(self.name, try_count))
            try_count += 1
            
            sub_range_index = None
            sub_range = None
            if satisfy_existence == False:
                sub_range_index = choice(len(restricted_remainder))
                sub_range = restricted_remainder[sub_range_index]
            else:
                sub_range_index = choice(len(remainder))
                sub_range = remainder[sub_range_index]
                
            sub_start_seconds, sub_end_seconds = sub_range[0].total_seconds(), sub_range[1].total_seconds()
            start_time_seconds = rand() * (sub_end_seconds - sub_start_seconds) + sub_start_seconds
            # prevent to generate a short time range
            if start_time_seconds - sub_start_seconds < rejection_seconds:
                continue
            if start_time_seconds + duration_time.total_seconds() + rejection_seconds >= sub_end_seconds:
                continue
                
            ### consider the repulsive constraints
            start_time = timedelta(seconds = start_time_seconds)
            end_time = start_time + duration_time
            satisfy_constraints = True
            for c in self.repulsive_constraints:
                for act in schedule:
                    if act[0].name == c[0]:
                        if (start_time < act[2] + c[1]) and (act[1] - c[1] < end_time):
                            satisfy_constraints = False
            if satisfy_constraints == False:
                continue
            is_valid = True
            
        return timedelta(seconds = start_time_seconds)
        
    
    def modified_new_activity(self, *, name = None, num_mean = None, duration_mean = None, duration_sd = None, place = None, color = None, home_equipment = None):
        """
        This returns a new necessary activity.
        The parameters of None input inherit original parameters. 

        Parameters
        ----------
        name : string
            Name of the activity.
        num_mean : int
            Mean number of daily frequnecy of the activity.
        duration_mean : float
            Mean of the duration time of the activity, [minute].
        duration_sd : float
            Standard deviation of the duration time of the activity, [minute].
        place : list of string
            Places to do the activity.
            The earlier the index of the list is, the higher the priority is.
            If this list is empty, then the place is randomly chosen.
        color : string
            Color code that is used in an activity graph.
        home_equipment : dict, default None
            The key is a home appliances, home devices or home equipment that is used during this activity.
            The value is a cost of the home equipment per hour.
            For example, {'stove in the kitchen': 10, 'faucet in the kitchen': 1}.
        
        Returns
        -------
        new_activity : activity_model.NecessaryActivity
            Modified new activity.
        """
        if name == None:
            name = self.name
        if num_mean == None:
            num_mean = self.num_mean
        if duration_mean == None:
            duration_mean = self.duration_mean
        if duration_sd == None:
            duration_sd = self.duration_sd
        if place == None:
            place = self.place
        if color == None:
            color = self.color
        if home_equipment == None:
            home_equipment = self.home_equipment
        act = NecessaryActivity(name, num_mean, duration_mean, duration_sd, place, color, home_equipment)
        act.set_repulsive_constraints(self.repulsive_constraints)
        act.set_existence_constraints(self.existence_constraints)
        return act
    

class RandomActivity(Activity):
            
    def __init__(self, name, duration_mean, duration_sd, place, color, home_equipment, weight = 1.0):
        """
        initialization of RandomActivity class
        
        Parameters
        ----------
        name : string
            Name of the activity.
        duration_mean : float
            Mean of the duration time of the activity, [minute].
        duration_sd : float
            Standard deviation of the duration time of the activity, [minute].
        place : list of string
            Places to do the activity.
            The earlier the index of the list is, the higher the priority is.
            If this list is empty, then the place is randomly chosen.
        color : string
            Color code that is used in an activity graph.
        home_equipment : dict, default None
            The key is a home appliances, home devices or home equipment that is used during this activity.
            The value is a cost of the home equipment per hour.
            For example, {'stove in the kitchen': 10, 'faucet in the kitchen': 1}.
        weight : float
            Weight to select this activity among all random activities.
            See RandomActivity.samplig_random_activity() in detail.
        """ 
        super().__init__(name, duration_mean, duration_sd, place, color, home_equipment)
        self.weight = weight
        if self.weight < 0:
            raise ValueError('The weight must be 0 or positive value.')
        
    @staticmethod
    def sampling_random_activity(activity_candidates):
        """
        This samples a random activity from all random activities.

        Parameters
        ----------
        activity_candidates : list of activity_model.Activity
             Candidates or random activities.

        Returns
        -------
        activity : activity_model.Activity
            Selected activity.
        """ 
        if activity_candidates == []:
            raise ValueError('The activity_candidates must include at least one element.')
        weights = [act.weight for act in activity_candidates]
        return random.choices(activity_candidates, k = 1, weights = weights)[0]
    
    def modified_new_activity(self, *, name = None, duration_mean = None, duration_sd = None, place = None, color = None, home_equipment = None, weight = None):
        """
        This returns a new random activity.
        The parameters of None input inherit original parameters. 

        Parameters
        ----------
        name : string
            Name of the activity.
        duration_mean : float
            Mean of the duration time of the activity, [minute].
        duration_sd : float
            Standard deviation of the duration time of the activity, [minute].
        place : list of string
            Places to do the activity.
            The earlier the index of the list is, the higher the priority is.
            If this list is empty, then the place is randomly chosen.
        color : string
            Color code that is used in an activity graph.
        home_equipment : dict, default None
            The key is a home appliances, home devices or home equipment that is used during this activity.
            The value is a cost of the home equipment per hour.
            For example, {'stove in the kitchen': 10, 'faucet in the kitchen': 1}.
        weight : float
            Weight to select this activity among all random activities.
            See RandomActivity.samplig_random_activity() in detail.
            
        Returns
        -------
        new_activity : activity_model.RandomActivity
            Modified new activity.
        """
        if name == None:
            name = self.name
        if duration_mean == None:
            duration_mean = self.duration_mean
        if duration_sd == None:
            duration_sd = self.duration_sd
        if place == None:
            place = self.place
        if color == None:
            color = self.color
        if home_equipment == None:
            home_equipment = self.home_equipment
        if weight == None:
            weight = self.weight
        return RandomActivity(name, duration_mean, duration_sd, place, color, home_equipment, weight)
    

### ------------------------- Meta Activity --------------------------
# MetaActivity consists of multiple activities.

class MetaActivity:

    def __init__(self, name, sub_activities):
        """
        initialization of MetaActivity class

        Parameters
        ----------
        name : string
            Name of the activity.
        sub_activities : list of Activity
            sub_activities[i] = i-th activity in this meta activity
        """ 
        self.__name = name
        self.sub_activities = deepcopy(sub_activities)
        
    def __str__(self):
        return "<MetaActivity>" + self.name
    
    # getter
    @property
    def name(self):
        return self.__name
    @property
    def duration_mean(self):
        return self.__duration_mean
    @property
    def duration_sd(self):
        return self.__duration_sd
    
    def sampling_sub_activities(self, start_time, lower_limit = None, upper_limit = None, max_try = 100):
        """
        This samples duration time of the activities from a normal distribution with a rejection range.

        Parameters
        ----------
        start_time : datetime.timedelta
            Start time of the activity.
        lower_limit : datetime.timedelta, default None
            Lower limit of the output.
            If this is None then this function uses min_ = self.__duration_mean / 10.
        upper_limit: datetime.timedelta, default None
            Upper limit of the output.
            If this is None then this function uses max_ = self.__duration_mean * 19 / 10.     
        max_try: int
            Maximum number of time to resample.

        Returns
        -------
        sub_activites: list of tuple
            sub_activities[i] = (class, start time, duration time) of i-th sub activities
        """ 
        sub_activities = []
        sub_start = start_time
        for sub_act in self.sub_activities:
            sub_duration = sub_act.sampling_duration_time()
            sub_activities.append((sub_act, sub_start, sub_start + sub_duration))
            sub_start += sub_duration
        
        return sub_activities
    
    def determine_place(self, existing_places):
        """
        This determines the place to do the activities.
        
        Parameters
        ----------
        existing_places : list of str
            Names of places that exist in the target layout.
        
        Returns
        -------
        place : True or None
            If the all sub activities have an appropritate place, the this returns True, else returns None.
            
        Note
        ----
        Random determination can be introduced by modification of this function. 
        """
        is_valid = True
        for sub_act in self.sub_activities:
            if sub_act.determine_place(existing_places) == None:
                is_valid = None
        return is_valid
    
    
class MetaFundamentalActivity(MetaActivity):
    
    def __init__(self, name, sub_activities, time_mean, time_sd):
        """
        initialization of MetaFundamentalActivity class

        Parameters
        ----------
        name : string
            Name of the activity.
        sub_activities : list of Activity
            sub_activities[i] = i-th activity in this meta activity
        time_mean : float
            Mean of the start time of the first activity, [minutes from 0:00].
        time_sd : float
            Standard deviation of the start time of the first activity, [minute].
        """ 
        super().__init__(name, sub_activities)
        self.__time_mean = time_mean
        self.__time_sd = time_sd
    
    # getter
    @property
    def time_mean(self):
        return self.__time_mean
    @property
    def time_sd(self):
        return self.__time_sd
    
    def sampling_start_time(self, num_days, lower_limit = None, upper_limit = None, max_try = 100):
        """
        This samples start time from a normal distribution with a rejection range.

        Parameters
        ----------
        num_days : int
            Number of days.
        lower_limit : datetime.timedelta, default None
            Lower limit of the output.
            If this is None then this function uses self.__time_mean - 4 * self.__time_sd.
        upper_limit: datetime.timedelta, default None
            Upper limit of the output.
            If this is None then this function uses self.__time_mean + 4 * self.__time_sd.
        max_try: int, default 100
            Maximum number of time to resample.

        Returns
        -------         
        start_time : datetime.timedelta
            Start time of the first activity.
            
        Raises
        ------
        1. lower_limit > 0
        2. upper_limit < 2 * 24 * 60
        3. lower_limit <= upper_limit
        4. fail to sampling max_try times
        """ 

        mean = self.__time_mean
        sd = self.__time_sd
        if lower_limit == None:
            lower_limit = self.__time_mean - 4 * self.__time_sd
        if upper_limit == None:
            upper_limit = self.__time_mean + 4 * self.__time_sd
        
        # Especially for sleep activity, the start time is shifted if the lower_limit is less than 0.
        # So, start time 01:00 may be converted into start time 25:00.
        # To avoid this exception clearly, please specify time_mean over 24:00, e.g., 01:34 -> 25:34.
        if lower_limit < 0:
            minutes_of_day = 24 * 60
            lower_limit += minutes_of_day
            upper_limit += minutes_of_day
            mean += minutes_of_day
        if (lower_limit <= 0) or (upper_limit < lower_limit) or (2 * 24 * 60 <= upper_limit):
            raise ValueError('The lower_limit and upper_limit must satisfy the following conditions. lower_limit > 0, upper_limit < 2 * 24 * 60, and lower_limit <= upper_limit. If you don\'t specify each of them, lower_limit or upper_limit are defined as another way. See detail in the decostring of this functions.')

        is_valid = False
        try_count = 0
        while is_valid == False:
            start_time = normal(mean, sd)
            if lower_limit <= start_time <= upper_limit:
                is_valid = True
            try_count += 1
            if try_count >= max_try:
                raise ValueError("MetaFundamentalActivity.sampling_start_time() failed to sample start time of {} {} times.".format(self.__name, try_count))

        return timedelta(days = num_days, minutes = start_time)
    
    
class MetaNecessaryActivity(MetaActivity):
    
    def __init__(self, name, sub_activities, num_mean):
        """
        initialization of MetaNecessaryActivity class

        Parameters
        ----------
        name : string
            Name of the activity.
        sub_activities : list of Activity
            sub_activities[i] = i-th activity in this meta activity
        num_mean : int
            Mean number of daily frequency.
        """ 
        super().__init__(name, sub_activities)
        self.__num_mean = num_mean
    
    # getter
    @property
    def num_mean(self):
        return self.__num_mean
    
    def sampling_number_of_activity(self):
        """
        This samples a daily frequency of the necessary activity from a Poisson distribution.

        Returns
        -------
        act_num : int
            A daily frequency of this activity.
        """ 
        return poisson(lam = self.__num_mean)
    
    def sampling_start_time(self, duration_time, remainder, rejection_range = None, max_try = 100):
        """
        This samples start time from an uniform distribution between the blank times and rejection conditions.

        Parameters
        ----------
        duration_time : datetime.timedelta
            Duration time of the activity.
            
        remainder : list of tuple of datetime.timedelta
            Time ranges in which any activity are not done yet.
            remainder[i] = (start_time, end_time)
                start_time : datetime.timedelta 
                end_time : datetime.timedelta 
        rejection_range : datetime.timedelta, default None
            To prevent to generate a short time range, this rejects a sample if the sample finally makes time range that is shorter than rejection_range.
            If this is None then this uses rejection_range = timedelta(minutes = 1).
        max_try: int, default 100
            Maximum number of time to resample.
        
        Returns
        -------
        start_time : datetime.timedelta
            Start time of the activity.

        Raises
        ------
        1. fail to sampling max_try times
        """ 
        is_valid = False
        try_count = 0
        
        rejection_range = timedelta(minutes = 1)
        rejection_seconds = rejection_range.total_seconds()
        
        while is_valid == False:
            if try_count >= max_try:
                raise ValueError("MetaNecessaryActivity.sampling_start_time() failed to sample start time of {} {} times.".format(self.name, try_count))
            try_count += 1
            sub_range_index = choice(len(remainder))
            sub_range = remainder[sub_range_index]
            sub_start_seconds, sub_end_seconds = sub_range[0].total_seconds(), sub_range[1].total_seconds()
            start_time_seconds = rand() * (sub_end_seconds - sub_start_seconds) + sub_start_seconds
            # prevent to generate a short time range
            if start_time_seconds - sub_start_seconds < rejection_seconds:
                continue
            if start_time_seconds + duration_time.total_seconds() + rejection_seconds >= sub_end_seconds:
                continue
            is_valid = True
            
        return timedelta(seconds = start_time_seconds)
    
    
    
class MetaRandomActivity(MetaActivity):
    
    def __init__(self, name, sub_activities, weight = 1):
        """
        initialization of MetaRandomActivity class

        Parameters
        ----------
        name : string
            Name of the activity.
        sub_activities : list of Activity
            sub_activities[i] = i-th activity in this meta activity
        weight : float
            Weight to select this activity among all random activities
            see function RandomActivity.samplig_random_activity
        """ 
        super().__init__(name, sub_activities)
        self.weight = weight
        if self.weight < 0:
            raise ValueError('The weight must be 0 or positive value.')
    
    
def determine_activity_set(path, original_act_model):
    """
    Thie checks the presence or absence of activities based on the presence or absence of furniture.

    Parameters
    ----------
    path : pathlib.Path
        Path to the floor plan data.
    original_act_model : dictionary
        For example, see activity_model.basic_activity_model.
    Returns
    -------
    new_act_model : dictionary
        Nodified activity model that except for the activities that do not have the corresponded places.
        For example, see activity_model.basic_activity_model.
    """
    fp = floor_plan.FloorPlan()
    fp.load_layout(path)
    existing_places = fp.existing_places()
    new_act_model = deepcopy(original_act_model)
    new_act_model['fundamental_activities'] = [a for a in new_act_model['fundamental_activities'] if a.determine_place(existing_places) != None]
    new_act_model['necessary_activities'] = [a for a in new_act_model['necessary_activities'] if a.determine_place(existing_places) != None]
    new_act_model['random_activities'] = [a for a in new_act_model['random_activities'] if a.determine_place(existing_places) != None]    
    return new_act_model
        

# walking activity
# class for walking
#  Attributes
#  ----------
#  name : string
#      Name of the activity.
#  stride : float
#      Stride length of 1 step, [cm].
#  speed : float
#      Walking speed, [seconds/step].
#  prefer_foot : string
#      Preferable foot the resident push out when to start to walk.
walking_activity = namedtuple('walking_activity', 'name stride step_speed prefer_foot')
indoor_movement = walking_activity('indoor movement', 55, 0.8, 'left')

# example of concrete settings ----------------------------------------------------------------------------------------------------------

# basic activity model
# (timedelta object) / timedelta(minutes = 1) returns float value whose unit is minute
have_breakfast = FundamentalActivity('Have breakfast', timedelta(hours = 7, minutes = 17) / timedelta(minutes = 1), 30, 34, 10, ['Dinner_Table_Chair', 'Entrance'], '#ffd700', {})
have_lunch = FundamentalActivity('Have lunch', timedelta(hours = 12) / timedelta(minutes = 1), 30, 42, 10, ['Dinner_Table_Chair', 'Entrance'], '#ffa500', {})
have_dinner = FundamentalActivity('Have dinner', timedelta(hours = 18, minutes = 18) / timedelta(minutes = 1), 30, 45, 10, ['Dinner_Table_Chair', 'Entrance'], '#b22222', {})
sleep = FundamentalActivity('Sleep', timedelta(hours = 21, minutes = 29) / timedelta(minutes = 1), 40, 482, 30, ['Bed'], '#696969', {})
# sleep = FundamentalActivity('Sleep', timedelta(hours=25, minutes=29) / timedelta(minutes = 1), 40, 382, 30, ['Bed'], '#696969', {})

defecation = NecessaryActivity('Defecation', 1, 10, 3, ['Toilet_Door'], '#4b0082', {})
urination = NecessaryActivity('Urination', 5, 3, 0.5, ['Toilet_Door'], '#6a5acd', {})
take_a_bath = NecessaryActivity('Take a bath', 1, 30, 10, ['Bathroom_Door'], '#00ffff', {sensor_model.FLOW_BATHROOM: 1})
change_clothes = NecessaryActivity('Change clothes', 2, 5, 1, ['Wardrobe', 'Bed'], '#008000', {})
brush_teeth = NecessaryActivity('Brush teeth', 2, 1.5, 0.5, ['Bathroom_Door'], '#2f4f4f', {sensor_model.FLOW_BATHROOM:1})

rest = RandomActivity('Rest', 30, 10, ['Sofa', 'Desk_Chair', 'Bed'], '#d3d3d3', {})
go_out = RandomActivity('Go out', 40, 20, ['Entrance'], '#ffe4c4', {})
cooking = RandomActivity('Cooking', 30, 10, ['Kitchen_Stove'], '#ff0000', {sensor_model.POWER_KITCHEN: 1, sensor_model.FLOW_KITCHEN :1})
clean = RandomActivity('Clean', 30, 10, ['Trash_Bin'], '#bc8f8f', {})
do_laundry = RandomActivity('Do laundry', 5, 1, ['Wash_Machine'], '#008080', {})
watch_TV = RandomActivity('Watch TV', 40, 20, ['Sofa'], '#7cfc00', {sensor_model.POWER_TV: 1})
read = RandomActivity('Read', 40, 20, ['Desk_Chair', 'Desk'], '#808000', {})
take_a_snack = RandomActivity('Take a snack', 10, 3, ['Dinner_Table_Chair', 'Sofa'], '#ff69b4', {})
use_the_phone = RandomActivity('Use the phone', 10, 3, ['Desk_Chair', 'Desk', 'Bed'], '#800000', {})

# extra activities
wandering = NecessaryActivity(WANDERING_NAME, 0, 5, 1, [], '#ffff00', {})
# place == [] means the start point of the wandering is randomly chosen in the layout, see Activity.determine_place() in detail
nap = RandomActivity('Nap', 40, 13, ['Bed', 'Sofa'], '#696969', {})
go_out_necessary = NecessaryActivity(go_out.name, 4, go_out.duration_mean, go_out.duration_sd, go_out.place, go_out.color, go_out.home_equipment)
use_the_phone_necessary = NecessaryActivity(use_the_phone.name, 1, use_the_phone.duration_mean, use_the_phone.duration_sd, use_the_phone.place, use_the_phone.color, use_the_phone.home_equipment)

# dictionary of basic activity model that contains fundamental activities, necessary activities, random activities and walking activities.
basic_activity_model = {
    'fundamental_activities': [have_breakfast, have_lunch, have_dinner, sleep],
    'necessary_activities': [defecation, urination, take_a_bath, change_clothes, brush_teeth],
    'random_activities': [rest, go_out, cooking, clean, do_laundry, watch_TV, read, take_a_snack, use_the_phone],
    'walking_activities': [indoor_movement],
    # The order of the fundamental activities: the smaller activity's index is, the earlier the activity is done. 
    'fundamental_activity_order': [have_breakfast, have_lunch, have_dinner, sleep],
    'sleep': [sleep]
}
    

# applied activity model 1 -----------------------------------------------------------------------------------
applied_have_lunch_0 = Activity('(sub 1) Take foods', 0.5, 0.001, ['Refrigerator', 'Kitchen_Stove'], '#ffdacc', {})
applied_have_lunch_1 = Activity('(sub 2) Cook', 15, 3, ['Kitchen_Stove'], '#ffb499', {})
applied_have_lunch_2 = Activity('(sub 3) Take tableware', 0.4, 0.001, ['Cupboard', 'Kitchen_Stove'], '#ff8f66', {})
applied_have_lunch_3 = Activity('(sub 4) Eat', 30, 5, ['Dinner_Table'], '#ff6933', {})
applied_have_lunch_4 = Activity('(sub 5) Wash tableware', 5, 0.1, ['Kitchen_Stove'], '#ff4500', {})
applied_have_lunch_list = [applied_have_lunch_0, applied_have_lunch_1, applied_have_lunch_2, applied_have_lunch_3, applied_have_lunch_4]
applied_have_breakfast = MetaFundamentalActivity('Applied Have breakfast', applied_have_lunch_list, timedelta(hours = 7, minutes = 17) / timedelta(minutes = 1), 30)
applied_have_lunch = MetaFundamentalActivity('Applied Have lunch', applied_have_lunch_list, timedelta(hours = 12) / timedelta(minutes = 1), 30)
applied_have_dinner = MetaFundamentalActivity('Applied Have dinner', applied_have_lunch_list, timedelta(hours = 18, minutes = 18) / timedelta(minutes = 1), 30)

applied_take_a_bath_1 = Activity('(sub 1) Take a bath', 30, 10, ['Bathroom_Door'], '#00ffff', {sensor_model.FLOW_BATHROOM: 1})
applied_take_a_bath_2 = Activity('(sub 2) Wash clothes', 1, 0.1, ['Wash_Machine'], '#008080', {})
applied_take_a_bath_list = [applied_take_a_bath_1, applied_take_a_bath_2]
applied_take_a_bath = MetaNecessaryActivity('Applied Take a bath', applied_take_a_bath_list, 1)

applied_go_out_1 = Activity('(sub 1) Put on a coat', 2, 0.1, ['Wardrobe'], '#008000', {})
applied_go_out_2 = Activity('(sub 2) Go out', 40, 20, ['Entrance'], '#ffe4c4', {})
applied_go_out_3 = Activity('(sub 3) Put on a coat', 2, 0.1, ['Wardrobe'], '#008000', {})
applied_go_out_list = [applied_go_out_1, applied_go_out_2, applied_go_out_3]
applied_go_out = MetaRandomActivity('Applied Go out', applied_go_out_list)

applied_urination = NecessaryActivity('Urination', 5, 2, 0.5, ['Toilet_Door'], '#6a5acd', {})
applied_urination.set_repulsive_constraints([('Urination', timedelta(hours = 1))])

applied_brush_teeth = NecessaryActivity('Brush teeth', 2, 1.5, 0.5, ['Bathroom_Door'], '#2f4f4f', {sensor_model.FLOW_BATHROOM: 1})
applied_brush_teeth.set_existence_constraints([(applied_have_dinner, sleep)])

applied_activity_model = {
    'fundamental_activities': [applied_have_breakfast, applied_have_lunch, applied_have_dinner, sleep],
    'necessary_activities': [defecation, applied_urination, applied_take_a_bath, change_clothes, applied_brush_teeth],
    'random_activities': [rest, applied_go_out, clean, do_laundry, watch_TV, read, take_a_snack, use_the_phone],
    'walking_activities': [indoor_movement],
    'fundamental_activity_order': [applied_have_breakfast, applied_have_lunch, applied_have_dinner, sleep],
    'sleep': [sleep]
}


# applied activity model 2 -----------------------------------------------------------------------------------
applied_urination = NecessaryActivity('Urination', 5, 2, 0.5, ['Toilet_Door'], '#6a5acd', {})
applied_urination.set_repulsive_constraints([('Urination', timedelta(hours = 1))])
applied_defecation = NecessaryActivity('Defecation', 1, 10, 3, ['Toilet_Door'], '#4b0082', {})
applied_defecation.set_repulsive_constraints([('Defecation', timedelta(hours = 1))])

applied_brush_teeth = NecessaryActivity('Brush teeth', 2, 1.5, 0.5, ['Bathroom_Door'], '#2f4f4f', {sensor_model.FLOW_BATHROOM: 1})
applied_brush_teeth.set_existence_constraints([(have_dinner, sleep), (sleep, have_breakfast)])

applied_change_clothes = NecessaryActivity('Change clothes', 2, 2, 1, ['Wardrobe'], '#008000', {})
applied_change_clothes.set_existence_constraints([(have_dinner, sleep), (sleep, have_breakfast)])

applied_activity_model_2 = {
    'fundamental_activities': [have_breakfast, have_lunch, have_dinner, sleep],
    'necessary_activities': [applied_defecation, applied_urination, take_a_bath, applied_change_clothes, applied_brush_teeth],
    'random_activities': [rest, cooking, go_out, clean, do_laundry, watch_TV, read, take_a_snack, use_the_phone],
    'walking_activities': [indoor_movement],
    'fundamental_activity_order': [have_breakfast, have_lunch, have_dinner, sleep],
    'sleep': [sleep]
}

