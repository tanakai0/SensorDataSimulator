# functions to generate information about floor plans
# almost all functions refers to the program of SISG4HEIAlpha from, [1] C. Jiang and A. Mita, "SISG4HEI_Alpha: Alpha version of simulated indoor scenario generator for houses with elderly individuals." Journal of Building Engineering, 35-101963(2021), and [2] https://github.com/Idontwan/SISG4HEI_Alpha.

import datetime
import json
import math
import os
import random
import sys
from copy import deepcopy
from pathlib import Path
from datetime import timedelta

import numpy as np
from numpy.random import normal
from numpy.random import poisson
import matplotlib.patches as patches
import matplotlib.pyplot as plt

all_furniture = [[1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1]]

    
class FloorPlan:
    # Class of a floor plan.
    
    def __init__(self):
        """
        Initialization of FloorPlan class.
        
        Attributes
        ----------
        layout_database_folder : pathlib.Path
            Folder name to save this floor plan data.
        topology : str
            Candidate topology of this floor plan.
            
        #
        topo : list of int
            For example, topo = [0, 1, 0, 0, 1].
            This represents whether the topology is a candidate.
        House = [Bedroom, Kitchen, Living]
        Toil_Bath = [Toilet, Bathroom]
        Furnitures = [Bedroom furniture, Kitchen furniture, Living furniture]
        Doors = [[x, y, L, W, label], ], label in {Entrance, Toilet_Door, Bathroom_Door}
        Walls = [[[x,y], [x + g, y]], ]
        T_con : str
            T_con is selected from {'Bedroom', 'Kitchen', 'Livingroom'}.
        B_con : str
            B_con is selected from {'Bedroom', 'Kitchen', 'Livingroom'}.
        type : int
            type is selected from {0, 1, ..., 5}.
            4 and 5 are 2 type for Triangles topology.
        Boundary : [X_lim, Y_lim] = [[minX, maxX], [minY, maxY]]
        height_data : dict
            e.g., { '0': f_H
                    '1': [[Bx, By, BL, BW], ..., [Lx, Ly, LL, LW]]}
                    '2': [[Tx, Ty, TL, TW, f_H], [Bax, Bay, BaL, BaW, f_H], [Bedx, Bedy, BedL, BedW, BedH], ...]
                    '3': Sofa and TV consists of multiple cube, so it shows [[Sofa1x, Sofa1y, Sofa1L, Sofa1W, Sofa1H], [Sofa2x, Sofa2y, Sofa2L, Sofa2W, Sofa2H], ...]}
        """
        self.layout_database_folder = None
        self.topology = None
        self.furs = None
        
        # temp
        self.topo = None
        self.House = None
        self.Toil_Bath = None
        self.Furnitures = None
        self.Doors = None
        self.Walls = None
        self.T_con = None
        self.B_con = None
        self.type = None
        self.Boundary = None
        self.height_data = None
        
        # additional variables
        self.bed = Furniture('Bed', 'Bed')
        self.wardrobe = Furniture('Wardrobe', 'WR')
        self.nightstand = Furniture('Nightstand', 'NS')
        self.desk = Furniture('Desk', 'Desk')
        self.sofa = Furniture('Sofa', 'Sofa')
        self.TV = Furniture('TV', 'TV')
        self.dinner_table = Furniture('Dinner_Table', 'DT')
        self.kitchen_stove = Furniture('Kitchen_Stove', 'KS')
        self.cupboard = Furniture('Cupboard', 'CB')
        self.refrigerator = Furniture('Refrigerator', 'RF')
        self.wash_machine = Furniture('Wash_Machine', 'WM')
        self.trash_bin = Furniture('Trash_Bin', 'T')
        self.chair = Furniture('Chair', 'C')
        self.dinner_table_chair = Furniture('Dinner_Table_Chair', 'C')
        self.desk_chair = Furniture('Desk_Chair', 'C')
        
        self.bedroom = Zone('Bedroom', None)
        self.livingroom = Zone('Livingroom', None)
        self.kitchen = Zone('Kitchen', None)
        self.toilet = Zone('Toilet', 'WC')
        self.bathroom = Zone('Bathroom', 'Bath')
        
        self.entrance = Door('Entrance')
        self.bathroom_door = Door('Bathroom_Door')
        self.toilet_door = Door('Toilet_Door')
        
        # for class variables
        self.B_A_min, self.B_A_max = 80000, 160000
        self.K_A_min, self.K_A_max = 60000, 120000
        self.L_A_min, self.L_A_max = 120000, 240000
        self.len_step = 20
        self.To_L, self.To_W = 90, 120
        self.Ba_L, self.Ba_W = 180, 120
        self.Size = [self.To_L, self.To_W, self.Ba_L, self.Ba_W]
        self.B_A0, self.B_A1 = 80000, 160000
        self.K_A0, self.K_A1 = 60000, 120000
        self.L_A0, self.L_A1 = 120000, 240000
        self.Pass_W = 60
        self.Bed_S = [[210, 150], [210, 180]]
        self.Wardrobe_S = [[60, 60], [80, 60], [100, 60], [120, 60]]
        self.Writing_T_C_S = [[100, 100], [120, 100]]
        self.Nstand_S = [40, 40]
        self.Dinner_T_S = [[80, 80], [100, 80], [120, 80], [140, 80], [160, 80]]
        self.Chair = [40, 40]
        self.Sofa_TV_S = [[300, 90], [300, 140], [300, 190]]
        self.Kitchen_S_S = [[60, 60], [120, 60], [180, 60]]
        self.Cupboard_S = [[100, 50], [120, 50], [150, 50]]
        self.Fridge_S = [60, 60]
        self.Washer_S = [60, 60]
        self.T_Bin_S = [[30, 30], [60, 30]]
        
        # for Height_Function
        self.edge, self.g_L, self.g_W = 50, 5, 5  # size of grid, [cm]
        # IF you want to change g_L or g_W, be careful to adjust conditions in new_functions.direct_path()
        self.FloorHeight = [280, 300, 320, 340]
        self.Bed_H = [5, 30, 35, 40, 45, 50]
        self.Poss_Hs = {self.wardrobe.name: [140, 155, 170], self.nightstand.name: [45, 50, 55], self.desk.name: [60, 65, 70], self.chair.name: [40, 45],
                        self.dinner_table_chair.name: [40, 45], self.desk_chair.name: [40, 45],
                        self.sofa.name: [40, 45], self.TV.name: [40, 45, 50, 55, 60, 65, 70], self.dinner_table.name: [60, 65, 70],
                        self.kitchen_stove.name: [75, 80, 85], self.cupboard.name: [100, 105, 110, 115, 120], self.refrigerator.name: [125, 145, 165],
                        self.wash_machine.name: [85], self.trash_bin.name: [30, 45, 60]}
        self.Sofa_Plus_H = [[20, 25], [0, 20, 25, 30]]
        self.TV_Sizes = [[40, 25], [50, 30], [70, 40], [90, 55], [110, 65]]
        self.W_TV, self.W_Sofa = 10, 20
        
        # from FSave
        self.Foldername = {self.wardrobe.name: [0, 'WAR'], self.desk.name: [1, 'De'], self.sofa.name: [2, 'So'], self.dinner_table.name: [3, 'DTA'],
              self.kitchen_stove.name: [4, 'KS'], self.cupboard.name: [5, 'CB'], self.refrigerator.name: [6, 'RFA'],
              self.wash_machine.name: [7, 'WM'], self.trash_bin.name: [8, 'TB']}
        self.layout_save_folder = None
        
        # from Plot
        self.Code = {self.toilet.name: self.toilet.plot_name, self.bathroom.name: self.bathroom.plot_name, self.bed.name: self.bed.plot_name,
                     self.nightstand.name: self.nightstand.plot_name, self.wardrobe.name: self.wardrobe.plot_name,
                     self.desk.name: self.desk.plot_name, self.chair.name: self.chair.plot_name, 
                     self.dinner_table_chair.name: self.dinner_table_chair.plot_name, self.desk_chair.name:  self.desk_chair.plot_name,
                     self.sofa.name: self.sofa.plot_name,
                     self.TV.name: self.TV.plot_name, self.dinner_table.name: self.dinner_table.plot_name,
                     self.kitchen_stove.name: self.kitchen_stove.plot_name, self.cupboard.name: self.cupboard.plot_name,
                     self.refrigerator.name: self.refrigerator.plot_name, self.wash_machine.name: self.wash_machine.plot_name,
                     self.trash_bin.name: self.trash_bin.plot_name}
        
        # from Human_Path_generation
        self.local_cood = [[-2, -1], [-2, 1], [-1, -2], [-1, -1], [-1, 0], [-1, 1], [-1, 2], [0, -1],
                           [0, 1], [1, -2], [1, -1], [1, 0], [1, 1], [1, 2], [2, -1], [2, 1]]
        
    def __str__(self):
        return '<FloorPlan>'
    
    def convert_topology_into_topo(self):
        if self.topology == 'RC':
            return [1, 0, 0, 0, 0]
        elif self.topology == 'RL':
            return [0, 1, 0, 0, 0]
        elif self.topology == 'RCL':
            return [0, 0, 1, 0, 0]
        elif self.topology == 'RLC':
            return [0, 0, 0, 1, 0]
        elif self.topology == 'RCLR':
            return [0, 0, 0, 0, 1]
        elif self.topology == 'random':
            return [1, 1, 1, 1, 1]
        else:
            raise ValueError('topology is inappropriate')
            
    def convert_furniture_into_furs(self, furs):
        # inputs;
        # furs: [['Wardrobe', 'Nightstand', 'Desk and Chair', 'Random'], ['Cupboard', 'Refrigerator', 'Trash Bin', 'Wash Machine', 'Random'] ['Table and Chair', 'Sofa and TV', 'Random'], ], each value takes in {0, 1}
        # outputs;
        # n_furs: [['Wardrobe', 'Nightstand', 'Desk and Chair'], ['Table and Chair', 'Sofa and TV'], ['Cupboard', 'Refrigerator', 'Trash Bin', 'Wash Machine']], each value takes in {0, 1, 2}, 0:no, 1:may, 2:must
        n_furs = []
        for i in range(3):
            if furs[i][-1]: n_fur = [furs[i][j] + 1 for j in range(len(furs[i])-1)]
            else: n_fur = [2 * furs[i][j] for j in range(len(furs[i])-1)]
            n_furs.append(n_fur)
        return n_furs
    
    def existing_places(self):
        # return the names of places (furniture, door, toilet, bathroom) that exist in the target layoutret
        ret = []
        for room in self.Furnitures:
            for f in room:
                ret.append(f[4])
        for d in self.Doors:
            ret.append(d[4])
        return ret
    
    def generate_layout_with_necessary_furniture(self, layout_database_folder, folder_name = None, topology = 'RCLR', furniture = all_furniture, necessary_furniture = [], max_iter = 1000):
        """
        This function generates a layout data with a limitaitions of the existence of the furniture.

        Parameters
        ----------
        layout_database_folder : pathlib.Path
            Folder name to save all layout data.
        folder_name : pathlib.Path
            Folder name to save this floor plan data.
            This is a child of layout_database_folder.
        topology : str
            The type of the layout topology.
            To explain it visually, use symbols '-', '/', '\' to represent the connection of zones, as follows.
            if topology == 'RC':
                Resting zone - Cooking zone
            if topology == 'RL':
                Resting zone - Living zone
            if topology == 'RCL':
                Resting zone - Cooking zone - Living zone
            if topology == 'RLC':
                Resting zone - Living zone - cooking zone
            if topology == 'RCLR':
                    Resting zone
                      /       \      (triangle connection)
               Living zone - cooking zone
            if topology == 'random':
                Random type is selected from types written above {'RC', 'RL', 'RCL', 'RLC', 'RCLR'}.

        furniture : list of list of int
            furs = [ Bed[Wardrobe, Nightstand, Desk and Chair, Random], 
                     Kitchen[Cupboard, Refrigerator, Trash Bin, Wash Machine, Random],
                     Living[Table and Chair, Sofa and TV, Random] ]
        necessary_furniture : list of str
            Names of furniture that the layout must have.
        max_iter : int
            Nonstraint about maximum iteration.

        Returns
        -------
        layout_path : str or None
            If a layout is generated successfuly, then return its path, else return None.
            layout_path = self.layout_database_folder / self.folder_name
        """
        self.layout_database_folder = layout_database_folder
        if folder_name == None:
            folder_name = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        self.folder_name = folder_name
        
        ret = None
        for _ in range(max_iter):
            ret = self.generate_layout(self.layout_database_folder, folder_name, topology, furniture, save = False)
            if ret == True and set(necessary_furniture).issubset(set(self.existing_places())):
                return self.save_layout_data(self.layout_database_folder / self.folder_name)
        return None
        

    def generate_layout(self, layout_database_folder, folder_name = None, topology = 'RCLR', furniture = all_furniture, error_message = False, save = True):
        """
        This function generates a layout data.

        Parameters
        ----------
        layout_database_folder : pathlib.Path
            Folder name to save all layout data.
        folder_name : pathlib.Path
            Folder name to save this floor plan data.
            This is a child of layout_database_folder.
        topology : str
            The type of the layout topology.
            To explain it visually, use symbols '-', '/', '\' to represent the connection of zones, as follows.
            if topology == 'RC':
                Resting zone - Cooking zone
            if topology == 'RL':
                Resting zone - Living zone
            if topology == 'RCL':
                Resting zone - Cooking zone - Living zone
            if topology == 'RLC':
                Resting zone - Living zone - cooking zone
            if topology == 'RCLR':
                    Resting zone
                      /       \      (triangle connection)
               Living zone - cooking zone
            if topology == 'random':
                Random type is selected from types written above {'RC', 'RL', 'RCL', 'RLC', 'RCLR'}.
        error_message : boolean, default False
            Whether the error message will be print.    
        save : boolean, default True
            Whether to save the semantics file.
        furniture : list of list of int
            furs = [ Bed[Wardrobe, Nightstand, Desk and Chair, Random], 
                     Kitchen[Cupboard, Refrigerator, Trash Bin, Wash Machine, Random],
                     Living[Table and Chair, Sofa and TV, Random] ]
            If the value is 1, then the furnituire is regarded as candidate of this floor plan.

        Returns
        -------
        layout_path : str or None
            If the layout is generated successfuly and save it, then this returns its path.
            e.g., layout_path = self.layout_database_folder / self.folder_name
            elif the layout is generated successfuly and not save it, then return the flag True
            else generation is not success, then return None.

        See Also
        --------
        SISG4HEI_Alpha-main/Floorplan_Generation/FP_main/generate_house in [2]

        Notes
        -----
        The programs about 'Content.json' that records statistics of multiple layouts in the original program is delelted in this program.
        """
            
        # check the validity of inputs
        if not os.path.exists(layout_database_folder):
            os.makedirs(layout_database_folder)
        if topology not in ['RC', 'RL', 'RCL', 'RLC', 'RCLR', 'random']:
            raise ValueError('topology must be in [\'RC\', \'RL\', \'RCL\', \'RLC\', \'RCLR\', \'random\'].')

        # set instance variables
        self.layout_database_folder = layout_database_folder
        self.topology = topology
        self.furs = self.convert_furniture_into_furs(furniture)
        self.topo = self.convert_topology_into_topo()
        if folder_name == None:
            folder_name = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        self.folder_name = folder_name

        # generate a layout
        try:
            self.generate_floor_plan()
        except Exception as e:
            if error_message == True:
                print(sys.exc_info())
                print("exception args:", e.args)
                print(e)
                import traceback
                traceback.print_exc()
            return None
        else:
            if save == True:
                return self.save_layout_data(self.layout_database_folder / self.folder_name)
            else:
                return True
    
        
    def generate_floor_plan(self):
        """
        This generates information of this floor plan.

        See Also
        --------
        SISG4HEI_Alpha-main/Floorplan_Generation/Floor_Plan/Floor_Plan in [2]

        Notes
        -----
        !!! Be careful, there is no guarantee that input topology and furniture correctly reflects on the result, so that the result may be different with input's topology and may not contain some input's furniture.
        """
        
        def sizes_sample(topo, B_A_min, B_A_max, K_A_min, K_A_max, L_A_min, L_A_max, len_step, To_W = 120):
            """
            sample sizes of this floor plan

            Parameters
            ----------
            topo : 
            B_A_min : 
            B_A_max :
            K_A_min :
            K_A_max :
            L_A_min :
            L_A_max :
            len_step :
            To_W : 

            Returns
            -------
            type : 
            [B_L, B_W] :
            B_A :
            [K_L, K_W] :
            K_A :
            [L_L, L_W] :
            L_A : 
            K_cut :
            L_add : 

            See Also
            --------
            SISG4HEI_Alpha-main/Floorplan_Generation/Size_Sample/sizes_sample in [2]
            
            Notes
            -----
            Abbreviations
            B: bedroom(resting zone), K: kitchen (cooking zone), L: living zone, A: area, L: length, To_W: width of toilet

            """

            def bi_side(A_min, A_max, L_step, ratio = 2):
                # generate random height and width of zones
                L_min, L_max = math.sqrt(A_min), math.sqrt(2 * A_max)
                i_min, i_max = int(L_min / L_step) + 1, int(L_max / L_step)
                i = np.random.randint(i_min, i_max)
                j_min = int(max(i / ratio, A_min / i / L_step / L_step)) + 1
                j_max = int(min(i, A_max / i / L_step / L_step)) + 1
                j = np.random.randint(j_min, j_max)
                return i*L_step, j*L_step

            def si_side(A_min, A_max, L_step, L, kitchen = True, ratio = 2, To_W = 120):
                L_ = L - To_W if kitchen else L + To_W
                j_min = int(max(L/L_step/ratio, A_min/L/L_step)) + 1
                j_max = int(min(L/L_step*ratio, A_max/L/L_step))
                j_len = j_max - j_min if j_max > j_min else 0
                j_min_ = int(max(L_/L_step/ratio, A_min/L_/L_step)) + 1
                j_max_ = int(min(L_/L_step*ratio, A_max/L_/L_step))
                j_len_ = j_max_ - j_min_ if j_max_>j_min_ else 0
                t = np.random.rand()
                if t < 0.7*j_len_/(0.7*j_len_+j_len):
                # increse the possibility of two zone has same length
                    done = True
                    j = np.random.randint(j_min_, j_max_)
                else:
                    done = False
                    j = np.random.randint(j_min, j_max)
                return j*L_step, done

            B_L_max = int(math.sqrt(2*B_A_max) / len_step) * len_step - len_step
            B_L, B_W = bi_side(B_A_min, B_A_max, len_step)
            K_L, K_W, K_A, L_L, L_W, L_A = 0, 0, 0, 0, 0, 0

            K_cut, L_add = None, None
            B_A = B_L * B_W
            # Area of zones are correlated from the principles and intervation for designing homes for elderly individuals.
            K_A_min = max(K_A_min, 0.6 * B_A)
            K_A_max = min(K_A_max, 0.9 * B_A)
            L_A_min = max(L_A_min, 1.2 * B_A)
            L_A_max = min(L_A_max, 1.8 * B_A)

            o_inds = [0.1, 0.15, 0.2, 0.3, 0.25]  # [0: 10%, 1:15%, 2:20%, 3:30%, 4 or 5: 25%]
            inds = [o_inds[i]*topo[i] for i in range(5)]
            ind2k, ind2l = inds[0]+inds[2]+inds[4], inds[1]+inds[3]
            ind_sum = ind2k + ind2l
            t = np.random.rand()
            se_is_K = True if t < (ind2k/ind_sum) else False # index0=(0+2+4)/all

            if not se_is_K:
                L_L = B_L
                L_W, L_add = si_side(L_A_min, L_A_max, len_step, L_L, kitchen=False, To_W = To_W)
                t = np.random.rand()
                self.type = 1 if t < (inds[1]/ind2l) else 3 # index1=1/(1+3)
                if self.type == 3:
                    K_L = B_L
                    K_W, K_cut = si_side(K_A_min, K_A_max, len_step, K_L, To_W = To_W)

            else:
                K_L = B_L
                K_W, K_cut = si_side(K_A_min, K_A_max, len_step, K_L, To_W = To_W)
                t = np.random.rand()
                if t < inds[0]/ind2k: self.type = 0 # index2=0/(0+2+4)
                elif t < (inds[0]+inds[2])/ind2k: # index3=(0+2)/(0+2+4)
                    self.type = 2
                    L_L = B_L
                    L_W, L_add = si_side(L_A_min, L_A_max, len_step, L_L, kitchen=False, To_W = To_W)
                else:
                    self.type = 4
                    L_L, L_W = bi_side(L_A_min, L_A_max, len_step)
                    t = np.random.randint(0, 2)
                    L_add = True if t else False
                    if L_L > B_L_max or (L_L + To_W) >= 2 * L_W: L_add = False
                    if (L_L + To_W) * L_W > L_A_max: L_add = False

            if K_cut: K_L -= To_W
            if L_add: L_L += To_W
            K_A, L_A = K_L * K_W, L_L * L_W
            if self.type == 4:
                if np.random.randint(0, 2): self.type = 5
                temp = L_W
                L_W = L_L
                L_L = temp

            return self.type, [B_L, B_W], B_A, [K_L, K_W], K_A, [L_L, L_W], L_A, K_cut, L_add
        
        
        def TB_LD(_type, B_L, B_W, K_W, L_W, L_L, Size, K_cut):
            """
            generate the layout of toilet and bathroom
            
            Parameters
            ----------
            _type : 
            B_L : 
            B_W : 
            K_W : 
            L_W : 
            L_L : 
            Size : 
            K_cut : 

            Returns
            -------
            TB_x : float
            TB_y : float
            (TB_x, TB_y) : a bottom left point of the zone
            is_y : boolean
                (maybe) the direction of the toilet and bathroom. If this is True then the toilet and bathroom are along with y-axis.
            To_on : boolean
                (maybe) If this is True, the room which connect with (TB_x, TB_y) is Bathroom, else toilet.
            T_B_in : boolean
                (maybe) whether the toilet or bathroom is in the other zone.

            See Also
            --------
            SISG4HEI_Alpha-main/Floorplan_Generation/Tol_Ba/TB_LD in [2]
            """
            def TB_in_ou(Value):
                # Value maybe B_L, B_W, K_W, L_W, L_L
                T_B_in = False
                t = np.random.rand()
                Q = (Value-360) / 300
                if t<Q: T_B_in = True
                return T_B_in

            [To_L, To_W, Ba_L, Ba_W] = Size
            # To_L, To_W, Ba_L, Ba_W = 90, 120, 180, 120
            B_ri, B_up = B_L//2, B_W//2
            B_le, B_do = -B_ri, -B_up

            if _type == 0:
                is_y, To_on = True, True
                TB_x, TB_y = B_ri, B_up-Ba_L
                T_B_in = TB_in_ou(B_L)
                if T_B_in: TB_x -= To_W
                return TB_x, TB_y, is_y, To_on, T_B_in

            if _type == 1:
                l_1 = [[B_ri, B_up-Ba_L], [B_ri, B_up], [B_ri, B_up+L_W-Ba_L-To_L]]
                l_2 = [[B_ri, B_up], [B_ri, B_up+L_W-Ba_L-To_L]]
                if L_W<(To_L+Ba_L): l_1, l_2 = [[B_ri, B_up-Ba_L]], []
                l_3 = [[B_ri-Ba_L-To_L, B_up+L_W]]
                l_4 = [[B_ri-Ba_L-To_L, B_up+L_W]]
            elif _type == 2:
                l_1 = [[B_ri, B_up-Ba_L], [B_ri, B_up+K_W], [B_ri, B_up+K_W+L_W-Ba_L-To_L]]
                l_2 = [[B_ri, B_up+K_W-To_L], [B_ri, B_up+K_W], [B_ri, B_up+K_W+L_W-Ba_L-To_L]]
                if L_W<(To_L+Ba_L): l_1, l_2 = [[B_ri, B_up-Ba_L]], [[B_ri, B_up+K_W-To_L]]
                l_3 = [[B_ri-Ba_L-To_L, B_up+K_W+L_W]]
                l_4 = [[B_ri-Ba_L-To_L, B_up+K_W+L_W]]
            elif _type == 3:
                l_1 = [[B_ri, B_up-Ba_L], [B_ri, B_up], [B_ri, B_up+L_W-Ba_L-To_L], [B_ri, B_up+L_W-Ba_L]]
                l_2 = [[B_ri, B_up], [B_ri, B_up+L_W-Ba_L-To_L]]
                if L_W<(To_L+Ba_L): l_1, l_2 = [[B_ri, B_up-Ba_L], [B_ri, B_up+L_W-Ba_L]], []
                l_3 = []
                l_4 = []
            elif _type == 4:
                l_1 = [[B_le-Ba_W, B_up-Ba_L], [B_ri+L_L, B_up+K_W-L_W], [B_ri+L_L, B_up+K_W-Ba_L-To_L]]
                # Attention on l_1[0]
                l_2 = [[B_ri+L_L, B_up+K_W-L_W], [B_ri+L_L, B_up+K_W-Ba_L-To_L]]
                l_3 = [[B_ri, B_up+K_W], [B_ri+L_L-Ba_L-To_L, B_up+K_W], [B_ri, B_up+K_W-L_W-To_W],
                       [B_ri+L_L-Ba_L-To_L, B_up+K_W-L_W-To_W]]
                l_4 = [[B_ri-To_L, B_up+K_W], [B_ri, B_up+K_W], [B_ri+L_L-Ba_L-To_L, B_up+K_W], [B_ri, B_up+K_W-L_W-To_W],
                       [B_ri+L_L-Ba_L-To_L, B_up+K_W-L_W-To_W]]
                if L_L<(To_L+Ba_L): l_3, l_4 = [], [[[B_ri-To_L, B_up+K_W]]]
            elif _type == 5:
                l_1 = [[B_le-Ba_W, B_up-Ba_L], [B_ri+L_L, B_do], [B_ri+L_L, B_do+L_W-Ba_L-To_L]]
                l_2 = [[B_ri+L_L, B_do], [B_ri+L_L, B_do+L_W-Ba_L-To_L]]
                l_3 = [[B_ri-Ba_L, B_do-Ba_W], [B_ri, B_do-Ba_W], [B_ri+L_L-Ba_L-To_L, B_do-Ba_W],
                        [B_ri, B_do+L_W], [B_ri+L_L-Ba_L-To_L, B_do+L_W]]
                l_4 = [[B_ri, B_do-Ba_W], [B_ri+L_L-Ba_L-To_L, B_do-Ba_W], [B_ri, B_do+L_W],
                        [B_ri+L_L-Ba_L-To_L, B_do+L_W]]
                if L_L<(To_L+Ba_L): l_3, l_4 = [[B_ri-Ba_L, B_do-Ba_W]], []

            t1, t2, t3, t4 = len(l_1), len(l_2), len(l_3), len(l_4)
            TT = t1+t2+t3+t4
            t = np.random.randint(0, TT)
            if t < t1:
                is_y, To_on = True, True
                [TB_x, TB_y] = l_1[t]
            elif t < t1+t2:
                is_y, To_on = True, False
                [TB_x, TB_y] = l_2[t-t1]
            elif t < t1+t2+t3:
                is_y, To_on = False, True
                [TB_x, TB_y] = l_3[t-t1-t2]
            else:
                is_y, To_on = False, False
                [TB_x, TB_y] = l_4[t-t1-t2-t3]

            if is_y and _type > 3 and TB_x<0 and K_cut:
                T_B_in = True
                TB_x += To_W
                return TB_x, TB_y, is_y, To_on, T_B_in

            if is_y:
                Value, Cut = B_L, -To_W
                if _type > 3:
                    if TB_x > 0: Value = L_L
                    else: Cut = To_W
            else:
                Value, Cut = L_W, -To_W
                if _type==4:
                    if t==t1+t2+t3: Value = K_W
                    if TB_y<(B_up+K_W): Cut = To_W
                if _type==5:
                    if t==t1+t2: Value = B_W
                    if TB_y<B_do: Cut = To_W

            T_B_in = TB_in_ou(Value)
            if T_B_in:
                if is_y: TB_x += Cut
                else: TB_y += Cut

            return TB_x, TB_y, is_y, To_on, T_B_in
    
        
        def loc_TB(Size, TB_x, TB_y, is_y, To_on):
            """
            Parameters
            ----------
            Size : 
            TB_x : 
            TB_y : 
            is_y : 
            To_on : 
            
            Returns
            -------
            To_L, To_W, Ba_L, Ba_W, To_x, To_y, Ba_x, Ba_y
            
            See Also
            --------
            SISG4HEI_Alpha-main/Floorplan_Generation/Tol_Ba/loc_TB in [2]
            """
            [To_L, To_W, Ba_L, Ba_W] = Size
            if is_y:
                temp = [To_W, Ba_W]
                To_W, Ba_W = To_L, Ba_L
                To_L, Ba_L = temp[0], temp[1]
                if To_on:
                    Ba_x, Ba_y = TB_x, TB_y
                    To_x, To_y = TB_x, TB_y+Ba_W
                else:
                    To_x, To_y = TB_x, TB_y
                    Ba_x, Ba_y = TB_x, TB_y+To_W
            else:
                if To_on:
                    Ba_x, Ba_y = TB_x, TB_y
                    To_x, To_y = TB_x+Ba_L, TB_y
                else:
                    To_x, To_y = TB_x, TB_y
                    Ba_x, Ba_y = TB_x+To_L, TB_y
            return To_L, To_W, Ba_L, Ba_W, To_x, To_y, Ba_x, Ba_y
        
        def rect_com(_type, B_L, B_W, K_W, L_L, L_W):
            """
            This returns a laytout format for each room.
            
            Parameters
            ----------
            _type :
            B_L : 
            B_W : 
            K_W :
            L_L : 
            L_W : 
            
            Returns
            -------
            [Bedroom, Kitchen, Livingroom]
            Room = [x, y, L, W]
                (x,y): a bottom left point of the zone (the origin is the center of the bedroom)
                L: length (along with x axis)
                W: width (along with y axis)
            
            See Also
            --------
            SISG4HEI_Alpha-main/Floorplan_Generation/R_xy/rect_com in [2]
            """ 
            Bedroom = [-B_L//2, -B_W//2, B_L, B_W, self.bedroom.name]
            Kitchen, Livingroom = [0, 0, 0, 0, self.kitchen.name], [0, 0, 0, 0, self.livingroom.name]
            if _type == 0:
                Kitchen = [-B_L//2, B_W//2, B_L, K_W, self.kitchen.name]
            elif _type == 1:
                Livingroom = [-B_L//2, B_W//2, B_L, L_W, self.livingroom.name]
            elif _type == 2:
                Kitchen = [-B_L//2, B_W//2, B_L, K_W, self.kitchen.name]
                Livingroom = [-B_L//2, B_W//2+K_W, B_L, L_W, self.livingroom.name]
            elif _type == 3:
                Livingroom = [-B_L//2, B_W//2, B_L, L_W, self.livingroom.name]
                Kitchen = [-B_L//2, B_W//2+L_W, B_L, K_W, self.kitchen.name]
            elif _type == 4:
                Kitchen = [-B_L//2, B_W//2, B_L, K_W, self.kitchen.name]
                Livingroom = [B_L//2, B_W//2+K_W-L_W, L_L, L_W, self.livingroom.name]
            else:
                Kitchen = [-B_L//2, B_W//2, B_L, K_W, self.kitchen.name]
                Livingroom = [B_L//2, -B_W//2, L_L, L_W, self.livingroom.name]
            return Bedroom, Kitchen, Livingroom

        def who_con(_type, ToB_x, ToB_y, B_x, B_y, K_y, L_y):
            """
            This decides which room is adjacent to the toilet or bathroom.
            
            Parameters
            ----------
            _type : 
            ToB_x :
            ToB_y : 
            B_x :
            B_y :
            K_y : 
            L_y : 
            
            Returns
            -------
            room_name : str
            
            See Also
            --------
            SISG4HEI_Alpha-main/Floorplan_Generation/R_xy/who_con in [2]
            """ 
            if _type < 4:
                if ToB_y < -B_y:
                    return self.bedroom.name
                elif _type == 0:
                    return self.kitchen.name
                elif _type == 1:
                    return self.livingroom.name
                elif _type == 2:
                    return self.livingroom.name if ToB_y >= L_y else self.kitchen.name
                else:
                    return self.kitchen.name if ToB_y >= K_y else self.livingroom.name
            else:
                if ToB_x < -B_x:
                    return self.bedroom.name if ToB_y < -B_y else self.kitchen.name
                else:
                    return self.livingroom.name
                
        def cut_add(_type, is_y, T_B_in, T_con, B_con, K_cut, L_add, Ba_y, Kitchen, Livingroom, To_W = 120):
            """
            Parameters
            ----------
            _type : 
            is_y :
            T_B_in : 
            T_con : 
            B_con : 
            K_cut : 
            L_add : 
            Ba_y : 
            Kitchen : 
            Livingroom :
            To_W : 
            
            Returns
            ---------------------
            [Kitchen, Livingroom]
            
            See Also
            --------
            SISG4HEI_Alpha-main/Floorplan_Generation/R_xy/cut_add in [2]
            """
            [K_x, K_y, K_L, K_W, _] = Kitchen
            [L_x, L_y, L_L, L_W, _] = Livingroom
            t_l = np.random.randint(0, 2, size=2)
            if K_cut == True:
                K_L -= To_W
                if _type < 4:
                    if not T_B_in and T_con == self.kitchen.name:
                        K_x += To_W
                    else:
                        if t_l[0]: K_x += To_W
                else:
                    K_x += To_W
            if L_add == True:
                if _type < 4:
                    L_L += To_W
                    if (T_con == self.livingroom.name or B_con == self.livingroom.name) and (T_B_in or (not is_y)):
                        L_x -= To_W
                    else:
                        if t_l[1]: L_x -= To_W
                else:
                    L_W += To_W
                    if (T_con == self.livingroom.name or B_con == self.livingroom.name) and T_B_in and (not is_y):
                        if Ba_y > L_y: L_y -= To_W
                    else:
                        if t_l[1]: L_y -= To_W
            Kitchen = [K_x, K_y, K_L, K_W, self.kitchen.name]
            Livingroom = [L_x, L_y, L_L, L_W, self.livingroom.name]
            return Kitchen, Livingroom
        
        
        def Pla_Door(_type, Bedr, Kitc, Livi, Toil, Bath, DW = 60):
            """
            This determines the positions of doors and walls.
            
            Parameters
            ----------
            _type : 
            Bedr : 
            Kitc : 
            Livi : 
            Toil : 
            Bath : 
            DW : 
            
            Returns
            -------
            Doors : 
            F_Walls : 
            Be_con : 
            Ki_con : 
            Li_con : 
            
            See Also
            --------
            SISG4HEI_Alpha-main/Floorplan_Generation/R_xy/Pla_Door in [2]
            """

            def Cal_W(length, is_mi = False, D_W = DW):
                if is_mi: length += D_W
                if length < 200:
                    return (200-length)//10 + 3
                return (length-240)//30

            def ins_side(sp, ep, l0, l1, j_, i_, DW = DW, j1_ = 6):
                if j1_ > 2: j1_ = j_
                if 119 < l0 < 181 or l0 > 269:
                    Po_D_cen[j1_].append([[sp[0]+i_*DW, sp[1]+(1-i_)*DW], i_])
                    W = Cal_W(l0)
                    Weights[j1_].append(W)
                if 119 < l1 < 181 or l1 > 269:
                    Po_D_cen[j_].append([[ep[0]-i_*DW, ep[1]-(1-i_)*DW], i_])
                    W = Cal_W(l1)
                    Weights[j_].append(W)

            def Cal_j1(T, j, i_, L):
                if L == 4: return 0
                elif i_ == 1: return 5-T
                elif T < 2: return 0
                elif T == 2: return j-1
                elif T == 3: return 0 if j==2 else 2
                else: return 0

            def ins_mild(p, j01, l0, l1, i_):
                if (59<l0<121 or l0>209) and (59<l1<121 or l1>209):
                    Po_D_cen[j01].append([p, i_])
                    W0 = Cal_W(l0, is_mi=True)
                    W1 = Cal_W(l1, is_mi=True)
                    Weights[j01].append(2*min(W0, W1))

            def Cal_C_W(l, j, i):
                L = len(l)
                i_ = i//2
                if L == 2:
                    [p0, p1] = l
                    leng = p1[1-i_] - p0[1-i_]
                    ins_side(p0, p1, leng, leng, j, i_)
                    Walls.append([p0, p1])
                elif L == 3:
                    [p0, p1, p2] = l
                    leng0, leng1 = p1[1-i_]-p0[1-i_], p2[1-i_]-p1[1-i_]
                    j1 = Cal_j1(_type, j, i_, L)
                    ins_side(p0, p2, leng0, leng1, j, i_, j1_=j1)
                    if [j, j1] == [1, 0]: j01 = 3
                    elif [j, j1] == [2, 0]: j01 = 4
                    else: j01 = 5
                    ins_mild(p1, j01, leng0, leng1, i_)
                    Walls.append([p0, p2])
                elif L == 4:
                    [p0, p1, p2, p3] = l
                    leng0, leng1, leng2 = p1[1]-p0[1], p2[1]-p1[1], p3[1]-p2[1]
                    ins_side(p0, p3, leng0, leng2, j, 0, j1_=0)
                    j01 = 3 if _type == 2 else 4
                    ins_mild(p1, j01, leng0, leng1, 0)
                    ins_mild(p2, 5, leng1, leng2, 0)
                    Walls.append([p0, p3])

            def origin_bounds(Bedr, Kitc, Livi, Toil, Bath):
                # Returns
                # -------
                # x_y_lims : [[[Bx, Bx + BL], [By, By + BW]], ..., [[Bax, Bax + BaL], [Bay, Bay + BaW]]]
                # bounds : list of the 4 corners of each zone,
                #     [B[[[[x, y], [x, y+W]]], [[[x+L, y], [x+L, y+W]]], [[[x, y], [x+L, y]]], [[[x, y+W], [x+L, y+W]]]], ...., 
                #      Ba[[[[x, y], [x, y+W]]], [[[x+L, y], [x+L, y+W]]], [[[x, y], [x+L, y]]], [[[x, y+W], [x+L, y+W]]]] ]
                house = [Bedr, Kitc, Livi, Toil, Bath]
                x_y_lims, bounds = [], []
                for room in house:
                    [x, y, L, W, _] = room
                    x_y_lim = [[x, x+L], [y, y+W]]
                    bound = [[[[x, y], [x, y+W]]], [[[x+L, y], [x+L, y+W]]], [[[x, y], [x+L, y]]],
                             [[[x, y+W], [x+L, y+W]]]]
                    x_y_lims.append(x_y_lim)
                    bounds.append(bound)
                return x_y_lims, bounds


            def wall_and_Obs(x_y_lims, bounds, _type):

                def Cut(strings0, strings1, yD):
                    # strings0 has 1 string and strings1 has 2 or string0 2, string1
                    interL = [[0, 0], [0, 0]]
                    if strings1 != []:
                        [p00, p01], [p10, p11] = strings0.pop(0), strings1.pop(0)
                        if p01[yD] <= p10[yD] or p00[yD] >= p11[yD]:
                            strings0.append([p00, p01])
                            strings1.append([p10, p11])
                            [p00, p01], [p10, p11] = strings0.pop(0), strings1.pop(0)
                        if p00[yD] < p10[yD] < p01[yD] or p00[yD] < p11[yD] < p01[yD] or \
                                p10[yD] < p00[yD] < p11[yD] or p10[yD] < p01[yD] < p11[yD]:
                            if p10[yD] > p00[yD]:
                                strings0.append([p00, p10])
                                interL[0] = p10
                            if p10[yD] < p00[yD]:
                                strings1.append([p10, p00])
                                interL[0] = p00
                            if p11[yD] < p01[yD]:
                                strings0.append([p11, p01])
                                interL[1] = p11
                            if p11[yD] > p01[yD]:
                                strings1.append([p01, p11])
                                interL[1] = p01
                            if p00[yD] == p10[yD]: interL[0] = p00
                            if p01[yD] == p11[yD]: interL[1] = p01
                        elif p00[yD] != p10[yD] or p01[yD] != p11[yD]:
                            strings0.append([p00, p01])
                            strings1.append([p10, p11])
                        else: interL = [p00, p01]
                    return interL

                def Cut_TB_R(x_y_lims, bounds, is_T):
                    # cut toilet and bathroom from ? zone
                    # Parameters
                    # ------------
                    # is_T : boolean
                    #     If True then Toilet else bathroom.
                    interLs = []
                    # x0 = x, y0 = y, x1 = x + L, y1 = y + W
                    x0, y0 = bounds[4 - is_T][0][0][0][0], bounds[4 - is_T][0][0][0][1]
                    x1, y1 = bounds[4 - is_T][-1][-1][-1][0], bounds[4 - is_T][-1][-1][-1][1]
                    for I in range(3):
                        for j in range(2):
                            for k in range(2):
                                for t in range(2):
                                    if x_y_lims[4-is_T][j][k] == x_y_lims[I][j][t]:
                                        interL = Cut(bounds[I][2 * j + t], bounds[4 - is_T][2 * j + k], 1 - j)
                                        bounds[4-is_T] = [[[[x0, y0], [x0, y1]]], [[[x1, y0], [x1, y1]]], [[[x0, y0], [x1, y0]]],
                                                            [[[x0, y1], [x1, y1]]]]
                                        interLs.append(interL)
                    return interLs

                def Con_R_R(x_y_lims, bounds, R0, R1, i):
                    for j in range(2):
                        if x_y_lims[R0][i][j] == x_y_lims[R1][i][j]:
                            if bounds[R0][2*i+j] != [] and bounds[R1][2*i+j] != []:
                                Con(bounds[R0][2*i+j], bounds[R1][2*i+j], 1-i)

                def Con(strings0, strings1, yD):
                    N = len(strings0[0])
                    if N > 0:
                        [p10, p11] = strings1.pop(0)
                        string0 = strings0.pop(0)
                        if N == 2:
                            [p00, p01] = string0
                        else:
                            [p, p00, p01] = string0
                        if p01[yD] == p10[yD]:
                            strings0.append([])
                            if N == 2:
                                strings1.append([p00, p10, p11])
                            else:
                                strings1.append([p, p00, p10, p11])
                        else:
                            strings1.append([p10, p11])
                            strings0.append(string0)

                def add_obs_area(X_con, interL, W=self.Pass_W//2):
                    if interL != [[0, 0], [0, 0]]:
                        [[p00, p01], [p10, p11]] = interL
                        if p00 == p10: X_con.append([[p00-W, p01],[p10+W, p11]])
                        elif p01 == p11: X_con.append([[p00, p01-W],[p10, p11+W]])


                Be_con, Ki_con, Li_con = [], [], []
                interLs_T = Cut_TB_R(x_y_lims, bounds, 1)
                interLs_B = Cut_TB_R(x_y_lims, bounds, 0)

                if _type == 1 or _type == 3:
                    interL = Cut(bounds[0][3], bounds[2][2], 0)
                    add_obs_area(Be_con, interL)
                    add_obs_area(Li_con, interL)
                    if _type == 3:
                        interL = Cut(bounds[2][3], bounds[1][2], 0)
                        add_obs_area(Li_con, interL)
                        add_obs_area(Ki_con, interL)
                else:
                    interL = Cut(bounds[0][3], bounds[1][2], 0)
                    add_obs_area(Be_con, interL)
                    add_obs_area(Ki_con, interL)
                    if _type == 2:
                        interL = Cut(bounds[1][3], bounds[2][2], 0)
                        add_obs_area(Ki_con, interL)
                        add_obs_area(Li_con, interL)
                    elif _type > 3:
                        interL = Cut(bounds[1][1], bounds[2][0], 1)
                        add_obs_area(Ki_con, interL)
                        add_obs_area(Li_con, interL)
                        interL = Cut(bounds[0][1], bounds[2][0], 1)
                        add_obs_area(Be_con, interL)
                        add_obs_area(Li_con, interL)

                if _type == 1 or _type == 3:
                    Con_R_R(x_y_lims, bounds, 0, 2, 0)
                    if _type == 3:
                        Con_R_R(x_y_lims, bounds, 2, 1, 0)
                else:
                    Con_R_R(x_y_lims, bounds, 0, 1, 0)
                    if _type == 2:
                        Con_R_R(x_y_lims, bounds, 1, 2, 0)
                    elif _type > 3:
                        Con_R_R(x_y_lims, bounds, 1, 2, 1)
                        Con_R_R(x_y_lims, bounds, 0, 2, 1)
                return bounds[:-2], Be_con, Ki_con, Li_con

            def sample(n_list, T = None):
                if not T: T = sum(n_list)
                list = [e/T for e in n_list]
                choice = [0]
                L = len(list)
                for i in range(1, L):
                    choice.append(choice[i-1]+list[i-1])
                t = np.random.rand()
                T, N = L//2, L//2 - 1
                while N < L-1 and (choice[N]>t or choice[N+1]<=t):
                    T = max(T//2, 1)
                    if choice[N]>t: N = N - T
                    else: N = N + T
                return N
            
            # Pla_Door --------------------------------------------------
            Doors = []
            x_y_lims, bounds = origin_bounds(Bedr, Kitc, Livi, Toil, Bath)
            bounds, Be_con, Ki_con, Li_con = wall_and_Obs(x_y_lims, bounds, _type)

            Po_D_cen = [[], [], [], [], [], []]
            # in B, K, L, BaK, BaL, KaL
            Weights = [[], [], [], [], [], []]
            Walls = []
            for i in range(4):
                for j in range(3):
                    if bounds[j][i] != []:
                        for l in bounds[j][i]:
                            Cal_C_W(l, j, i)

            Choices = []
            for i in range(6):
                Choices.append(sum(Weights[i]))

            I = sample(Choices)
            J = sample(Weights[I])
            [x, y], i_ = Po_D_cen[I][J][0], Po_D_cen[I][J][1]
            if i_ == 1:
                Door = [x - DW // 2, y, DW, 0, self.entrance.name]
            else:
                Door = [x, y - DW // 2, 0, DW, self.entrance.name]
            Doors.append(Door)
            Do_obs = [[x - DW, y - DW], [x + DW, y + DW]]
            if I == 0:
                Be_con.append(Do_obs)
            elif I == 1:
                Ki_con.append(Do_obs)
            elif I == 2:
                Li_con.append(Do_obs)
            elif I == 3:
                Be_con.append(Do_obs)
                Ki_con.append(Do_obs)
            elif I == 4:
                Be_con.append(Do_obs)
                Li_con.append(Do_obs)
            elif I == 5:
                Ki_con.append(Do_obs)
                Li_con.append(Do_obs)

            F_Walls = []
            for wall in Walls:
                if wall != [[0,0],[0,0]]: F_Walls.append(wall)
            return Doors, F_Walls, Be_con, Ki_con, Li_con
        
        
        def add_TB_obs(Be_con, Ki_con, Li_con, Toil, Bath, T_con, B_con, W):
            """
            Parameters
            ----------
            Be_con : 
            Ki_con : 
            Li_con : 
            Toil : 
            Bath : 
            T_con : 
            B_con : 
            W : 
            
            Returns
            -------
            Be_con : 
            Ki_con : 
            Li_con : 
            
            See Also
            --------
            SISG4HEI_Alpha-main/Floorplan_Generation/R_xy/add_TB_obs in [2]
            """
            To_temp = [[Toil[0]-W, Toil[1]-W], [Toil[0]+Toil[2]+W, Toil[1]+Toil[3]+W]]
            Ba_temp = [[Bath[0]-W, Bath[1]-W], [Bath[0]+Bath[2]+W, Bath[1]+Bath[3]+W]]

            if T_con == self.livingroom.name:
                Li_con.append(To_temp)
            else:
                Ki_con.append(To_temp)
            if B_con == self.livingroom.name:
                Li_con.append(Ba_temp)
            else:
                Be_con.append(Ba_temp)
            return Be_con, Ki_con, Li_con

        def TB_Door(Bedr, Kitc, Livi, Toil, Bath, T_con, B_con, TB_W = 120, DW = 60):
            """
            This determines the positions of doors and walls in toilet and bathroom.
            Parameters
            ----------
            Bedr : 
            Kitc : 
            Livi : 
            Toil : 
            Bath : 
            T_con : 
            B_con : 
            TB_W : 
            DW : 
            
            Returns
            -------
            T_Door : 
            B_Door : 
            
            See Also
            --------
            SISG4HEI_Alpha-main/Floorplan_Generation/R_xy/TB_Door in [2]
            """
            Bedr_C = [Bedr[0]+Bedr[2]//2, Bedr[1]+Bedr[3]//2]
            Kitc_C = [Kitc[0]+Kitc[2]//2, Kitc[1]+Kitc[3]//2]
            Livi_C = [Livi[0]+Livi[2]//2, Livi[1]+Livi[3]//2]
            T_R_C = Livi_C if T_con == self.livingroom.name else Kitc_C
            B_R_C = Livi_C if B_con == self.livingroom.name else Bedr_C
            if Toil[2] == TB_W:
                T_Door = [Toil[0], Toil[1]+Toil[3]//2-DW//2, 0, DW, self.toilet_door.name] if T_R_C[0] < Toil[0] \
                    else [Toil[0]+Toil[2], Toil[1]+Toil[3]//2-DW//2, 0, DW, self.toilet_door.name]
                B_Door = [Bath[0], Bath[1]+Bath[3]//2-DW//2, 0, DW, self.bathroom_door.name] if B_R_C[0] < Bath[0] \
                    else [Bath[0]+Bath[2], Bath[1]+Bath[3]//2-DW//2, 0, DW, self.bathroom_door.name]
            elif Toil[3] == TB_W:
                T_Door = [Toil[0]+Toil[2]//2-DW//2, Toil[1], DW, 0, self.toilet_door.name] if T_R_C[1] < Toil[1] \
                    else [Toil[0]+Toil[2]//2-DW//2, Toil[1]+Toil[3], DW, 0, self.toilet_door.name]
                B_Door = [Bath[0]+Bath[2]//2-DW//2, Bath[1], DW, 0, self.bathroom_door.name] if B_R_C[1] < Bath[1] \
                    else [Bath[0]+Bath[2]//2-DW//2, Bath[1]+Bath[3], DW, 0, self.bathroom_door.name]
            return T_Door, B_Door
    
        
        def flip_rot(House, Toil_Bath, Furnitures, Doors, Walls):
            """
            This changes the coordinates.
            Parameters
            ----------
            House : 
            Toil_Bath : 
            Furnitures : 
            Doors : 
            Walls : 
            
            See Also
            --------
            SISG4HEI_Alpha-main/Floorplan_Generation/Floor_Plan/flip_rot in [2]
            """
            def change_R(i, j, k, room):
                [x, y, L, W, _] = room
                if i :
                    room[0] = - (x + L)
                    x = room[0]
                if j :
                    room[1] = - (y + W)
                    y = room[1]
                if k :
                    room[0], room[1] = y, x
                    room[2], room[3] = W, L

            [i, j, k] = np.random.randint(0, 2, size = 3)
            for room in House:
                change_R(i, j, k, room)
            for room in Toil_Bath:
                change_R(i, j, k, room)
            for room in Furnitures:
                for furniture in room:
                    change_R(i, j, k, furniture)
            for door in Doors:
                change_R(i, j, k, door)
            for wall in Walls:
                [[p0x, p0y], [p1x, p1y]] = wall
                if i:
                    if p0x == p1x:
                        wall[0], wall[1] = [-p0x, p0y], [-p1x, p1y]
                    else:
                        wall[0], wall[1] = [-p1x, p1y], [-p0x, p0y]
                    [[p0x, p0y], [p1x, p1y]] = wall
                if j:
                    if p0y == p1y:
                        wall[0], wall[1] = [p0x, -p0y], [p1x, -p1y]
                    else:
                        wall[0], wall[1] = [p1x, -p1y], [p0x, -p0y]
                    [[p0x, p0y], [p1x, p1y]] = wall
                if k:
                    wall[0], wall[1] = [p0y, p0x], [p1y, p1x]

        def set_boundary(House, Toil_Bath):
            """
            This changes the coordinates.
            
            Parameters
            ----------
            House = [Bedroom, Kitchen, Living] : 
            Toil_Bath = [Toilet, Bathroom] 
            
            Returns
            -------
            [X_lim, Y_lim] = [[minX, maxX], [minY, maxY]] : 
            
            See Also
            --------
            SISG4HEI_Alpha-main/Floorplan_Generation/Floor_Plan/Boundary in [2]
            """
            X_lims = [[], []]
            Y_lims = [[], []]
            for room in House:
                [x, y, L, W, _] = room
                X_lims[0].append(x), X_lims[1].append(x + L)
                Y_lims[0].append(y), Y_lims[1].append(y + W)
            for room in Toil_Bath:
                [x, y, L, W, _] = room
                X_lims[0].append(x), X_lims[1].append(x + L)
                Y_lims[0].append(y), Y_lims[1].append(y + W)
            X_lim = [min(X_lims[0]), max(X_lims[1])]
            Y_lim = [min(Y_lims[0]), max(Y_lims[1])]
            return [X_lim, Y_lim]
    
    
        # generate_floor_plan --------------------------------------------------
        _type, [B_L, B_W], B_A, [K_L, K_W], K_A, [L_L, L_W], L_A, K_cut, L_add = \
            sizes_sample(self.topo, self.B_A_min, self.B_A_max, self.K_A_min, self.K_A_max, self.L_A_min, self.L_A_max, self.len_step, self.To_W)
        if _type > 3 and L_add: L_W = L_W - self.To_W
        TB_x, TB_y, is_y, To_on, T_B_in = TB_LD(_type, B_L, B_W, K_W, L_W, L_L, self.Size, K_cut)
        To_L, To_W, Ba_L, Ba_W, To_x, To_y, Ba_x, Ba_y = \
            loc_TB(self.Size, TB_x, TB_y, is_y, To_on)
        Bedroom, Kitchen, Livingroom = rect_com(_type, B_L, B_W, K_W, L_L, L_W)
        T_con = who_con(_type, To_x, To_y, Bedroom[0], Bedroom[1], Kitchen[1], Livingroom[1])
        B_con = who_con(_type, Ba_x, Ba_y, Bedroom[0], Bedroom[1], Kitchen[1], Livingroom[1])
        Kitchen, Livingroom = cut_add(_type, is_y, T_B_in, T_con, B_con, K_cut, L_add, Ba_y, Kitchen, Livingroom, To_W = To_W)
        Toilet = [To_x, To_y, To_L, To_W, self.toilet.name]
        Bathroom = [Ba_x, Ba_y, Ba_L, Ba_W, self.bathroom.name]
        # Room = [x, y, L, W, name]
        # (x,y): a bottom left point of the zone (origin is the center of bedroom)
        # L: length(along with x axis)
        # W: width(along with y axis)
        # name in {'Bedroom', 'Kitchen', 'Livingroom'}
        Toil_Bath = [Toilet, Bathroom]
        Doors, Walls, Be_con, Ki_con, Li_con = Pla_Door(_type, Bedroom, Kitchen, Livingroom, Toilet, Bathroom)
        # Door = [x, y, L, W, label], label in {Entrance, Toilet_Door, Bathroom_Door}
        # Doors is the list of Door.
        # Wall is, e.g., [(x,y), (x + g, y)].
        # Walls is the list of Wall.
        # con, e.g., [[x - DW, y - DW], [x + DW, y + DW]].
        # Be_con and Ki_con, Li_con are the list of con. These may mean obstacle zones.
        Be_con, Ki_con, Li_con = add_TB_obs(Be_con, Ki_con, Li_con, Toilet, Bathroom, T_con, B_con, self.Pass_W)
        T_Door, B_Door = TB_Door(Bedroom, Kitchen, Livingroom, Toilet, Bathroom, T_con, B_con)
        Doors.append(T_Door)
        Doors.append(B_Door)
        
        furs = self.furs
        B_F = self.PLF_B(furs[0], Bedroom, Bathroom, Be_con, Doors, self.B_A0, self.B_A1)
        # furniture = [x, y, L, W, name], name in {'Bed', 'Desk', ..., 'Wri_T_C'}
        if _type == 0:
            House = [Bedroom, Kitchen]
            K_F = self.PLF_K(furs[1], Kitchen, Toilet, Ki_con, self.K_A0, self.K_A1)
            Furnitures = [B_F, K_F]
        elif _type == 1:
            House = [Bedroom, Livingroom]
            L_F = self.PLF_L(furs[2], Livingroom, Toilet, Bathroom, Li_con, Doors, self.L_A0, self.L_A1)
            Furnitures = [B_F, L_F]
        else:
            House = [Bedroom, Kitchen, Livingroom]
            K_F = self.PLF_K(furs[1], Kitchen, Toilet, Ki_con, self.K_A0, self.K_A1)
            L_F = self.PLF_L(furs[2], Livingroom, Toilet, Bathroom, Li_con, Doors, self.L_A0, self.L_A1)
            Furnitures = [B_F, K_F, L_F]
        flip_rot(House, Toil_Bath, Furnitures, Doors, Walls)
        Boundary = set_boundary(House, Toil_Bath)
        self.House, self.Toil_Bath, self.Furnitures, self.Doors, self.Walls, self.T_con, self.B_con, self.type \
            = House, Toil_Bath, Furnitures, Doors, Walls, T_con, B_con, _type
        self.Boundary = Boundary
        
    
    def PLF_B(self, fur_B, Bedr, Bath, B_con, Doors, B_A0, B_A1):
        """
        This function places furniture in a bedroom.
            
        Parameters
        ----------
        fur_B : 
        Bedr : 
        Bath : 
        B_con : 
        Doors : 
        B_A0 : 
        B_A1 : 
            
        Returns
        -------
        B_furniture  
        
        See Also
        --------
        SISG4HEI_Alpha-main/Floorplan_Generation/Furniture_Place/PLF_B in [2]
        """
        
        def Cal_Ai(Room, TB, R_A0, R_A1, TB1=None):
            R_A = Room[2] * Room[3]
            TB_cen = [TB[0]+TB[2]//2, TB[1]+TB[3]//2]
            if Room[0]<TB_cen[0]<(Room[0]+Room[2]) and Room[1]<TB_cen[1]<(Room[1]+Room[3]):
                T_or_B_A = TB[2]*TB[3]
                R_A -= T_or_B_A
            if TB1:
                TB1_cen = [TB1[0]+TB1[2]//2, TB1[1]+TB1[3]//2]
                if Room[0]<TB1_cen[0]<(Room[0]+Room[2]) and Room[1]<TB1_cen[1]<(Room[1]+Room[3]):
                    T_or_B_A = TB1[2] * TB1[3]
                    R_A -= T_or_B_A
            Ai = (R_A-R_A0) / (R_A1-R_A0)
            return Ai
        
        def Sample_Size_F(Sizes, A_i):
            t = np.random.rand()
            L = len(Sizes)
            p = 1 / L
            i = int(A_i/p)
            if i == 0:
                return Sizes[i] if t<0.67 else Sizes[i+1]
            elif i > L-2:
                return Sizes[L-1] if t<0.67 else Sizes[L-2]
            else:
                if t < 0.25: return Sizes[i-1]
                elif t < 0.75: return Sizes[i]
                else: return Sizes[i+1]
                
        def Sam_far_XY(N, Ai):
            if Ai <0.2: N = N*2//5
            elif Ai<0.5: N = N*3//5
            elif Ai<0.8: N = N*4//5
            i_l = np.random.randint(0, N+1, size=2)
            far_X, far_Y = i_l[0] * 10, i_l[1] * 10
            return far_X, far_Y
            
        def pla_f(exist, room, F_size, F_name, room_con, far_X=0, far_Y =0):
            
            def poss_strings(room, F_size, far_X, far_Y):
                # in most cast far_X = 0, far_Y = 0
                R_x, R_y, R_L, R_W, _ = room
                X, Y = F_size[0]//2, F_size[1]//2
                XYS = [[R_x+X+far_X, R_x+R_L-X-far_X], [R_y+Y+far_Y, R_y+R_W-Y-far_Y]]
                Ps = [[XYS[0][i],XYS[1][j]] for i in range(2) for j in range(2)]
                Strings = [[Ps[0],Ps[1]], [Ps[1], Ps[3]], [Ps[2], Ps[3]], [Ps[0], Ps[2]]]
                return Strings
            
            def complement(strings, areas):
                '''strings = [string1, string2, ...]
                   string1 = [start_point, end_point]
                   start_point on the left down of end_point
                   start_point = [x_cor, y_cor]
                   area = [bottle_left_point, top_right_point]
                   bottle_left_point = [x_cor, y_cor]'''

                for area in areas:
                    for i in range(len(strings)):
                        string = strings.pop(0)
                        if string[0][0] == string[1][0]:
                            if area[0][0] < string[0][0] < area[1][0]:
                                if area[0][1] >= string[1][1] or area[1][1] <= string[0][1]:
                                    strings.append(string)
                                else:
                                    if area[1][1] < string[1][1]:
                                        strings.append([[string[0][0], area[1][1]], string[1]])
                                    if string[0][1] < area[0][1]:
                                        strings.append([string[0], [string[1][0], area[0][1]]])
                            else:
                                strings.append(string)
                        else:
                            if area[0][1] < string[0][1] < area[1][1]:
                                if area[0][0] >= string[1][0] or area[1][0] <= string[0][0]:
                                    strings.append(string)
                                else:
                                    if area[1][0] < string[1][0]:
                                        strings.append([[area[1][0], string[0][1]], string[1]])
                                    if string[0][0] < area[0][0]:
                                        strings.append([string[0], [area[0][0], string[1][1]]])
                            else:
                                strings.append(string)
                    
            def select_f_c(strings0, strings1):
                '''strings = [string1, string2, ...]
                   string1 = [start_point, end_point]
                   start_point on the left down of end_point'''
                
                def place_f_c(strings, strings_len, T_len):
                    
                    def sample_fcstr(n_list, T=None):
                        # Decide which wall does the furniture be against
                        if not T: T = sum(n_list)
                        list = [e/T for e in n_list]
                        choice = [0]
                        L = len(list)
                        for i in range(1, L):
                            choice.append(choice[i-1]+list[i-1])
                        t = np.random.rand()
                        T, N = L//2, L//2 - 1
                        while N < L-1 and (choice[N]>t or choice[N+1]<=t):
                            T = max(T//2, 1)
                            if choice[N]>t: N = N - T
                            else: N = N + T
                        return N
                
                    N = sample_fcstr(strings_len, T=T_len)
                    string = strings[N]
                    t = np.random.rand()
                    if t < 0.4: return string[0]
                    elif t < 0.8: return string[1]
                    else: return [(string[0][0]+string[1][0])//2, (string[0][1]+string[1][1])//2]
                
                flag, Place = 0, True
                strings0_len, strings1_len = [], []
                for string in strings0:
                    _len = string[1][0]+string[1][1]-string[0][0]-string[0][1]
                    strings0_len.append(_len)
                for string in strings1:
                    _len = string[1][0] + string[1][1] - string[0][0] - string[0][1]
                    strings1_len.append(_len)
                T0, T1 = sum(strings0_len), sum(strings1_len)
                if T0+T1==0: return None, None
                else:
                    T_len0_r = T0 / (T0 + T1)
                    t = np.random.rand()
                    if t < T_len0_r:
                        f_center = place_f_c(strings0, strings0_len, T0)
                    else:
                        flag = 1
                        f_center = place_f_c(strings1, strings1_len, T1)

                return f_center, flag

            if exist == 1:
                if np.random.rand()<0.45: return None
            F_s = [F_size, [F_size[1], F_size[0]]]
            L0, W0, L1, W1 = F_s[0][0]//2, F_s[0][1]//2, F_s[1][0]//2, F_s[1][1]//2
            temp_con0, temp_con1 = [], []
            for rect in room_con:
                [x0, y0], [x1, y1] = rect[0], rect[1]
                temp_con0.append([[x0-L0-far_X, y0-W0-far_Y], [x1+L0+far_X, y1+W0+far_Y]])
                temp_con1.append([[x0-L1-far_X, y0-W1-far_Y], [x1+L1+far_X, y1+W1+far_Y]])

            strings0 = poss_strings(room, F_s[0], far_X, far_Y)
            complement(strings0, temp_con0)
            strings1 = poss_strings(room, F_s[1], far_X, far_Y)
            complement(strings1, temp_con1)
            
            F_cen, flag = select_f_c(strings0, strings1)
            # !
            if F_cen == None: return None
            F_size = F_s[flag]
            X, Y = F_size[0]//2, F_size[1]//2
            return [F_cen[0]-X, F_cen[1]-Y, F_size[0], F_size[1], F_name]
        
        def add_obstacle(Obastcles, New_Ob, PW = self.Pass_W):
            # New_Ob = [x, y, L, W], Obatacle = [[x,y],[x,y]]
            [x, y, L, W, name] = New_Ob
            Obastcle = [[x-PW, y-PW], [x+L+PW, y+W+PW]]
            Obastcles.append(Obastcle)
            
        def Pla_Nstands(NS_e, Bedr, Bed, Doors, N=self.Nstand_S):
            
            def sample(OPs_list, tr):
                OPs_sum = sum(OPs_list)
                N = len(OPs_list)
                OPsI = [sum(OPs_list[:i+1]) for i in range(N)]
                PsI = [OPsI[i]/OPs_sum for i in range(N)]
                for i in range(N):
                    if tr<PsI[i]: return i
                    
            def Cal_Dist(R, F):
                D0, D1 = F[0]-R[0], F[1]-R[1]
                D2, D3 = R[0]+R[2]-F[0]-F[2], R[1]+R[3]-F[1]-F[3]
                return [D0, D1, D2, D3]
            
            
            def Cal_D_D(P, Ds):
                [Px, Py] = P
                Dist = []
                for D in Ds:
                    [x, y, L, W, _] = D
                    CP = [x+L//2, y+W//2]
                    D = math.sqrt((Px-CP[0])**2+(Py-CP[1])**2)
                    Dist.append(D)
                return min(Dist)
        
            if min(Bed[2], Bed[3]) < 150: return None
            rts = [0.3, 0.4, 0.3] if NS_e==2 else [0, 0.4, 0.3]
            rt = np.random.rand()
            t = sample(rts, rt)
            if t == 0: return None
            [Bx, By, BL, BW, _] = Bed
            [NL, NW] = N
            N_S_centers = [[[Bx-NL//2, By+NW//2], [Bx-NL//2, By+BW-NW//2]], [[Bx+NL//2, By-NW//2], [Bx+NL-NL//2, By-NW//2]],
                             [[Bx+BL+NL//2, By+NW//2], [Bx+BL+NL//2, By+BW-NW//2]], [[Bx+NL//2, By+BW+NW//2], [Bx+NL-NL//2, By+BW+NW//2]]]
            Dist = Cal_Dist(Bedr, Bed)
            for i in range(4):
                temp0 = N_S_centers.pop(0)
                if Dist[i]>=40:
                    for i in range(2):
                        temp1 = temp0.pop(0)
                        if Cal_D_D(temp1, Doors)>=90: temp0.append(temp1)
                        else: temp0.append([])
                    N_S_centers.append(temp0)
                else: N_S_centers.append([[],[]])
            if Bed[2] == 210: N_S_cen = [N_S_centers[1], N_S_centers[3]]
            else: N_S_cen = [N_S_centers[0], N_S_centers[2]]
            if t == 1:
                N_S_Cp = []
                for i in range(2):
                    for P in N_S_cen[i]:
                        if P != []: N_S_Cp.append(P)
                if N_S_Cp == []: return None
                else:
                    N_S_C = random.choice(N_S_Cp)
                    return [[N_S_C[0]-NL//2, N_S_C[1]-NW//2, NL, NW, self.nightstand.name]]
            else:
                N_S_C2 = []
                for i in range(2):
                    if N_S_cen[0][i]!=[] and N_S_cen[1][i]!=[]:
                        N_S_C2.append([N_S_cen[0][i], N_S_cen[1][i]])
                if N_S_C2 == []: return None
                else:
                    N_S_Cs = random.choice(N_S_C2)
                    return [[N_S_Cs[0][0]-NL//2, N_S_Cs[0][1]-NW//2, NL, NW, self.nightstand.name],
                             [N_S_Cs[1][0]-NL//2, N_S_Cs[1][1]-NW//2, NL, NW, self.nightstand.name]]
        
        def Pla_Wri_T_C(T_C, DW=50, CL=40, CW=40):
            [x, y, L, W, name] = T_C
            i = np.random.randint(0, 2)
            Desks = [[x, y, L, DW, self.desk.name], [x, y+W-DW, L, DW, self.desk.name], [x, y, DW, W, self.desk.name],
                     [x+L-DW, y, DW, W, self.desk.name]]
            Chairs = [[x+L//2-CL//2, y+W-CW, CL, CW, self.desk_chair.name], [x+L//2-CL//2, y, CL, CW, self.desk_chair.name],
                      [x+L-CW, y+W//2-CL//2, CW, CL, self.desk_chair.name], [x, y+W//2-CL//2, CW, CL, self.desk_chair.name]]
            if T_C[2] > T_C[3]:
                if i: Desk, Chair = Desks[1], Chairs[1]
                else: Desk, Chair = Desks[0], Chairs[0]
            elif T_C[3] > T_C[2]:
                if i: Desk, Chair = Desks[3], Chairs[3]
                else: Desk, Chair = Desks[2], Chairs[2]
            else:
                j = np.random.randint(0, 2)
                N = 2 * i + j
                Desk, Chair = Desks[N], Chairs[N]
            return Desk, Chair

        # PLF_B-----------------------------------------------------------------------
        [WR_e, NS_e, De_e] = fur_B # e means exist
        B_furniture = []
        Ai = Cal_Ai(Bedr, Bath, B_A0, B_A1)
        Bed_Size = Sample_Size_F(self.Bed_S, Ai)
        
        Bed = pla_f(2, Bedr, Bed_Size, self.bed.name, B_con)
        add_obstacle(B_con, Bed)
        B_furniture.append(Bed)
        if NS_e > 0:
            N_stands = Pla_Nstands(NS_e, Bedr, Bed, Doors)
            if N_stands:
                for N_stand in N_stands:
                    add_obstacle(B_con, N_stand)
                    B_furniture.append(N_stand)
        if WR_e > 0:
            Wardr_Size = Sample_Size_F(self.Wardrobe_S, Ai)
            Wardr = pla_f(WR_e, Bedr, Wardr_Size, self.wardrobe.name, B_con)
            if Wardr:
                add_obstacle(B_con, Wardr)
                B_furniture.append(Wardr)
        if De_e > 0:
            Wri_T_C_Size = Sample_Size_F(self.Writing_T_C_S, Ai)
            far_X, far_Y = Sam_far_XY(5, Ai)
            Wri_T_C = pla_f(De_e, Bedr, Wri_T_C_Size, 'Wri_T_C', B_con, far_X=far_X, far_Y=far_Y)
            if Wri_T_C:
                Wri_T, Wri_C = Pla_Wri_T_C(Wri_T_C)
                B_furniture.append(Wri_T)
                B_furniture.append(Wri_C)
        return B_furniture     
                
    
    def PLF_K(self, fur_K, Kitc, Toil, K_con, K_A0, K_A1):
        """
        This function places furniture in a kitchen.
            
        Parameters
        ----------
        fur_K : 
        Kitc : 
        Toil : 
        K_con : 
        K_A0 : 
        K_A1 :  
            
        Returns
        -------
        K_furniture  
        
        See Also
        --------
        SISG4HEI_Alpha-main/Floorplan_Generation/Furniture_Place/PLF_K in [2]
        """
                
        def Cal_Ai(Room, TB, R_A0, R_A1, TB1 = None):
            R_A = Room[2] * Room[3]
            TB_cen = [TB[0]+TB[2]//2, TB[1]+TB[3]//2]
            if Room[0]<TB_cen[0]<(Room[0]+Room[2]) and Room[1]<TB_cen[1]<(Room[1]+Room[3]):
                T_or_B_A = TB[2]*TB[3]
                R_A -= T_or_B_A
            if TB1:
                TB1_cen = [TB1[0]+TB1[2]//2, TB1[1]+TB1[3]//2]
                if Room[0]<TB1_cen[0]<(Room[0]+Room[2]) and Room[1]<TB1_cen[1]<(Room[1]+Room[3]):
                    T_or_B_A = TB1[2] * TB1[3]
                    R_A -= T_or_B_A
            Ai = (R_A-R_A0) / (R_A1-R_A0)
            return Ai
        
        def Sample_Size_F(Sizes, A_i):
            t = np.random.rand()
            L = len(Sizes)
            p = 1 / L
            i = int(A_i/p)
            if i == 0:
                return Sizes[i] if t<0.67 else Sizes[i+1]
            elif i > L-2:
                return Sizes[L-1] if t<0.67 else Sizes[L-2]
            else:
                if t < 0.25: return Sizes[i-1]
                elif t < 0.75: return Sizes[i]
                else: return Sizes[i+1]
            
        def pla_f(exist, room, F_size, F_name, room_con, far_X=0, far_Y =0):
            
            def poss_strings(room, F_size, far_X, far_Y):
                # in most cast far_X = 0, far_Y = 0
                R_x, R_y, R_L, R_W, _ = room
                X, Y = F_size[0]//2, F_size[1]//2
                XYS = [[R_x+X+far_X, R_x+R_L-X-far_X], [R_y+Y+far_Y, R_y+R_W-Y-far_Y]]
                Ps = [[XYS[0][i],XYS[1][j]] for i in range(2) for j in range(2)]
                Strings = [[Ps[0],Ps[1]], [Ps[1], Ps[3]], [Ps[2], Ps[3]], [Ps[0], Ps[2]]]
                return Strings
            
            def complement(strings, areas):
                '''strings = [string1, string2, ...]
                   string1 = [start_point, end_point]
                   start_point on the left down of end_point
                   start_point = [x_cor, y_cor]
                   area = [bottle_left_point, top_right_point]
                   bottle_left_point = [x_cor, y_cor]'''

                for area in areas:
                    for i in range(len(strings)):
                        string = strings.pop(0)
                        if string[0][0] == string[1][0]:
                            if area[0][0] < string[0][0] < area[1][0]:
                                if area[0][1] >= string[1][1] or area[1][1] <= string[0][1]:
                                    strings.append(string)
                                else:
                                    if area[1][1] < string[1][1]:
                                        strings.append([[string[0][0], area[1][1]], string[1]])
                                    if string[0][1] < area[0][1]:
                                        strings.append([string[0], [string[1][0], area[0][1]]])
                            else:
                                strings.append(string)
                        else:
                            if area[0][1] < string[0][1] < area[1][1]:
                                if area[0][0] >= string[1][0] or area[1][0] <= string[0][0]:
                                    strings.append(string)
                                else:
                                    if area[1][0] < string[1][0]:
                                        strings.append([[area[1][0], string[0][1]], string[1]])
                                    if string[0][0] < area[0][0]:
                                        strings.append([string[0], [area[0][0], string[1][1]]])
                            else:
                                strings.append(string)
                    
            def select_f_c(strings0, strings1):
                '''strings = [string1, string2, ...]
                   string1 = [start_point, end_point]
                   start_point on the left down of end_point'''
                
                def place_f_c(strings, strings_len, T_len):
                    
                    def sample_fcstr(n_list, T=None):
                        # Decide which wall does the furniture be against
                        if not T: T = sum(n_list)
                        list = [e/T for e in n_list]
                        choice = [0]
                        L = len(list)
                        for i in range(1, L):
                            choice.append(choice[i-1]+list[i-1])
                        t = np.random.rand()
                        T, N = L//2, L//2 - 1
                        while N < L-1 and (choice[N]>t or choice[N+1]<=t):
                            T = max(T//2, 1)
                            if choice[N]>t: N = N - T
                            else: N = N + T
                        return N
                
                    N = sample_fcstr(strings_len, T=T_len)
                    string = strings[N]
                    t = np.random.rand()
                    if t < 0.4: return string[0]
                    elif t < 0.8: return string[1]
                    else: return [(string[0][0]+string[1][0])//2, (string[0][1]+string[1][1])//2]
                
                flag, Place = 0, True
                strings0_len, strings1_len = [], []
                for string in strings0:
                    _len = string[1][0]+string[1][1]-string[0][0]-string[0][1]
                    strings0_len.append(_len)
                for string in strings1:
                    _len = string[1][0] + string[1][1] - string[0][0] - string[0][1]
                    strings1_len.append(_len)
                T0, T1 = sum(strings0_len), sum(strings1_len)
                if T0+T1==0: return None, None
                else:
                    T_len0_r = T0 / (T0 + T1)
                    t = np.random.rand()
                    if t < T_len0_r:
                        f_center = place_f_c(strings0, strings0_len, T0)
                    else:
                        flag = 1
                        f_center = place_f_c(strings1, strings1_len, T1)

                return f_center, flag

            if exist == 1:
                if np.random.rand()<0.45: return None
            F_s = [F_size, [F_size[1], F_size[0]]]
            L0, W0, L1, W1 = F_s[0][0]//2, F_s[0][1]//2, F_s[1][0]//2, F_s[1][1]//2
            temp_con0, temp_con1 = [], []
            for rect in room_con:
                [x0, y0], [x1, y1] = rect[0], rect[1]
                temp_con0.append([[x0-L0-far_X, y0-W0-far_Y], [x1+L0+far_X, y1+W0+far_Y]])
                temp_con1.append([[x0-L1-far_X, y0-W1-far_Y], [x1+L1+far_X, y1+W1+far_Y]])

            strings0 = poss_strings(room, F_s[0], far_X, far_Y)
            complement(strings0, temp_con0)
            strings1 = poss_strings(room, F_s[1], far_X, far_Y)
            complement(strings1, temp_con1)
            
            F_cen, flag = select_f_c(strings0, strings1)
            # !
            if F_cen == None: return None
            F_size = F_s[flag]
            X, Y = F_size[0]//2, F_size[1]//2
            return [F_cen[0]-X, F_cen[1]-Y, F_size[0], F_size[1], F_name]
        
        def add_obstacle(Obastcles, New_Ob, PW = self.Pass_W):
            # New_Ob = [x, y, L, W], Obatacle = [[x,y],[x,y]]
            [x, y, L, W, name] = New_Ob
            Obastcle = [[x-PW, y-PW], [x+L+PW, y+W+PW]]
            Obastcles.append(Obastcle)
        
        # PLF_K--------------------------------------------------------
        [CB_e, Re_e, TB_e, WM_e] = fur_K
        K_furniture = []
        Ai = Cal_Ai(Kitc, Toil, K_A0, K_A1)
        Kitc_Stove_Size = Sample_Size_F(self.Kitchen_S_S, Ai)
        Kitc_Stove = pla_f(2, Kitc, Kitc_Stove_Size, self.kitchen_stove.name, K_con)
        add_obstacle(K_con, Kitc_Stove)
        K_furniture.append(Kitc_Stove)
        if CB_e > 0:
            Cupboard_Size = Sample_Size_F(self.Cupboard_S, Ai)
            Cupboard = pla_f(CB_e, Kitc, Cupboard_Size, self.cupboard.name, K_con)
            if Cupboard:
                add_obstacle(K_con, Cupboard)
                K_furniture.append(Cupboard)
        if Re_e > 0:
            Fridge = pla_f(Re_e, Kitc, self.Fridge_S, self.refrigerator.name, K_con)
            if Fridge:
                add_obstacle(K_con, Fridge)
                K_furniture.append(Fridge)
        if WM_e > 0:
            Washer = pla_f(WM_e, Kitc, self.Washer_S, self.wash_machine.name, K_con)
            if Washer:
                add_obstacle(K_con, Washer)
                K_furniture.append(Washer)
        if TB_e > 0:
            T_Bin_Size = Sample_Size_F(self.T_Bin_S, Ai)
            T_Bin = pla_f(TB_e, Kitc, T_Bin_Size, self.trash_bin.name, K_con)
            if T_Bin: K_furniture.append(T_Bin)
        return K_furniture
    
    
    def PLF_L(self, fur_L, Livi, Toil, Bath, L_con, Doors, L_A0, L_A1):
        """
        This function places furniture in a kitchen.
            
        Parameters
        ----------
        fur_L : 
        Livi :
        Toil : 
        Bath : 
        L_con : 
        Doors : 
        L_A0 : 
        L_A1 :  
            
        Returns
        -------
        L_furniture  
        
        See Also
        --------
        SISG4HEI_Alpha-main/Floorplan_Generation/Furniture_Place/PLF_L in [2]
        """
        
        def Cal_Ai(Room, TB, R_A0, R_A1, TB1=None):
            R_A = Room[2] * Room[3]
            TB_cen = [TB[0]+TB[2]//2, TB[1]+TB[3]//2]
            if Room[0]<TB_cen[0]<(Room[0]+Room[2]) and Room[1]<TB_cen[1]<(Room[1]+Room[3]):
                T_or_B_A = TB[2]*TB[3]
                R_A -= T_or_B_A
            if TB1:
                TB1_cen = [TB1[0]+TB1[2]//2, TB1[1]+TB1[3]//2]
                if Room[0]<TB1_cen[0]<(Room[0]+Room[2]) and Room[1]<TB1_cen[1]<(Room[1]+Room[3]):
                    T_or_B_A = TB1[2] * TB1[3]
                    R_A -= T_or_B_A
            Ai = (R_A-R_A0) / (R_A1-R_A0)
            return Ai
        
        def Sample_Size_F(Sizes, A_i):
            t = np.random.rand()
            L = len(Sizes)
            p = 1 / L
            i = int(A_i/p)
            if i == 0:
                return Sizes[i] if t<0.67 else Sizes[i+1]
            elif i > L-2:
                return Sizes[L-1] if t<0.67 else Sizes[L-2]
            else:
                if t < 0.25: return Sizes[i-1]
                elif t < 0.75: return Sizes[i]
                else: return Sizes[i+1]
            
        def pla_f(exist, room, F_size, F_name, room_con, far_X=0, far_Y =0):
            
            def poss_strings(room, F_size, far_X, far_Y):
                # in most cast far_X = 0, far_Y = 0
                R_x, R_y, R_L, R_W, _ = room
                X, Y = F_size[0]//2, F_size[1]//2
                XYS = [[R_x+X+far_X, R_x+R_L-X-far_X], [R_y+Y+far_Y, R_y+R_W-Y-far_Y]]
                Ps = [[XYS[0][i],XYS[1][j]] for i in range(2) for j in range(2)]
                Strings = [[Ps[0],Ps[1]], [Ps[1], Ps[3]], [Ps[2], Ps[3]], [Ps[0], Ps[2]]]
                return Strings
            
            def complement(strings, areas):
                '''strings = [string1, string2, ...]
                   string1 = [start_point, end_point]
                   start_point on the left down of end_point
                   start_point = [x_cor, y_cor]
                   area = [bottle_left_point, top_right_point]
                   bottle_left_point = [x_cor, y_cor]'''

                for area in areas:
                    for i in range(len(strings)):
                        string = strings.pop(0)
                        if string[0][0] == string[1][0]:
                            if area[0][0] < string[0][0] < area[1][0]:
                                if area[0][1] >= string[1][1] or area[1][1] <= string[0][1]:
                                    strings.append(string)
                                else:
                                    if area[1][1] < string[1][1]:
                                        strings.append([[string[0][0], area[1][1]], string[1]])
                                    if string[0][1] < area[0][1]:
                                        strings.append([string[0], [string[1][0], area[0][1]]])
                            else:
                                strings.append(string)
                        else:
                            if area[0][1] < string[0][1] < area[1][1]:
                                if area[0][0] >= string[1][0] or area[1][0] <= string[0][0]:
                                    strings.append(string)
                                else:
                                    if area[1][0] < string[1][0]:
                                        strings.append([[area[1][0], string[0][1]], string[1]])
                                    if string[0][0] < area[0][0]:
                                        strings.append([string[0], [area[0][0], string[1][1]]])
                            else:
                                strings.append(string)
                    
            def select_f_c(strings0, strings1):
                '''strings = [string1, string2, ...]
                   string1 = [start_point, end_point]
                   start_point on the left down of end_point'''
                
                def place_f_c(strings, strings_len, T_len):
                    
                    def sample_fcstr(n_list, T=None):
                        # Decide which wall does the furniture be against
                        if not T: T = sum(n_list)
                        list = [e/T for e in n_list]
                        choice = [0]
                        L = len(list)
                        for i in range(1, L):
                            choice.append(choice[i-1]+list[i-1])
                        t = np.random.rand()
                        T, N = L//2, L//2 - 1
                        while N < L-1 and (choice[N]>t or choice[N+1]<=t):
                            T = max(T//2, 1)
                            if choice[N]>t: N = N - T
                            else: N = N + T
                        return N
                
                    N = sample_fcstr(strings_len, T=T_len)
                    string = strings[N]
                    t = np.random.rand()
                    if t < 0.4: return string[0]
                    elif t < 0.8: return string[1]
                    else: return [(string[0][0]+string[1][0])//2, (string[0][1]+string[1][1])//2]
                
                flag, Place = 0, True
                strings0_len, strings1_len = [], []
                for string in strings0:
                    _len = string[1][0]+string[1][1]-string[0][0]-string[0][1]
                    strings0_len.append(_len)
                for string in strings1:
                    _len = string[1][0] + string[1][1] - string[0][0] - string[0][1]
                    strings1_len.append(_len)
                T0, T1 = sum(strings0_len), sum(strings1_len)
                if T0+T1==0: return None, None
                else:
                    T_len0_r = T0 / (T0 + T1)
                    t = np.random.rand()
                    if t < T_len0_r:
                        f_center = place_f_c(strings0, strings0_len, T0)
                    else:
                        flag = 1
                        f_center = place_f_c(strings1, strings1_len, T1)

                return f_center, flag

            if exist == 1:
                if np.random.rand()<0.45: return None
            F_s = [F_size, [F_size[1], F_size[0]]]
            L0, W0, L1, W1 = F_s[0][0]//2, F_s[0][1]//2, F_s[1][0]//2, F_s[1][1]//2
            temp_con0, temp_con1 = [], []
            for rect in room_con:
                [x0, y0], [x1, y1] = rect[0], rect[1]
                temp_con0.append([[x0-L0-far_X, y0-W0-far_Y], [x1+L0+far_X, y1+W0+far_Y]])
                temp_con1.append([[x0-L1-far_X, y0-W1-far_Y], [x1+L1+far_X, y1+W1+far_Y]])

            strings0 = poss_strings(room, F_s[0], far_X, far_Y)
            complement(strings0, temp_con0)
            strings1 = poss_strings(room, F_s[1], far_X, far_Y)
            complement(strings1, temp_con1)
            
            F_cen, flag = select_f_c(strings0, strings1)
            # !
            if F_cen == None: return None
            F_size = F_s[flag]
            X, Y = F_size[0]//2, F_size[1]//2
            return [F_cen[0]-X, F_cen[1]-Y, F_size[0], F_size[1], F_name]
                
        def Sam_far_XY(N, Ai):
            if Ai <0.2: N = N*2//5
            elif Ai<0.5: N = N*3//5
            elif Ai<0.8: N = N*4//5
            i_l = np.random.randint(0, N+1, size=2)
            far_X, far_Y = i_l[0] * 10, i_l[1] * 10
            return far_X, far_Y


        def add_obstacle(Obastcles, New_Ob, PW=self.Pass_W):
            # New_Ob = [x, y, L, W], Obatacle = [[x,y],[x,y]]
            [x, y, L, W, name] = New_Ob
            Obastcle = [[x-PW, y-PW], [x+L+PW, y+W+PW]]
            Obastcles.append(Obastcle)
            
        def Pla_Din_Cs(Dinner_T, Livi, Doors, far=10, CL=40, CW=40):
            
            def sam_cha_num(L):
                t = np.random.rand()
                if L == self.Dinner_T_S[0][0]:
                    return 1 if t < 0.67 else 2
                elif L == self.Dinner_T_S[1][0]:
                    if t<0.25: return 1
                    else: return 2 if t<0.75 else 3
                elif L == self.Dinner_T_S[2][0]:
                    if t<0.15: return 1
                    elif t<0.5: return 2
                    else: return 3 if t< 0.85 else 4
                elif L == self.Dinner_T_S[3][0]:
                    if t<0.25: return 2
                    else: return 3 if t<0.75 else 4
                else:
                    return 3 if t < 0.33 else 4
                
            def Cal_Dist(R, F):
                D0, D1 = F[0]-R[0], F[1]-R[1]
                D2, D3 = R[0]+R[2]-F[0]-F[2], R[1]+R[3]-F[1]-F[3]
                return [D0, D1, D2, D3]
            
            def Cal_D_D(P, Ds):
                [Px, Py] = P
                Dist = []
                for D in Ds:
                    [x, y, L, W, _] = D
                    CP = [x+L//2, y+W//2]
                    D = math.sqrt((Px-CP[0])**2+(Py-CP[1])**2)
                    Dist.append(D)
                return min(Dist)
    
            [x, y, L, W, _] = Dinner_T
            T_L = max(Dinner_T[2], Dinner_T[3])
            n = sam_cha_num(T_L)
            Chairs_cen_P = [[x-far-CL//2, y+W//2], [x+L//2, y-far-CW//2], [x+L+far+CL//2, y+W//2], [x+L//2, y+W+far+CW//2]]
            Dist = Cal_Dist(Livi, Dinner_T)
            for i in range(4):
                center = Chairs_cen_P.pop(0)
                if Dist[i] >= 50:
                    Dist_D = Cal_D_D(center, Doors)
                    if Dist_D >= 90: Chairs_cen_P.append(center)
            if n > len(Chairs_cen_P): n = len(Chairs_cen_P)
            centers = random.sample(Chairs_cen_P, n)
            Chairs = []
            for center in centers:
                Chairs.append([center[0]-CL//2, center[1]-CW//2, CL, CW, self.dinner_table_chair.name])
            return Chairs
        
        def Pla_Sofa_TV(L_center, S_TV, S_W=80):
            
            def Sam_TV_size(L):
                P_Ws = [40, 60]
                P_Ls = [60, 80, 100, 120]
                t = np.random.rand()
                i = np.random.randint(0, 2)
                if L == self.Sofa_TV_S[0][1]:
                    TV_W = P_Ws[0]
                    TV_L = P_Ls[0+i]
                elif L == self.Sofa_TV_S[1][1]:
                    TV_W = P_Ws[0] if t < 0.67 else P_Ws[1]
                    TV_L = P_Ls[1+i]
                else:
                    TV_W = P_Ws[1] if t < 0.67 else P_Ws[0]
                    TV_L = P_Ls[2+i]
                return TV_L, TV_W

            [x, y, L, W, name] = S_TV
            TV_L, TV_W = Sam_TV_size(min(L, W))
            [xl, yl] = L_center
            if L > W:
                if abs(x-xl) > abs(x+L-xl):
                    Sofa, TV = [x+L-S_W, y, S_W, W, self.sofa.name], [x, y+W//2-TV_L//2, TV_W, TV_L, self.TV.name]
                else:
                    Sofa, TV = [x, y, S_W, W, self.sofa.name], [x+L-TV_W, y+W//2-TV_L//2, TV_W, TV_L, self.TV.name]
            else:
                if abs(y-yl) > abs(y+W-yl):
                    Sofa, TV = [x, y+W-S_W, L, S_W, self.sofa.name], [x+L//2-TV_L//2, y, TV_L, TV_W, self.TV.name]
                else:
                    Sofa, TV = [x, y, L, S_W, self.sofa.name], [x+L//2-TV_L//2, y+W-TV_W, TV_L, TV_W, self.TV.name]
            return Sofa, TV
             
        # PLF_L--------------------------------------------------------
        [Ta_e, ST_e] = fur_L
        [x, y, L, W, _] = Livi
        Livi_c = [x+L//2, y+W//2]
        L_furniture = []
        Ai = Cal_Ai(Livi, Bath, L_A0, L_A1, TB1=Toil)
        if Ta_e>ST_e: I = 1
        elif Ta_e<ST_e: I = 0
        else: I = np.random.randint(0, 2)
        if I:
            Dinner_T_Size = Sample_Size_F(self.Dinner_T_S, Ai)
            far_X, far_Y = Sam_far_XY(10, Ai)
            Dinner_T = pla_f(2, Livi, Dinner_T_Size, self.dinner_table.name, L_con, far_X=far_X, far_Y=far_Y)
            add_obstacle(L_con, Dinner_T)
            L_furniture.append(Dinner_T)
            Dinner_Cs = Pla_Din_Cs(Dinner_T, Livi, Doors)
            for Dinner_C in Dinner_Cs:
                add_obstacle(L_con, Dinner_C)
                L_furniture.append(Dinner_C)
            if ST_e > 0:
                Sofa_TV_Size = Sample_Size_F(self.Sofa_TV_S, Ai)
                Sofa_TV = pla_f(ST_e, Livi, Sofa_TV_Size, 'Sofa_TV', L_con)
                if Sofa_TV:
                    Sofa, TV = Pla_Sofa_TV(Livi_c, Sofa_TV)
                    L_furniture.append(Sofa)
                    L_furniture.append(TV)
        else:
            Sofa_TV_Size = Sample_Size_F(self.Sofa_TV_S, Ai)
            Sofa_TV = pla_f(2, Livi, Sofa_TV_Size, 'Sofa_TV', L_con)
            add_obstacle(L_con, Sofa_TV)
            Sofa, TV = Pla_Sofa_TV(Livi_c, Sofa_TV)
            L_furniture.append(Sofa)
            L_furniture.append(TV)
            if Ta_e > 0:
                Dinner_T_Size = Sample_Size_F(self.Dinner_T_S, Ai)
                far_X, far_Y = Sam_far_XY(5, Ai)
                Dinner_T = pla_f(Ta_e, Livi, Dinner_T_Size, self.dinner_table.name, L_con, far_X=far_X, far_Y=far_Y)
                if Dinner_T:
                    Dinner_Cs = Pla_Din_Cs(Dinner_T, Livi, Doors)
                    if Dinner_Cs:
                        add_obstacle(L_con, Dinner_T)
                        L_furniture.append(Dinner_T)
                        for Dinner_C in Dinner_Cs:
                            add_obstacle(L_con, Dinner_C)
                            L_furniture.append(Dinner_C)
        return L_furniture

    
    class MyEncoder(json.JSONEncoder):
        """
        See Also
        --------
        SISG4HEI_Alpha-main/Floorplan_Generation/FSave/MyEncoder in [2]
        """
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return super(MyEncoder, self).default(obj)
                
    
    def save_layout_data(self, save_folder, filename = 'Semantic.json'):
        """
        This function saves the layout data as a json file.
        
        Parameters
        ----------
        save_folder : pathlib.Path
            Folder name to save this floor plan data.
        filename : str
            Filename to save.
            
        Returns
        -------
        path : pathlib.Path
            Same as ''save_folder''.
            
        See Also
        --------
        SISG4HEI_Alpha-main/Floorplan_Generation/FSave/data_save in [2]
        """
                
        def creat_folder(path):
            if not os.path.exists(path): os.makedirs(path)
        
        topo_type, House, Toil_Bath, Furnitures, Doors, Walls, T_con, B_con, Boundary\
            = self.type, self.House, self.Toil_Bath, self.Furnitures, self.Doors, self.Walls, self.T_con, self.B_con, self.Boundary
        
        if topo_type > 4: topo_type = 4
        Json_File = {}
        for i in range(len(House)):
            x, y, L, W, N = House[i][0], House[i][1], House[i][2], House[i][3], House[i][4]  # N means name
            # Area = (L*W)//100
            Json_File[N] = {}
            Json_File[N]['Size'] = [x, y, L, W]
            Json_File[N]['Furnitures'] = {}
            if T_con == N:
                x, y, L, W, N0 = Toil_Bath[0][0], Toil_Bath[0][1], Toil_Bath[0][2], \
                                Toil_Bath[0][3], Toil_Bath[0][4]
                Json_File[N][N0] = {'Size': [x, y, L, W]}
            if B_con == N:
                x, y, L, W, N0 = Toil_Bath[1][0], Toil_Bath[1][1], Toil_Bath[1][2], \
                                Toil_Bath[1][3], Toil_Bath[1][4]
                Json_File[N][N0] = {'Size': [x, y, L, W]}
            for furniture in Furnitures[i]:
                x, y, L, W, N1 = furniture[0], furniture[1], furniture[2], \
                                furniture[3], furniture[4]
                if N1 not in Json_File[N]['Furnitures']:
                    Json_File[N]['Furnitures'][N1] = {'Size': [[x, y, L, W]]}
                else: Json_File[N]['Furnitures'][N1]['Size'].append([x, y, L, W])
        Json_File['Doors'] = {}
        for door in Doors:
            x, y, L, W, N = door[0], door[1], door[2], door[3], door[4]
            Json_File['Doors'][N] = {'Size': [x, y, L, W]}
        Json_File['Walls'] = Walls
        Json_File['Boundary'] = Boundary
        
        Json_File['T_con'] =  T_con
        Json_File['B_con'] =  B_con
        [lx0, lx1], [ly0, ly1] = Boundary[0], Boundary[1]
        Ratio = (lx1-lx0)/(ly1-ly0)
        if Ratio < 1: Ratio = 1 / Ratio
        Ratio = int(100 * Ratio)
        Json_File['Ratio'] = Ratio
        Json_File['topo_type'] = topo_type
        Js_Obj = json.dumps(Json_File, cls = self.MyEncoder)

        self.layout_save_folder = save_folder
        if not os.path.exists(self.layout_save_folder):
            os.makedirs(self.layout_save_folder)
        with open(self.layout_save_folder / filename, 'w') as f:
            f.write(Js_Obj)
        return self.layout_save_folder
    
    def load_layout(self, folder_path, file_name = 'Semantic.json'):
        """
        This function loads the saved layout data.
        
        Parameters
        ----------
        folder_path : pathlib.Path
            An absolute path to the folder that contains the target file.
        file_name : str
            Filename to load.
            
        See Also
        --------
        SISG4HEI_Alpha-main/Load/load_se_map in [2]
        SISG4HEI_Alpha-main/Load/dic2house in [2]
        """
        data = None
        with open(folder_path / file_name, 'r') as f:
            data = json.load(f)

        hou_dict = data
        rooms, T_Bs, furnitures, doors = [], [], [], []
        def expand_room(hou_dict, key):
            if key in hou_dict:
                list_temp = deepcopy(hou_dict[key]['Size'])
                list_temp.append(key)
                rooms.append(list_temp)
                furnitures.append([])
                for f in hou_dict[key]['Furnitures']:
                    for i in range(len(hou_dict[key]['Furnitures'][f]['Size'])):
                        list_temp = deepcopy(hou_dict[key]['Furnitures'][f]['Size'][i])
                        list_temp.append(f)
                        furnitures[-1].append(list_temp)
                for T_B in [self.toilet.name, self.bathroom.name]:
                    if T_B in hou_dict[key]:
                        list_temp = deepcopy(hou_dict[key][T_B]['Size'])
                        list_temp.append(T_B)
                        T_Bs.append(list_temp)

        expand_room(hou_dict, self.bedroom.name)
        expand_room(hou_dict, self.kitchen.name)
        expand_room(hou_dict, self.livingroom.name)
        for d in hou_dict['Doors']:
            list_temp = deepcopy(hou_dict['Doors'][d]['Size'])
            list_temp.append(d)
            doors.append(list_temp)
        walls = hou_dict['Walls']
        lims = hou_dict['Boundary']
        
        topo_type = hou_dict['topo_type']
        T_con, B_con = hou_dict['T_con'], hou_dict['B_con']
        
        # T_con, B_con = hou_dict['T_con'][0], hou_dict['B_con'][0]
        # House_name = [self.bedroom.name, self.kitchen.name, self.livingroom.name]
        # for i in range(3):
        #     if T_con == House_name[i][0]: T_con = House_name[i]
        #     if B_con == House_name[i][0]: B_con = House_name[i]
            
        self.House, self.Toil_Bath, self.Furnitures, self.Doors, self.Walls, self.Boundary \
            = rooms, T_Bs, furnitures, doors, walls, lims
        self.type, self.T_con, self.B_con = int(topo_type), T_con, B_con
        self.layout_save_folder = Path(folder_path)
        
    def load_height(self, folder_path, file_name = 'Height_Function.json'):
        """
        This function loads the saved height data.
        
        Parameters
        ----------
        folder_path : pathlib.Path
            absolute path of the folder that contains the target height data file
        file_name : str
            filename to load
        """
        data = None
        with open(folder_path / file_name, 'r') as f:
            data = f.read()
        self.height_data = json.loads(data)
        

    def sample_height_data(self, abs_path):
        """
        This function generates the height data and save it.
        
        Parameters
        ----------
        abs_path : Pathlib.Path
            An absolute path to the folder that contains the target Semantic.json file.
            
        See Also
        --------
        SISG4HEI_Alpha-main/Floorplan_Generation/FSave/Height_SampleSave in [2]
        """
        
        def floorH(House, _type):
            """
            Parameters
            ----------
            House = [Bedroom, Kitchen, Living] : 
            type : int
                in {0, 1, ..., 4}
            
            Returns
            -------
            height : int
            
            See Also
            --------
            SISG4HEI_Alpha-main/Floorplan_Generation/FSave/floorH in [2]
            """
            Ps = [[0.5,0.8,1,1],[0.2,0.5,0.8,1],[0,0.2,0.5,1]]
            B_A = House[0][2] * House[0][3]
            B_A_i = (B_A-self.B_A0)/(self.B_A1-self.B_A0)
            if _type == 0:
                K_A = House[1][2] * House[1][3]
                K_A_i = (K_A-self.K_A0)/(self.K_A1-self.K_A0)
                index = (B_A_i + K_A_i) / 2
            elif _type == 1:
                L_A = House[1][2] * House[1][3]
                L_A_i = (L_A-self.L_A0)/(self.L_A1-self.L_A0)
                index = (B_A_i + L_A_i) / 2
            else:
                K_A = House[1][2] * House[1][3]
                K_A_i = (K_A-self.K_A0)/(self.K_A1-self.K_A0)
                L_A = House[2][2] * House[2][3]
                L_A_i = (L_A-self.L_A0)/(self.L_A1-self.L_A0)
                index = (B_A_i + K_A_i + L_A_i) / 3
            if index < 0.33: P = Ps[0]
            elif index < 0.66: P = Ps[1]
            else: P = Ps[2]
            rf = random.random()
            for i in range(4):
                if P[i] > rf: return self.FloorHeight[i]
            
        def furnitureH(f_H, House, T_B, Furnitures):
            """
            Parameters
            ----------
            f_H : int
            House = [Bedroom, Kitchen, Living] : 
            T_B = [Toilet, Bathroom] : 
            Furnitures = [Bedroom furniture, Kitchen furniture, Living furniture] : 
            
            Returns
            -------
            Heights(dict): {'0': f_H,
                            '1': [[Bx, By, BL, BW], ..., [Lx, Ly, LL, LW]]},
                            '2': [[Tx, Ty, TL, TW, f_H], [Bax, Bay, BaL, BaW, f_H], [Bedx, Bedy, BedL, BedW, BedH], ...],
                            '3': Sofa and TV consists of multiple cube, so it shows [[Sofa1x, Sofa1y, Sofa1L, Sofa1W, Sofa1H], [Sofa2x, Sofa2y, Sofa2L, Sofa2W, Sofa2H], ...]}
                            
            See Also
            --------
            SISG4HEI_Alpha-main/Floorplan_Generation/FSave/furnitureH in [2]
            """ 
            def fur_height(na, loca, arrs, has_nist):
                """
                Parameters
                ----------
                na :
                    name of the furniture
                loca :
                    location[x, y, L ,W]
                arrs : list
                has_nist : boolean
                    If the layout contains nightstand, then this is True.
                    
                Returns
                -------
                f_height : 
                """  
                if na == self.bed.name:
                    if has_nist:
                        f_height = random.choice(self.Bed_H[1:])
                    else:
                        rj = random.random()
                        if rj < 0.35: f_height = self.Bed_H[0]
                        else: f_height = random.choice(self.Bed_H)
                else:
                    f_height = random.choice(self.Poss_Hs[na])
                arr = loca + [f_height]
                arrs.append(arr)
                return f_height
            
            def Sofa_TV_height(S_T, arrs):
                """
                Parameters
                ----------
                S_T : dict
                    If the layout has sofa and TV, S_T = [[Sofax], Sofay, SofaL, SofaW, SofaH],[TVx, TVy, TVL, TVW, TVH]]
                arrs : list
                
                See Also
                --------
                SISG4HEI_Alpha-main/Floorplan_Generation/FSave/Sofa_TV_height in [2]
                """
                
                def Sample_TV_Size(L, W):
                    LL = max(L, W)
                    if LL == 60: return random.choice(self.TV_Sizes[:2])
                    if LL == 80: return random.choice(self.TV_Sizes[:3])
                    if LL == 100: return random.choice(self.TV_Sizes[1:4])
                    if LL == 120: return random.choice(self.TV_Sizes[2:])

                W_Sofa = self.W_Sofa
                W_TV = self.W_TV
                Sofa = S_T[self.sofa.name]
                TV = S_T[self.TV.name]
                [x0, y0, L0, W0, HSo] = Sofa
                [x1, y1, L1, W1, HT] = TV
                [L_TV, H_TV] = Sample_TV_Size(L1, W1)
                Sofa_H1 = HSo + random.choice(self.Sofa_Plus_H[0])
                Sofa_H2 = Sofa_H1 + random.choice(self.Sofa_Plus_H[1])
                TV_H3 = HT + H_TV
                S_c, TV_c = [x0+L0//2, y0+W0//2], [x1+L1//2, y1+W1//2]
                if S_c[0] == TV_c[0]:
                    S_3_w_0 = [x0, y0, W_Sofa, W0, Sofa_H1]
                    S_3_w_1 = [x0+L0-W_Sofa, y0, W_Sofa, W0, Sofa_H1]
                    if S_c[1] > TV_c[1]:
                        TV_3 = [x1+L1//2-L_TV//2, y1, L_TV, W_TV, TV_H3]
                        S_3_r = [x0, y0+W0-W_Sofa, L0, W_Sofa, Sofa_H2]
                    elif S_c[1] < TV_c[1]:
                        TV_3 = [x1+L1//2-L_TV//2, y1+W1-W_TV, L_TV, W_TV, TV_H3]
                        S_3_r = [x0, y0, L0, W_Sofa, Sofa_H2]
                elif S_c[1] == TV_c[1]:
                    S_3_w_0 = [x0, y0, L0, W_Sofa, Sofa_H1]
                    S_3_w_1 = [x0, y0+W0-W_Sofa, L0, W_Sofa, Sofa_H1]
                    if S_c[0] > TV_c[0]:
                        TV_3 = [x1, y1+W1//2-L_TV//2, W_TV, L_TV, TV_H3]
                        S_3_r = [x0+L0-W_Sofa, y0, W_Sofa, W0, Sofa_H2]
                    elif S_c[0] < TV_c[0]:
                        TV_3 = [x1+L1-W_TV, y1+W1//2-L_TV//2, W_TV, L_TV, TV_H3]
                        S_3_r = [x0, y0, W_Sofa, W0, Sofa_H2]
                arrs.append(S_3_r)
                arrs.append(S_3_w_0)
                arrs.append(S_3_w_1)
                rt = random.randint(0, 1)
                if rt:
                    arrs.pop(0)
                    arrs.append(S_3_r)
                arrs.append(TV_3)


            Heights = {'0': f_H}
            Has_NS = 0
            Sofa_TV = {}
            if len(Furnitures[0]) > 1:
                if Furnitures[0][1][4] == self.nightstand.name: Has_NS = 1
            height_1 = []
            for room in House:
                height_1.append(room[0:4])
            Heights['1'] = height_1
            height_2 = []
            height_3 = []
            for room in T_B:
                temp = room[0:4] + [f_H]
                height_2.append(temp)
            for room in Furnitures:
                for furni in room:
                    name = furni[-1]
                    temp = furni[0:4]
                    fur_H = fur_height(name, temp, height_2, Has_NS)
                    if name == self.sofa.name or name == self.TV.name:
                        temp.append(fur_H)
                        Sofa_TV[name] = temp
            Heights['2'] = height_2
            if Sofa_TV != {}:
                Sofa_TV_height(Sofa_TV, height_3)
            Heights['3'] = height_3
            return Heights
        
        def save_normaljson(path, data, name):
            Js_obj = json.dumps(data, cls = self.MyEncoder)
            file_name =  path + '/' + name + '.json'
            with open(file_name, 'w') as f:
                f.write(Js_obj)
        
        # sample_height_data--------------------------------------------------------------------------------
        _type, House, Toil_Bath, Furnitures = self.type, self.House, self.Toil_Bath, self.Furnitures
        floor_H = floorH(House, _type)
        fur_Hs = furnitureH(floor_H, House, Toil_Bath, Furnitures)
        self.height_data = fur_Hs
        save_normaljson(str(abs_path), fur_Hs, 'Height_Function')
        

    def save_layout_figure(self, folder_path, file_name = 'Layout', show = False, save = True, close = True):
        """
        This function saves the layout data as a figure.
        
        Parameters
        ----------
        folder_path : Pathlib.Path
            folder to save the figure
        file_name : str
            file name of the layout figure (,omitting the filename extension)
        show : boolean
            whether this plt is shown
        save : boolean
            whther plt.savefig() will be done 
        close : boolean
            whether plt.close() will be done
            
        Returns
        -------
        ax : matplotlib.pyplot.gca
            Current axes on this figure.
        
        See Also
        --------
        SISG4HEI_Alpha-main/Plot/layout_plot in SISG4HEIAlpha in [2]
        """
        Toil_Bath, Furnitures, Doors, Walls, Lims \
            = self.Toil_Bath, self.Furnitures, self.Doors, self.Walls, self.Boundary
        edge = self.edge
        plt.figure()
        ax = plt.gca()
        ax.set_aspect(1)
        X_lim = [Lims[0][0] - edge, Lims[0][1] + edge]
        Y_lim = [Lims[1][0] - edge, Lims[1][1] + edge]
        plt.xlim((X_lim[0], X_lim[1]))
        plt.ylim((Y_lim[0], Y_lim[1]))
        for room in Toil_Bath:
            [x, y, L, W, name] = room
            rect = plt.Rectangle((x, y), L, W, edgecolor='k', facecolor='none')
            ax.add_patch(rect)
            ax.text(x+L//2, y+W//2, self.Code[name], fontsize = 10, va='center', ha='center')
        for room in Furnitures:
            for furniture in room:
                [x, y, L, W, name] = furniture
                rect = plt.Rectangle((x, y), L, W, edgecolor='k', facecolor='none')
                ax.add_patch(rect)
                ax.text(x+L//2, y+W//2, self.Code[name], fontsize=10, va='center', ha='center')
        for wall in Walls:
            x = [wall[0][0], wall[1][0]]
            y = [wall[0][1], wall[1][1]]
            plt.plot(x, y, 'k-')
        for door in Doors:
            x = [door[0], door[0] + door[2]]
            y = [door[1], door[1] + door[3]]
            plt.plot(x, y, 'w-')
        
        if save: plt.savefig(folder_path / (file_name + '.png'), bbox_inches = 'tight', dpi = 500)
        if show: plt.show()
        if close: plt.close()
        return ax
    
        
    def save_height_figure(self, folder_path, file_name = 'Height_Function', show = False):
        """
        This function saves the height data as a figure.
        
        Parameters
        ----------
        folder_path : Pathlib.Path
            folder to save this figure
        file_name : str
            file name of the height figure (,omitting the filename extension)
        show : boolean
            whether this plt will be shown
        
        See Also
        --------
        SISG4HEI_Alpha-main/Plot/height2feild in SISG4HEIAlpha in [2]
        SISG4HEI_Alpha-main/Plot/filed_plot in SISG4HEIAlpha in [2]
        """
        
        def fill_value(_martix, x, y, L, W, x0, y0, value=0):
            I0, J0, II, JJ = (x-x0)//g_L, (y-y0)//g_W, L//g_L, W//g_W
            for i in range(I0, I0+II):
                for j in range(J0, J0+JJ):
                    _martix[i][j] = value
            
        lims, Heights = self.Boundary, self.height_data
        edge, g_L, g_W = self.edge, self.g_L, self.g_W
        X_lim = [lims[0][0] - edge, lims[0][1] + edge]
        Y_lim = [lims[1][0] - edge, lims[1][1] + edge]
        l_x = np.arange(X_lim[0], X_lim[1] + g_L, g_L)
        l_y = np.arange(Y_lim[0], Y_lim[1] + g_W, g_W)
        data = Heights['0'] * np.ones((len(l_x)-1, len(l_y)-1), dtype = np.int16)
        for [x, y, L, W] in Heights['1']:
            fill_value(data, x, y, L, W, X_lim[0], Y_lim[0])
        for [x, y, L, W, H] in Heights['2']:
            fill_value(data, x, y, L, W, X_lim[0], Y_lim[0], value=H)
        if Heights['3'] != []:
            for [x, y, L, W, H] in Heights['3']:
                fill_value(data, x, y, L, W, X_lim[0], Y_lim[0], value=H)
        
        cbmin, cbmax, masked_v = 0, 340, None
        plt.figure()
        ax = plt.gca()
        ax.set_aspect(1)
        X_lim = [lims[0][0] - edge, lims[0][1] + edge]
        Y_lim = [lims[1][0] - edge, lims[1][1] + edge]
        l_x = np.arange(X_lim[0], X_lim[1] + g_L, g_L)
        l_y = np.arange(Y_lim[0], Y_lim[1] + g_W, g_W)
        XX, YY = np.meshgrid(l_x, l_y)
        if masked_v == None: Z = data.T
        else: Z = np.ma.masked_greater(data.T, masked_v)
        plt.pcolor(XX, YY, Z, cmap = plt.cm.rainbow, vmin = cbmin, vmax = cbmax)
        plt.colorbar()
        plt.savefig(folder_path / (file_name + '.png'), bbox_inches = 'tight', dpi = 500)
        if show: plt.show()
        plt.close()
        
        
    def save_distance(self, folder_path):
        """
        This function saves the information about pass planning.
        
        Parameters
        ----------
        folder_path : Pathlib.Path
            Folder to save the information.
        
        See Also
        --------
        SISG4HEI_Alpha-main/Human_Path_Generation/HP_main/all_distance in SISG4HEIAlpha in [2]
        """
        
        def T_B2obj_W(T_B, rooms):
            """
            This function saves the information about pass planning.

            Parameters
            ----------
            T_B = [[toilet], [bathroom]] :
            rooms : 
            
            Returns
            -------
            T_B_in : list
                place of the toilet and bathroom zone, [x0, min(y0, y1), L0, W0+W1, 'T_B_con'], if those are in the resting zone, cooking zone, or living zone.
            T_B_walls : None or list
                This is a sublist of walls in the toilet and bathroom, [[[x,y],[x,y+W]], [[x,y],[x+L,y]], [[x+L,y],[x+L,y+W]], [[x,y+W],[x+L,y+W]]], if the toilet and bathroom are not in the resting zone, cooking zone, nor living zone. If the walls are in the resting zone, cooking zone, or living zone, then T_B_walls contains it.
            
            See Also
            --------
            SISG4HEI_Alpha-main/Human_Path_Generation/Discom/T_B2obj_W in SISG4HEIAlpha in [2]
            """            
            def com_T_B(T_B):
                [x0, y0, L0, W0, _] = T_B[0]
                [x1, y1, L1, W1, _] = T_B[1]
                # along with y axis
                if x0==x1: return [x0, min(y0, y1), L0, W0+W1, 'T_B_con']
                # along with x axis
                else: return [min(x0, x1), y0, L0+L1, W0, 'T_B_con']

            flag = [0, 0]
            T_B_walls = []
            room_lims = []
            for room in rooms:
                [rx, ry, rL, rW, _] = room
                room_lims.append([[rx, rx+rL], [ry, ry+rW]])
            for n in range(2):
                [x, y, L, W, _] = T_B[n]
                cx, cy = x+L//2, y+W//2
                wall_cens = [[x,y+W//2], [x+L//2,y], [x+L,y+W//2], [x+L//2,y+W]]
                walls = [[[x,y],[x,y+W]], [[x,y],[x+L,y]], [[x+L,y],[x+L,y+W]], [[x,y+W],[x+L,y+W]]]
                for lim in room_lims:
                    if lim[0][0]<cx<lim[0][1] and lim[1][0]<cy<lim[1][1]: flag[n] = 1
                if flag[n] == 0:
                    for i in range(4):
                        for lim in room_lims:
                            if lim[0][0]<=wall_cens[i][0]<=lim[0][1] and lim[1][0]<=wall_cens[i][1]<=lim[1][1]:
                                T_B_walls.append(walls[i])
                                break
            if flag==[0, 0]: return None, T_B_walls
            if flag==[1, 0]: return T_B[0], T_B_walls
            if flag==[0, 1]: return T_B[1], T_B_walls
            return com_T_B(T_B), None
        
        def mod_walls(walls, T_B_Walls):
            # T_B_Walls are added to walls.
            if T_B_Walls != None:
                for wall in T_B_Walls:
                    walls.append(wall)
                    
        def cal_dis_val(rooms, furnitures , walls, lims, t_b_in = None):
            """
            This calculates the discomfort value caused by nearby obstacles (e.g., furniture or walls).
            
            Parameters
            ----------
            rooms = [Bedroom, Kitchen, Living] : 
            furnitures = [Bedroom furniture, Kitchen furniture, Living furniture] : 
            walls = [[[x,y], [x + g, y]], ] + T_B_walls : 
            lims = [X_lim, Y_lim] = [[minX, maxX], [minY, maxY]] : 
            t_b_in : 
                toilet and bathroom zone [x0, min(y0, y1), L0, W0+W1, 'T_B_con'] which is in the resting zone, cokking zone, or living zone.
            
            Returns
            -------
            data : numpy.ndarray
                a discomfort value matrix,
                be careful that the indexes of the matrix do not correspond with (x,y) coordinate
                
            See Also
            --------
            SISG4HEI_Alpha-main/Human_Path_Generation/Discom/cal_dis_val in SISG4HEIAlpha in [2]
            """
            def cal_p_xy(room, i, j):
                """
                This function calclulates the center of the (i,j)-th unit zone.
                
                Parameters
                ----------
                room = [x, y, L, W, name] : 
                i : int
                j : int
                    (i,j): index
                
                Returns
                -------
                center : list of float
                """
                [x, y] = room[:2]
                return [x+self.g_L/2+self.g_L*i, y+self.g_W/2+self.g_W*j]
            
            def p2wdis(p,wall):
                """
                Parameters
                ----------
                p : list of float
                    center point
                wall : list of list of float
                    e.g., [[x,y], [x + l, y]]
                
                Returns
                -------
                dist : float
                    minimum distance between center point and wall
                """  
                [[w00,w01],[w10,w11]], [px, py] = wall, p
                dx = min(abs(px-w00), abs(px-w10))
                dy = min(abs(py-w01), abs(py-w11))
                if w00 == w10:  # door attached to the y-axis
                    if w01 <= py <= w11: return dx
                if w01 == w11:  # door attached to the x-axis
                    if w00 <= px <= w10: return dy
                return math.sqrt(dx*dx + dy*dy)
            
            def p2odis(p, obj):
                """
                Parameters
                ----------
                p : list of float
                    center point
                obj = [x, y, L, W, name] : 
                
                Returns
                -------
                dist : float
                    minimum distance between center point and obj zone if the center point is out of obj zone
                """
                [x, y, L, W, _], [px, py] = obj, p
                if x <= px <= x + L and y <= py <= y + W: return 0
                dy = min(abs(py-y),abs(y+W-py))
                dx = min(abs(px-x),abs(x+L-px))
                if x <= px <= x + L: return dy
                if y <= py <= y + W: return dx
                return math.sqrt(dx*dx + dy*dy)
            
            def sub_dis_val_c(d, _type = 'obj'):
                """
                Parameters
                ----------
                d : float
                    distance
                _type : str, default 'obj'
                
                Returns
                -------
                dist : float
                   
                """
                if _type == 'wall': return max(math.sqrt(250/d)-1, 0)
                return max(math.sqrt(150/d)-1, 0)

            dis_val, obstacles = [], []
            for r in furnitures:
                for f in r: obstacles.append(f)
            for n in range(len(rooms)):
                _, _, L, W, _ = rooms[n]
                sub_dis_val = np.ones((L//self.g_L, W//self.g_W), dtype = np.float32)
                for i in range(L//self.g_L):
                    for j in range(W//self.g_W):
                        p = cal_p_xy(rooms[n], i, j)  # center coordinates of unit zone
                        disws = []
                        for wall in walls:
                            dis = p2wdis(p, wall)
                            disws.append(dis)
                        if t_b_in:
                            dis = p2odis(p, t_b_in)
                            disws.append(dis)
                        disw = min(disws)
                        if disw<25: sub_dis_val[i][j] = 100
                        else: sub_dis_val[i][j] += sub_dis_val_c(disw, _type = 'wall')
                        for obj in obstacles:
                            dis = p2odis(p, obj)
                            if dis<20: sub_dis_val[i][j] = 100
                            elif sub_dis_val[i][j] != 100:
                                sub_dis_val[i][j] += sub_dis_val_c(dis)
                dis_val.append(sub_dis_val)

            X_lim = [lims[0][0] - self.edge, lims[0][1] + self.edge]
            Y_lim = [lims[1][0] - self.edge, lims[1][1] + self.edge]
            l_x = np.arange(X_lim[0], X_lim[1] + self.g_L, self.g_L)
            l_y = np.arange(Y_lim[0], Y_lim[1] + self.g_W, self.g_W)
            data = 100 * np.ones((len(l_x)-1, len(l_y)-1), dtype = np.float32)
            for n in range(len(rooms)):
                [x, y, L, W, _] = rooms[n]
                N_i, N_j, N_L, N_W = (x-X_lim[0])//self.g_L, (y-Y_lim[0])//self.g_W, L//self.g_L, W//self.g_W
                for i in range(N_L):
                    for j in range(N_W):
                        data[i+N_i][j+N_j] = dis_val[n][i][j]  # indexes of this matrix do not correspond with (x,y) coordinate
            return data
        
        
        def cal_destinations(rooms, furnitures, doors, dis_val, X0, Y0):
            """
            Parameters
            ----------
            rooms = [Bedroom, Kitchen, Living] : 
            furnitures = [Bedroom furniture, Kitchen furniture, Living furniture] : 
            doors :
                Doors is the list of Door
                label in {Entrance, Toilet_Door, Bathroom_Door}
            dis_val : numpy.ndarray
                discomfort value matrix, but matrix indexes do not correspond with (x,y) coordinate
            X0 : float
                minX - edge
            Y0 : float
                minY - edge
            
            Returns
            -------
            Destinations : dict fo list of list
                {'Bed': [[x1, y1], [x2, y2], ..., [xn, yn]],
                 'Desk': ...}
                
            Notes
            -----
            The original code in SISG4HEIAlpha in [2] is used activity name as keys of destinations.
            In this function, furniture name is used as keys instead of activity name.
            
            See Also
            --------
            SISG4HEI_Alpha-main/Human_Path_Generation/Dest_nodes/cal_destinations in SISG4HEIAlpha in [2]
            """
            
            def calc_key_centers(rooms, furnitures):
                # calculate the bedroom center, kitchen center, Desk center, Table center and TV center.
                Bed_c, Kit_c, Des_c, Tab_c, TV_c = None, None, None, None, None
                for i in range(len(rooms)):
                    rx, ry, rL, rW, rn = rooms[i]
                    if rn == self.bedroom.name:
                        Bed_c = [rx+rL//2, ry+rW//2]
                    elif rn == self.kitchen.name:
                        Kit_c = [rx+rL//2, ry+rW//2]
                    for fur in furnitures[i]:
                        fx, fy, fL, fW, fn = fur
                        if fn == self.desk.name:
                            Des_c = [fx+fL//2, fy+fW//2]
                        elif fn == self.dinner_table.name:
                            Tab_c = [fx+fL//2, fy+fW//2]
                        elif fn == self.TV.name:
                            TV_c = [fx+fL//2, fy+fW//2]
                return Bed_c, Kit_c, Des_c, Tab_c, TV_c
            
            def correct_side(x, y, L, W, p, flag = 'longest_nearst'):
                """
                Parameters
                ----------
                [x, y, L, W] :
                    furnitures information
                p : float
                    related object's center point
                flag : str, default 'longest_nearst'
                    
                Returns
                -------
                index : list of int
                     retrun the index of center point of furniture
                """
                def distance(p0, p1):
                    [x0, y0], [x1, y1] = p0, p1
                    return math.sqrt((x1-x0)*(x1-x0)+(y1-y0)*(y1-y0))

                side_centers = [[x+L//2, y], [x, y+W//2], [x+L//2, y+W], [x+L, y+W//2]]
                # return nearest long sides
                v1 = [L * 10, W * 10, L * 10, W * 10]
                v2 = [distance(e, p) for e in side_centers]
                v = [v1[i] - v2[i]/10 for i in range(4)]
                if flag == 'longest_nearst': return [v.index(max(v))]
                # As primary metrics, the longer the side is the more it will be returned, then as secondary metrics, the nearer the side is related object's center point, the more it will be returned.
                # else return middle sides
                v2_min_i, v2_max_i = v2.index(min(v2)), v2.index(max(v2))
                sides = [0, 1, 2, 3]
                sides.remove(v2_min_i)
                sides.remove(v2_max_i)
                return sides
            
            def fur_dest(X0, Y0, x, y, L, W, index, g_L = 5, g_W = 5):
                """
                Parameters
                ----------
                X0 : float
                    minX - edge
                Y0 : float
                    minY - edge
                [x, y, L, W] : 
                    furnitures information
                index :
                    index of furniture's side
                g_L : 
                g_W : 
                    
                Returns
                -------
                dest : 
                    outline point of furniture destination zone
                """
                nodes = [[], [], [], []]
                dest = []
                I0, J0 = (x-X0)//g_L, (y-Y0)//g_W
                I1, J1 = I0+L//g_L, J0+W//g_W
                for t in range(I0, I1):
                    nodes[0].append([t, J0-5])
                    nodes[2].append([t, J1+4])
                for t in range(J0, J1):
                    nodes[1].append([I0-5, t])
                    nodes[3].append([I1+4, t])
                for i in index:
                    for [I, J] in nodes[i]: dest.append([I, J])
                return dest
            
            def door_dest(X0, Y0, x, y, L, W, g_L=5, g_W=5):
                nodes = []
                I0, J0 = (x-X0)//g_L, (y-Y0)//g_W
                I1, J1 = I0+L//g_L, J0+W//g_W
                if L == 0:
                    for t in range(J0, J1):
                        nodes.append([I0-6, t])
                        nodes.append([I0+5, t])
                elif W == 0:
                    for k in range(I0, I1):
                        nodes.append([k, J0-6])
                        nodes.append([k, J0+5])
                return nodes
            
            Bed_c, Kit_c, Des_c, Tab_c, TV_c = calc_key_centers(rooms, furnitures)  # center points
            Destinations = {}
            for i in range(len(rooms)):
                _, _, _, _, r_n = rooms[i]
                for furniture in furnitures[i]:
                    x, y, L, W, name = furniture
                    key = name
                    if key not in Destinations:
                        Destinations[key] = []
                    index = [0, 1, 2, 3]
                    if name == self.wardrobe.name:
                        index = correct_side(x, y, L, W, Bed_c)
                    elif name == self.kitchen_stove.name:
                        index = correct_side(x, y, L, W, Kit_c)
                    elif name == self.cupboard.name:
                        index = correct_side(x, y, L, W, Kit_c)
                    elif name == self.refrigerator.name:
                        index = correct_side(x, y, L, W, Kit_c)
                    elif name == self.sofa.name:
                        index = correct_side(x, y, L, W, TV_c)
                    elif name == self.desk_chair.name:
                        index = correct_side(x, y, L, W, Des_c, flag = 'middle')
                    elif name == self.dinner_table.name:
                        index = correct_side(x, y, L, W, Tab_c, flag = 'middle')
                    nodes = fur_dest(X0, Y0, x, y, L, W, index)
                    for [k, t] in nodes:
                        if dis_val[k][t] < 99:
                            Destinations[key].append([k, t])  # destination points around the furniture
            for door in doors:
                x, y, L, W, d_n = door
                key = d_n
                Destinations[key] = []
                nodes = door_dest(X0, Y0, x, y, L, W)
                for [k, t] in nodes:
                    if dis_val[k][t] < 99:
                        Destinations[key].append([k, t])
            return Destinations

        def topology(dis_v):
            """
            Parameters
            ----------
            dis_v : numpy.ndarray
                discomfort value matrix
                be carefult that the indexes of the matrix do not correspond (x,y) coordinate
            
            Returns
            -------
            Topology : dict of list of int
                a dictionary of nearest cells' indexes, e.g., {(2,2): [[0,1], [0,3], [1,0], ..., [4,3]], (2,3):...},
                SISG4HEIAlpha used 16 neighbors mentioned in [1]
                
            See Also
            --------
            SISG4HEI_Alpha-main/Human_Path_Generation/Distance/topology in SISG4HEIAlpha in [2]
            """
            Topology = {}
            for i in range(1, dis_v.shape[0]-1):
                for j in range(1, dis_v.shape[1]-1):
                    if dis_v[i][j] < 99:
                        Topology[(i, j)] = []
                        for [ii, jj] in self.local_cood:
                            if dis_v[i+ii][j+jj] < 99:
                                Topology[(i, j)].append([i+ii, j+jj])
            return Topology
        
        def weighted_dijkstra(s_nodes, topo_graph, dis_val):
            """
            Parameters
            ----------
            s_nodes : list of list
                start or destination points of some place, e.g., [[x1, y1], [x2, y2], ..., [xn, yn]]
            topo_graph : dict of list of int
                a dictionary of nearest cells' indexes, e.g., {(2,2): [[0,1], [0,3], [1,0], ..., [4,3]], (2,3):...}
            dis_val : numpy.ndarray
                a discomfort value matrix
                be carefult that the indexes of the matrix do not correspond (x,y) coordinate
            
            Returns
            -------
            distance : numpy.ndarray
                distance from a place
                The shape is same with dis_val
            max_dis : float
                maximum distance among points that can be arrived from a place
                
            See Also
            --------
            SISG4HEI_Alpha-main/Human_Path_Generation/Distance/weighted_dijistra in SISG4HEIAlpha in [2]
            """
            def real_distance(P0, P1):
                # input is a node number
                [I0, J0], [I1, J1] = P0, P1
                return math.sqrt(self.g_L*self.g_L*(I1-I0)*(I1-I0)+self.g_W*self.g_W*(J1-J0)*(J1-J0))

            distance = 99999 * np.ones(dis_val.shape, dtype = np.float32)
            boundry, max_dis = {}, 0
            for [k, t] in s_nodes:
                distance[k][t] = 0.0
                boundry[(k, t)] = 0.0
            while boundry != {}:
                (I, J) = min(boundry, key = boundry.get)
                Od = boundry.pop((I, J))
                neighbors = topo_graph.pop((I, J))
                if neighbors != []:
                    for [II, JJ] in neighbors:
                        real_d = real_distance([II, JJ],[I, J])  # Euclidean distance
                        n_dis = Od + real_d * dis_val[II][JJ]
                        if n_dis < distance[II][JJ]:
                            distance[II][JJ] = n_dis
                            if n_dis > max_dis: max_dis = n_dis
                        boundry[(II, JJ)] = distance[II][JJ]
                        topo_graph[(II, JJ)].remove([I, J])
            return distance, round(max_dis, 2)
        
        def connect_check(distance, max_dis, destinations):
            """
            Parameters
            ----------
            distance : 
            max_dis : 
            destinations : 
            
            Returns
            -------
            is_connected : boolean
                whether the place is conncted
            error_furniture : str
                the furniture that cannnot reach from some furniture
                
            See Also
            --------
            SISG4HEI_Alpha-main/Human_Path_Generation/Distance/connect_check in SISG4HEIAlpha in [2]
            """
            for f in destinations:
                e_nodes = destinations[f]
                value = 99998
                for node in e_nodes:
                    [I, J] = node
                    if distance[I][J] < value: value = distance[I][J]
                if value > max_dis: return False, f
            return True, None

        def save_filed(path, name, _format, data):
            file_name = path / name 
            np.savetxt(file_name, data, fmt = _format, delimiter = ',')
                
        def save_normaljson(path, data, name):
            Js_obj = json.dumps(data, cls = self.MyEncoder)
            file_name =  path / (name + '.json')
            with open(file_name, 'w') as f:
                f.write(Js_obj)

        # save_distance--------------------------------------------------------
        rooms, T_Bs, furnitures, doors, walls, lims \
            = self.House, self.Toil_Bath, self.Furnitures, self.Doors, self.Walls, self.Boundary
        
        T_B_in, T_B_walls = T_B2obj_W(T_Bs, rooms)
        # T_B_in: place of the toilet and bathroom [x0, min(y0, y1), L0, W0+W1, 'T_B_con']
        # T_B_walls: sublist of [[[x,y],[x,y+W]], [[x,y],[x+L,y]], [[x+L,y],[x+L,y+W]], [[x,y+W],[x+L,y+W]]]
        mod_walls(walls, T_B_walls)  # T_B_walls are added to walls.
        discom = cal_dis_val(rooms, furnitures, walls, lims, t_b_in = T_B_in)
        # discom: discomfort value matrix, but matrix index doesn't correspond (x,y) coordinate
        X0, Y0 = lims[0][0] - self.edge, lims[1][0] - self.edge
        destinations = cal_destinations(rooms, furnitures, doors, discom, X0, Y0)
        # destinations: {'Bed': [[x1, y1], [x2, y2], ..., [xn, yn]], 'Kitchen_Stove': ...}
        max_diss_d = {}  # max distance, thus, it corresponds to max LTJ1(x, the destinations) mentioned in [1]
        connectflag = 0
        
        for key in destinations.keys():
            s_nodes = destinations[key]
            topo = topology(discom)
            # topo : dictionary of nearest cells' indexes, e.g., {(2,2): [[0,1], [0,3], [1,0], ..., [4,3]], (2,3):...}
            distance, max_dis = weighted_dijkstra(s_nodes, topo, discom)
            # This distance is LTJ1 in the paper[1].
            if connectflag == 0:
                che_con, e_f = connect_check(distance, max_dis, destinations)
                if che_con:
                    connectflag = 1
                else:
                    raise ValueError("{} CAN NOT be arrived in {}!".format(e_f, folder_path))
            max_diss_d[key] = max_dis
            f_name = key + '_distance.csv'
            save_filed(folder_path, f_name, '%7.2f', distance)
        save_filed(folder_path, 'Discomfortable_value.csv', '%5.2f', discom)
        save_normaljson(folder_path, destinations, 'Destinations')
        save_normaljson(folder_path, max_diss_d, 'Max_Distances')

        
    def load_file(self, layout_path, name, extension):
        """
        This function loads a csv file.

        Parameters
        ----------
        layout_path : pathlib.Path
            Path to the layout data.
        name : str
            Filename except for the filename extensions.
        extension : str
            Filename extension.
        """
        file_name = layout_path / (name + '.' + extension)
        ret = None
        if extension == 'csv':
            ret = np.loadtxt(file_name, dtype = np.float32, delimiter = ',')
        elif extension == 'json':
            with open(file_name, 'r') as f:
                data = f.read()
            ret = json.loads(data)
        else:
            raise ValueError('The filename extensions is not appropriate.')
        return ret
        
        

class Zone:
    # class of a zone
    
    def __init__(self, name, plot_name):
        """
        initialization of Zone class

        Parameters
        ----------

        """
        #   y-axis     -----------
        #   ^         |          |   
        #   |         |   area   |  l_y
        #   |         |          |
        #             p-----------       ------> x-axis
        #                l_x
        #  A coordinate p = (self.x, self.y) is left bottom of the rectangle.
        self.__name = name
        self.__plot_name = plot_name
        # self.x, self.y = 0, 0
        # self.l_x, self.l_y, self.l_z, self.area = 0, 0, 0, 0
        
    def __str__(self):
        return "<Zone> name:{}".format(self.name)
    
    # getter
    @property
    def name(self):
        return self.__name
    @property
    def plot_name(self):
        return self.__plot_name
    
    
class Door:
    # class of a door
    
    def __init__(self, name):
        """
        initialization of this class

        Parameters
        ----------
        name : str
            name of this door

        """
        self.__name = name
        
    def __str__(self):
        return "<Door> name:{}".format(self.name)
    
    # getter
    @property
    def name(self):
        return self.__name
    
class Wall:
    # class of a wall
    
    def __init__(self):
        """
        initialization of thisFurniture class

        Parameters
        ----------

        """
        
    def __str__(self):
        return "<Wall>"
    
    
class Furniture:
    # class of furniture
    
    def __init__(self, name, plot_name):
        """
        initialization of Furniture class

        Parameters
        ----------
        name : str
            name of this furniture
        plot_name : str
            abbreviation of name to plot this furniture in a figure

        """
        self.__name = name
        self.__plot_name = plot_name
        
    def __str__(self):
        return "<Furniture> name:{}".format(self.name)
    
    # getter
    @property
    def name(self):
        return self.__name
    @property
    def plot_name(self):
        return self.__plot_name
    
    
    
