import bisect
import re
import tkinter
from datetime import timedelta
from pathlib import Path
from tkinter import ttk

# New pickled data can be loaded as below.
import src.floor_plan as floor_plan
import src.utils as utils

# Old pickled sensor_model are made when the sensor_model.py was located in the same folder hierarchy as main.py.
# New sensor_model.py is located in the src/sensor_model.py, that is different from the folder hierarchy of main.py.
# How to load both old and newly created data is below. But linter error may arise.
# import sys
# sys.path.append("./src")
# import floor_plan
# import utils
# import sensor_model


class FootprintSimulator:
    N, W, E, S = tkinter.N, tkinter.W, tkinter.E, tkinter.S

    def __init__(self, root, path):
        """
        Parameters
        ----------
        root : tkinter.Tk
        path : pathlib.Path
            A data folder path that includes layout folders.

        Returns
        -------
        None
        """

        self.root = root
        self.path = path
        self.layout_path = Path()
        self.data_path = Path()
        self.can_start = False  # whether to be able to start the simulation
        self.playing = False  # whether the simulation is playing
        self.simulation_id = None  # an id of the timer event
        self.AS_i = 0  # index in an activity sequence data
        self.WT_i = 0
        self.SD_i = 0
        self.sensor2id = {}  # map the sensor index to the id on the canvas
        self.simulation_time = timedelta(days=0)
        self.body_radius = 10
        self.foot_radius = 4
        self.center_id = None
        self.left_step_id = None
        self.right_step_id = None

        self.root.title("Footprint Simulator")

        # widgets
        self.mainframe = ttk.Frame(root, padding=(3, 3, 12, 12))

        self.time_l = ttk.Label(self.mainframe, text="Time")  # l means label
        self.time_textvariable = tkinter.StringVar()
        self.time_textvariable.set(str(self.simulation_time))
        self.time = ttk.Label(self.mainframe, textvariable=self.time_textvariable)

        self.time_input_l = ttk.Label(self.mainframe, text="Time input")
        self.time_input_textvariable = tkinter.StringVar()
        self.placeholder_text = "dddd-hh-mm-ss"
        self.time_input_textvariable.set(self.placeholder_text)
        self.time_input = ttk.Entry(
            self.mainframe, textvariable=self.time_input_textvariable
        )
        self.time_input.config(foreground="gray")
        self.time_input.bind("<FocusIn>", self.on_time_input_click)
        self.time_input.bind("<FocusOut>", self.on_focusout)
        self.on_focusout(None)

        self.activity_l = ttk.Label(self.mainframe, text="Activity")
        self.activity_textvariable = tkinter.StringVar()
        self.activity_textvariable.set("")
        self.activity = ttk.Label(
            self.mainframe, textvariable=self.activity_textvariable
        )

        self.figure = tkinter.Canvas(
            self.mainframe, borderwidth=5, relief="ridge", bg="white"
        )

        self.layout_l = ttk.Label(self.mainframe, text="Layout")
        self.layouts = []
        for p in self.path.iterdir():
            if p.name != ".ipynb_checkpoints":
                self.layouts.append(p)
        self.layouts_name = [p.name for p in self.layouts]
        self.layout_listvariable = tkinter.StringVar(value=self.layouts_name)
        self.layout = tkinter.Listbox(
            self.mainframe, listvariable=self.layout_listvariable, height=3
        )
        self.layout.bind("<<ListboxSelect>>", self.show_data_list)

        self.data_l = ttk.Label(self.mainframe, text="Data")
        self.data_paths = []
        self.data_name = []
        self.data_name_listvariable = tkinter.StringVar(value=self.data_name)
        self.data = tkinter.Listbox(
            self.mainframe, listvariable=self.data_name_listvariable, height=3
        )
        self.data.bind("<<ListboxSelect>>", self.draw_layout)

        self.simulation_l = ttk.Label(self.mainframe, text="Simulation")
        self.simulation = ttk.Button(
            self.mainframe, text="Start", command=self.start_stop_simulation
        )

        # grid geometry manager
        self.mainframe.grid(row=0, column=0, sticky=(self.N, self.S, self.E, self.W))
        self.time_l.grid(row=0, column=0)
        self.time.grid(row=0, column=1)
        self.time_input_l.grid(row=0, column=2)
        self.time_input.grid(row=0, column=3)
        self.activity_l.grid(row=0, column=4)
        self.activity.grid(row=0, column=5)
        self.figure.grid(
            row=1, column=0, columnspan=6, sticky=(self.N, self.S, self.E, self.W)
        )
        self.layout_l.grid(row=2, column=0)
        self.layout.grid(row=2, column=1)
        self.data_l.grid(row=2, column=2)
        self.data.grid(row=2, column=3)
        self.simulation_l.grid(row=2, column=4)
        self.simulation.grid(row=2, column=5)

        # resizable
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.mainframe.columnconfigure(0, weight=1)
        self.mainframe.columnconfigure(1, weight=1)
        self.mainframe.columnconfigure(2, weight=1)
        self.mainframe.columnconfigure(3, weight=1)
        self.mainframe.rowconfigure(1, weight=1)

    def on_time_input_click(self, event):
        self.time_input_textvariable.set("")
        self.time_input.config(foreground="black")

    def on_focusout(self, event):
        pattern = r"\d{4}-\d{2}-\d{2}-\d{2}"
        text = self.time_input_textvariable.get()
        valid = False
        if re.match(pattern, text):
            parts = text.split("-")
            days = int(parts[0])
            hours = int(parts[1])
            minutes = int(parts[2])
            seconds = int(parts[3])
            time = timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds)
            if self.AS[0].start < time <= self.AS[-1].end:
                self.simulation_time = time
                valid = True
        if valid and self.can_start:
            self.AS_i = 0
            self.WT_i = 0
            self.SD_i = 0
            self.update_activity()
            self.update_sensor()
            self.update_walking_trajectory()
            self.simulation["text"] = "Stop"
            self.root.after(50, self.draw_data)
            self.playing = True

        self.time_input_textvariable.set(self.placeholder_text)
        self.time_input.config(foreground="gray")

    def show_data_list(self, *args):
        indexes = self.layout.curselection()
        if len(indexes) == 0:
            return
        index = int(indexes[0])
        self.layout_path = self.layouts[index]
        self.data_paths = []
        for p in self.layout_path.iterdir():
            if p.is_dir() and p.name != ".ipynb_checkpoints":
                self.data_paths.append(p)
        self.data_name = [p.name for p in self.data_paths]
        self.data_name_listvariable.set(self.data_name)
        self.figure.delete("all")
        self.can_start = False
        self.reset()

    @staticmethod
    def closure_of_point2canvas(reference, scale):
        """
        This is a closure of a function that converts a coordinate in the layout into the coordinate on the canvas.

        Parameters
        ----------
        reference : tuple of float
            This is the reference point as the top left coordinate in the layout.
        scale : float
            length in the layout : length on the canvas = 1 : scale.


        Returns
        -------
        ret : function

            This function returns ret = (xx, yy) is the calculated coordinate on the canvas.
        """

        def point2canvas(xy):
            """
            This converts a coordinate in the layout into the coordinate on the canvas.

            Parameters
            ----------
            xy : tuple of float
                xy = (x, y) is the target point in the layout.

            Returns
            -------
            ret : tuple of float
                ret = (xx, yy) is the calculated coordinate on the canvas.
            """
            return (scale * (xy[0] - reference[0]), scale * (reference[1] - xy[1]))

        return point2canvas

    def draw_layout(self, *args):
        """
        This draws a layout and sensor arrangements on the canvas.
        """
        self.reset()
        indexes = self.data.curselection()
        if len(indexes) == 0:
            return
        index = int(indexes[0])
        self.data_path = self.data_paths[index]

        # layout
        fp = floor_plan.FloorPlan()
        fp.load_layout(self.layout_path)

        scale = 1  # length in the layout : length on the canvas = 1 : scale.
        # reference is the reference point as the top left coordinate in the layout
        reference = (fp.Boundary[0][0] - fp.edge, fp.Boundary[1][1] + fp.edge)
        self.point2canvas = self.closure_of_point2canvas(reference, scale)

        # layout
        fp.canvas(self.figure, self.point2canvas)

        # sensor
        sensors = utils.pickle_load(self.data_path, "SD_model")
        for i, s in enumerate(sensors):
            self.sensor2id[i] = s.canvas(self.figure, self.point2canvas)

        # load activity sequence and walking trajectories
        self.AS = utils.pickle_load(self.data_path, "AS")
        self.WT = utils.pickle_load(self.data_path, "WT")
        self.SD = utils.pickle_load(self.data_path, "SD")
        self.body_radius = utils.pickle_load(self.data_path, "WT_model").body_radius
        self.can_start = True

    def start_stop_simulation(self, *args):
        """
        This starts the footprint simulation.
        """
        if not (self.can_start):
            return
        if self.playing:
            self.simulation["text"] = "Start"
            self.root.after_cancel(self.simulation_id)
        else:
            self.simulation["text"] = "Stop"
            self.root.after(50, self.draw_data)
        self.playing = not (self.playing)

    def reset(self):
        """
        This resets the simulation time.
        """
        if self.simulation_id is None:
            return
        self.simulation_time = timedelta(days=0)
        self.time_textvariable.set(str(self.simulation_time))
        self.playing = False
        self.simulation["text"] = "Start"
        self.root.after_cancel(self.simulation_id)
        self.can_start = False
        self.AS_i = 0
        self.WT_i = 0
        self.SD_i = 0
        self.sensor2id = {}

    def draw_data(self):
        """
        This draws time, activities, walking trajectories, and sensor activations.
        """
        if self.playing:
            self.simulation_time += self.determine_simulation_step()
            self.time_textvariable.set(str(self.simulation_time))
            self.update_activity()
            self.update_sensor()
            self.update_walking_trajectory()

        self.simulation_id = root.after(50, self.draw_data)

    def determine_simulation_step(self):
        is_walking = (
            self.WT[self.WT_i].start_time
            <= self.simulation_time
            < self.WT[self.WT_i].end_time
        )
        if is_walking:
            step = timedelta(microseconds=100000)
        else:
            step = min(
                timedelta(minutes=2),
                self.WT[self.WT_i].start_time - self.simulation_time,
            )
        return step

    def update_activity(self):
        while not (
            self.AS[self.AS_i].start <= self.simulation_time < self.AS[self.AS_i].end
        ):
            self.AS_i += 1
        self.activity_textvariable.set(self.AS[self.AS_i].activity.name)

    def update_sensor(self):
        data = []
        while self.SD[self.SD_i][0] <= self.simulation_time:
            data.append(self.SD[self.SD_i])
            self.SD_i += 1
        for d in data:
            _id = self.sensor2id[d[1]]
            if d[2]:
                color = self.figure.itemcget(_id, "outline")
            else:
                color = ""
            self.figure.itemconfigure(_id, fill=color)
        pass

    def update_walking_trajectory(self):
        """
        Notes
        -----
        self.WT_i means the index of the closest walking trajectories after the present time.

           None  | WT0 |    None     |   WT1    | None  |WT2 |    None: no walking
        ---------------|------------------------|------------|----> time
              WT_i = 0 |         WT_i = 1       |  WT_i = 2  |
        """

        if self.center_id is not None:
            self.figure.delete(self.center_id)
        # if self.left_step_id != None:
        #     self.figure.delete(self.left_step_id)
        # if self.right_step_id != None:
        #     self.figure.delete(self.right_step_id)
        while self.WT[self.WT_i].end_time <= self.simulation_time:
            self.WT_i += 1
        if self.WT[self.WT_i].start_time <= self.simulation_time:
            # draw the closest position before self.simulation_time
            timestamp_i = (
                bisect.bisect_right(self.WT[self.WT_i].timestamp, self.simulation_time)
                - 1
            )
            center = self.WT[self.WT_i].centers[timestamp_i]
            center_top_left = self.point2canvas(
                (center[0] - self.body_radius, center[1] + self.body_radius)
            )
            center_bottom_right = self.point2canvas(
                (center[0] + self.body_radius, center[1] - self.body_radius)
            )
            self.center_id = self.figure.create_oval(
                center_top_left[0],
                center_top_left[1],
                center_bottom_right[0],
                center_bottom_right[1],
                outline="black",
                fill="black",
            )
            if timestamp_i % 2 == 1:
                left_step = self.WT[self.WT_i].left_steps[timestamp_i // 2 + 1]
                left_step_left = self.point2canvas(
                    (left_step[0] - self.foot_radius, left_step[1] + self.foot_radius)
                )
                left_step_bottom_right = self.point2canvas(
                    (left_step[0] + self.foot_radius, left_step[1] - self.foot_radius)
                )
                self.left_step_id = self.figure.create_oval(
                    left_step_left[0],
                    left_step_left[1],
                    left_step_bottom_right[0],
                    left_step_bottom_right[1],
                    outline="black",
                    fill="red",
                )
            else:
                right_step = self.WT[self.WT_i].right_steps[timestamp_i // 2]
                right_step_left = self.point2canvas(
                    (right_step[0] - self.foot_radius, right_step[1] + self.foot_radius)
                )
                right_step_bottom_right = self.point2canvas(
                    (right_step[0] + self.foot_radius, right_step[1] - self.foot_radius)
                )
                self.right_step_id = self.figure.create_oval(
                    right_step_left[0],
                    right_step_left[1],
                    right_step_bottom_right[0],
                    right_step_bottom_right[1],
                    outline="black",
                    fill="blue",
                )

    def interpolate_positions(self):
        pass


root = tkinter.Tk()
path = Path("./layout_data")
FootprintSimulator(root, path)
root.mainloop()
