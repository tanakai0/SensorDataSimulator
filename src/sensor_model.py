import random

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance

# self-made
import src.floor_plan as floor_plan
import src.constants as constants



def generate_test_sensor_arrange(
    path, th_discomfort=90.0, num_PIR=20, num_pressure=3, min_d=50.0, max_iter=1000
):
    """
    This generates test arrangement of sensors in a layout.

    Parameters
    ----------
    path : pathlib.Path
        Path to the layout files.
    th_discomfort : float, default 90
        Threshold of the dicomfort values.
        The points that exceed this threshold are not candidates for the sensor centers.
    num_PIR : int, default 20
        Maximum number of the PIR sensors.
        Be careful that the final number of the sensors are sometimes lower than this value.
    num_pressure : int, default 3
        Maximum number of the pressure sensors.
        Be careful that the final number of the sensors are sometimes lower than this value.
    min_d : float, default 50.0
        Minimum distance between the centers of the sensors.
    max_iter : int
        Maximum number of the iteration to try to arrange the sensors.

    Returns
    -------
    sensors : list of Sensor

    See Also
    --------
    SISG4HEI_Alpha-main/Interface/Tools_/PIR_position in SISG4HEIAlpha in [2] as a reference.
    """
    fp = floor_plan.FloorPlan()
    fp.load_layout(path)
    rooms, lims = fp.House, fp.Boundary
    discomfort = fp.load_file(path, "Discomfortable_value", "csv")

    # randomly generate positions of sensors, using discomfort value in the layout.
    positions = []
    num, i = 0, 0
    while (num < num_PIR + num_pressure) and (i < max_iter):
        is_valid = True
        room_index = random.randint(0, len(rooms) - 1)
        x = rooms[room_index][0] + random.random() * rooms[room_index][2]
        y = rooms[room_index][1] + random.random() * rooms[room_index][3]
        p = (x, y)
        pos_I = int((x - lims[0][0] + fp.edge) // fp.g_L)
        pos_J = int((y - lims[1][0] + fp.edge) // fp.g_W)

        if discomfort[pos_I][pos_J] >= th_discomfort:
            is_valid = False
        for _p in positions:
            if distance.euclidean(p, _p) < min_d:
                is_valid = False
                break

        if is_valid:
            positions.append(p)
            num += 1
        i += 1

    if (i >= max_iter) and (num < num_PIR + num_pressure):
        print(
            "The number of sensors is {} and it is less than the input num_PIR + num_pressure.".format(
                num
            )
        )

    sensors = []
    for i, p in enumerate(positions):
        if i < num_PIR:
            sensors.append(
                CircularPIRSensor("normal PIR", i, p[0], p[1], constants.LIGHT_RED, 50)
            )
        else:
            sensors.append(
                SquarePressureSensor(
                    "vertical oblong pressure",
                    i,
                    p[0] - 50,
                    p[1] - 50,
                    constants.LIGHT_GRAY,
                    100,
                    100,
                    0,
                )
            )
    return sensors


class Sensor:
    """
    Class of a sensor.

    Attributes
    ----------
    name : str
        Name of the sensor.
    index : int
        Index of this sensor.
    x : float
        A x-axis value of 2D coordinate.
    y : float
        A y-axis value of 2D coordinate.
        The (x, y) is a reference point of this sensor.
    color : str
        Color code that is used to plot a figure.
    type_name : str
        Name of this class.
    """

    type_name = "Sensor"

    def __init__(self, name, index, x, y, color):
        """
        Initialize of an object of Sensor.

        Parameters
        ----------
        Same with parameters of Attributes.
        """
        self.__name = name
        self.__index = index
        self.__x = x
        self.__y = y
        self.__color = color

    def __str__(self):
        return "<Sensor>{}(#{})({:.2f}, {:.2f})".format(
            self.name, self.index, self.x, self.y
        )

    # getter
    @property
    def name(self):
        return self.__name

    @property
    def index(self):
        return self.__index

    @property
    def x(self):
        return self.__x

    @property
    def y(self):
        return self.__y

    @property
    def color(self):
        return self.__color

    def draw(self, ax):
        """
        This plots the figure of this sensor.

        Parameters
        ----------
        ax : matplotlib.pyplot.gca
            Current axes on this figure.
        """
        pass

    def save_text(self):
        """
        This returns string that represents the information of this sensor.

        Returns
        -------
        ret : str
            Text of this sensor.
        """
        pass


class CircularPIRSensor(Sensor):
    """
    Class of a circuler passive infrared (PIR) sensor.

    Attributes
    ----------
    name : str
        Name of the sensor.
    index : int
        Index of this sensor.
    x : float
        A x-axis value of 2D coordinate.
    y : float
        A y-axis value of 2D coordinate.
        The (x, y) is the center point of this sensor.
    color : str
        Color code that is used to plot a figure.
    radius : float
        Radius [cm] of this sensor.
    type_name : str
        Name of this class.
    """

    type_name = "PIR"

    def __init__(self, name, index, x, y, color, radius):
        """
        Initialize of an object of CircularPIRSensor.

        Parameters
        ----------
        Same with parameters of Attributes.

        Raises
        ------
        1. radius < 0
        """
        super().__init__(name, index, x, y, color)
        if radius < 0:
            raise ValueError(
                "The radius of a CircluarPIRSensor must be positive value."
            )
        self.__radius = radius

    # getter
    @property
    def radius(self):
        return self.__radius

    def collide(self, xy, body_radius=0):
        """
        This function calculates whether the resident collides with the range of a sensor.
        The sensor has its circle range.

        Parameters
        ----------
        xy : tuple of float
            The (x,y) is a center of body.
        body_radius : float, default 0
            Radius[cm] of resident's body on floor.

        Returns
        -------
        is_collided : boolean
            Whether the resident is in the range of a sensor.
        """
        r = self.radius + body_radius
        return distance.euclidean(xy, (self.x, self.y)) < r

    def is_activated(self, p, past_p, body_radius, min_d):
        """
        This function determines how this sensor is activated.

        Parameters
        ----------
        p : tuple of float
            Present 2d coordinate (x,y) of the resident.
        past_p : tuple of float
            Past 2d coordinate (x,y) of the resident.
        body_radius : float, default 0
            Radius [cm] of resident's body on floor.
        min_d : float
            Minimum distance between p and past_p.
            If the distance between p and past_p is shorter than min_d, then the sensor is not activated.

        Returns
        -------
        is_activated : boolean
            Whether this sensor is activated.
        """
        # If the resident moves shorter than min_d [cm], then PIR sensor does not be activated.
        if distance.euclidean(p, past_p) <= min_d:
            return False
        else:
            return self.collide(p, body_radius=body_radius)


    def draw(self, ax):
        """
        This plots the figure of this sensor.

        Parameters
        ----------
        ax : matplotlib.pyplot.gca
            Current axes on this figure.
        """
        circle = patches.Circle(
            xy=(self.x, self.y),
            radius=self.radius,
            fc=self.color,
            ec=self.color,
            alpha=0.2,
        )
        ax.add_patch(circle)
        ax.text(
            self.x,
            self.y,
            "{}".format(self.index),
            fontsize=10,
            va="center",
            ha="center",
        )

    def canvas(self, canvas, point2canvas):
        """
        This draws the figure of this sensor on the tkinter.Canvas.

        Parameters
        ----------
        canvas : tkinter.Canvas
        point2canvas : function
            This converts a coordinate in the layout into the coordinate on the canvas.
            Parameters
            ----------
            xy : tuple of float
                xy = (x, y) is the target point in the layout.
            Returns
            -------
            ret : tuple of float
                ret = (xx, yy) is the calculated coordinate on the canvas.
        Returns
        -------
        _id : int
            The id of the drawn object.
        """
        top_left = point2canvas((self.x - self.radius, self.y + self.radius))
        bottom_right = point2canvas((self.x + self.radius, self.y - self.radius))
        return canvas.create_oval(
            top_left[0],
            top_left[1],
            bottom_right[0],
            bottom_right[1],
            outline=self.color,
        )

    def save_text(self):
        """
        This returns string that represents the information of this sensor.

        Returns
        -------
        ret : str
            Text of this sensor.
        """
        return "name = {}, index = {}, center coordinates = ({}, {}), radius = {}[cm]".format(
            self.name, self.index, self.x, self.y, self.radius
        )


class SquarePressureSensor(Sensor):
    """
    Class of a square pressure sensor.

    Attributes
    ----------
    name : str
        Name of the sensor.
    index : int
        Index of this sensor.
    x : float
        A x-axis value of 2D coordinate.
    y : float
        A y-axis value of 2D coordinate.
        The (x, y) is a bottom left point of this sensor.
    color : str
        Color code that is used to plot a figure.
    vertical : float
        Length[cm] of the vertical edge.
    horizontal ; float
        Length[cm] of the horizontal edge.
    angle : float
        Angle[deg] of the sensor.
        This takes a positive value in clockwise direction.
        The center of rotation is (x, y).
    type_name : str
        Name of this class.
    """

    type_name = "pressure"

    def __init__(self, name, index, x, y, color, vertical, horizontal, angle):
        """
        Initialize of an object of SquarePressureSensor class.

        Parameters
        ----------
        Same with the Attributes.

        Notes
        -----
        The point of (x, y) is a bottom left point of the pressure sensor.
        The sensor rotates with (x,y) as the center.

        Raises
        ------
        1. vertical < 0
        2. horizontal < 0
        """
        super().__init__(name, index, x, y, color)
        if (vertical < 0) or (horizontal < 0):
            raise ValueError(
                "The vertical length and horizontal length of CircularSensor must be positive value."
            )
        self.__vertical = vertical
        self.__horizontal = horizontal
        self.__angle = angle

    # getter
    @property
    def vertical(self):
        return self.__vertical

    @property
    def horizontal(self):
        return self.__horizontal

    @property
    def angle(self):
        return self.__angle

    def cover(self, xy):
        """
        This function calculates whether the target point is in the range of a sensor.
        The sensor has its rectangle range with the angle.

        Parameters
        ----------
        xy : tuple of float
            The (x,y) is a point of a target.

        Returns
        -------
        is_covered : boolean
            Whether the target point is in the range of a sensor.

        Example
        -------
            The following figure illustrates indexes of vertexes.
            The index of self.center is 1.
            4---------3   y
            |         |   ^
            |         |   |
            1---------2   --> x
        """
        temp_xy = xy
        if self.angle != 0:
            temp_xy = self.rotate(temp_xy, (self.x, self.y), -self.angle)
        (x, y) = temp_xy
        return (
            self.x <= x <= self.x + self.horizontal
            and self.y <= y <= self.y + self.vertical
        )

    def collide(self, xy, body_radius):
        """
        This function calculates whether the xy with body_radius collides with the range of a sensor.
        The sensor has its rectangle range with the angle.
        The body is represented as a circle whose center is xy and whose radius is body_radius.

        Parameters
        ----------
        xy : tuple of float
            The (x,y) is a center of the body.

        Returns
        -------
        is_collided : boolean
            Whether the resident collides with the range of the sensor.
        body_radius : float
            Radius[cm] of resident's body on floor.
        """
        temp_xy = xy
        if self.angle != 0:
            temp_xy = self.rotate(temp_xy, (self.x, self.y), -self.angle)
        (x, y) = temp_xy
        flag1 = (
            self.x <= x <= self.x + self.horizontal
            and self.y - body_radius <= y <= self.y + self.vertical + body_radius
        )
        flag2 = (
            self.x - body_radius <= x <= self.x + self.horizontal + body_radius
            and self.y <= y <= self.y + self.vertical
        )
        flag3 = distance.euclidean(temp_xy, (self.x, self.y)) < body_radius
        flag4 = (
            distance.euclidean(temp_xy, (self.x + self.horizontal, self.y))
            < body_radius
        )
        flag5 = (
            distance.euclidean(
                temp_xy, (self.x + self.horizontal, self.y + self.vertical)
            )
            < body_radius
        )
        flag6 = (
            distance.euclidean(temp_xy, (self.x, self.y + self.vertical)) < body_radius
        )
        return flag1 or flag2 or flag3 or flag4 or flag5 or flag6

    @staticmethod
    def rotate(xy, center, angle):
        """
        This rotates a point in 2d coordinate.

        Parameters
        ----------
        xy : tuple of float
            2d coordinates of a target point.
        center : tuple of float
            2d coordinates of the center of this rotation.
        angle : float
            Angle[deg] to rotate.

        Returns
        -------
        rotated_point : tuple of float
            Rotated 2d coordinates.
        """
        theta = np.deg2rad(angle)
        temp_xy = np.array(xy) - np.array(center)
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        rotated_temp_xy = np.dot(R, temp_xy)
        return tuple(rotated_temp_xy + center)

    @staticmethod
    def in_convex_polygon(self, xy, vertex):
        """
        Determine whether a point is in the range of a convex polygon.

        Parameters
        ----------
        xy : tuple of float
            2d coordinate of the target point.
        vertex : list of tuple of float
            List of the vertex of the rectangle, the order must be one-way (the rectangle can be drawn by connecting each vertex in order and connecting the last point and the first point).

        Returns
        -------
        is_in : boolean
            Whether the target point is in (or on the border of) the convex polygon.

        Notes
        -----
        The signs of cross products are used for this collision detection.
        """

        vertex = [np.array(x) for x in vertex]
        xy = np.array(xy)
        n = len(vertex)
        vectors, xy_vectors = [], []
        for i, v in enumerate(vertex):
            if i == n - 1:
                vectors.append(vertex[0] - vertex[n - 1])
            else:
                vectors.append(vertex[i + 1] - vertex[i])
            xy_vectors.append(xy - vertex[i])
        cross = [a[0] * b[1] - a[1] * b[0] for (a, b) in zip(vectors, xy_vectors)]
        sign = None
        for c in cross:
            if c == 0:
                return True
            if sign is None:
                sign = c > 0
            if sign != (c > 0):
                return False
        return True

    def draw(self, ax):
        """
        This plots the figure of this sensor.

        Parameters
        ----------
        ax : matplotlib.pyplot.gca
            Current axes on this figure.
        """
        rectangle = patches.Rectangle(
            xy=(self.x, self.y),
            width=self.horizontal,
            height=self.vertical,
            angle=self.angle,
            fc=self.color,
            ec=self.color,
            alpha=0.2,
        )
        ax.add_patch(rectangle)
        x, y = self.x + self.horizontal / 2, self.y + self.vertical / 2
        if self.angle != 0:
            (x, y) = self.rotate((x, y), (self.x, self.y), self.angle)
        ax.text(x, y, "{}".format(self.index), fontsize=10, va="center", ha="center")

    def canvas(self, canvas, point2canvas):
        """
        This draws the figure of this sensor on the tkinter.Canvas.

        Parameters
        ----------
        canvas : tkinter.Canvas
        point2canvas : function
            This converts a coordinate in the layout into the coordinate on the canvas.
            Parameters
            ----------
            xy : tuple of float
                xy = (x, y) is the target point in the layout.
            Returns
            -------
            ret : tuple of float
                ret = (xx, yy) is the calculated coordinate on the canvas.
        Returns
        -------
        _id : int
            The id of the drawn object.
        """
        center = (self.x, self.y)
        top_left = point2canvas(
            self.rotate((self.x, self.y + self.vertical), center, self.angle)
        )
        top_right = point2canvas(
            self.rotate(
                (self.x + self.horizontal, self.y + self.vertical), center, self.angle
            )
        )
        bottom_left = point2canvas((self.x, self.y))
        bottom_right = point2canvas(
            self.rotate((self.x + self.horizontal, self.y), center, self.angle)
        )
        return canvas.create_polygon(
            top_left[0],
            top_left[1],
            top_right[0],
            top_right[1],
            bottom_right[0],
            bottom_right[1],
            bottom_left[0],
            bottom_left[1],
            fill="",
            outline=self.color,
        )

    def save_text(self):
        """
        This returns string that represents the information of this sensor.

        Returns
        -------
        ret : str
            Text of this sensor.
        """
        return "name = {}, index = {}, bottom left coordinates = ({}, {}), vertical = {}[cm], horizontal = {}[cm], angle = {}[deg]".format(
            self.name,
            self.index,
            self.x,
            self.y,
            self.vertical,
            self.horizontal,
            self.angle,
        )


class CostSensor(Sensor):
    """
    Class of a cost sensor.

    Attributes
    ----------
    name : str
        Name of the sensor.
    index : int
        Index of this sensor.
    x : float
        A x-axis value of 2D coordinate.
    y : float
        A y-axis value of 2D coordinate.
        The (x, y) is a bottom left point of this sensor.
    color : str
        Color code that is used to plot a figure.
    type_name : str
        Name of this class.
    """

    type_name = "cost"

    def __init__(self, name, index, x, y, color):
        """
        Initialize of an object of CostSensor class.

        Parameters
        ----------
        Same with the Attributes.

        Notes
        -----
        The point of (x, y) is the center of the sensor.
        """
        super().__init__(name, index, x, y, color)

    def draw(self, ax):
        """
        This plots the figure of this sensor.

        Parameters
        ----------
        ax : matplotlib.pyplot.gca
            Current axes on this figure.
        """
        ax.scatter(
            self.x,
            self.y,
            c=self.color,
            marker="*",
            alpha=0.6,
            s=(plt.rcParams["lines.markersize"] ** 2) * 4,
        )
        ax.text(
            self.x,
            self.y,
            "{}".format(self.index),
            fontsize=10,
            va="center",
            ha="center",
        )

    def canvas(self, canvas, point2canvas):
        """
        This draws the figure of this sensor on the tkinter.Canvas.

        Parameters
        ----------
        canvas : tkinter.Canvas
        point2canvas : function
            This converts a coordinate in the layout into the coordinate on the canvas.
            Parameters
            ----------
            xy : tuple of float
                xy = (x, y) is the target point in the layout.
            Returns
            -------
            ret : tuple of float
                ret = (xx, yy) is the calculated coordinate on the canvas.
        Returns
        -------
        _id : int
            The id of the drawn object.
        """
        size = 10
        top_left = point2canvas((self.x - size / 2, self.y + size / 2))
        bottom_right = point2canvas((self.x + size / 2, self.y - size / 2))
        return canvas.create_rectangle(
            top_left[0],
            top_left[1],
            bottom_right[0],
            bottom_right[1],
            outline=self.color,
        )

    def save_text(self):
        """
        This returns string that represents the information of this sensor.

        Returns
        -------
        ret : str
            Text of this sensor.
        """
        return "name = {}, index = {}, center coordinate = ({}, {})".format(
            self.name, self.index, self.x, self.y
        )


class DoorSensor(Sensor):
    """
    Class of a door sensor.

    Attributes
    ----------
    name : str
        Name of the sensor.
    index : int
        Index of this sensor.
    x : float
        A x-axis value of 2D coordinate.
    y : float
        A y-axis value of 2D coordinate.
        The (x, y) is a bottom left point of this sensor.
    color : str
        Color code that is used to plot a figure.
    type_name : str
        Name of this class.
    door_name : str
        Name of the door.
    """

    type_name = "door"

    def __init__(self, name, index, x, y, color, door_name):
        """
        Initialize of an object of SquarePressureSensor class.

        Parameters
        ----------
        Same with the Attributes.

        Notes
        -----
        The point of (x, y) is the center of the sensor.
        """
        super().__init__(name, index, x, y, color)
        self.door_name = door_name

    def draw(self, ax):
        """
        This plots the figure of this sensor.

        Parameters
        ----------
        ax : matplotlib.pyplot.gca
            Current axes on this figure.
        """
        ax.scatter(
            self.x,
            self.y,
            c=self.color,
            marker="^",
            alpha=0.5,
            s=(plt.rcParams["lines.markersize"] ** 2) * 4,
        )
        ax.text(
            self.x,
            self.y,
            "{}".format(self.index),
            fontsize=10,
            va="center",
            ha="center",
        )

    def canvas(self, canvas, point2canvas):
        """
        This draws the figure of this sensor on the tkinter.Canvas.

        Parameters
        ----------
        canvas : tkinter.Canvas
        point2canvas : function
            This converts a coordinate in the layout into the coordinate on the canvas.
            Parameters
            ----------
            xy : tuple of float
                xy = (x, y) is the target point in the layout.
            Returns
            -------
            ret : tuple of float
                ret = (xx, yy) is the calculated coordinate on the canvas.
        Returns
        -------
        _id : int
            The id of the drawn object.
        """
        size = 10
        top_left = point2canvas((self.x - size / 2, self.y + size / 2))
        bottom_right = point2canvas((self.x + size / 2, self.y - size / 2))
        return canvas.create_rectangle(
            top_left[0],
            top_left[1],
            bottom_right[0],
            bottom_right[1],
            outline=self.color,
        )

    def save_text(self):
        """
        This returns string that represents the information of this sensor.

        Returns
        -------
        ret : str
            Text of this sensor.
        """
        return "name = {}, index = {}, center coordinate = ({}, {})".format(
            self.name, self.index, self.x, self.y
        )


def check_indexes_of_sensors(sensors):
    """
    Check whether indexes of sensors are different from each other.

    Parameters
    ----------
    sensors : list of sensor_model.Sensor
        Target sensors.
    """
    indexes = []
    for s in sensors:
        if s.index not in indexes:
            indexes.append(s.index)
        else:
            raise ValueError(
                "Some indexes of sensors are same. Indexes of sensors must be different from each other."
            )


# test sensor model of test_layout
test_sensors = [
    CircularPIRSensor("normal PIR", 0, 150, 150, constants.LIGHT_RED, 50),
    CircularPIRSensor("normal PIR", 1, 400, 120, constants.LIGHT_RED, 50),
    CircularPIRSensor("normal PIR", 2, 100, 55, constants.LIGHT_RED, 50),
    CircularPIRSensor("normal PIR", 3, 200, 55, constants.LIGHT_RED, 50),
    CircularPIRSensor("normal PIR", 4, 300, 40, constants.LIGHT_RED, 50),
    CircularPIRSensor("normal PIR", 5, 400, 40, constants.LIGHT_RED, 50),
    CircularPIRSensor("normal PIR", 6, 100, -40, constants.LIGHT_RED, 50),
    CircularPIRSensor("normal PIR", 7, 200, -40, constants.LIGHT_RED, 50),
    CircularPIRSensor("normal PIR", 8, 450, -40, constants.LIGHT_RED, 50),
    CircularPIRSensor("normal PIR", 9, 550, -40, constants.LIGHT_RED, 50),
    CircularPIRSensor("normal PIR", 10, 650, -40, constants.LIGHT_RED, 50),
    CircularPIRSensor("normal PIR", 11, -10, -70, constants.LIGHT_RED, 50),
    CircularPIRSensor("normal PIR", 12, 200, -140, constants.LIGHT_RED, 50),
    CircularPIRSensor("normal PIR", 13, 300, -140, constants.LIGHT_RED, 50),
    CircularPIRSensor("normal PIR", 14, 400, -140, constants.LIGHT_RED, 50),
    CircularPIRSensor("normal PIR", 15, 500, -140, constants.LIGHT_RED, 50),
    CircularPIRSensor("normal PIR", 16, 610, -140, constants.LIGHT_RED, 50),
    CircularPIRSensor("normal PIR", 17, 670, 70, constants.LIGHT_RED, 50),
    CircularPIRSensor("normal PIR", 18, 750, 80, constants.LIGHT_RED, 50),
    CircularPIRSensor("normal PIR", 19, 850, 90, constants.LIGHT_RED, 50),
    CircularPIRSensor("normal PIR", 20, 850, 10, constants.LIGHT_RED, 50),
    CircularPIRSensor("normal PIR", 21, 920, 50, constants.LIGHT_RED, 50),
    SquarePressureSensor("vertical oblong pressure", 22, -30, 0, constants.LIGHT_GRAY, 200, 50, 0),
    SquarePressureSensor(
        "horizontal oblong pressure", 23, -160, -60, constants.LIGHT_GRAY, 50, 150, 0
    ),
    CostSensor(constants.FLOW_BATHROOM, 24, 600, -310, constants.BLUE),
    CostSensor(constants.FLOW_KITCHEN, 25, 990, 60, constants.BLUE),
    CostSensor(constants.POWER_TV, 26, 320, 160, constants.YELLOW),
    CostSensor(constants.POWER_KITCHEN, 27, 985, 180, constants.YELLOW),
    DoorSensor("door", 28, 195, 205, constants.LIGHT_GREEN, "Entrance"),
]

test_sensors2 = [
    CircularPIRSensor("normal PIR", 0,  60, 140, constants.LIGHT_RED, 60),
    CircularPIRSensor("normal PIR", 1, 110, 140, constants.LIGHT_RED, 60),
    CircularPIRSensor("normal PIR", 2, 160, 140, constants.LIGHT_RED, 60),
    CircularPIRSensor("normal PIR", 3, 210, 140, constants.LIGHT_RED, 60),
    CircularPIRSensor("normal PIR", 4,  60,  90, constants.LIGHT_RED, 60),
    CircularPIRSensor("normal PIR", 5, 110,  90, constants.LIGHT_RED, 60),
    CircularPIRSensor("normal PIR", 6, 160,  90, constants.LIGHT_RED, 60),
    CircularPIRSensor("normal PIR", 7, 210,  90, constants.LIGHT_RED, 60),

    
    CircularPIRSensor("normal PIR", 71, 400, 120, constants.LIGHT_RED, 50),
    CircularPIRSensor("normal PIR", 72, 100, 55, constants.LIGHT_RED, 50),
    CircularPIRSensor("normal PIR", 73, 200, 55, constants.LIGHT_RED, 50),
    CircularPIRSensor("normal PIR", 74, 300, 40, constants.LIGHT_RED, 50),
    CircularPIRSensor("normal PIR", 75, 400, 40, constants.LIGHT_RED, 50),
    CircularPIRSensor("normal PIR", 76, 100, -40, constants.LIGHT_RED, 50),
    CircularPIRSensor("normal PIR", 77, 200, -40, constants.LIGHT_RED, 50),
    CircularPIRSensor("normal PIR", 78, 450, -40, constants.LIGHT_RED, 50),
    CircularPIRSensor("normal PIR", 79, 550, -40, constants.LIGHT_RED, 50),
    CircularPIRSensor("normal PIR", 80, 650, -40, constants.LIGHT_RED, 50),
    CircularPIRSensor("normal PIR", 81, -10, -70, constants.LIGHT_RED, 50),
    CircularPIRSensor("normal PIR", 82, 200, -140, constants.LIGHT_RED, 50),
    CircularPIRSensor("normal PIR", 83, 300, -140, constants.LIGHT_RED, 50),
    CircularPIRSensor("normal PIR", 84, 400, -140, constants.LIGHT_RED, 50),
    CircularPIRSensor("normal PIR", 85, 500, -140, constants.LIGHT_RED, 50),
    CircularPIRSensor("normal PIR", 86, 610, -140, constants.LIGHT_RED, 50),
    CircularPIRSensor("normal PIR", 87, 670, 70, constants.LIGHT_RED, 50),
    CircularPIRSensor("normal PIR", 88, 750, 80, constants.LIGHT_RED, 50),
    CircularPIRSensor("normal PIR", 89, 850, 90, constants.LIGHT_RED, 50),
    CircularPIRSensor("normal PIR", 90, 850, 10, constants.LIGHT_RED, 50),
    CircularPIRSensor("normal PIR", 91, 920, 50, constants.LIGHT_RED, 50),
    CircularPIRSensor("normal PIR", 92, 930, 55, constants.LIGHT_RED, 50),
    SquarePressureSensor("vertical oblong pressure", 100, -30, 0, constants.LIGHT_GRAY, 200, 80, 0),
    SquarePressureSensor(
        "horizontal oblong pressure", 101, -160, -90, constants.LIGHT_GRAY, 80, 150, 0
    ),
    CostSensor(constants.FLOW_BATHROOM, 201, 600, -310, constants.BLUE),
    CostSensor(constants.FLOW_KITCHEN, 202, 990, 60, constants.BLUE),
    CostSensor(constants.POWER_TV, 203, 320, 160, constants.YELLOW),
    CostSensor(constants.POWER_KITCHEN, 204, 985, 180, constants.YELLOW),
    DoorSensor("door", 301, 195, 205, constants.LIGHT_GREEN, "Entrance"),
]

check_indexes_of_sensors(test_sensors)
check_indexes_of_sensors(test_sensors2)


def duration_time_in_activity(mean, std, log_scale = False):
    """
    Sampling duration time of sensor activation in an activity.
    This follows a log normal distribution.
    The time units are seconds.

    Parameters
    ----------
    mean : float
        Mean.
    std : float
        Standard deviantion.
    scale : bool, default False
        Whether the parameter is log scale.
        False : mean of the data
        True : mean of the log(data)

    Returns
    -------
    seconds : float
        Duration time.
    """
    if not(log_scale):
        mean2 = np.log(mean**2/np.sqrt(mean**2+std**2))
        std2 = np.sqrt(np.log(1 + std**2/mean**2))
        mean, std = mean2, std2
    return np.random.lognormal(mean, std)


def interval_time_in_activity(mean):
    """
    Sampling interval time between sensor activations in an activity.
    This follows an exponential distribution.
    The time units are seconds.

    Parameters
    ----------
    mean : float
        Mean of the data.

    Returns
    -------
    seconds : float
        Duration time.
    """
    # scale = 1/lambda
    return np.random.exponential(scale = mean)