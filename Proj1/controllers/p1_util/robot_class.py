from math import cos, sin
import numpy as np
from controller import Supervisor, DistanceSensor, Motor

WHEEL_DISTANCE = 0.112            # Distance between wheels in Thymio Robot

class SimulationEndedError(Exception):
    """Custom exception to indicate that the simulation has ended."""
    pass


class Agent:
# Inits
    def __init__(self, SENSOR_TYPE, timestep_multiplier):
        if SENSOR_TYPE == "SIMPLE":
            self.read_sensors = self.get_ground_sensors_values
            self.run_step = self._run_step_braiternberg
        else:
            self.read_sensors = self.get_frontal_sensors_values
        
        self.supervisor : Supervisor = Supervisor()
        self.timestep = int(self.supervisor.getBasicTimeStep() * timestep_multiplier)
    
        self.rotation = self.supervisor.getFromDef("ROBOT").getField("rotation")             
        self.translation = self.supervisor.getFromDef("ROBOT").getField("translation")

        self._init_sensors()
        self._init_motors()

        self._init_black_line()

    def _init_sensors(self):
        sensors = ["prox.horizontal.0", "prox.horizontal.1", # Left
                   "prox.horizontal.2",                      # Middle
                   "prox.horizontal.3", "prox.horizontal.4", # Right
                   "prox.horizontal.5", "prox.horizontal.6", # Back
                   "prox.ground.0", "prox.ground.1"]         # Vertical

        self.sensors : list[DistanceSensor] = []

        for sensor_name in sensors:
            sensor : DistanceSensor = self.supervisor.getDevice(sensor_name)
            sensor.enable(self.timestep)
            self.sensors.append(sensor)

    def _init_motors(self):
        self.left_motor : Motor = self.supervisor.getDevice('motor.left')      
        self.left_motor.setPosition(float('inf'))
        
        self.right_motor : Motor = self.supervisor.getDevice('motor.right')
        self.right_motor.setPosition(float('inf'))

        self.left_motor.setVelocity(0)
        self.right_motor.setVelocity(0)

    def _init_black_line(self):
        self.black_line = []
        children = self.supervisor.getRoot().getField('children')

        for i in range(children.getCount()):
            node = children.getMFNode(i)
            field = node.getField("name")
            if field and node.getTypeName() == "Solid" and "BlackArea" in field.getSFString():
                translation = node.getField("translation").getSFVec3f()
                rotation = node.getField("rotation").getSFVec3f()
                rotation = rotation[-2] * rotation[-1]

                children_field = node.getField("children")
                for i in range(children_field.getCount()):
                    child_node = children_field.getMFNode(i)
                    if child_node.getTypeName() == "Shape":
                        geometry_node = child_node.getField("geometry").getSFNode()
                        if geometry_node and geometry_node.getTypeName() == "Box":
                            size = geometry_node.getField("size").getSFVec3f()
                
                self.black_line.append((translation, size, rotation))

# Readings
    def get_frontal_sensors_values(self):
        """
        Returns the values of the left, center and right frontal sensors.

        Returns
        -------
        tuple
            A tuple containing the values of the left, center and right frontal sensors.
        """
        return (self.sensors[0].getValue(), self.sensors[2].getValue(), self.sensors[4].getValue())

    def get_ground_sensors_values(self):
        """
        Returns the values of the left and right ground sensors.

        Returns
        -------
        tuple
            A tuple containing the values of the left and right ground sensors.
        """
        return (self.is_not_on_black_line(self.sensors[-2].getValue()), self.is_not_on_black_line(self.sensors[-1].getValue()))

    def _get_horizontal_sensors(self):
        """
        Returns the values of the seven horizontal sensors.

        Returns
        -------
        list
            A list containing the values of the seven horizontal sensors.
        """
        return [sensor.getValue() for sensor in self.sensors[:7]]

    def get_average_velocity(self):
        """
        Returns the average velocity of the two wheels.

        Returns
        -------
        float
            The average velocity of the two wheels.
        """
        return (self.left_motor.getVelocity() + self.right_motor.getVelocity()) / 2

    def get_angular_velocity(self):
        """
        Returns the angular velocity of the robot.

        Returns
        -------
        float
            The angular velocity of the robot.
        """
        v_left = self.left_motor.getVelocity()
        v_right = self.right_motor.getVelocity()

        return (v_right - v_left) / WHEEL_DISTANCE
    
    def get_max_velocity(self):
        """
        Returns the maximum velocity of the left motor.

        Returns
        -------
        float
            The maximum velocity of the left motor.
        """
        return self.left_motor.getMaxVelocity()

# State
    def is_on_black_line_map(self):
        def is_point_in_rotated_rectangle(px, py, cx, cy, width, height, angle):
            tx, ty = px - cx, py - cy

            cos_a = cos(-angle)
            sin_a = sin(-angle)

            rx = tx * cos_a - ty * sin_a
            ry = tx * sin_a + ty * cos_a

            return abs(rx) <= width / 2 and abs(ry) <= height / 2

        current_position = self.supervisor.getSelf().getPosition()[:2]
        for translation, size, angle in self.black_line:
            if is_point_in_rotated_rectangle(current_position[0], current_position[1],
                                            translation[0], translation[1],
                                            size[0], size[1],
                                            angle):
                return True
            
        return False

    def collided(self, max_limit = 4300):
        """
        Checks if any of the horizontal sensors have detected an obstacle.

        Parameters
        ----------
        max_limit : int, optional
            The threshold value for sensor detection. If any sensor's value exceeds this limit, a collision is detected.
            Default is 4300.

        Returns
        -------
        bool
            True if any horizontal sensor's value exceeds the max_limit, indicating a collision. False otherwise.
        """
        return any(value >= max_limit for value in self._get_horizontal_sensors())

    def is_not_on_black_line(self, ground_sensor_value):
        """
        Checks if a ground sensor's value indicates that the robot is not on a black line.

        Parameters
        ----------
        ground_sensor_value : int
            The value of the ground sensor.

        Returns
        -------
        bool
            True if the value of the ground sensor indicates that the robot is not on a black line. False otherwise.
      """
        return (ground_sensor_value / 1023 - .6) / .2 > .3
    
    def _reset_velocity(self):
        """
        Resets the velocity of the left and right motors to 0.

        This is useful when the robot needs to be stopped or reset.
        """
        self.left_motor.setVelocity(0)
        self.right_motor.setVelocity(0)

    def get_rotation_translation(self):
        """
        Returns a random rotation and translation for the robot.

        Returns
        -------
        tuple
            A tuple containing a random rotation and translation for the robot.
        """
        return [0, 0, 1, np.random.uniform(0, 2 * np.pi)], [0, 0, 0]

    def reset_params(self, rotation = [0, 0, 1, 2], translation = [0, 0, 0]):
        """
        Resets the robot's rotation and translation to the given values.

        Parameters
        ----------
        rotation : list, optional
            A list of 4 floats representing the rotation of the robot in the form [x, y, z, angle].
            Default is [0, 0, 1, 2] which corresponds to a 2 radian rotation around the z-axis.
        translation : list, optional
            A list of 3 floats representing the translation of the robot in the form [x, y, z].
            Default is [0, 0, 0] which corresponds to no translation.
        """
        self.rotation.setSFRotation(rotation)
        self.translation.setSFVec3f(translation)
        self._reset_velocity()

    def reset(self):
        rotation, translation = self.get_rotation_translation()
        self.reset_params(rotation, translation)

        self.supervisor.simulationResetPhysics()


# Velocity
    def _limit_velocity(self, velocity, weights):
        """
        Limits the velocity of the motor to ensure that it doesn't exceed the maximum velocity.

        Parameters
        ----------
        velocity : float
            The velocity to limit.
        weights : list
            The weights of the sensors.

        Returns
        -------
        float
            The limited velocity.
        """
        return self.get_max_velocity() / (sum(weights) + 1) * velocity

    def set_velocity_left_motor(self, velocity, weights):
        """
        Sets the velocity of the left motor to the given value after limiting it to the maximum velocity allowed.

        Parameters
        ----------
        velocity : float
            The velocity to set.
        weights : list
            The weights of the sensors.
        """
        self.left_motor.setVelocity(self._limit_velocity(velocity, weights))

    def set_velocity_right_motor(self, velocity, weights):
        """
        Sets the velocity of the right motor to the given value after limiting it to the maximum velocity allowed.

        Parameters
        ----------
        velocity : float
            The velocity to set.
        weights : list
            The weights of the sensors.
        """
        self.right_motor.setVelocity(self._limit_velocity(velocity, weights))

# Movement
    def run_individual(self, individual):
        # Read Sensors
        sensors_inputs = self.read_sensors()

        # Control Motors
        self.run_step(individual.weights, sensors_inputs)

    def _run_step_braiternberg(self, weights, sensors_inputs):
        p_1_e, p_2_e, p_3_e, p_1_d, p_2_d, p_3_d = weights
        s_e, s_d = sensors_inputs

        left_speed =  s_e * p_1_e + s_d * p_2_e + p_3_e
        right_speed = s_e * p_1_d + s_d * p_2_d + p_3_d

        self.set_velocity_left_motor(left_speed, sensors_inputs)
        self.set_velocity_right_motor(right_speed, sensors_inputs)