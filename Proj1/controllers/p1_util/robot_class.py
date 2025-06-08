from math import cos, sin
import math
import numpy as np
from controller import Supervisor, DistanceSensor, Motor

WHEEL_DISTANCE = 0.112            # Distance between wheels in Thymio Robot
MAX_OMEGA = 9.53
MAX_SENSOR_FRONT = 4308.0
ANGLE_ACTIONS = [-0.6981, 0, 0.7854]
ROBOT_LENGTH = 0.112
ROBOT_WIDTH = 0.117

OBSTACLES_NUMBER = 15
OBSTACLES_MIN_RADIUS = 0.5
OBSTACLES_MAX_RADIUS = 1.4

RESET_MIN_RADIUS = 0
RESET_MAX_RADIUS = 1.5

class SimulationEndedError(Exception):
    """Custom exception to indicate that the simulation has ended."""
    pass


class Agent:
# Inits
    def __init__(self, INDIVIDUAL_TYPE, timestep_multiplier):

        """
        Initializes the agent with the given type and timestep multiplier.

        Parameters
        ----------
        INDIVIDUAL_TYPE : str
            The type of the individual. It can be "BRAITENBERG" or "NETWORKS_SIMPLE".
        timestep_multiplier : int
            The multiplier for the basic time step of the simulation.

        Sets the robot's sensors, motors, and reset function according to the type of the individual.
        """
        self.supervisor : Supervisor = Supervisor()
        self.timestep = int(self.supervisor.getBasicTimeStep() * timestep_multiplier)
        
        match INDIVIDUAL_TYPE:
            case "BRAITENBERG": 
                self.read_sensors = self._get_ground_sensors_values
                self.run_step = self._run_step_braiternberg
                self._limit_velocity = self._limit_velocity_braitenberg
                self.reset = self._reset_without_obstacles
            case "NETWORKS_SIMPLE":
                self.read_sensors = self._get_ground_sensors_values
                # self.read_sensors = self.get_frontal_sensors_values
                self.run_step = self._run_step_net
                self._limit_velocity = self._limit_velocity_net
                self.reset = self._reset_without_obstacles
                # self.reset = self._reset_with_obstacles
                # self._init_boxes()
            case _:
                self.read_sensors = self._get_frontal_and_ground_sensors_values
                self.run_step = self._run_step_complex_net
                self._limit_velocity = self._limit_velocity_net
                # self.reset = self._reset_without_obstacles
                self.reset = self._reset_with_obstacles
                self._init_boxes()
    
        self.rotation = self.supervisor.getFromDef("ROBOT").getField("rotation")             
        self.translation = self.supervisor.getFromDef("ROBOT").getField("translation")
        self.last_will_collide = False

        self._init_sensors()
        self._init_motors()

        self._init_black_line()


    def _init_sensors(self):
        """
        Initializes the sensors of the robot.

        Retrieves the sensors from the supervisor by name, enables them, and stores them in the self.sensors list.
        The sensors are the ones labeled "prox.horizontal.*" and "prox.ground.*".

        No return value.
        """
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
        """
        Initializes the motors of the robot.

        Retrieves the motors from the supervisor by name, sets their positions to infinity (no limit), and stores them in the self.left_motor and self.right_motor attributes.

        No return value.
        """
        self.left_motor : Motor = self.supervisor.getDevice('motor.left')      
        self.left_motor.setPosition(float('inf'))
        
        self.right_motor : Motor = self.supervisor.getDevice('motor.right')
        self.right_motor.setPosition(float('inf'))

        self.left_motor.setVelocity(0)
        self.right_motor.setVelocity(0)

    def _init_black_line(self):
        """
        Initializes the black line of the robot.

        Retrieves the black line from the supervisor by name, calculates the orientation of the black line in radians, and stores it in the self._black_line list.

        No return value.
        """
        self._black_line = []
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
                
                self._black_line.append((translation, size, rotation))

    def _init_boxes(self):
        """
        Initializes the obstacles of the robot.

        Retrieves the root of the supervisor, retrieves the children field of the root, and
        then creates a specified number of obstacles. Each obstacle is assigned a random
        position, orientation, length, and width within specified ranges. The obstacles are
        then added to the supervisor and stored in the self.obstacles list.

        No return value.
        """
        root = self.supervisor.getRoot()
        children_field = root.getField('children')
        self.obstacles = []

        for i in range(OBSTACLES_NUMBER):
            position = self._random_position(OBSTACLES_MIN_RADIUS, OBSTACLES_MAX_RADIUS, 0)
            orientation = self._random_orientation()
            length = np.random.uniform(0.05, 0.2)
            width = np.random.uniform(0.05, 0.2)
            
            box_string = f"""
            DEF WHITE_BOX_{i} Solid {{
            translation {position[0]} {position[1]} {position[2]}
            rotation {orientation[0]} {orientation[1]} {orientation[2]} {orientation[3]}
            physics Physics {{
                density 1000.0
            }}
            children [
                Shape {{
                appearance Appearance {{
                    material Material {{
                    diffuseColor 1 1 1
                    }}
                }}
                geometry Box {{
                    size {length} {width} 0.2  
                }}
                }}
            ]
            boundingObject Box {{
                size {length} {width} 0.2  
            }}
            }}
            """
            
            children_field.importMFNodeFromString(-1, box_string)

            self.obstacles.append(self.supervisor.getFromDef(f"WHITE_BOX_{i}"))

            

# Readings
    def get_frontal_sensors_values(self):
        """
        Returns the values of the left, middle and right frontal sensors.

        The values are normalized by the maximum sensor value.

        Returns
        -------
        tuple
            A tuple containing the normalized values of the left, middle and right frontal sensors.
        """
        return (self.sensors[0].getValue()/MAX_SENSOR_FRONT, self.sensors[2].getValue()/MAX_SENSOR_FRONT, self.sensors[4].getValue()/MAX_SENSOR_FRONT)

    def _get_ground_sensors_values(self):
        """
        Returns the values of the left and right ground sensors.

        Returns
        -------
        tuple
            A tuple containing the values of the left and right ground sensors.
        """
        return (self.is_not_on_black_line(self.sensors[-2].getValue()), self.is_not_on_black_line(self.sensors[-1].getValue()))
    
    def _get_frontal_and_ground_sensors_values(self):
        """
        Returns the combined values of the frontal and ground sensors.

        This method retrieves the normalized values of the left, middle, and right frontal sensors,
        and the values of the left and right ground sensors, concatenating them into a single tuple.

        Returns
        -------
        tuple
            A tuple containing the normalized values of the frontal sensors followed by the values of the ground sensors.
        """

        return self.get_frontal_sensors_values() + self._get_ground_sensors_values()

    def _get_horizontal_sensors(self):
        """
        Returns the values of the seven horizontal sensors.

        Returns
        -------
        list
            A list containing the values of the seven horizontal sensors.
        """
        return [sensor.getValue() for sensor in self.sensors[:5]] + [sensor.getValue() + 300 for sensor in self.sensors[5:-2]]

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
    # Utils
    def _random_orientation(self):                                      
        """
        Returns a random orientation quaternion.

        Returns
        -------
        list
            A quaternion [x, y, z, angle] representing a random orientation.
        """
        angle = np.random.uniform(0, 2 * np.pi)
        return [0, 0, 1, angle]

    def _random_position(self, min_radius, max_radius, z):                
        """
        Returns a random position as [x, y, z] coordinates within a specified
        radius range and at a specified z-coordinate.

        Parameters
        ----------
        min_radius : float
            Minimum radius of the random position.
        max_radius : float
            Maximum radius of the random position.
        z : float
            Z-coordinate of the random position.

        Returns
        -------
        list
            A list containing the x, y, and z coordinates of the random position.
        """
        
        radius = np.random.uniform(min_radius, max_radius)
        angle = self._random_orientation()
        x = radius * np.cos(angle[3])
        y = radius * np.sin(angle[3])
        return [x, y, z]
    
    # Black Line Reward
    def is_on_black_line_map(self):
        """
        Determines if the robot is currently on the black line within the environment.

        This function checks whether any corner of the robot's bounding box, defined by 
        its current position and dimensions, is inside any of the black line segments 
        specified in the environment. Each black line segment is defined by its translation, 
        size, and rotation angle. The function iterates over each corner of the robot and 
        uses a helper function to check for intersection with the rotated rectangles 
        representing the black lines.

        Returns
        -------
        bool
            True if the robot is on the black line, False otherwise.
        """

        def is_point_in_rotated_rectangle(px, py, cx, cy, width, height, angle):
            tx, ty = px - cx, py - cy

            cos_a = cos(-angle)
            sin_a = sin(-angle)

            rx = tx * cos_a - ty * sin_a
            ry = tx * sin_a + ty * cos_a

            return abs(rx) <= width / 2 and abs(ry) <= height / 2

        current_position = self.supervisor.getSelf().getPosition()[:2]

        half_length = ROBOT_LENGTH / 2 + 0.015
        half_width = ROBOT_WIDTH / 2 + 0.015
        corners = [
            (current_position[0] - half_length, current_position[1] - half_width),
            (current_position[0] + half_length, current_position[1] - half_width),
            (current_position[0] - half_length, current_position[1] + half_width),
            (current_position[0] + half_length, current_position[1] + half_width),
            ]
        for translation, size, angle in self._black_line:
            for corner in corners:
                if is_point_in_rotated_rectangle(corner[0], corner[1],
                                            translation[0], translation[1],
                                            size[0], size[1],
                                            angle):
                    return True
            
        return False

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
    
    # Colliding
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

    # 
    def _get_rotation_translation(self):
        """
        Returns a random orientation quaternion and the origin translation.

        Returns
        -------
        tuple
            A tuple containing a list of 4 floats representing the rotation quaternion
            and a list of 3 floats representing the translation, which is [0, 0, 0] in this
            case.
        """
        return self._random_orientation(), [0, 0, 0]

    # Reset
    def _reset_velocity(self):
        """
        Resets the velocity of the left and right motors to 0.

        This is useful when the robot needs to be stopped or reset.
        """
        self.left_motor.setVelocity(0)
        self.right_motor.setVelocity(0)

    def _reset_params(self, rotation = [0, 0, 1, 2], translation = [0, 0, 0]):
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
    
    def _reset_boxes(self):
        """
        Randomly resets the rotation and translation of each obstacle.

        Iterates through the list of obstacles and assigns a new random rotation and
        translation to each one. The random position is constrained within the minimum
        and maximum radius defined for obstacles.

        No return value.
        """

        for obs in self.obstacles:
            obs.getField("rotation").setSFRotation(self._random_orientation())
            obs.getField("translation").setSFVec3f(self._random_position(OBSTACLES_MIN_RADIUS, OBSTACLES_MAX_RADIUS, 0))

    def _reset_with_obstacles(self):
        """
        Resets the robot's position, orientation, and the positions of obstacles.

        This function randomizes and resets the robot's rotation and translation using
        specified methods for obtaining random orientations and translations. It also 
        resets the position and rotation of each obstacle in the environment to new random
        values. After updating these positions, the physics of the simulation is reset 
        to ensure that the changes take effect immediately.

        No return value.
        """

        rotation, translation = self._get_rotation_translation()
        self._reset_params(rotation, translation)
        # self._reset_params(self._random_orientation(), self._random_position(RESET_MIN_RADIUS, RESET_MAX_RADIUS, 0))
        self._reset_boxes()
        self.supervisor.simulationResetPhysics()

    def _reset_without_obstacles(self):
        """
        Resets the robot's position, orientation, and the physics of the simulation.

        This function randomizes and resets the robot's rotation and translation using
        specified methods for obtaining random orientations and translations. The physics
        of the simulation is then reset to ensure that the changes take effect immediately.

        No return value.
        """
        rotation, translation = self._get_rotation_translation()
        self._reset_params(rotation, translation)
        self.supervisor.simulationResetPhysics()

    def will_next_position_collide(self):
        """
        Checks if the robot's next position will result in a collision.

        This function retrieves the value of the last_will_collide variable, which represents
        whether the next position of the robot will result in a collision with an obstacle.

        Returns
        -------
        bool
            True if the next position will result in a collision, False otherwise.
        """
        return self.last_will_collide
    
    def update_sensors_past(self):
        """
        Updates the collision status based on the current frontal sensor values.

        This method checks the values of the frontal sensors to determine if the robot is 
        likely to collide in its next position. The result is stored in the `last_will_collide` 
        variable, which is set to `True` if any frontal sensor detects a potential obstacle, 
        indicating a possible collision.

        No parameters or returns.
        """

        self.last_will_collide = any(self.get_frontal_sensors_values())

# Velocity
    def _limit_velocity_braitenberg(self, velocity, weights):
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
    
    def _limit_velocity_net(self, velocity, weights):
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
        return self.get_max_velocity() * max(min(velocity, 1), -1)

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

    def _set_wheel_velocities(self, angle, velocity, sensors_inputs):
        """
        Sets the velocities of the left and right motors based on the given angle and velocity.

        Parameters
        ----------
        angle : float
            The angle to set the motors to.
        velocity : float
            The velocity to set the motors to.
        sensors_inputs : list
            The inputs from the sensors.

        Notes
        -----
        The velocities are calculated using the following formula:

        v_left = velocity - (WHEEL_DISTANCE / 2.0) * omega
        v_right = velocity + (WHEEL_DISTANCE / 2.0) * omega

        where omega is the angular velocity and WHEEL_DISTANCE is the distance between the wheels.
        """
        k_angle = MAX_OMEGA / math.pi
        omega = k_angle * angle

        v_left  = velocity - (WHEEL_DISTANCE / 2.0) * omega
        v_right = velocity + (WHEEL_DISTANCE / 2.0) * omega

        self.set_velocity_left_motor(v_left, sensors_inputs)
        self.set_velocity_right_motor(v_right, sensors_inputs)

# Movement
    def run_individual(self, individual):
        # Read Sensors
        """
        Runs the individual in the environment.

        Parameters
        ----------
        individual : Individual
            The individual to run.

        Notes
        -----
        This function reads the sensors of the robot, and then runs the individual's
        control function to control the motors of the robot.
        """
        sensors_inputs = self.read_sensors()

        # Control Motors
        self.run_step(individual, sensors_inputs)

    def _run_step_braiternberg(self, individual, sensors_inputs):
        """
        Runs the individual in the environment using the Braitenberg control algorithm.

        Parameters
        ----------
        individual : Individual
            The individual to run.
        sensors_inputs : list
            The inputs from the sensors.

        Notes
        -----
        This function takes the individual's weights and the sensors' inputs to control the motors of the robot.
        The Braitenberg control algorithm is used to calculate the left and right motor velocities based on the inputs.
        The velocities are then set to the left and right motors of the robot.
        """
        p_1_e, p_2_e, p_3_e, p_1_d, p_2_d, p_3_d = individual.weights
        s_e, s_d = sensors_inputs

        left_speed =  s_e * p_1_e + s_d * p_2_e + p_3_e
        right_speed = s_e * p_1_d + s_d * p_2_d + p_3_d

        self.set_velocity_left_motor(left_speed, sensors_inputs)
        self.set_velocity_right_motor(right_speed, sensors_inputs)

    def _run_step_net(self, individual, sensors_inputs):
        """
        Runs the individual in the environment using the simple neural network control algorithm.

        Parameters
        ----------
        individual : Individual
            The individual to run.
        sensors_inputs : list
            The inputs from the sensors.

        Notes
        -----
        This function takes the individual's neural network and the sensors' inputs to control the motors of the robot.
        The neural network is used to calculate the left and right motor velocities based on the inputs.
        The velocities are then set to the left and right motors of the robot.
        """
        left_speed, right_speed = individual.forward(sensors_inputs)

        self.set_velocity_left_motor(left_speed, sensors_inputs)
        self.set_velocity_right_motor(right_speed, sensors_inputs)

    def _run_step_complex_net(self, individual, sensors_inputs):
        """
        Runs the individual in the environment using the complex neural network control algorithm.

        Parameters
        ----------
        individual : Individual
            The individual to run.
        sensors_inputs : list
            The inputs from the sensors.

        Notes
        -----
        This function takes the individual's neural network and the sensors' inputs to control the motors of the robot.
        The neural network is used to calculate the left and right motor velocities based on the inputs.
        The velocities are then set to the left and right motors of the robot.
        """
        left_speed, right_speed = individual.forward(sensors_inputs)

        self.set_velocity_left_motor(left_speed, sensors_inputs)
        self.set_velocity_right_motor(right_speed, sensors_inputs)