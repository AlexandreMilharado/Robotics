import numpy as np
from controller import DistanceSensor, Motor

WHEEL_DISTANCE = 0.112            # Distance between wheels in Thymio Robot

class Agent:
# Inits
    def __init__(self, supervisor, timestep):
        """
        Class constructor.

        Parameters
        ----------
        supervisor : Supervisor
            The simulation supervisor.
        timestep : int
            The simulation timestep.

        Attributes
        ----------
        rotation : Field
            The supervisor field for the robot's rotation.
        translation : Field
            The supervisor field for the robot's translation.
        sensors : list
            List of distance sensors.
        left_motor : Motor
            The left wheel motor.
        right_motor : Motor
            The right wheel motor.
        """

        self.rotation = supervisor.getFromDef("ROBOT").getField("rotation")             
        self.translation = supervisor.getFromDef("ROBOT").getField("translation")

        self._init_sensors(supervisor, timestep)
        self._init_motors(supervisor)

    def _init_sensors(self, supervisor, timestep):
        """
        Inits the distance sensors.

        Parameters
        ----------
        supervisor : Supervisor
            The simulation supervisor.
        timestep : int
            The simulation timestep.

        Attributes
        ----------
        sensors : list
            List of distance sensors.
        """
        sensors = ["prox.horizontal.0", "prox.horizontal.1", # Left
                   "prox.horizontal.2",                      # Middle
                   "prox.horizontal.3", "prox.horizontal.4", # Right
                   "prox.horizontal.5", "prox.horizontal.6", # Back
                   "prox.ground.0", "prox.ground.1"]         # Vertical

        self.sensors : list[DistanceSensor] = []

        for sensor_name in sensors:
            sensor : DistanceSensor = supervisor.getDevice(sensor_name)
            sensor.enable(timestep)
            self.sensors.append(sensor)

    def _init_motors(self, supervisor):
        """
        Initializes the motors by retrieving them from the supervisor, setting their positions 
        to infinity to enable velocity control, and initializing their velocities to zero.

        Parameters
        ----------
        supervisor : Supervisor
            The simulation supervisor used to access the robot's devices.

        Attributes
        ----------
        left_motor : Motor
            The left motor of the robot.
        right_motor : Motor
            The right motor of the robot.
        """
        self.left_motor : Motor = supervisor.getDevice('motor.left')      
        self.left_motor.setPosition(float('inf'))
        
        self.right_motor : Motor = supervisor.getDevice('motor.right')
        self.right_motor.setPosition(float('inf'))

        self.left_motor.setVelocity(0)
        self.right_motor.setVelocity(0)


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
        return (self.sensors[-2].getValue(), self.sensors[-1].getValue())

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
        return any(value > max_limit for value in self._get_horizontal_sensors())

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

    def reset(self, rotation = [0, 0, 1, 2], translation = [0, 0, 0]):
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
