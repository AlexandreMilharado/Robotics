import numpy as np
from controller import Supervisor
from controller import Robot, DistanceSensor, Motor
import random
import math

WHEEL_DISTANCE = 0.112 

class Agent:
    def __init__(self, supervisor, timestep):

        self.rotation = supervisor.getFromDef("ROBOT").getField("rotation")             
        self.translation = supervisor.getFromDef("ROBOT").getField("translation")

        self.init_sensors(supervisor, timestep)
        self.init_motors(supervisor)

    # Inits
    def init_sensors(self, supervisor, timestep):
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

    def init_motors(self, supervisor):
        self.left_motor : Motor = supervisor.getDevice('motor.left')      
        self.left_motor.setPosition(float('inf'))
        
        self.right_motor : Motor = supervisor.getDevice('motor.right')
        self.right_motor.setPosition(float('inf'))

        self.left_motor.setVelocity(0)
        self.right_motor.setVelocity(0)


    # Readings
    def get_frontal_sensors_values(self):
        return (self.sensors[0].getValue(), self.sensors[2].getValue(), self.sensors[4].getValue())

    def get_ground_sensors_values(self):
        return (self.sensors[-2].getValue(), self.sensors[-1].getValue())

    def _get_horizontal_sensors(self):
        return [sensor.getValue() for sensor in self.sensors[:7]]

    def get_average_velocity(self):
        return (self.left_motor.getVelocity() + self.right_motor.getVelocity()) / 2

    def get_angular_velocity(self):
        v_left = self.left_motor.getVelocity()
        v_right = self.right_motor.getVelocity()

        return (v_right - v_left) / WHEEL_DISTANCE
    
    def get_max_velocity(self):
        return self.left_motor.getMaxVelocity()

    # State
    def collided(self, max_limit = 4300):
        return any(value > max_limit for value in self._get_horizontal_sensors())

    def is_not_on_black_line(self, ground_sensor_value):
        return (ground_sensor_value / 1023 - .6) / .2 > .3
    
    def reset(self, seed=None, options=None):
        def get_translation(random_number):
            if random_number < 0.25:
                return [-0.650024, 0.10793, -0.00453284]
            elif random_number < 0.5:
                return [-0.029315, 0.529102, -0.00453284]
            elif random_number < 0.75:
                return [0.498481 -0.438083 -0.00453284]
            else:
                return [0, 0, 0]

        def get_rotation(random_number):
            if random_number < 0.25:
                return [-0.647181, 0.377113, -0.662527, 0.0126894]
            elif random_number < 0.5:
                return [0.00647158, 0.00170631, 0.999978, -1.57923]
            elif random_number < 0.75:
                return [-0.0006930518083526099 -0.005106128588018666 -0.9999867234768839 -2.3477953071795863]
            else:
                return [0, 0, 1, np.random.uniform(0, 2 * np.pi)]       
            
        random_number = 1
        if options:
            random_number = np.random.uniform(0, 1)

        self.rotation.setSFRotation(get_rotation(random_number))
        self.translation.setSFVec3f(get_translation(random_number))
        
        self.left_motor.setVelocity(0)
        self.right_motor.setVelocity(0)


    # Velocity
    def _limit_velocity(self, velocity, weights):
        return self.get_max_velocity() / (sum(weights) + 1) * velocity

    def set_velocity_left_motor(self, velocity, weights):
        self.left_motor.setVelocity(self._limit_velocity(velocity, weights))

    def set_velocity_right_motor(self, velocity, weights):
        self.right_motor.setVelocity(self._limit_velocity(velocity, weights))
