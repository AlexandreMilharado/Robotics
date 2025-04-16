import numpy as np
from controller import Supervisor
from controller import Robot, DistanceSensor, Motor
import random
import math

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
        return (sensor.getValue() for sensor in self.sensors[:7])

    def get_average_velocity(self):
        return (self.left_motor.getVelocity() + self.right_motor.getVelocity()) / 2

    # State
    def collided(self, max_limit = 4300):
        return any(value > max_limit for value in self._get_horizontal_sensors())

    def is_not_on_black_line(self, ground_sensor_value):
        return (ground_sensor_value / 1023 - .6) / .2 > .3
    
    def reset(self, seed=None, options=None):
        random_rotation = [0, 0, 1, np.random.uniform(0, 2 * np.pi)]

        self.rotation.setSFRotation(random_rotation)
        self.translation.setSFVec3f([0, 0, 0])
        
        self.left_motor.setVelocity(0)
        self.right_motor.setVelocity(0)


    # Velocity
    def _limit_velocity(self, velocity):
        return self.left_motor.getMaxVelocity() / (3) * velocity

    def set_velocity_left_motor(self, velocity):
        self.left_motor.setVelocity(self._limit_velocity(velocity))

    def set_velocity_right_motor(self, velocity):
        self.right_motor.setVelocity(self._limit_velocity(velocity))
