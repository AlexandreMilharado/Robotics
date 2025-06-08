#
# ISCTE-IUL, IAR, 2024/2025.
#
# Template to use SB3 to train a Thymio in Webots.
#



try:
    import os
    import time
    import gymnasium as gym
    import torch
    import numpy as np
    import math
    import sys
    from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback, CallbackList
    from sb3_contrib import RecurrentPPO
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.utils import get_schedule_fn
    from controller import Supervisor, Motor, DistanceSensor
    import csv


except ImportError:
    sys.exit('Please make sure you have all dependencies installed.')

MODEL_LOAD = "experiments/RPPO/RPPO.zip"
MODEL_PATH = "RPPO.zip"
CSV_PATH = "rollout.csv"

TIME_STEP = 5
EPISODE_STEPS = 256
LEDGE_THRESHOLD = 100
PROX_THRESHOLD = 0.9
ACC_THRESHOLD = 2
DEGREES_INCLINED = 7

OBSTACLES_NUMBER = 10
OBSTACLES_MIN_RADIUS = 0.35
OBSTACLES_MAX_RADIUS = 0.67

GRID_RESOLUTION = 0.117 * 5
VELOCITY_THRESHOLD = 2
WHEEL_DIFF_THRESHOLD = 0.1

ROLLOUT_STEPS = 1000
TOTAL_TIMESTEPS = 20000
BATCH_SIZE = 16
ENTROPY_COEFICIENT=0.005
CLIP_RANGE=0.2
VF_COEFICIENT=0.5
LEARNING_RATE=3e-4
MAX_GRAD_NORM=0.5

# Com penalização a recuar da berma
# Sem penalização around quando encontra obstáculo
REWARD_GROUNDED_POSITVE_VELOCITY = 0
VELOCITY_REWARD = 1
REWARD_VISITED = 30
PENALTY_BACKTRACK = 0
PENALTY_VISITED = 0
PENALTY_PROX_OBSTACLES = 15
PENALTY_LEDGE_OBSTACLES = 20
PENALTY_LEDGE_FALL = 100
PENALTY_TURN = 1
PENALTY_STUCK = 0

#TODO Mudar
IS_RECURRENT = True

# Structure of a class to create an OpenAI Gym in Webots.

class OpenAIGymEnvironment(Supervisor, gym.Env):
    
    def __init__(self, max_episode_steps = EPISODE_STEPS):
        """
        Initialize the OpenAIGymEnvironment.

        This constructor sets up the environment by registering it with the Gym library and configuring
        the action and observation spaces according to the Thymio robot's specifications.

        Parameters
        ----------
        max_episode_steps : int, optional
            Maximum number of steps in an episode. Default is EPISODE_STEPS.

        Notes
        -----
        - The environment is registered with id 'WebotsEnv-v0'.
        - The action space is configured as a continuous space with values ranging from -1 to 1.
        - The observation space is configured based on Thymio's sensors, with values ranging from 0 to 1.
        - The environment resets to an initial state upon initialization.
        """

        super().__init__()

        gym.register(
            id='WebotsEnv-v0',
            entry_point=OpenAIGymEnvironment,
            max_episode_steps=max_episode_steps
        )
        self.spec = gym.spec('WebotsEnv-v0')
        self.__timestep = int(self.getBasicTimeStep())

        # Do all other required initializations
        self.reset()

        # Fill in according to the action space of Thymio
        # See: https://www.gymlibrary.dev/api/spaces/
        self.action_space = gym.spaces.Box(
            low=np.array([-1, -1]),
            high=np.array([1, 1]),
            dtype=np.float32)                                   # MUDANÇA

        # Fill in according to Thymio's sensors
        # See: https://www.gymlibrary.dev/api/spaces/
        self.observation_space = gym.spaces.Box(
            low=np.array([0, 0,
                        0,
                        0, 0,
                        0, 0]), 
            high=np.array([1, 1,
                        1,
                        1, 1,
                        1, 1]),
            dtype=np.float32)                                   # MUDANÇA

        self.__n = 0
        self.__nn = 0
        self.__max_n = max_episode_steps

    #
    # Reset the environment to an initial internal state, returning an initial observation and info.
    #
    def reset(self, seed=None, options=None):
        """
        Reset the environment to an initial internal state, returning an initial observation and info.

        Parameters:
        seed (int): the seed used for the random number generator. If None, the environment will use the Webots random seed.
        options (dict): additional information to pass to the environment. If None, no additional information is passed.

        Returns:
        tuple: a tuple of the initial observation and additional information (info). The initial observation is a numpy array of shape (7,) with the sensors' values, and the additional information is a dictionary with the following keys:

        - 'obstacles': a list of the obstacles' positions and radiuses
        - 'visited': a set of the visited positions
        - 'last_grid_position': the last grid position
        - 'stuck_counter': the number of steps the robot is stuck
        - 'last_reading': the last reading of the front sensors
        - 'last_break_action': the last action that made the robot break

        The info dictionary can be used to store additional information about the environment, such as the obstacles' positions and radiuses, the visited positions, the last grid position, the number of steps the robot is stuck, the last reading of the front sensors, and the last action that made the robot break.
        """
        super().reset(seed=seed)

        # Reset Simulation
        self._sim_reset()
        self.__n = 0
        self.total_episode_reward = 0

        # initialize the sensors, reset the actuators, randomize the environment
        # See how in Lab 1 code
        self.obstacles = []
        self.visited = set()
        self.last_grid_position = None
        self.stuck_counter = 0

        # Sensors
        self._init_sensors()

        # Actuators
        self._init_actuators()

        # Randomize Enviroment
        self._init_position()
        self._init_obstacles()

        # you may need to iterate a few times to let physics stabilize
        for _ in range(15):
            super().step(self.__timestep)

        # set the initial state vector to return
        init_state = self._get_obs()

        # aditional info
        # info = self._get_info()

        return init_state, {}


    #
    # Run one timestep of the environment’s dynamics using the agent actions.
    #   
    def step(self, action):

        """
        Run one timestep of the environment's dynamics using the agent actions.

        Parameters
        ----------
        action : numpy array of shape (2,)
            The action to take in the current state.

        Returns
        -------
        state : numpy array of shape (7,)
            The state that resulted from applying the action in the current state.
        reward : float
            The reward that results from applying the action in the current state.
        terminated : bool
            A boolean indicating whether the episode has terminated.
        truncated : bool
            A boolean indicating whether the episode has been truncated.
        info : dict
            Additional information about the environment, such as the obstacles' positions and radiuses, the visited positions, the last grid position, the number of steps the robot is stuck, the last reading of the front sensors, and the last action that made the robot break.

        Notes
        -----
        - The reward is computed based on the movement of the robot.
        - The episode is terminated if the robot is stuck or if the maximum number of steps is reached.
        - The episode is truncated if the robot has moved for a certain number of steps without being stuck.
        """
        self.__n = self.__n + 1
        self.__nn = self.__nn + 1

        # start by applying the action in the robot actuators
        # See how in Lab 1 code
        self._set_velocities(action)

        # let the action to effect for a few timesteps
        self._act()

        # set the state that resulted from applying the action (consulting the robot sensors)
        self.state = self._get_obs()

        # set termination and truncation flags (bools)
        self._wheel_stuck(action)
        terminated = self._determine_terminated()
        truncated = self._determine_truncated()
        done = terminated or truncated

        # compute the reward that results from applying the action in the current state
        reward = self._get_reward(terminated)
        self.total_episode_reward += reward

        # aditional info
        info = self._get_info(done, self.state)

        return self.state.astype(np.float64), reward, terminated, truncated, info
    
    def _get_obs(self):
        """
        Retrieve the current observation of the environment's state.

        This function collects and normalizes data from the robot's horizontal and ground sensors.
        It then aggregates these readings into a numpy array to represent the current state of 
        the environment in a standardized format.

        Returns
        -------
        np.ndarray
            A numpy array of dtype float32, containing the normalized sensor readings that represent
            the current state of the environment.
        """

        state = []
        state += self._get_normalize_h_sensors()
        state += self._get_normalize_g_sensors()

        return np.array(state, dtype=np.float32)
    
    def _get_info(self, done, obs):
        """
        Construct additional information dictionary for the environment.

        Parameters
        ----------
        done : bool
            A boolean indicating whether the episode has terminated.
        obs : np.ndarray
            The terminal observation of the environment.

        Returns
        -------
        dict
            A dictionary containing additional information about the environment, such as the terminal observation, the total reward of the episode and the number of steps taken in the episode.
        """
        info = {}
        info['terminal_observation'] = obs
        if done:
            info['episode'] = {
                'r': self.total_episode_reward,
                'l': self.__n,
            }
        return info

    def _set_velocities(self, action):
        """
        Sets the velocities for the left and right motors based on the provided action.

        This function clamps the action values between -1 and 1, scales them by the 
        maximum velocity of each motor, and sets the motor velocities accordingly.

        Parameters
        ----------
        action : list of float
            A list containing two elements representing the desired velocity scale 
            for the left and right motors, respectively.
        """

        self.left_motor.setVelocity(max(min(action[0], 1), -1) * self.left_motor.getMaxVelocity() )
        self.right_motor.setVelocity(max(min(action[1], 1), -1) * self.right_motor.getMaxVelocity())
        #self.left_motor.setVelocity(2)
        #self.right_motor.setVelocity(5)

    def _act(self):
        """
        Simulates the physics for 10 steps.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        
        for i in range(10):
            super().step(self.__timestep)

    def _get_info_vs(self, done):
        """
        Provides additional information about the state of the environment.

        Parameters
        ----------
        done : bool
            A boolean indicating whether the episode has terminated.

        Returns
        -------
        dict
            A dictionary containing additional information about the environment, such as the reward given by the 
            exploration, obstacles, and ledge detectors.
        """

        frontal_readings = self._get_normalize_h_sensors()
        ground_readings = self._get_normalize_g_sensors()

        info = {}
        if done:
            info = {
                'exploration': self._reward_exploration(),
                'obstacles': self._penalty_prox_obstacles(frontal_readings),
                'ledge': self._penalty_ledge(ground_readings)
            }

        return info

    def _get_reward(self, done):
        """
        Calculates the total reward for the current state of the environment.

        This function aggregates various reward and penalty components to determine
        the total reward for the current state of the environment. The reward is 
        calculated based on exploration, proximity to obstacles, proximity to ledges,
        positive velocity, linear velocity, backtracking, and falling.

        Parameters
        ----------
        done : bool
            A boolean indicating whether the episode has terminated.

        Returns
        -------
        float
            The computed total reward for the current state of the environment.
        """

        frontal_readings = self._get_normalize_h_sensors()
        
        total = 0
        total += self._reward_exploration()
        total += self._penalty_prox_obstacles(frontal_readings)
        total += self._penalty_ledge(self._get_normalize_g_sensors())
        total += self._reward_positive_velocity()
        total += self._reward_linear_velocity(frontal_readings)
        total += self._penalty_backtrack()
        total += self._penalty_fall(total, done)
        #total += self._penalty_stuck()

        #print("Reward: ", total)

        return float(total)

    
    def _wheel_stuck(self, action):

        """
        Determines if the robot's wheels are stuck.

        This function checks whether the robot is stuck based on its current position
        and motor activity. If the robot remains in the same position while its motors
        are active, it is considered stuck.

        Parameters
        ----------
        action : list of float
            A list containing two elements representing the desired velocity for 
            the left and right motors, respectively.

        Returns
        -------
        bool
            True if the robot is stuck, otherwise False.

        Updates
        -------
        self.stuck_counter : int
            Increments if the robot is stuck, otherwise resets to 0.
        """
        left_speed, right_speed = action[0], action[1]

        current_position = self._get_position_on_grid()

        same_pos = (current_position == self.last_grid_position)

        motor_active = abs(left_speed) > 0.1 or abs(right_speed) > 0.1

        stuck = same_pos and motor_active

        if stuck:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0
        
        return stuck
    

    def _determine_terminated(self):
        """
        Determines whether the episode should be terminated.

        This method checks if the robot has fallen below a certain height or is stuck, 
        and returns a boolean indicating whether the episode is terminated.

        Returns
        -------
        bool
            True if the episode is terminated (robot has fallen or is stuck), otherwise False.
        """

        fall = (self.getFromDef("ROBOT").getField("translation").getSFVec3f()[2] < 0.9)
        stuck = self.stuck_counter > 30
        terminated = fall

        if(terminated):
            print(f"--- EPISODE ENDED ---")
        return terminated


    def _determine_truncated(self):
        """
        Determines whether the episode should be truncated.

        Checks if the robot has reached the maximum number of steps allowed in an episode.

        Returns
        -------
        bool
            True if the episode should be truncated, otherwise False.
        """
        return self.__n > self.__max_n

# Reset Auxiliar Methods
    def _sim_reset(self):
        """
        Resets the simulation and physics.

        Resets the Webots simulation and physics, then steps the simulation once to 
        ensure that the physics is properly initialized.

        Returns
        -------
        None
        """
        self.simulationReset()
        self.simulationResetPhysics()
        
        super().step(self.__timestep)

    def _init_sensors(self):
        """
        Initializes the sensors of the robot.

        Retrieves the sensors from the supervisor by name, enables them, and stores them in the self.sensors list.
        The sensors are the ones labeled "prox.horizontal.*" and "prox.ground.*".

        No return value.
        """
        self.h_sensors : list[DistanceSensor] = [
            super().getDevice('prox.horizontal.0'), # Left
            super().getDevice('prox.horizontal.1'), # Left
            super().getDevice('prox.horizontal.2'), # Middle
            super().getDevice('prox.horizontal.3'), # Right
            super().getDevice('prox.horizontal.4'), # Right
        ]
        self.g_sensors : list[DistanceSensor] = [
            super().getDevice('prox.ground.0'),     # Vertical
            super().getDevice('prox.ground.1')      # Vertical
        ]
        self.acc_sensor = super().getDevice('acc')

        for sensor in self.h_sensors:
            sensor.enable(self.__timestep)
        for sensor in self.g_sensors:
            sensor.enable(self.__timestep)
        self.acc_sensor.enable(self.__timestep)

    def _init_actuators(self):
        """
        Initializes the actuators of the robot.

        Retrieves the left and right motors from the robot by name, sets their positions 
        to infinity (allowing continuous rotation), and initializes their velocities to zero.

        No return value.
        """
        self.left_motor : Motor= self.getDevice('motor.left')
        self.right_motor : Motor = self.getDevice('motor.right')
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        self.left_motor.setVelocity(0)
        self.right_motor.setVelocity(0)

    def _init_position(self):
        """
        Initializes the position of the robot.

        Sets the initial position of the robot to a random orientation and a translation of [0, 0, 1].

        No return value.
        """
        self.getFromDef('ROBOT').getField('rotation').setSFRotation(self._random_orientation())
        self.getFromDef('ROBOT').getField('translation').setSFVec3f([0, 0, 1])
    
    def _random_orientation(self):                                      
        """
        Generates a random orientation quaternion.

        The orientation is represented as a quaternion [x, y, z, angle], 
        where the axis of rotation is fixed as the z-axis and the angle is 
        randomly selected between 0 and 2*pi radians.

        Returns
        -------
        list
            A quaternion representing the random orientation.
        """
        angle = np.random.uniform(0, 2 * np.pi)
        return [0, 0, 1, angle]

    def _random_position(self, min_radius, max_radius, z):                
        """
        Generates a random position within a specified radius range and at a specified z-coordinate.

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

    def _init_obstacles(self):                
        """
        Initializes the obstacles in the environment.

        This function creates a specified number of obstacles, each with a random position, 
        orientation, length, and width. The obstacles are defined as Solid nodes with a 
        white appearance and a density of 1000.0. They are added to the children field of 
        the root node in the simulation environment. Each obstacle is stored in the 
        self.obstacles list for later reference.

        No return value.
        """
        
        root = self.getRoot()
        children_field = root.getField('children')

        for i in range(OBSTACLES_NUMBER):
            position = self._random_position(OBSTACLES_MIN_RADIUS, OBSTACLES_MAX_RADIUS, 1.2)
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

            self.obstacles.append(self.getFromDef(f"WHITE_BOX_{i}"))

    def _inital_obs(self):
        """
        Returns the initial observation of the environment.

        The initial observation is a numpy array of shape (7,) with all values set to 0.

        Returns:
            np.ndarray: the initial observation
        Notes - DEPRECATED
        """
        return np.array([0, 0, 0, 0, 0, 0, 0]).astype(np.float32)

# Observations Auxiliar Methods
    def _get_normalize_h_sensors(self):
        """
        Returns the normalized values of the horizontal sensors.

        The normalized values are obtained by dividing the raw sensor values by the maximum
        sensor value.

        Returns:
            list: a list of 5 normalized values of the horizontal sensors
        """
        return [sensor.getValue()/sensor.getMaxValue() for sensor in self.h_sensors]

    def _get_normalize_g_sensors(self):
        """
        Returns the normalized values of the ground sensors.

        The normalized values are obtained by dividing the raw sensor values by the maximum
        sensor value.

        Returns:
            list: a list of 2 normalized values of the ground sensors
        """
        return [sensor.getValue()/sensor.getMaxValue() for sensor in self.g_sensors]

# Reward Auxiliar Methods
    def _get_position_on_grid(self):
        """
        Returns the position of the robot on the grid.

        The position is represented as a tuple (gx, gy) where gx and gy are the
        coordinates of the robot in the grid. The grid coordinates are obtained by
        dividing the position in meters by the grid resolution.

        Returns
        -------
        tuple
            A tuple containing the x and y coordinates of the robot in the grid.
        """
        pos = self.getFromDef("ROBOT").getField("translation").getSFVec3f()
        gx = int(pos[0] / GRID_RESOLUTION)
        gy = int(pos[1] / GRID_RESOLUTION)
        return (gx, gy)

    def _reward_positive_velocity(self):
        """
        Calculates the reward for maintaining positive velocity while grounded.

        This function evaluates the robot's motor velocities and ground sensor readings
        to determine a reward based on whether both wheels are moving forward and in 
        contact with the ground. A reward is given when both wheels are moving forward 
        and the robot is grounded.

        Returns
        -------
        float
            The computed reward for the robot's current velocity state.
        """

        left_vel = self.left_motor.getVelocity()
        right_vel = self.right_motor.getVelocity()
        both_forward = left_vel > 0 and right_vel > 0

        gs = self._get_normalize_g_sensors()
        on_ground = gs[0] != 0 and gs[1] != 0

        return REWARD_GROUNDED_POSITVE_VELOCITY * int(both_forward and on_ground)
    
    def _penalty_stuck(self):
        """
        Calculates the penalty for the robot being stuck.

        This function assesses whether the robot is stuck by evaluating the 
        stuck counter. If the counter exceeds a predefined threshold, it 
        returns a penalty value.

        Returns
        -------
        int
            A penalty value if the robot is determined to be stuck, otherwise 0.
        """

        stuck = self.stuck_counter > 30
        return PENALTY_STUCK * int(stuck)

    def _penalty_backtrack(self):
        """
        Calculates the penalty for the robot backtracking.

        This function evaluates the velocities of the robot's motors to determine
        if both wheels are moving backwards, which indicates backtracking. A penalty
        is applied if the robot is backtracking.

        Returns
        -------
        int
            The penalty value if backtracking is detected, otherwise 0.
        """

        left_vel = self.left_motor.getVelocity()
        right_vel = self.right_motor.getVelocity()
        both_back = left_vel < 0 and right_vel < 0
        return -PENALTY_BACKTRACK * int(both_back)

    def _penalty_prox_obstacles(self, frontal_readings):
        """
        Calculates the penalty for the robot's proximity to obstacles.

        This function takes the values of the frontal sensors and returns a penalty
        based on whether the sensors are reading any obstacles. The penalty is higher
        when the sensors are closer to an obstacle.

        Parameters
        ----------
        frontal_readings : list
            The current readings of the frontal sensors.

        Returns
        -------
        float
            The computed penalty for the robot's current proximity to obstacles.
        """

        return sum([-PENALTY_PROX_OBSTACLES * reading for reading in frontal_readings])
    
    def _penalty_ledge(self, ground_sensors):
        """
        Calculates the penalty for the robot's proximity to the ledge.

        This function takes the values of the ground sensors and returns a penalty
        based on whether the sensors are reading any obstacles. The penalty is higher
        when the sensors are closer to an obstacle.

        Parameters
        ----------
        ground_sensors : list
            The current readings of the ground sensors.

        Returns
        -------
        float
            The computed penalty for the robot's current proximity to the ledge.
        """
        avg_vel = (self.left_motor.getVelocity() + self.right_motor.getVelocity()) / (2 * 9.53)
        return sum([-PENALTY_LEDGE_OBSTACLES * int(1 - reading) for reading in ground_sensors])

    def _reward_exploration(self):
        """
        Calculates the exploration reward for the robot.

        This function determines a reward based on the robot's movement to new grid positions.
        The reward is given when the robot moves to a new position on the grid. If the robot
        revisits a position, a penalty is applied.

        Returns
        -------
        float
            The computed reward or penalty for exploration based on the robot's movement.
        """

        x, y = self._get_position_on_grid()

        if (x, y) == self.last_grid_position:
            return 0

        self.last_grid_position = (x, y)

        if (x, y) in self.visited:
            return -PENALTY_VISITED
        else:
            self.visited.add((x, y))
            return REWARD_VISITED
            
    def _reward_linear_velocity(self, frontal_readings):
        """
        Calculates the reward for linear positive velocity.

        This function evaluates the robot's motor velocities and frontal sensor readings to 
        determine a reward based on whether both wheels are moving forward, aligned, and not
        obscured by obstacles. The reward is higher when both wheels are moving forward and
        aligned, and a penalty is applied if they are not.

        Returns
        -------
        float
            The computed reward or penalty for the robot's current velocity state.
        """
        leftV = self.left_motor.getVelocity()
        rightV = self.right_motor.getVelocity()
        maxV = max(abs(leftV), abs(rightV))
        diff = min(abs(leftV-rightV)/maxV, 1)
        percentV = ((leftV + rightV)/2)/maxV
        
        thresholdPass = (leftV + rightV)/2 > VELOCITY_THRESHOLD
        diffPass = diff < WHEEL_DIFF_THRESHOLD

        obscured = sum([reading for reading in frontal_readings]) > 0

        #print("-")
        #print("Passed velocity:", (leftV + rightV)/2)
        #print("Passed diff:", diff)
        #print("Condition:", thresholdPass and diffPass)

        #return VELOCITY_REWARD * int(thresholdPass and diffPass) - PENALTY_AROUND * int(not diffPass)
        return VELOCITY_REWARD * percentV - PENALTY_TURN * diff
        
    def _penalty_fall(self, current_reward, done):
        """
        Calculates the penalty for the robot falling off the ledge.

        This function takes the current total episode reward and whether the episode is done
        and returns a penalty based on whether the robot has fallen off the ledge.

        Parameters
        ----------
        current_reward : int
            The current total episode reward.
        done : bool
            Whether the episode is done.

        Returns
        -------
        int
            The computed penalty for the robot falling off the ledge.
        """
        return -PENALTY_LEDGE_FALL * int(done)
        #return -((PENALTY_LEDGE_FALL*100) + PENALTY_LEDGE_FALL * abs(self.total_episode_reward + current_reward)) * int(done)
        

class RolloutCSVLogger(BaseCallback):
    def __init__(self, csv_path: str, verbose=0):
        """
        Initializes the RolloutCSVLogger.

        Parameters
        ----------
        csv_path : str
            The path to the CSV file where the rollout data will be saved.
        verbose : int, optional
            The verbosity level (default: 0).

        Attributes
        ----------
        header_written : bool
            Tracks whether the header has been written to the file.
        filename : str
            Stores the path to the CSV file.
        """

        super().__init__(verbose)
        self.header_written = False
        self.filename = csv_path

    def _on_step(self) -> bool:
        """
        Called after each step in the environment.

        This function is invoked after every step in the environment and is used to
        determine whether the rollout should continue or if the environment needs to be reset.

        Returns
        -------
        bool
            True if the rollout should continue, otherwise False.
        """

        return True
    
    def _on_rollout_end(self) -> None:
        """
        Called after each rollout.

        This function writes out the metrics gathered during the rollout to a CSV file.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        logger = self.model.logger
        name_to_value = getattr(logger, 'name_to_value', {})


        print("Episode reward:", sum(self.training_env.envs[0].get_episode_rewards()))
        ep_rew_mean = sum(self.training_env.envs[0].get_episode_rewards())/len(self.training_env.envs[0].get_episode_rewards())
        ep_len_mean = sum(self.training_env.envs[0].get_episode_lengths())/len(self.training_env.envs[0].get_episode_lengths())
        data = {
            'timesteps': self.model.num_timesteps,
            'learning_rate': self.model.lr_schedule(1 - (self.model.num_timesteps / TOTAL_TIMESTEPS)),
            'entropy_loss': name_to_value.get('train/entropy_loss', 0),
            'approx_kl': name_to_value.get('train/approx_kl', 0),
            'loss': name_to_value.get('train/loss', 0),
            'policy_gradient_loss': name_to_value.get('train/policy_gradient_loss', 0),
            'clip_fraction': name_to_value.get('train/clip_fraction', 0),
            'value_loss': name_to_value.get('train/value_loss', 0),
            'explained_variance': name_to_value.get('train/explained_variance', 0),
            'std': name_to_value.get('train/std', 0),
            'n_updates': name_to_value.get('train/n_updates', self.num_timesteps // self.model.n_steps),
            'clip_range': self.model.clip_range(self.model.num_timesteps),
            'ep_len_mean': ep_len_mean,
            'ep_rew_mean': ep_rew_mean,
        }

        
        # Write metrics after every rollout (training metrics will be 0 on first rollout)
        file_exists = os.path.isfile(self.filename)
        with open(self.filename, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=data.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(data)
            
def exp_schedule(initial_value, decay_rate=0.9):
    """
    Returns a function that computes a learning rate schedule according to an exponential decay scheme.

    Parameters
    ----------
    initial_value : float
        The initial learning rate.
    decay_rate : float, optional
        The decay rate. Defaults to 0.9.

    Returns
    -------
    scheduler : function
        A function that takes a single argument, progress_remaining, and returns the learning rate.
    """
    def scheduler(progress_remaining):
        progress_remaining = max(0.0, min(1.0, progress_remaining))
        lr = initial_value * (progress_remaining ** decay_rate)
        return lr

    return scheduler

def main():
    # Create the environment to train / test the robot
    env = OpenAIGymEnvironment()
    env = Monitor(env) 
    # Initializing environment
    #env = gym.make('WebotsEnv-v0')

    # Code to train and save a model
    # For the PPO case, see how in Lab 7 code
    # For the RecurrentPPO case, consult its documentation
    if ('WebotsEnv-v0' not in gym.registry):
        raise Exception('Environment not registered correctly')
    
    print("PyTorch CUDA available:", torch.cuda.is_available())
    print("CUDA device count:", torch.cuda.device_count())
    print("CUDA device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")

    if os.path.exists(MODEL_LOAD):
        if IS_RECURRENT:
            model = RecurrentPPO.load(MODEL_LOAD, env, device="cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            model = PPO.load(MODEL_PATH, env, device="cpu")
        
    else:
        if IS_RECURRENT:
            model = RecurrentPPO(
                "MlpLstmPolicy", env, device="cuda:0" if torch.cuda.is_available() else "cpu",
                n_steps=ROLLOUT_STEPS,
                batch_size=BATCH_SIZE,
                ent_coef=ENTROPY_COEFICIENT,
                clip_range=CLIP_RANGE,
                vf_coef=VF_COEFICIENT,
                learning_rate=LEARNING_RATE,
                max_grad_norm=MAX_GRAD_NORM,
                verbose=1
            )
        else:
            model = PPO(
                "MlpPolicy", env, device="cpu",
                batch_size=BATCH_SIZE,
                ent_coef=ENTROPY_COEFICIENT,
                clip_range=CLIP_RANGE,
                vf_coef=VF_COEFICIENT,
                learning_rate=exp_schedule(LEARNING_RATE),
                max_grad_norm=MAX_GRAD_NORM,
                verbose=1
            )

        checkpoint_callback = CheckpointCallback(save_freq=TOTAL_TIMESTEPS/4, save_path='./models/')
        csv_logger = RolloutCSVLogger(CSV_PATH)

        callback = CallbackList([checkpoint_callback, csv_logger])
        model.learn(TOTAL_TIMESTEPS, callback=callback)

        model.save(MODEL_PATH)

        ###

    # Code to load a model and run it
    if IS_RECURRENT:
        obs = env.reset()  # Handle new reset return format
        for _ in range(100000):
            action, _states = model.predict(obs)
            obs, reward, terminated, info = env.step(action)  # Handle new step return format
            if terminated or truncated:
                obs = env.reset()
    else:
        obs, _ = env.reset()  # Handle new reset return format
        for _ in range(100000):
            action, _states = model.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)  # Handle new step return format
            if terminated or truncated:
                obs, _ = env.reset()


if __name__ == '__main__':
    main()