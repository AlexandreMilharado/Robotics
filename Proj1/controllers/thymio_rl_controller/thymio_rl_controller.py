#
# ISCTE-IUL, IAR, 2024/2025.
#
# Template to use SB3 to train a Thymio in Webots.
#

try:
    import time
    import gymnasium as gym
    import numpy as np
    import math
    import sys
    from stable_baselines3.common.callbacks import CheckpointCallback
    from sb3_contrib import RecurrentPPO
    from controller import Supervisor, Motor, DistanceSensor

except ImportError:
    sys.exit('Please make sure you have all dependencies installed.')


TIME_STEP = 5
EPISODE_STEPS = 10
LEDGE_THRESHOLD = 100
PROX_THRESHOLD = 0.9
ACC_THRESHOLD = 2

OBSTACLES_NUMBER = 10
OBSTACLES_MIN_RADIUS = 0.35
OBSTACLES_MAX_RADIUS = 0.67

GRID_RESOLUTION = 0.117/4

REWARD_POSITVE_LINEAR_VELOCITY = 10
REWARD_VISITED = 10
PENALTY_PROX_OBSTACLES = 10
PENALTY_LEDGE_OBSTACLES = 10

#
# Structure of a class to create an OpenAI Gym in Webots.
#
class OpenAIGymEnvironment(Supervisor, gym.Env):
    
    def __init__(self, max_episode_steps = EPISODE_STEPS):
        super().__init__()

        gym.register(
            id='WebotsEnv-v0',
            entry_point=OpenAIGymEnvironment,
            max_episode_steps=max_episode_steps
        )
        self.spec = gym.spec('WebotsEnv-v0')
        self.__timestep = int(self.getBasicTimeStep())

        # Do all other required initializations
        self.obstacles = []
        self.visited = set()
        self.reset()

        # Fill in according to the action space of Thymio
        # See: https://www.gymlibrary.dev/api/spaces/
        self.action_space = gym.spaces.Box(
            low=np.array([-self.left_motor.getMaxVelocity(), -self.right_motor.getMaxVelocity()]),
            high=np.array([self.left_motor.getMaxVelocity(), self.right_motor.getMaxVelocity()]),
            dtype=np.float64)

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
            dtype=np.float64)

        self.__n = 0

        # To remove
        # self.acc_sensor = super().getDevice('acc')
        # self.acc_sensor.enable(self.__timestep)

    #
    # Reset the environment to an initial internal state, returning an initial observation and info.
    #
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset Simulation
        self._sim_reset()

        # initialize the sensors, reset the actuators, randomize the environment
        # See how in Lab 1 code
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
        init_state = self._inital_obs()

        # aditional info
        # info = self._get_info()

        return init_state, {}


    #
    # Run one timestep of the environment’s dynamics using the agent actions.
    #   
    def step(self, action):

        self.__n = self.__n + 1

        # start by applying the action in the robot actuators
        # See how in Lab 1 code
        self._set_velocities(action)

        # let the action to effect for a few timesteps
        self._act()

        # set the state that resulted from applying the action (consulting the robot sensors)
        self.state = self._get_obs()

        # compute the reward that results from applying the action in the current state
        reward = self._get_reward()

        # set termination and truncation flags (bools)
        terminated = self._determine_terminated()
        truncated = self._determine_truncated()

        # aditional info
        info = self._get_info()

        return self.state.astype(np.float64), reward, terminated, truncated, info
    
    def _get_obs(self):
        # To remove
        # acc_x, acc_y, acc_z = self.acc_sensor.getValues()
        # To remove
        # state += [acc_x, acc_y, acc_z]
        state = []
        state += self._get_normalize_h_sensors()
        state += self._get_normalize_g_sensors()

        return np.array(state)
    
    def _get_info(self):
        return {}

    def _set_velocities(self, action):
        self.left_motor.setVelocity(max(min(action[0], 9), -9))
        self.right_motor.setVelocity(max(min(action[1], 9), -9))

    def _act(self):
        for i in range(10):
            super().step(self.__timestep)

    def _get_reward(self):
        total = 0
        total += self._reward_exploration()
        total += self._penalty_prox_obstacles(self._get_normalize_h_sensors())
        total += self._penalty_ledge(self._get_normalize_g_sensors())
        total += self._reward_linear_positive_velocity()

        # To remove
        # acc_x, acc_y, acc_z = self.acc_sensor.getValues()
        # acc = math.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
        # if acc < ACC_THRESHOLD:
        #     total -= 40

        return total

    def _determine_terminated(self):
        if self.getFromDef("ROBOT").getField("translation").getSFVec3f()[2] < 0:
            print("ENDING")
        return self.getFromDef("ROBOT").getField("translation").getSFVec3f()[2] < 0
        # To remove
        # acc_x, acc_y, acc_z = self.acc_sensor.getValues()
        # acc = math.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
        # print("ACC", acc)
        # return acc < ACC_THRESHOLD

    def _determine_truncated(self):
        return False

# Reset Auxiliar Methods
    def _sim_reset(self):
        print("---RESET---")

        self.simulationReset()
        self.simulationResetPhysics()
        
        super().step(self.__timestep)

    def _init_sensors(self):
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
        for sensor in self.h_sensors:
            sensor.enable(self.__timestep)
        for sensor in self.g_sensors:
            sensor.enable(self.__timestep)

    def _init_actuators(self):
        self.left_motor : Motor= self.getDevice('motor.left')
        self.right_motor : Motor = self.getDevice('motor.right')
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        self.left_motor.setVelocity(0)
        self.right_motor.setVelocity(0)

    def _init_position(self):
        self.getFromDef('ROBOT').getField('rotation').setSFRotation(self._random_orientation())
        self.getFromDef('ROBOT').getField('translation').setSFVec3f([0, 0, 1])
    
    def _random_orientation(self):                                      
        angle = np.random.uniform(0, 2 * np.pi)
        return [0, 0, 1, angle]

    def _random_position(self, min_radius, max_radius, z):                
        radius = np.random.uniform(min_radius, max_radius)
        angle = self._random_orientation()
        x = radius * np.cos(angle[3])
        y = radius * np.sin(angle[3])
        return [x, y, z]

    def _init_obstacles(self):                
        root = self.getRoot()
        children_field = root.getField('children')

        for i in range(OBSTACLES_NUMBER):
            position = self._random_position(OBSTACLES_MIN_RADIUS, OBSTACLES_MAX_RADIUS, 1)
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
        return np.array([0, 0, 0, 0, 0, 0, 0]).astype(np.float32)

# Observations Auxiliar Methods
    def _get_normalize_h_sensors(self):
        return [sensor.getValue()/sensor.getMaxValue() for sensor in self.h_sensors]

    def _get_normalize_g_sensors(self):
        return [sensor.getValue()/sensor.getMaxValue() for sensor in self.g_sensors]

# Reward Auxiliar Methods
    def _get_position_on_grid(self):
        pos = self.getFromDef("ROBOT").getField("translation").getSFVec3f()
        gx = int(pos[0] / GRID_RESOLUTION)
        gy = int(pos[2] / GRID_RESOLUTION)
        return (gx, gy)
    
    def _reward_linear_positive_velocity(self):
        return (REWARD_POSITVE_LINEAR_VELOCITY *
                int(self.left_motor.getVelocity() > 0 and self.right_motor.getVelocity() > 0))
    
    def _penalty_prox_obstacles(self, frontal_readings):
        return sum([-PENALTY_PROX_OBSTACLES * reading for reading in frontal_readings])
    
    def _penalty_ledge(self, ground_sensors):
        return sum([-PENALTY_LEDGE_OBSTACLES * int(1 - reading) for reading in ground_sensors])

    def _reward_exploration(self):
        x, y = self._get_position_on_grid()
        if (x, y) in self.visited:
            return -REWARD_VISITED
        else:
            self.visited.add((x, y))
            return REWARD_VISITED


def main():
    # Create the environment to train / test the robot
    env = OpenAIGymEnvironment()

    # Initializing environment
    #env = gym.make('WebotsEnv-v0')

    # Code to train and save a model
    # For the PPO case, see how in Lab 7 code
    # For the RecurrentPPO case, consult its documentation
    if ('WebotsEnv-v0' not in gym.registry):
        raise Exception('Environment not registered correctly')
    model = RecurrentPPO("MlpLstmPolicy", env, device="cuda",
                         n_steps=2048,
                         batch_size=64,
                         ent_coef=0.02,
                         clip_range=0.2,
                         vf_coef=0.5,
                         learning_rate=3e-4,
                         max_grad_norm=0.5,
                         verbose=1)
    model.learn(500000)

    model.save("RecurrentPPO_test_1")
    

    # Code to load a model and run it
    # For the RecurrentPPO case, consult its documentation
    obs, _ = env.reset()  # Handle new reset return format
    for _ in range(100000):
        action, _states = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)  # Handle new step return format
        print(obs, reward, terminated, truncated, info)
        if terminated or truncated:
            obs, _ = env.reset()


if __name__ == '__main__':
    main()