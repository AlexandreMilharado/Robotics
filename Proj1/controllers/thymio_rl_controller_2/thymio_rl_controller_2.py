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
        state = []
        state += self._get_normalize_h_sensors()
        state += self._get_normalize_g_sensors()

        return np.array(state, dtype=np.float32)
    
    def _get_info(self, done, obs):
        info = {}
        info['terminal_observation'] = obs
        if done:
            info['episode'] = {
                'r': self.total_episode_reward,
                'l': self.__n,
            }
        return info

    def _set_velocities(self, action):
        self.left_motor.setVelocity(max(min(action[0], 1), -1) * self.left_motor.getMaxVelocity() )
        self.right_motor.setVelocity(max(min(action[1], 1), -1) * self.right_motor.getMaxVelocity())
        #self.left_motor.setVelocity(2)
        #self.right_motor.setVelocity(5)

    def _act(self):
        for i in range(10):
            super().step(self.__timestep)

    def _get_info_vs(self, done):
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
        fall = (self.getFromDef("ROBOT").getField("translation").getSFVec3f()[2] < 0.9)
        stuck = self.stuck_counter > 30
        terminated = fall

        if(terminated):
            print(f"--- EPISODE ENDED ---")
        return terminated


    def _determine_truncated(self):
        return self.__n > self.__max_n

# Reset Auxiliar Methods
    def _sim_reset(self):
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
        self.acc_sensor = super().getDevice('acc')

        for sensor in self.h_sensors:
            sensor.enable(self.__timestep)
        for sensor in self.g_sensors:
            sensor.enable(self.__timestep)
        self.acc_sensor.enable(self.__timestep)

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
        gy = int(pos[1] / GRID_RESOLUTION)
        return (gx, gy)

    def _reward_positive_velocity(self):
        left_vel = self.left_motor.getVelocity()
        right_vel = self.right_motor.getVelocity()
        both_forward = left_vel > 0 and right_vel > 0

        gs = self._get_normalize_g_sensors()
        on_ground = gs[0] != 0 and gs[1] != 0

        return REWARD_GROUNDED_POSITVE_VELOCITY * int(both_forward and on_ground)
    
    def _penalty_stuck(self):
        stuck = self.stuck_counter > 30
        return PENALTY_STUCK * int(stuck)

    def _penalty_backtrack(self):
        left_vel = self.left_motor.getVelocity()
        right_vel = self.right_motor.getVelocity()
        both_back = left_vel < 0 and right_vel < 0
        return -PENALTY_BACKTRACK * int(both_back)

    def _penalty_prox_obstacles(self, frontal_readings):
        return sum([-PENALTY_PROX_OBSTACLES * reading for reading in frontal_readings])
    
    def _penalty_ledge(self, ground_sensors):
        avg_vel = (self.left_motor.getVelocity() + self.right_motor.getVelocity()) / (2 * 9.53)
        return sum([-PENALTY_LEDGE_OBSTACLES * int(1 - reading) for reading in ground_sensors])

    def _reward_exploration(self):
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
        return -PENALTY_LEDGE_FALL * int(done)
        #return -((PENALTY_LEDGE_FALL*100) + PENALTY_LEDGE_FALL * abs(self.total_episode_reward + current_reward)) * int(done)
        

class RolloutCSVLogger(BaseCallback):
    def __init__(self, csv_path: str, verbose=0):
        super().__init__(verbose)
        self.header_written = False
        self.filename = csv_path

    def _on_step(self) -> bool:
        return True
    
    def _on_rollout_end(self) -> None:
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