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
    from controller import Supervisor, Motor

except ImportError:
    sys.exit('Please make sure you have all dependencies installed.')


TIME_STEP = 5
WHEEL_DISTANCE = 0
WHEEL_RADIUS = 0
MAX_VELOCITY = 9.53


#
# Structure of a class to create an OpenAI Gym in Webots.
#
class OpenAIGymEnvironment(Supervisor, gym.Env):
    
    def __init__(self, max_episode_steps = 3000):
        
        super().__init__()
        gym.register(
            id='WebotsEnv-v0',
            entry_point=OpenAIGymEnvironment,
            max_episode_steps=max_episode_steps
        )
        self.spec = gym.spec('WebotsEnv-v0')
        self.__timestep = int(self.getBasicTimeStep())

        # Fill in according to the action space of Thymio
        # See: https://www.gymlibrary.dev/api/spaces/
        self.action_space = gym.spaces.Box(
            low=np.array([-MAX_VELOCITY, -MAX_VELOCITY]),
            high=np.array([MAX_VELOCITY, MAX_VELOCITY]),
            dtype=np.float64)

        # Fill in according to Thymio's sensors
        # See: https://www.gymlibrary.dev/api/spaces/
        self.observation_space = gym.spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]), 
            high=np.array([100, 100, 100, 100, 100, 100, 100, 1000, 1000]),
            dtype=np.float64)
        
        # Do all other required initializations
        self.sensors = [
            super().getDevice('prox.horizontal.0'),
            super().getDevice('prox.horizontal.1'),
            super().getDevice('prox.horizontal.2'),
            super().getDevice('prox.horizontal.3'),
            super().getDevice('prox.horizontal.4'),
            super().getDevice('prox.horizontal.5'),
            super().getDevice('prox.horizontal.6'),
            super().getDevice('prox.ground.0'),
            super().getDevice('prox.ground.1')
        ]

        for sensor in self.sensors:
            sensor.enable(self.__timestep)

        self.left_motor = super().getDevice('motor.left')
        self.right_motor = super().getDevice('motor.right')

        self.__n = 0
        self.__max_n = max_episode_steps

        self.reset()


    #
    # Reset the environment to an initial internal state, returning an initial observation and info.
    #
    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        self._sim_reset()

        # initialize the sensors, reset the actuators, randomize the environment
        # See how in Lab 1 code
        ...

        # you may need to iterate a few times to let physics stabilize
        for i in range(15):
            super().step(self.__timestep)

        # set the initial state vector to return
        init_state = self._get_obs()
        ...

        # aditional info
        info = self._get_info()

        return np.array(init_state).astype(np.float64), info


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
        return np.array([sensor.getValue() for sensor in self.sensors])
    
    def _get_info(self):
        return {}
        ...

    def _set_velocities(self, action):
        self.left_motor.setVelocity(max(min(action[0], MAX_VELOCITY), -MAX_VELOCITY))
        self.right_motor.setVelocity(max(min(action[1], MAX_VELOCITY), -MAX_VELOCITY))

    def _act(self):
        for i in range(10):
            super().step(self.__timestep)

    def _get_reward(self):
        return 0#TODO implement

        ...

    def _determine_terminated(self):
        ...

    def _determine_truncated(self):
        return self.getFromDef('ROBOT').getField('translation').getSFVec3f()[2] < -10
        #return self.__n > self.__max_n ??????????????????????????????????????????

    def _sim_reset(self):
        print("---RESET---")
        
        self.simulationReset()
        self.simulationResetPhysics()
        
        super().step(self.__timestep)

        random_rotation = [0, 0, 1, np.random.uniform(0, 2 * np.pi)]
        self.getFromDef('ROBOT').getField('rotation').setSFRotation(random_rotation)
        self.getFromDef('ROBOT').getField('translation').setSFVec3f([0, 0, 1])
        
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))

        self.left_motor.setVelocity(0)
        self.right_motor.setVelocity(0)


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
    model = RecurrentPPO("MlpLstmPolicy", env, verbose=1)
    model.learn(5000)
    ...

    # Code to load a model and run it
    # For the RecurrentPPO case, consult its documentation
    ...


if __name__ == '__main__':
    main()