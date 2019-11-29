import numpy as np

#from stable_baselines.common.base_class import BaseRLModel

class Heristic():
    def __init__(self, env):
        self.ac_space = env.action_space


    def predict(self, observation, state=None, mask=None, deterministic=False):
        """
        Get the model's action from an observation

        :param observation: (np.ndarray) the input observation
        :param state: (np.ndarray) The last states (can be None, used in recurrent policies)
        :param mask: (np.ndarray) The last masks (can be None, used in recurrent policies)
        :param deterministic: (bool) Whether or not to return deterministic actions.
        :return: (np.ndarray, np.ndarray) the model's action and the next state (used in recurrent policies)
        """
        actions,_,states = self.step(obs=observation,state=state)
        return actions, states
    def step(self, obs, state=None, mask=None, deterministic=True):
        """
        Returns the q_values for a single step

        :param obs: (np.ndarray float or int) The current observation of the environment
        :param state: (np.ndarray float) The last states (used in recurrent policies)
        :param mask: (np.ndarray float) The last masks (used in recurrent policies)
        :param deterministic: (bool) Whether or not to return deterministic actions.
        :return: (np.ndarray int, np.ndarray float, np.ndarray float) actions, q_values, states
        """

        frames = obs[:,:-1]
        frame = frames[:,:,:-2]


        comp = []
        aver = []
        for comp_i in range(int(self.ac_space.shape[0]/2)):
            top = []
            bottom = []
            for i in range(frame.shape[1]):
                top.append(frame[0][i][int(4*comp_i+2)])
                bottom.append(frame[0][i][int(4*comp_i)+3])
            comp.append(np.array(top) / np.array(bottom))
            aver.append(np.mean(comp))
        comp = np.array(comp)
        aver = np.array(aver)
        actions_index = np.argmax(aver - comp[:,-1])

        actions = np.zeros(self.ac_space.shape)
        actions[::2] = 2
        actions[1::2] = 255
        actions[2*actions_index] = 1
        actions = np.array(actions).reshape(1,-1)
        return actions, None, None
