import tensorflow as tf
import numpy as np
from models.lstm import Model
import glob
#from stable_baselines.common.base_class import BaseRLModel

class Baseline():

    def __init__(self, env):

        # init 
        self.ac_space = env.action_space
        self.processed_obs = tf.placeholder(shape=[None,env.buf_obs[None].shape[1],env.buf_obs[None].shape[2]],dtype=tf.float32)
        self.comp = ['TROW', 'CMA', 'BEN', 'WFC', 'JPM', 'BK', 'NTRS', 'AXP', 'BAC', 'USB', 'MS', 'RJF', 'C', 'STT', 'SCHW', 'COF', 'IVZ','ETFC','AMG','GS','BLK','AMP','DFS']
        # basic model
        self.models = {}
        for c in self.comp:
            self.models[c] = Model(ticker=c, batch_size=c, epochs=0, lr=0, lookback_days=30, prediction_days=1, dim=1)
            self.models[c].init_model((30,1))
            filename = glob.glob("./saved_weights/"+c+"_*")[0]
            self.models[c].load_model(filename)

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
        output = []
        obs = obs[:,:-1]
        obs = obs[:,:,:-2]
        obs = obs[:,:,0::4]

        for i,c in zip(range(len(self.comp)),self.comp):
            s = obs[:,:,i]
            s1= s[:,:,np.newaxis]
            output.append(self.models[c].model.predict(obs[:,:,i][:,:,np.newaxis])[0,0]-obs[:,-1,i])

        comp_i = np.argmax(output,axis=-1)
        
        actions = np.zeros(self.ac_space.shape)
        actions[::2] = 2
        actions[1::2] = 255
        actions[2*comp_i] = 1
        return np.array([actions]), None, None
