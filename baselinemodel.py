import tensorflow as tf
import numpy as np

#from stable_baselines.common.base_class import BaseRLModel

class Baseline():
    def __init__(self, env):

        # init 
        self.ac_space = env.action_space
        self.processed_obs = tf.placeholder(shape=[None,env.buf_obs[None].shape[1],env.buf_obs[None].shape[2]],dtype=tf.float32)

        # basic model
        with tf.variable_scope("model", reuse=False):
            activ = tf.nn.relu
            frame, _ = self.processed_obs[:,:-1],self.processed_obs[:,-1]
            frame_features = tf.keras.layers.LSTM(units=64,activation=activ)(frame[:,:-2])
            
            self.logit = tf.keras.layers.Dense(int(self.ac_space.shape[0]/2))(frame_features)

        # setup
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        self.sess = sess

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

        output = self.sess.run([self.logit], {self.processed_obs: obs})
        comp_i = np.argmax(output,axis=-1)
        actions = []
        for i in range(len(comp_i)):
            action = np.zeros(self.ac_space.shape)
            action[::2] = 2
            action[1::2] = 255
            action[2*comp_i] = 1
            actions.append(action)
        actions = np.array(actions)
        return actions, None, None
