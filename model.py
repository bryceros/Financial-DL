import tensorflow as tf

from stable_baselines.common.policies import ActorCriticPolicy, register_policy, nature_cnn


class CustomPolicy(ActorCriticPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **kwargs):
        super(CustomPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse, scale=True)

        with tf.variable_scope("model", reuse=reuse):
            activ = tf.nn.relu

            #extracted_features = tf.layers.flatten(self.processed_obs)

            #extracted_features = tf.keras.layers.LSTM(units=64,activation=activ)(self.processed_obs)

            frame, profile = self.processed_obs[:,:-1],self.processed_obs[:,-1]
            frame = frame[:,:,:-2]
            frame_features = tf.keras.layers.LSTM(units=64,activation=activ)(frame)
            profile_features = profile
            extracted_features = tf.keras.layers.concatenate([frame_features,profile_features])

            pi_h = extracted_features
            for i, layer_size in enumerate([3, 128]):
                pi_h = activ(tf.layers.dense(pi_h, layer_size, name='pi_fc' + str(i)))
            pi_latent = pi_h

            vf_h = extracted_features
            for i, layer_size in enumerate([3, 32]):
                vf_h = activ(tf.layers.dense(vf_h, layer_size, name='vf_fc' + str(i)))
            value_fn = tf.layers.dense(vf_h, 1, name='vf')
            vf_latent = vf_h

            self._proba_distribution, self._policy, self.q_value = \
                self.pdtype.proba_distribution_from_latent(pi_latent, vf_latent, init_scale=1)

        self._value_fn = value_fn
        self._setup_init()
    def save(self,filename):
        saver = tf.train.Saver()
        model_path = saver.save(self.sess, filename+".ckpt")
        print("Model saved in %s" % model_path)

    def load(self,filename):
        saver = tf.train.Saver()
        saver.restore(self.sess, filename+".ckpt")
        print("Model loaded in %s" % filename+".ckpt")

    def step(self, obs, state=None, mask=None, deterministic=False):
        if deterministic:
            action, value, neglogp = self.sess.run([self.deterministic_action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        else:
            action, value, neglogp = self.sess.run([self.action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        return action, value, self.initial_state, neglogp

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})

    def value(self, obs, state=None, mask=None):
        return self.sess.run(self.value_flat, {self.obs_ph: obs})
