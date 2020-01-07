import tensorflow as tf
import numpy as np
from replay_buffer import *
from loggers import loggers

class DDPG(object):
    def __init__(self,
                 s_dim,
                 a_dim,
                 a_bound,
                 num_robot,
                 lr=1e-4,
                 tau=1e-3,
                 gamma=0.99,
                 batch_size=64,
                 training=True,
                 save_path='./model/',
                 dump_root='./logs'):
        # Parameters Setup
        self.sess = tf.Session()
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.a_bound = a_bound
        self.num_robot = num_robot
        self.lr = lr
        self.tau = tau
        self.gamma = gamma
        self.batch_size = batch_size
        self.training = training
        self.save_path = save_path
        self.dump_root = dump_root
        self.actor = ActorNetwork(self.sess,
                                  self.s_dim,
                                  self.a_dim,
                                  self.a_bound,
                                  self.num_robot,
                                  self.lr,
                                  self.tau,
                                  self.batch_size)
        self.critic = CriticNetwork(self.sess,
                                    self.s_dim,
                                    self.a_dim,
                                    self.num_robot,
                                    self.lr,
                                    self.tau,
                                    self.gamma,
                                    self.actor.get_num_trainable_vars())
        self.replay_buffer = ReplayBuffer(1e6)
        self.a_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(self.num_robot*self.a_dim))
        self.loggers = loggers(self.sess, self.dump_root)

        # Initialization
        self.sess.run(tf.global_variables_initializer())
        self.init_update_target_network()
        self.saver = tf.train.Saver(max_to_keep=30)
        self.update_step = 0
        self.save_frequency = 120000

    def init_update_target_network(self):
        self.actor.init_update_target_network()
        self.critic.init_update_target_network()

    def save_model(self):
        self.saver.save(self.sess, self.save_path + 'ddpg', global_step=self.update_step)

    def load_model(self):
        latest_ckpt = tf.train.latest_checkpoint(self.save_path)
        if latest_ckpt != None:
            self.saver.restore(self.sess, latest_ckpt)
            return True
        else:
            return False

    def buffer_stack(self, state, action, terminal, next_state, reward):
        self.replay_buffer.add(np.reshape(state, (self.s_dim,)),
                               np.reshape(action, (self.num_robot*self.a_dim,)),
                               np.reshape(reward, (self.num_robot,)),
                               terminal,
                               np.reshape(next_state, (self.s_dim,)))

    def buffer_size(self):
        return self.replay_buffer.size()

    def predict(self, state):
        noise = 0
        if self.training:
            noise = self.a_noise()
        return self.actor.predict(np.reshape(state, (1, self.s_dim))) + noise

    def update(self):
        if self.replay_buffer.size() < self.batch_size:
            print("Not enough replay memory to update")
        else:
            s_batch, a_batch, r_batch, t_batch, s2_batch = self.replay_buffer.sample_batch(self.batch_size)
            target_q = self.critic.predict_target(s2_batch, self.actor.predict_target(s2_batch))
            y_i = []
            for k in range(self.batch_size):
                if t_batch[k]:
                    y_i.append(r_batch[k])
                else:
                    y_i.append(r_batch[k] + self.gamma * target_q[k])
            predicted_q_value, _ = self.critic.train(s_batch, a_batch,
                                                     np.reshape(y_i, (int(self.batch_size), self.num_robot)))
            self.loggers.add_ep_ave_max_q(np.amax(predicted_q_value, axis=0))
            a_outs = self.actor.predict(s_batch)
            grads = self.critic.action_gradients(s_batch, a_outs)
            self.actor.train(s_batch, grads[0])
            self.actor.update_target_network()
            self.critic.update_target_network()

            self.update_step += 1

            if self.update_step % self.save_frequency == 0:
                self.save_model()

class ActorNetwork(object):
    """
    Input to the network is the state, output is the action
    under a deterministic policy.
    The output layer activation is a tanh to keep the action
    between -action_bound and action_bound
    """

    def __init__(self, sess, state_dim, action_dim, action_bound, number_of_robots, learning_rate, tau, batch_size):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.action_bound = action_bound
        self.num_robot = number_of_robots
        self.learning_rate = learning_rate
        self.tau = tau
        self.batch_size = batch_size
        self.summary_list = []

        # Actor Network
        with tf.variable_scope('ActorNetwork'):
            self.inputs, self.out, self.scaled_out = self.create_actor_network()

        self.network_params = tf.trainable_variables()

        # Target Network
        with tf.variable_scope('ActorTargetNetwork'):
            self.target_inputs, self.target_out, self.target_scaled_out = self.create_actor_network()

        self.target_network_params = tf.trainable_variables()[len(self.network_params):]

        # Op for periodically updating target network with online network
        # weights
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) +
                                                  tf.multiply(self.target_network_params[i], 1. - self.tau))
                for i in range(len(self.target_network_params))]

        # This gradient will be provided by the critic network
        self.action_gradient = tf.placeholder(tf.float32, [None, self.num_robot*self.a_dim], name='action_gradient')

        # Combine the gradients here
        self.unnormalized_actor_gradients = tf.gradients(
            self.scaled_out, self.network_params, -self.action_gradient)
        self.actor_gradients = list(map(lambda x: tf.div(x, self.batch_size), self.unnormalized_actor_gradients))

        # Optimization Op
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).\
            apply_gradients(zip(self.actor_gradients, self.network_params))

        self.num_trainable_vars = len(self.network_params) + len(self.target_network_params)

        self.summary = tf.summary.merge(self.summary_list)

    def create_actor_network(self):
        inputs = tf.placeholder(tf.float32, shape=[None, self.s_dim], name='inputs')

        unit_outs = []
        unit_scaled_outs = []

        for i in range(self.num_robot):
            tag = 'agent_' + str(i)
            with tf.variable_scope(tag):
                unit_out, unit_scaled_out = self.create_actor_unit_network(inputs)
                unit_outs.append(unit_out)
                unit_scaled_outs.append(unit_scaled_out)

        out = tf.concat(unit_outs, axis=1)
        scaled_out = tf.concat(unit_scaled_outs, axis=1)

        return inputs, out, scaled_out

    def create_actor_unit_network(self, inputs):
        # Fully Connected Layer 1
        fc1_num_neurons = 400
        W_fc1 = tf.Variable(tf.truncated_normal([self.s_dim, fc1_num_neurons], stddev=0.02), name='W_fc1')
        self.summary_list.append(tf.summary.histogram("W_fc1_hist", W_fc1))
        b_fc1 = tf.Variable(tf.constant(0., shape=[fc1_num_neurons]), name='b_fc1')
        self.summary_list.append(tf.summary.histogram("b_fc1_hist", b_fc1))
        out_fc1 = tf.nn.relu(tf.matmul(inputs, W_fc1) + b_fc1, name='out_fc1')
        self.summary_list.append(tf.summary.histogram("out_fc1_hist", out_fc1))

        # Fully Connected Layer 2
        fc2_num_neurons = 300
        W_fc2 = tf.Variable(tf.truncated_normal([fc1_num_neurons, fc2_num_neurons], stddev=0.02), name='W_fc2')
        self.summary_list.append(tf.summary.histogram("W_fc2_hist", W_fc2))
        b_fc2 = tf.Variable(tf.constant(0., shape=[fc2_num_neurons]), name='b_fc2')
        self.summary_list.append(tf.summary.histogram("b_fc2_hist", b_fc2))
        out_fc2 = tf.nn.relu(tf.matmul(out_fc1, W_fc2) + b_fc2, name='out_fc2')
        self.summary_list.append(tf.summary.histogram("out_fc2_hist", out_fc2))

        # Output Layer
        W_out = tf.Variable(tf.random_uniform([fc2_num_neurons, self.a_dim], minval=-0.003, maxval=0.003), name='W_out')
        self.summary_list.append(tf.summary.histogram("W_out_hist", W_out))
        b_out = tf.Variable(tf.constant(0., shape=[self.a_dim], name='b_out'))
        self.summary_list.append(tf.summary.histogram("b_out_hist", b_out))
        out = tf.nn.tanh(tf.matmul(out_fc2, W_out) + b_out, name='out')
        self.summary_list.append(tf.summary.histogram("out_hist", out))

        scaled_out = tf.multiply(out, self.action_bound)
        return out, scaled_out

    def train(self, inputs, a_gradient):
        self.sess.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.action_gradient: a_gradient
        })

    def predict(self, inputs):
        return self.sess.run(self.scaled_out, feed_dict={
            self.inputs: inputs
        })

    def predict_target(self, inputs):
        return self.sess.run(self.target_scaled_out, feed_dict={
            self.target_inputs: inputs
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars

    def init_update_target_network(self):
        init_update_target_network_params = [self.target_network_params[i].assign(self.network_params[i])
                                             for i in range(len(self.target_network_params))]
        self.sess.run(init_update_target_network_params)

class CriticNetwork(object):
    """
    Input to the network is the state and action, output is Q(s,a).
    The action must be obtained from the output of the Actor network.
    """

    def __init__(self, sess, state_dim, action_dim, number_of_robots, learning_rate, tau, gamma, num_actor_vars):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.num_robot = number_of_robots
        self.split_window = [action_dim for i in range(number_of_robots)]
        self.learning_rate = learning_rate
        self.tau = tau
        self.gamma = gamma
        self.summary_list = []

        # Create the critic network
        with tf.variable_scope('CriticNetwork'):
            self.inputs, self.action, self.out = self.create_critic_network()

        self.network_params = tf.trainable_variables()[num_actor_vars:]

        # Target Network
        with tf.variable_scope('CriticTargetNetwork'):
            self.target_inputs, self.target_action, self.target_out = self.create_critic_network()

        self.target_network_params = tf.trainable_variables()[(len(self.network_params) + num_actor_vars):]

        # Op for periodically updating target network with online network
        # weights with regularization
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) \
            + tf.multiply(self.target_network_params[i], 1. - self.tau))
                for i in range(len(self.target_network_params))]

        # Network target (y_i)
        self.predicted_q_value = tf.placeholder(tf.float32, [None, self.num_robot], name='predicted_q_value')

        # Define loss and optimization Op
        self.loss = tf.losses.mean_squared_error(self.out, self.predicted_q_value)
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        # Get the gradient of the net w.r.t. the action.
        # For each action in the minibatch (i.e., for each x in xs),
        # this will sum up the gradients of each critic output in the minibatch
        # w.r.t. that action. Each output is independent of all
        # actions except for one.
        self.action_grads = tf.gradients(self.out, self.action, name='action_grads')

        self.summary = tf.summary.merge(self.summary_list)

    def create_critic_network(self):
        inputs = tf.placeholder(tf.float32, shape=[None, self.s_dim], name='inputs')
        action = tf.placeholder(tf.float32, shape=[None, self.num_robot*self.a_dim], name='action')

        actions = tf.split(action, self.split_window, 1)

        unit_outs = []

        for i in range(self.num_robot):
            tag = 'agent_' + str(i)
            with tf.variable_scope(tag):
                unit_out = self.create_critic_unit_network(inputs, actions[i])
                unit_outs.append(unit_out)

        out = tf.concat(unit_outs, axis=1)

        return inputs, action, out

    def create_critic_unit_network(self, inputs, action):
        # Fully Connected Layer 1
        fc1_num_neurons = 400
        W_fc1 = tf.Variable(tf.truncated_normal([self.s_dim, fc1_num_neurons], stddev=0.02), name='W_fc1')
        self.summary_list.append(tf.summary.histogram("W_fc1_hist", W_fc1))
        b_fc1 = tf.Variable(tf.constant(0., shape=[fc1_num_neurons]), name='b_fc1')
        self.summary_list.append(tf.summary.histogram("b_fc1_hist", b_fc1))
        out_fc1 = tf.nn.relu(tf.matmul(inputs, W_fc1) + b_fc1, name='out_fc1')
        self.summary_list.append(tf.summary.histogram("out_fc1_hist", out_fc1))

        # Fully Connected Layer 2 for inputs and action
        fc2_num_neurons = 300
        W_fc2_x = tf.Variable(tf.truncated_normal([fc1_num_neurons, fc2_num_neurons], stddev=0.02), name='W_fc2_x')
        self.summary_list.append(tf.summary.histogram("W_fc2_x_hist", W_fc2_x))
        W_fc2_a = tf.Variable(tf.truncated_normal([self.a_dim, fc2_num_neurons], stddev=0.02), name='W_fc2_a')
        self.summary_list.append(tf.summary.histogram("W_fc2_a_hist", W_fc2_a))
        b_fc2 = tf.Variable(tf.constant(0., shape=[fc2_num_neurons]), name='b_fc2')
        self.summary_list.append(tf.summary.histogram("b_fc2_hist", b_fc2))
        out_fc2 = tf.nn.relu(tf.matmul(out_fc1, W_fc2_x) + tf.matmul(action, W_fc2_a) + b_fc2, name='out_fc2')
        self.summary_list.append(tf.summary.histogram("out_fc2_hist", out_fc2))

        # Output Layer
        W_out = tf.Variable(tf.random_uniform([fc2_num_neurons, 1], minval=-0.003, maxval=0.003), name='W_out')
        self.summary_list.append(tf.summary.histogram("W_out_hist", W_out))
        b_out = tf.Variable(tf.constant(0., shape=[1], name='b_out'))
        self.summary_list.append(tf.summary.histogram("W_fc1_hist", b_out))
        out = tf.add(tf.matmul(out_fc2, W_out), b_out, name='out')
        self.summary_list.append(tf.summary.histogram("out_hist", out))

        return out


    def train(self, inputs, action, predicted_q_value):
        return self.sess.run([self.out, self.optimize], feed_dict={
            self.inputs: inputs,
            self.action: action,
            self.predicted_q_value: predicted_q_value
        })

    def predict(self, inputs, action):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs,
            self.action: action
        })

    def predict_target(self, inputs, action):
        return self.sess.run(self.target_out, feed_dict={
            self.target_inputs: inputs,
            self.target_action: action
        })

    def action_gradients(self, inputs, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.inputs: inputs,
            self.action: actions
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def init_update_target_network(self):
        init_update_target_network_params = [self.target_network_params[i].assign(self.network_params[i])
                                             for i in range(len(self.target_network_params))]
        self.sess.run(init_update_target_network_params)

# Taken from https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py, which is
# based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma=0.2, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)


if __name__ == "__main__":
    sess = tf.Session()
    agent = DDPG(sess, 180, 2, [1, 1])
