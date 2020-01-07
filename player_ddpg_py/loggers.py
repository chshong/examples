from __future__ import division
import tensorflow as tf
import numpy as np

class loggers(object):
    def __init__(self, sess, dump_root, period=100):
        self.sess = sess
        self.dump_root = dump_root
        self.writer = tf.summary.FileWriter(dump_root, self.sess.graph)
        self.summary_ops, self.summary_vars = self.build_summaries()
        self.ep_reward = np.array([0., 0., 0., 0., 0.])
        self.ep_ave_max_q = np.array([0., 0., 0., 0., 0.])
        self.period = period

    def build_summaries(self):
        episode_reward = tf.Variable([0., 0., 0., 0., 0.])
        episode_ave_max_q = tf.Variable([0., 0., 0., 0., 0.])
        summary_vars = [episode_reward, episode_ave_max_q]
        summary_ops = tf.summary.merge([
                                        tf.summary.scalar("Avg_Reward_0", episode_reward[0]),
                                        tf.summary.scalar("Avg_Reward_1", episode_reward[1]),
                                        tf.summary.scalar("Avg_Reward_2", episode_reward[2]),
                                        tf.summary.scalar("Avg_Reward_3", episode_reward[3]),
                                        tf.summary.scalar("Avg_Reward_4", episode_reward[4]),
                                        tf.summary.scalar("Avg_Qmax_Value_0", episode_ave_max_q[0]),
                                        tf.summary.scalar("Avg_Qmax_Value_1", episode_ave_max_q[1]),
                                        tf.summary.scalar("Avg_Qmax_Value_2", episode_ave_max_q[2]),
                                        tf.summary.scalar("Avg_Qmax_Value_3", episode_ave_max_q[3]),
                                        tf.summary.scalar("Avg_Qmax_Value_4", episode_ave_max_q[4])])

        return summary_ops, summary_vars

    def write_summaries(self, episode_step, episode):
        if episode % self.period == 0:
            if episode_step != 0:
                summary_str = self.sess.run(self.summary_ops, feed_dict={
                    self.summary_vars[0]: self.ep_reward / self.period,
                    self.summary_vars[1]: self.ep_ave_max_q / (episode_step*self.period)
                })
                self.writer.add_summary(summary_str, episode)
                self.writer.flush()
                print('| Reward: {} | Episode: {:d} | Qmax: {}'.format(self.ep_reward/self.period, \
                                episode, (self.ep_ave_max_q / (episode_step*self.period))))
            self.reset()

    def add_ep_ave_max_q(self, ep_ave_max_q):
        self.ep_ave_max_q += ep_ave_max_q

    def add_reward(self, ep_reward):
        self.ep_reward += ep_reward

    def reset(self):
        self.ep_reward = np.array([0., 0., 0., 0., 0.])
        self.ep_ave_max_q = np.array([0., 0., 0., 0., 0.])
