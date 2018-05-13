import numpy as np
import tensorflow as tf
from tensorflow.python.training import optimizer

from base.util import update_target_graph, discount
from component.actor_critic_network import ActorCriticNetwork
from component.game import Game


class Worker:
    def __init__(self, game: Game, name: str, state_space_size: int, action_space_size: int,
                 trainer: optimizer.Optimizer, model_path: str, global_episodes: tf.Variable, memory_length: int):
        self.name = "worker_" + str(name)
        self.number = name
        self.model_path = model_path
        self.trainer = trainer
        self.memory_length = memory_length
        self.global_episodes = global_episodes
        self.increment = self.global_episodes.assign_add(1)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_mean_values = []
        self.summary_writer = tf.summary.FileWriter("train_" + str(self.number))

        # Create the local copy of the network and the tensorflow op to copy global parameters to local network
        self.local_network = ActorCriticNetwork(state_space_size, action_space_size, self.name, trainer)
        self.update_local_ops = update_target_graph('global', self.name)

        self.actions = np.identity(action_space_size, dtype=bool).tolist()
        self.env = game
        self.batch_rnn_state = None
        self.rewards_plus = 0
        self.value_plus = 0

    def train(self, rollout, sess, gamma, bootstrap_value):
        rollout = np.array(rollout)
        observations = rollout[:, 0]
        actions = rollout[:, 1]
        rewards = rollout[:, 2]
        next_observations = rollout[:, 3]
        values = rollout[:, 5]

        # Here we take the rewards and values from the rollout, and use them to
        # generate the advantage and discounted returns.
        # The advantage function uses "Generalized Advantage Estimation"
        self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        discounted_rewards = discount(self.rewards_plus, gamma)[:-1]
        self.value_plus = np.asarray(values.tolist() + [bootstrap_value])
        advantages = rewards + gamma * self.value_plus[1:] - self.value_plus[:-1]
        advantages = discount(advantages, gamma)

        # Update the global network using gradients from loss
        # Generate network statistics to periodically save
        feed_dict = {self.local_network.target_v: discounted_rewards,
                     self.local_network.inputs: np.vstack(observations),
                     self.local_network.actions: actions,
                     self.local_network.advantages: advantages,
                     self.local_network.state_in[0]: self.batch_rnn_state[0],
                     self.local_network.state_in[1]: self.batch_rnn_state[1]}
        v_l, p_l, e_l, g_n, v_n, self.batch_rnn_state, _ = sess.run([self.local_network.value_loss,
                                                                     self.local_network.policy_loss,
                                                                     self.local_network.entropy,
                                                                     self.local_network.grad_norms,
                                                                     self.local_network.var_norms,
                                                                     self.local_network.state_out,
                                                                     self.local_network.apply_grads],
                                                                    feed_dict=feed_dict)
        return v_l / len(rollout), p_l / len(rollout), e_l / len(rollout), g_n, v_n

    def work(self, gamma, sess, coord, saver):
        episode_count = sess.run(self.global_episodes)
        total_steps = 0
        print("Starting worker " + str(self.number))
        with sess.as_default(), sess.graph.as_default():
            while not coord.should_stop():
                # Copy variables from global network to local network
                sess.run(self.update_local_ops)
                episode_buffer = []
                episode_values = []
                episode_states = []
                episode_reward = 0
                episode_step_count = 0

                self.env.new_episode()
                state = self.env.get_state()
                episode_states.append(state)
                rnn_state = self.local_network.state_init
                self.batch_rnn_state = rnn_state
                # Play the game until it ends
                while not self.env.is_episode_finished():
                    # Take an action using probabilities from policy network output.
                    policy, value, rnn_state = sess.run(
                        [self.local_network.policy, self.local_network.value, self.local_network.state_out],
                        feed_dict={self.local_network.inputs: episode_states[-self.memory_length:],
                                   self.local_network.state_in[0]: rnn_state[0],
                                   self.local_network.state_in[1]: rnn_state[1]})
                    # action = np.random.choice(policy[-1], p=policy[-1])
                    # # noinspection PyTypeChecker
                    # action = np.argmax(policy == action)

                    reward, actions = self.env.perform_action(policy[-1]) / 100.0
                    episode_finished = self.env.is_episode_finished()

                    new_state = self.env.get_state()

                    if not episode_finished:
                        episode_states.append(new_state)

                    episode_buffer.append([state, actions, reward, new_state, episode_finished, value[-1, 0]])
                    episode_values.append(value[-1, 0])

                    episode_reward += reward
                    state = new_state
                    total_steps += 1
                    episode_step_count += 1

                    # # If the episode hasn't ended, but the experience buffer is full, then we
                    # # make an update step using that experience rollout.
                    # if len(episode_buffer) == 30 and not d and episode_step_count != max_episode_length - 1:
                    #     # Since we don't know what the true final return is, we "bootstrap" from our current
                    #     # value estimation.
                    #     rnn_inputs = episode_states[-self.memory_length+1:]+[s]
                    #     v1 = sess.run(self.local_network.value,
                    #                   feed_dict={self.local_network.inputs: rnn_inputs,
                    #                              self.local_network.state_in[0]: rnn_state[0],
                    #                              self.local_network.state_in[1]: rnn_state[1]})[-1, 0]
                    #     v_l, p_l, e_l, g_n, v_n = self.train(episode_buffer, sess, gamma, v1)
                    #     episode_buffer = []
                    #     sess.run(self.update_local_ops)
                    if episode_finished:
                        break

                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_step_count)
                self.episode_mean_values.append(np.mean(episode_values))

                # Update the network using the episode buffer at the end of the episode.
                if len(episode_buffer) != 0:
                    v_l, p_l, e_l, g_n, v_n = self.train(episode_buffer, sess, gamma, 0.0)

                # Periodically save model parameters, and summary statistics.
                if episode_count % 5 == 0 and episode_count != 0:
                    if episode_count % 250 == 0 and self.name == 'worker_0':
                        saver.save(sess, self.model_path + '/model-' + str(episode_count) + '.cptk')
                        print("Saved Model")

                    mean_reward = np.mean(self.episode_rewards[-5:])
                    mean_length = np.mean(self.episode_lengths[-5:])
                    mean_value = np.mean(self.episode_mean_values[-5:])
                    summary = tf.Summary()
                    summary.value.add(tag='Perf/Reward', simple_value=float(mean_reward))
                    summary.value.add(tag='Perf/Length', simple_value=float(mean_length))
                    summary.value.add(tag='Perf/Value', simple_value=float(mean_value))
                    summary.value.add(tag='Losses/Value Loss', simple_value=float(v_l))
                    summary.value.add(tag='Losses/Policy Loss', simple_value=float(p_l))
                    summary.value.add(tag='Losses/Entropy', simple_value=float(e_l))
                    summary.value.add(tag='Losses/Grad Norm', simple_value=float(g_n))
                    summary.value.add(tag='Losses/Var Norm', simple_value=float(v_n))
                    self.summary_writer.add_summary(summary, episode_count)

                    self.summary_writer.flush()
                if self.name == 'worker_0':
                    sess.run(self.increment)
                episode_count += 1
