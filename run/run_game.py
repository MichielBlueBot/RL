import multiprocessing
import os
import threading
from functools import partial
from time import sleep

import tensorflow as tf
from component.actor_critic_network import ActorCriticNetwork
from component.game import Game
from component.worker import Worker

max_episode_length = 300
memory_length = 5
gamma = .99  # discount rate for advantage estimation and reward discounting
state_space_size = 52 + 52  # Observations are cards in stack and cards in hand
action_space_size = 52 + 2  # pop deck, take stack, throw away card (52)
load_model = False
model_path = './model'

if __name__ == '__main__':
    tf.reset_default_graph()

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # Create a directory to save episode playback gifs to
    if not os.path.exists('./frames'):
        os.makedirs('./frames')

    with tf.device("/cpu:0"):
        global_episodes = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)
        trainer = tf.train.AdamOptimizer(learning_rate=1e-4)
        master_network = ActorCriticNetwork(state_space_size, action_space_size, 'global', None)  # global network
        num_workers = multiprocessing.cpu_count()  # Set workers to number of available CPU threads
        workers = []
        # Create worker classes
        for i in range(num_workers):
            workers.append(Worker(Game(), i, state_space_size, action_space_size,
                                  trainer, model_path, global_episodes, memory_length))
        saver = tf.train.Saver(max_to_keep=5)

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        if load_model:
            print('Loading Model...')
            checkpoint = tf.train.get_checkpoint_state(model_path)
            saver.restore(sess, checkpoint.model_checkpoint_path)
        else:
            sess.run(tf.global_variables_initializer())

        # This is where the asynchronous magic happens.
        # Start the "work" process for each worker in a separate threat.
        worker_threads = []
        for worker in workers:
            worker_work = partial(func=worker.work, gamme=gamma, sess=sess, coord=coord, saver=saver)
            t = threading.Thread(target=worker_work)
            t.start()
            sleep(0.5)
            worker_threads.append(t)
        coord.join(worker_threads)
