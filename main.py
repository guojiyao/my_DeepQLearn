import gym
import random
import logging
import tensorflow as tf
import sys

#from environments.environment import ToyEnvironment, AtariEnvironment
from environments.torcsenvironment import TorcsEnvironment
from networks.cnn import CNN
from agents.statistic import Statistic
from utils import get_model_dir
from agents.agent import Agent

flags = tf.app.flags
# env
flags.DEFINE_string('agent_type', 'DQN', 'The type of agent [DQN]')
flags.DEFINE_string('env_name', 'Torcs', 'The name of gym environment to use')
flags.DEFINE_integer('n_action_repeat', 4, 'The number of actions to repeat')
flags.DEFINE_integer('max_random_start', 30, 'The maximum number of NOOP actions at the beginning of an episode')
flags.DEFINE_string('observation_dims', '[64, 64]', 'The dimension of gym observation')
flags.DEFINE_boolean('use_gpu', False, 'Whether to use gpu or not. gpu use NHWC and gpu use NCHW for data_format')
flags.DEFINE_boolean('display', False, 'Whether to do display the game screen or not')

#modeling
flags.DEFINE_string('gpu_fraction', '1/1', 'idx / # of gpu fraction e.g. 1/3, 2/3, 3/3')
flags.DEFINE_integer('history_length', 4, 'The length of history of observation to use as an input to DQN')
flags.DEFINE_string('network_header_type', 'nips', 'The type of network header [mlp, nature, nips]')

flags.DEFINE_integer('t_test', 1, 'The maximum number of t while training (*= scale)')
flags.DEFINE_float('t_learn_start', 5, 'The time when to begin training (*= scale)')

# training agent
flags.DEFINE_float('ep_start', 1., 'The value of epsilon at start in e-greedy')
flags.DEFINE_float('ep_end', 0.01, 'The value of epsilnon at the end in e-greedy')
flags.DEFINE_integer('t_ep_end', 100, 'The time when epsilon reach ep_end (*= scale)')
flags.DEFINE_integer('t_train_freq', 4, '')
flags.DEFINE_integer('t_target_q_update_freq', 1, 'The frequency of target network to be updated (*= scale)')
flags.DEFINE_integer('discount_r', 0.99, 'The discount factor for reware')
flags.DEFINE_integer('max_r', +1, 'The maximum value of clipped reward')
flags.DEFINE_integer('min_r', -1, 'The minimum value of clipped reward')
flags.DEFINE_integer('max_delta', None, 'The maximum value of delta')
flags.DEFINE_integer('min_delta', None, 'The minimum value of delta')
flags.DEFINE_integer('max_grad_norm', None, 'The maximum norm of gradient while updating')
flags.DEFINE_float('learning_rate_decay_step', 5, 'The learning rate of training (*= scale)')

# Optimizer
flags.DEFINE_float('learning_rate', 0.00025, 'The learning rate of training')
flags.DEFINE_float('learning_rate_minimum', 0.00025, 'The learning rate of training')
flags.DEFINE_float('learning_rate_decay', 0.96, 'The learning rate of training')
flags.DEFINE_boolean('double_q', False, 'Whether to use double Q-learning')

flags.DEFINE_integer('batch_size', 32, 'The size of batch for minibatch training')
flags.DEFINE_integer('memory_size', 100, 'The size of experience memory (*= scale)')



def calc_gpu_fraction(fraction_string):
	idx, num = fraction_string.split('/')
	idx, num = float(idx), float(num)

	fraction = 1 / (num - idx + 1)
	print (" [*] GPU : %.4f" % fraction)
	return fraction


conf = flags.FLAGS
if conf.agent_type == 'DQN':
	from agents.deep_q import DeepQ
	TrainAgent = DeepQ
else:
	raise ValueError('Unknown agent_type: %s' % conf.agent_type)


#if conf.agent_type == 'DQN':
#	from agents.deep_q import DeepQ
#	TrainAgent = DeepQ
#else:
	#raise ValueError('Unknown agent_type: %s' % conf.agent_type)

if conf.use_gpu:
	conf.data_format = 'NCHW'
else:
	conf.data_format = 'NHWC'

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=calc_gpu_fraction(conf.gpu_fraction))

conf.observation_dims = eval(conf.observation_dims)

model_dir = get_model_dir(conf,
	['use_gpu', 'max_random_start', 'n_worker', 'is_train', 'memory_size', 'gpu_fraction',
	't_save', 't_train', 'display', 'log_level', 'random_seed', 'tag', 'scale'])

conf.env_name = "Torcs"

with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:


	if conf.env_name == "Torcs":
		env = TorcsEnvironment(observation_dims=conf.observation_dims)
		if conf.network_header_type in ['nature', 'nips']:
		    pred_network = CNN(sess=sess,
		                         data_format=conf.data_format,
		                         history_length=conf.history_length,
		                         observation_dims=conf.observation_dims,
		                         output_size=env.action_space,
		                         network_header_type=conf.network_header_type,
		                         name='pred_network', trainable=True)
		    target_network = CNN(sess=sess,
		                           data_format=conf.data_format,
		                           history_length=conf.history_length,
		                           observation_dims=conf.observation_dims,
		                           output_size=env.action_space,
		                           network_header_type=conf.network_header_type,
								   name='target_network', trainable=False)

		stat = Statistic(sess, conf.t_test, conf.t_learn_start, model_dir, list(pred_network.var.values()))
		agent = Agent(sess, pred_network, env, stat, conf, target_network=target_network)

		agent.play(conf.ep_end)