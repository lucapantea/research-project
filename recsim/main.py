# coding=utf-8
# coding=utf-8
# Copyright 2019 The RecSim Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""An example of main function for training in RecSim.

Use the interest evolution environment and a slateQ agent for illustration.

To run locally:

python main.py --base_dir=/tmp/interest_evolution \
  --gin_bindings=simulator.runner_lib.Runner.max_steps_per_episode=50

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
import numpy as np
from recsim.agents import random_agent
from recsim.agents import tabular_q_agent
from recsim.agents import full_slate_q_agent
from recsim.agents import slate_decomp_q_agent
from recsim.agents.slate_decomp_q_agent import select_slate_greedy, compute_target_greedy_q
from recsim.environments import preference_dynamics
from recsim.simulator import runner_lib

FLAGS = flags.FLAGS
flags.DEFINE_string('algorithm', 'full_slate', 'The algorithm to run the experiments with.')


def create_agent(sess, environment, eval_mode, summary_writer=None):
    """Creates an instance of FullSlateQAgent.

  Args:
    sess: A `tf.Session` object for running associated ops.
    environment: A recsim Gym environment.
    eval_mode: A bool for whether the agent is in training or evaluation mode.
    summary_writer: A Tensorflow summary writer to pass to the agent for
      in-agent training statistics in Tensorboard.

  Returns:
    An instance of FullSlateQAgent.
  """
    kwargs = {
        'observation_space': environment.observation_space,
        'action_space': environment.action_space,
        'summary_writer': summary_writer,
        'eval_mode': eval_mode,
    }
    return full_slate_q_agent.FullSlateQAgent(sess, **kwargs)

def create_random_agent(sess, environment, eval_mode, summary_writer=None):
    """Returns an instance of RandomAgent"""
    del sess
    del eval_mode
    del summary_writer
    kwargs = {
        'action_space': environment.action_space,
    }
    return random_agent.RandomAgent(**kwargs)

def create_tabular_q_agent(sess, environment, eval_mode, summary_writer=None):
    """Returns an instance of RandomAgent"""
    del sess
    kwargs = {
        'observation_space': environment.observation_space,
        'action_space': environment.action_space,
        'summary_writer': summary_writer,
        'eval_mode': eval_mode
    }
    return tabular_q_agent.TabularQAgent(**kwargs)

def create_full_slateQ_agent(sess, environment, eval_mode, summary_writer=None):
    """Returns an instance of FullSlateQAgent"""
    kwargs = {
        'observation_space': environment.observation_space,
        'action_space': environment.action_space,
        'summary_writer': summary_writer,
        'eval_mode': eval_mode
    }
    return full_slate_q_agent.FullSlateQAgent(sess, **kwargs)


def create_slate_decomp_agent(sess, environment, eval_mode, summary_writer=None):
    """Returns an instance of SlateDecompQAgent"""
    kwargs = {
        'observation_space': environment.observation_space,
        'action_space': environment.action_space,
        'summary_writer': summary_writer,
        'eval_mode': eval_mode,
        'select_slate_fn': select_slate_greedy,
        'compute_target_fn': compute_target_greedy_q
    }
    return slate_decomp_q_agent.SlateDecompQAgent(sess, **kwargs)


def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')

    runner_lib.load_gin_configs(FLAGS.gin_files, FLAGS.gin_bindings)
    base_dir = FLAGS.base_dir + '/inevolution'
    algorithm = FLAGS.algorithm
    seed = 0
    slate_size = 2
    num_candidates = 10
    np.random.seed(seed)
    env_config = {
        'num_candidates': num_candidates,
        'slate_size': slate_size,
        'resample_documents': True,
        'seed': seed,
    }
    for algorithm in ['full_slate_q', 'slate_decomp_q', 'tabular_q_agent', 'random']:
        if algorithm == 'full_slate_q':
            experiment_dir = base_dir + '/full_slate_q'
            train, evaluate = experiment_config(experiment_dir, create_full_slateQ_agent, env_config)
            train.run_experiment()
            evaluate.run_experiment()
        elif algorithm == 'slate_decomp_q':
            experiment_dir = base_dir + '/slate_decomp_q'
            train, evaluate = experiment_config(experiment_dir, create_slate_decomp_agent, env_config)
            train.run_experiment()
            evaluate.run_experiment()
        elif algorithm == 'tabular_q_agent':
            experiment_dir = base_dir + '/tabular_q_agent'
            train, evaluate = experiment_config(experiment_dir, create_tabular_q_agent, env_config)
            train.run_experiment()
            evaluate.run_experiment()
        elif algorithm == 'random':
            experiment_dir = base_dir + '/random'
            train, evaluate = experiment_config(experiment_dir, create_random_agent, env_config)
            train.run_experiment()
            evaluate.run_experiment()
        else:
            print(f'No algorithm found: {algorithm}.')

def experiment_config(base_dir, create_agent_fn, env_config):
    t_runner = runner_lib.TrainRunner(
        base_dir=base_dir,
        create_agent_fn=create_agent_fn,
        env=preference_dynamics.create_environment(env_config),
        episode_log_file=FLAGS.episode_log_file,
        max_training_steps=50,
        num_iterations=20)
    e_runner = runner_lib.EvalRunner(
        base_dir=base_dir,
        create_agent_fn=create_agent_fn,
        env=preference_dynamics.create_environment(env_config),
        max_eval_episodes=200,
        test_mode=True)
    return t_runner, e_runner


if __name__ == '__main__':
    flags.mark_flag_as_required('base_dir')
    app.run(main)
