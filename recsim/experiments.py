r""" Training and Evaluating the Interest Exploration environment.

Using the Interest Exploration environment and a slateQ agent.

This experiment illustrates the problem of active exploration of user interests.
It is meant to show the "popularity bias" in recommender systems, where myopic
maximization of engagement leads to bias towards particular documents that have wider appeal,
yet the more niche user interests are not explored.

Experimental setup:
- RL Algorithm slateQ: FullSlateQAgent / SlateDecompQAgent
- Document:
    - documents are generated from M topics (types). 1 doc -> 1 type
    - documents have a production quality score f_D(d) [Dependent on types]
- User:
    - users are generated from N types
    - users have an affinity score g_U(u, d) towards each document type
    - combined score for users (u, d) = g_U(u, d) + f_D(d)
- User selects a document according to a multinomial logit choice model

The premise: myopic agents will favour the types with high production quality score (high apriori probability)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import numpy as np
import tensorflow.compat.v1 as tf

from absl import app
from absl import flags
from recsim.agents import random_agent
from recsim.agents import tabular_q_agent
from recsim.agents import full_slate_q_agent
from recsim.agents import slate_decomp_q_agent
from recsim.agents.slate_decomp_q_agent import select_slate_greedy, select_slate_optimal, select_slate_topk, \
    compute_target_optimal_q, compute_target_greedy_q, compute_target_sarsa, compute_target_topk_q
from recsim.simulator import runner_lib
from recsim.environments import preference_dynamics

FLAGS = flags.FLAGS
flags.DEFINE_integer('slate_size', 2, 'The slate size presented to the user.')
flags.DEFINE_integer('num_candidates', 5, 'The number of candidates of documents.')
flags.DEFINE_bool('stationary', False, 'If set to true, user is stationary, ' +
                  'i.e. the preferences do not changeover time.')
flags.DEFINE_float('interest_update_rate', 0.001, 'The interest update rate parameter.')
flags.DEFINE_float('interest_update_prob', 0.005, 'The interest update probability.')
flags.DEFINE_string('slate_fn', 'greedy', 'The slate function.')
flags.DEFINE_string('target_fn', 'greedy', 'The target function.')
flags.DEFINE_string('algorithm', 'full_slate', 'The algorithm to run the experiments with.')

slate_fns = {
    'greedy': select_slate_greedy,
    'optimal': select_slate_optimal,
    'top-k': select_slate_topk,
}

target_fns = {
    'sarsa': compute_target_sarsa,
    'greedy': compute_target_greedy_q,
    'optimal': compute_target_optimal_q,
    'top-k': compute_target_topk_q,
}

slate_target_kwargs = {
    'select_slate_fn': None,
    'compute_target_fn': None
}

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
        **slate_target_kwargs
    }
    return slate_decomp_q_agent.SlateDecompQAgent(sess, **kwargs)


def experiment_config(base_dir, create_agent_fn, env_config, exp_config):
    t_runner = runner_lib.TrainRunner(
        base_dir=base_dir,
        create_agent_fn=create_agent_fn,
        env=preference_dynamics.create_environment(env_config),
        episode_log_file=FLAGS.episode_log_file,
        max_training_steps=exp_config['max_training_steps'],
        num_iterations=exp_config['num_iterations'])
    e_runner = runner_lib.EvalRunner(
        base_dir=base_dir,
        create_agent_fn=create_agent_fn,
        env=preference_dynamics.create_environment(env_config),
        max_eval_episodes=exp_config['max_eval_episodes'],
        test_mode=True)
    return t_runner, e_runner


def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')

    runner_lib.load_gin_configs(FLAGS.gin_files, FLAGS.gin_bindings)
    base_dir = FLAGS.base_dir
    if experiment_mode:
        # For the experimental setup
        env_config = {
            'num_candidates': FLAGS.num_candidates,
            'slate_size': FLAGS.slate_size,
            'resample_documents': True,
            'interest_update_rate': FLAGS.interest_update_rate,
            'interest_update_prob': FLAGS.interest_update_prob,
            'stationary': FLAGS.stationary,
            'slate_fn': FLAGS.slate_fn,
            'target_fn': FLAGS.target_fn,
            'seed': None
        }
    else:
        # For testing purposes
        env_config = {
            'num_candidates': 10,
            'slate_size': 2,
            'resample_documents': True,
            'interest_update_rate': 0.001,
            'interest_update_prob': 0.005,
            'stationary': False,
            'slate_fn': 'greedy',
            'target_fn': 'greedy',
            'seed': None
        }

    # Algorithms to use in experimentation & randomly generated seeds
    slate_algorithm = FLAGS.algorithm
    seeds = np.random.randint(low=0, high=100, size=3)

    for seed in seeds:
        # Set new seed for each experiment for statistical significance
        env_config['seed'] = seed

        # Algorithm selection for experiments
        stationary = 'stationary' if env_config['stationary'] else 'non_stationary'
        experiment_dir = f'{base_dir}/preference_dynamics_{stationary}'
        if not env_config['stationary']:
            experiment_dir += f'_iur_{env_config["interest_update_rate"]}' + \
                              f'_iup_{env_config["interest_update_prob"]}'
        experiment_dir += f'/{slate_algorithm}_iter_{exp_config["num_iterations"]}' + \
                          f'_steps_{exp_config["max_training_steps"]}' + \
                          f'_eval_{exp_config["max_eval_episodes"]}'

        if slate_algorithm == 'full_slate_q':
            tf.logging.info(f'Beginning preference dynamics experiments for {slate_algorithm}\n' +
                            f'Experiment parameters: \n' +
                            f'\tIterations: {exp_config["num_iterations"]}\n' +
                            f'\tTraining steps: {exp_config["max_training_steps"]}\n' +
                            f'\tEvaluation Episodes: {exp_config["max_eval_episodes"]}')
            experiment_dir += f'_seed_{env_config["seed"]}'
            train, evaluate = experiment_config(experiment_dir, create_full_slateQ_agent, env_config, exp_config)
            train.run_experiment()
            evaluate.run_experiment()
        elif slate_algorithm == 'slate_decomp_q':
            tf.logging.info(f'Beginning preference dynamics experiments for {slate_algorithm}\n' +
                            f'Experiment parameters: \n' +
                            f'\tIterations: {exp_config["num_iterations"]}\n' +
                            f'\tTraining steps: {exp_config["max_training_steps"]}\n' +
                            f'\tEvaluation Episodes: {exp_config["max_eval_episodes"]}')
            slate_fn = env_config['slate_fn']
            target_fn = env_config['target_fn']

            # Changing the output dir for slate decomposition
            experiment_dir += f'_slate_{slate_fn}_target_{target_fn}_seed_{env_config["seed"]}'

            # Slate Decomposition Agent function configuration
            slate_target_kwargs['select_slate_fn'] = slate_fns[slate_fn]
            slate_target_kwargs['compute_target_fn'] = target_fns[target_fn]

            train, evaluate = experiment_config(experiment_dir, create_slate_decomp_agent, env_config, exp_config)
            train.run_experiment()
            evaluate.run_experiment()
        elif slate_algorithm == 'random':
            tf.logging.info(f'Beginning preference dynamics experiments for {slate_algorithm}\n' +
                            f'Experiment parameters: \n' +
                            f'\tIterations: {exp_config["num_iterations"]}\n' +
                            f'\tTraining steps: {exp_config["max_training_steps"]}\n' +
                            f'\tEvaluation Episodes: {exp_config["max_eval_episodes"]}')
            experiment_dir += f'_seed_{env_config["seed"]}'
            train, evaluate = experiment_config(experiment_dir, create_random_agent, env_config, exp_config)
            train.run_experiment()
            evaluate.run_experiment()
        elif slate_algorithm == 'tabular_q_agent':
            tf.logging.info(f'Beginning preference dynamics experiments for {slate_algorithm}\n' +
                            f'Experiment parameters: \n' +
                            f'\tIterations: {exp_config["num_iterations"]}\n' +
                            f'\tTraining steps: {exp_config["max_training_steps"]}\n' +
                            f'\tEvaluation Episodes: {exp_config["max_eval_episodes"]}')
            experiment_dir += f'_seed_{env_config["seed"]}'
            train, evaluate = experiment_config(experiment_dir, create_tabular_q_agent, env_config, exp_config)
            train.run_experiment()
            evaluate.run_experiment()
        else:
            print('Experiments for algorithm {} not supported'.format(slate_algorithm))


experiment_mode = True  # change this to false when running experiments
exp_config = {
    'max_training_steps': 1000,
    'num_iterations': 400,
    'max_eval_episodes': 3000
}


if __name__ == '__main__':
    flags.mark_flags_as_required([
        'base_dir',  # where the experiment results will be stored
        'slate_size',               # slate size
        'num_candidates',           # number of document candidates
        'stationary',               # if the environment is stationary
        'interest_update_rate',     # the interest update rate parameter
        'interest_update_prob'      # the interest update probability
        'algorithm'                 # the used algorithm in the experiments
    ])
    app.run(main)
