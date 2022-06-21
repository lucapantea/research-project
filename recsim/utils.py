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
"""Utility functions for RecSim environment."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools


def aggregate_pd_metrics(responses, metrics, info=None):
    """Aggregates the preference dynamics documents

  Args:
    responses: a dictionary of names, observed responses.
    metrics: A dictionary mapping from metric_name to its value in float.
    info: Additional info for computing metrics (ignored here)
  """
    del info
    is_clicked = False
    metrics['impression'] += 1

    for response in responses:
        if not response['clicked']:
            continue
        is_clicked = True
        metrics['clicks'] += 1

        metrics['quality'] += response['quality']
        metrics['satisfaction'] += response['satisfaction']
        metrics['engagement'] += response['engagement']
        cluster_id = response['topic_id']
        metrics['engagement_topic_%d' % cluster_id] += 1

        # metrics['user_interests'] = response['user_interests']
        # metrics['interest_topic_%d' % cluster_id] = response['user_interests'][cluster_id]

    if not is_clicked:
        metrics['no_click_count'] += 1
    return metrics


def write_pd_metrics(metrics, add_summary_fn):
    """Writes average metrics using add_summary_fn."""
    add_summary_fn('CTR', metrics['clicks'] / metrics['impression'])
    if metrics['clicks'] > 0:
        add_summary_fn('AverageQuality', metrics['quality'] / metrics['clicks'])
        add_summary_fn('AverageSatisfaction', metrics['satisfaction'] / metrics['clicks'])
    for k in metrics:
        prefix = 'engagement_topic_'
        if k.startswith(prefix):
            add_summary_fn('engagement_topic_frac/topic_%s' % k[len(prefix):],
                           metrics[k] / metrics['impression'])
    add_summary_fn(
        'engagement_topic_frac/no_click',
        metrics['no_click_count'] / metrics['impression'])


def aggregate_video_cluster_metrics(responses, metrics, info=None):
    """Aggregates the video cluster metrics with one step responses.

  Args:
    responses: a dictionary of names, observed responses.
    metrics: A dictionary mapping from metric_name to its value in float.
    info: Additional info for computing metrics (ignored here)

  Returns:
    A dictionary storing metrics after aggregation.
  """
    del info  # Unused.
    is_clicked = False
    metrics['impression'] += 1

    for response in responses:
        if not response['click']:
            continue
        is_clicked = True
        metrics['click'] += 1
        metrics['quality'] += response['quality']
        cluster_id = response['cluster_id']
        metrics['cluster_watch_count_cluster_%d' % cluster_id] += 1

    if not is_clicked:
        metrics['cluster_watch_count_no_click'] += 1
    return metrics


def write_video_cluster_metrics(metrics, add_summary_fn):
    """Writes average video cluster metrics using add_summary_fn."""
    add_summary_fn('CTR', metrics['click'] / metrics['impression'])
    if metrics['click'] > 0:
        add_summary_fn('AverageQuality', metrics['quality'] / metrics['click'])
    for k in metrics:
        prefix = 'cluster_watch_count_cluster_'
        if k.startswith(prefix):
            add_summary_fn('cluster_watch_count_frac/cluster_%s' % k[len(prefix):],
                           metrics[k] / metrics['impression'])
    add_summary_fn(
        'cluster_watch_count_frac/no_click',
        metrics['cluster_watch_count_no_click'] / metrics['impression'])


def generate_experiments(module_config, param_grid):
    # Check for non-list elements in the param_gird
    for key in param_grid.keys():
        if not isinstance(param_grid[key], list):
            new_param = {key: [param_grid[key]]}
            param_grid.update(new_param)

    job_count = 0
    keys = list(param_grid)
    param_constraints = {
        'stationary_slate_decomp_q': len(param_grid['slate_fn']) * len(param_grid['target_fn']),
        'non_stationary_slate_decomp_q': len(param_grid['slate_fn']) * len(param_grid['target_fn'])
                                    * len(param_grid['interest_update_rate']) * len(param_grid['interest_update_prob'])
    }

    def accepts(constr, constrs):
        return constrs[constr] > 0

    for algorithm in param_grid['algorithm']:
        if algorithm == 'slate_decomp_q':
            continue
        param_constraints[f'stationary_{algorithm}'] = 1  # no other parameters to take into account
        param_constraints[f'non_stationary_{algorithm}'] = len(param_grid['interest_update_rate']) \
                                                           * len(param_grid['interest_update_prob'])

    for combination in list(itertools.product(*map(param_grid.get, keys))):
        param_combination = dict(zip(keys, combination))
        stationary = 'stationary' if param_combination['stationary'] else 'non_stationary'
        constraint = f'{stationary}_{param_combination["algorithm"]}'

        # If the parameter combination is acceptable, write to file
        if accepts(constraint, param_constraints):
            with open(f'../scripts/run_{job_count}.sh', 'w') as run_sh:
                file_contents = setup_experiment(module_config, param_combination)
                run_sh.write(file_contents)
                param_constraints[constraint] -= 1
                job_count += 1


def setup_experiment(module_config, params):
    """Creates executable with different parameter settings for the DHPC."""
    script = module_config['file']
    module_args = module_config['module_args']

    base_dir = params['base_dir']
    slate_size = params['slate_size']
    num_candidates = params['num_candidates']
    stationary = params['stationary']
    interest_update_rate = params['interest_update_rate']
    interest_update_prob = params['interest_update_prob']
    slate_fn = params['slate_fn']
    target_fn = params['target_fn']
    algorithm = params['algorithm']

    run_python_cmd = f'srun python -m {script} ' \
                     f'--algorithm=\"{algorithm}\" ' \
                     f'--base_dir=\"{base_dir}\" ' \
                     f'--slate_size={slate_size} ' \
                     f'--num_candidates={num_candidates} ' \
                     f'--stationary={stationary} ' \
                     f'--interest_update_rate={interest_update_rate} ' \
                     f'--interest_update_prob={interest_update_prob} ' \
                     f'--slate_fn=\"{slate_fn}\" ' \
                     f'--target_fn=\"{target_fn}\"'

    run_script = module_args + '\n\n' + run_python_cmd
    return run_script


if __name__ == '__main__':
    params = {
        'base_dir': 'recsim/experiments',
        'algorithm': ['slate_decomp_q', 'full_slate_q', 'random', 'tabular_q_agent'],
        'slate_size': 2,
        'num_candidates': 10,
        'stationary': [True, False],
        'interest_update_rate': [0.05, 0.1, 0.5, 1.0],
        'interest_update_prob': [0.001, 0.005, 0.05, 0.1, 0.5],
        'slate_fn': ['greedy', 'top-k'],
        'target_fn': ['greedy', 'top-k', 'sarsa']
    }
    # potential script for generating source experimental files
