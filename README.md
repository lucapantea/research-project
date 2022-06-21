# Adapting to Dynamic User Preferences in Recommendation Systems via Deep Reinforcement Learning

Recommender Systems play a significant part in filtering and 
efficiently prioritizing relevant information to alleviate the 
information overload problem and maximize user engagement. 
Traditional recommender systems employ a static approach towards 
learning the user's preferences, relying on logged previous interactions 
with the system, disregarding the sequential nature of the recommendation
 task and consequently, the user preference shifts occurring across interactions.
  In this study, we formulate the recommendation task as a slate Markov Decision
   Process (slate-MDP) and leverage deep reinforcement learning (DRL) to learn recommendation 
   policies through sequential interactions and maximize user engagement over 
   extended horizons in non-stationary environments. We construct the 
   simulated environment with various degrees of preferential dynamics 
   and benchmark two DRL-based algorithms: FullSlateQ, a non-decomposed 
   full slate Q-learning based on a DQN agent, and SlateQ, which implements
    DQN using slate decomposition. Our findings suggest that SlateQ outperforms 
    by 10.57% FullSlateQ in non-stationary environments and that with a moderate 
    discount factor, the algorithms behave myopically 
and fail to make an appropriate tradeoff to maximize long-term user engagement.




#### About
Please refer to the [paper](BSc_Thesis_Luca_Pantea.pdf) for an in-depth treatment on the subject matter.

Main Contributions: `recsim/environments`, `experiments.py`, `main.py`

This project of is a part of the 2022 Edition of CSE3000 Research Project course at Delft University of Technology. 

#### Installing necessary dependencies
Cd into the **parent** directory.
```console
> pip install -r requirements.txt
```

#### Running Experiments
Cd into the **parent** directory. This works on my windows machine. Could potentially differ on other OSs.
Example command:
```console
> $ python -m recsim.experiments --agent_name=full_slate_q --episode_log_file='episode_logs.tfrecord' --base_dir=/recsim/experiments/ 
```

#### Running Tensorboard
To view the results of the experiment, first cd into the recsim **sub**-directory, 
and decide on the log path. Example command:
```console
> $ tensorboard --logdir=recsim/tmp/recsim/ --host localhost --port 8088
```




#### Accessing all experimental results
Due to the high amounts of computation necessary for generating statistically significant data, the result files could not be published because of storage limit.

However, should one be interested in accessing the results, send an email to luca.p.pantea@gmail.com, and I will give access to the WandB project where all the run logs are stored. 

## RecSim
To best observe the effects of such dynamics on recommendation performance, we turn to 
[**RecSim**](https://arxiv.org/abs/1909.04847), a configurable simulation environment, which allows for 
the study of RL in the context of stylized recommender settings. This repository contains all the necessary source code for
the preference dynamics environment. For more examples of environments and further work in RecSim, we point the reader to
the [**RecSim GitHub repository**](https://github.com/google-research/recsim).

#### RecSim Tutorials

To get started, please check out our Colab tutorials. In
[**RecSim: Overview**](recsim/colab/RecSim_Overview.ipynb),
we give a brief overview about RecSim. We then talk about each configurable
component:
[**environment**](recsim/colab/RecSim_Developing_an_Environment.ipynb)
and
[**recommender agent**](recsim/colab/RecSim_Developing_an_Agent.ipynb).

