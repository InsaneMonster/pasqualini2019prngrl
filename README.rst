Pseudo Random Number Generation: a Reinforcement Learning Approach
******************************************************************

Luca Pasqualini, Maurizio Parton
################################################################

GitHub for a reinforcement learning research project consisting in simulating a novel Random Number Generator (RNG) by Deep Reinforcement Learning.

A RNG is an algorithm generating pseudo-random numbers and in this research project it is approximated by a Deep Neural Network using Reinforcement Learning.
The network is trained to "randomly" generate a novel algorithm by Reinforcement Learning, using a deep agent to solve a navigation problem.
This navigation task is defined by an N-dimensional environment in which said agent can "move".
Starting from a seed state the agent learns how to "move" in the N-dimensional environment in order to reach state with high rewards.
The reward is given by the result of the `NIST test battery <https://nvlpubs.nist.gov/nistpubs/legacy/sp/nistspecialpublication800-22r1a.pdf>`_ on the sequence at each time step.

Additional information are given in the related arXiv article.

The algorithm used are:
    - Dueling Double DQN (DDDQN) with Prioritized Experience Replay and Gradient-Clipping by using Huber loss
    - Vanilla Policy Gradient (VPG) with rewards-to-go and Generalized Advantage Estimation (GAE-Lambda) buffer
    - Proximal Policy Optimization (PPO) with rewards-to-go and Generalized Advantage Estimation (GAE-Lambda) buffer

**License**

*BSD 3-Clause License*

For additional information check the provided license file.

**Framework used**

- To run the NIST test battery `NistRng <https://github.com/InsaneMonster/NistRng>`_ (`nistrng package <https://pypi.org/project/nistrng/>`_) python implementation framework is used.
- To execute reinforcement learning the framework `USienaRL <https://github.com/InsaneMonster/USienaRL>`_ (`usienarl package <https://pypi.org/project/usienarl/>`_) is used.

**Compatible with Usienarl v0.5.0**

**Backend**

- *Python 3.6*
- *Tensorflow 1.10*

**Framework used**

- To run the NIST test battery `NistRng <https://github.com/InsaneMonster/NistRng>`_ (`nistrng package <https://pypi.org/project/nistrng/>`_) python implementation framework is used.
- To execute reinforcement learning the framework `USienaRL <https://github.com/InsaneMonster/USienaRL>`_ (`usienarl package <https://pypi.org/project/usienarl/>`_) is used.

**Compatible with Usienarl v0.5.0**

**Backend**

- *Python 3.6*
- *Tensorflow 1.10*