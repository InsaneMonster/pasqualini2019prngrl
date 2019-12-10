#
# Copyright (C) 2019 Luca Pasqualini
# University of Siena - Artificial Intelligence Laboratory - SAILab
#
#
# RngRL research project is licensed under a BSD 3-Clause.
#
# You should have received a copy of the license along with this
# work. If not, see <https://opensource.org/licenses/BSD-3-Clause>.

# Import packages

import tensorflow
import logging
import os
import math

# Import usienarl

from usienarl import Config, LayerType, run_experiment, command_line_parse
from usienarl.po_models import ProximalPolicyOptimization


# Import required src

from src.rng_discrete_environment_binary import RNGDiscreteBinaryEpisodicEnvironment
from src.ppo_rng_discrete_agent import PPORNGDiscreteAgent
from src.rng_discrete_pass_through_interface import RNGDiscretePassThroughInterface
from src.rng_discrete_experiment_episodic import RNGDiscreteEpisodicExperiment


# Define utility functions to run the experiment

def _define_ppo_model(config: Config) -> ProximalPolicyOptimization:
    # Define attributes
    learning_rate_policy: float = 0.0003
    learning_rate_advantage: float = 0.001
    discount_factor: float = 0.99
    value_steps_per_update: int = 80
    policy_steps_per_update: int = 80
    lambda_parameter: float = 0.97
    clip_ratio: float = 0.2
    target_kl_divergence: float = 0.01
    # Return the model
    return ProximalPolicyOptimization("model", discount_factor,
                                      learning_rate_policy, learning_rate_advantage,
                                      value_steps_per_update, policy_steps_per_update,
                                      config,
                                      lambda_parameter,
                                      clip_ratio,
                                      target_kl_divergence)


def _define_agent(model: ProximalPolicyOptimization) -> PPORNGDiscreteAgent:
    # Define attributes
    updates_per_training_volley: int = 2
    # Return the agent
    return PPORNGDiscreteAgent("ppo_agent", model, updates_per_training_volley)


if __name__ == "__main__":
    # Parse the command line arguments
    workspace_path, experiment_iterations_number, cuda_devices, render_during_training, render_during_validation, render_during_test = command_line_parse()
    # Define the CUDA devices in which to run the experiment
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices
    # Define the logger
    logger: logging.Logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    # Define Neural Network layers
    nn_config: Config = Config()
    nn_config.add_hidden_layer(LayerType.dense, [4096, tensorflow.nn.relu, True])
    nn_config.add_hidden_layer(LayerType.dense, [4096, tensorflow.nn.relu, True])
    nn_config.add_hidden_layer(LayerType.dense, [4096, tensorflow.nn.relu, True])
    # Define model
    inner_model: ProximalPolicyOptimization = _define_ppo_model(nn_config)
    # Define agent
    vpg_agent: PPORNGDiscreteAgent = _define_agent(inner_model)
    environment_name: str = 'RNGDiscreteBinary'
    # Generate RNG discrete environment
    sequence_size: int = 25
    range_min: int = -128
    range_max: int = 127
    acceptance_value: float = math.inf
    seed_range_min: int = 0
    seed_range_max: int = 0
    episode_length: int = 100
    threshold_value: float = 0.1
    reward_scale: float = 100
    environment: RNGDiscreteBinaryEpisodicEnvironment = RNGDiscreteBinaryEpisodicEnvironment(environment_name,
                                                                                             sequence_size,
                                                                                             range_min, range_max,
                                                                                             seed_range_min, seed_range_max,
                                                                                             acceptance_value,
                                                                                             episode_length,
                                                                                             threshold_value,
                                                                                             reward_scale)

    # Define interfaces
    interface: RNGDiscretePassThroughInterface = RNGDiscretePassThroughInterface(environment)
    # Define experiments
    success_threshold: float = 0.35 * reward_scale
    experiment: RNGDiscreteEpisodicExperiment = RNGDiscreteEpisodicExperiment("experiment",
                                                                              success_threshold,
                                                                              environment,
                                                                              vpg_agent, interface)
    # Define experiments data
    testing_episodes: int = 100
    test_cycles: int = 10
    training_episodes: int = 100
    validation_episodes: int = 100
    max_training_episodes: int = 50000
    episode_length_max: int = 100
    plot_sample_density: int = 10
    # Run experiment
    intro: str = "Data:\n" \
                 "\nProximal Policy Optimization with GAE buffer and early stopping" \
                 "\nThree dense layer with 4096 neurons each using xavier initialization" \
                 "\nLearning rate policy: 0.0003" \
                 "\nLearning rate advantage: 0.001" \
                 "\nDiscount factor: 0.99" \
                 "\nValue steps per update: 80" \
                 "\nPolicy steps per update: 80" \
                 "\nLambda parameter: 0.97" \
                 "\nClip ratio: 0.2" \
                 "\nTarget KL divergence: 0.01" \
                 "\nUpdates per training volley: 2" \
                 "\nSuccess threshold: 0.35 average total reward on the validation set episodes" \
                 "\nEpisodic: yes" \
                 "\nEpisode length: 100" \
                 "\nMax allowed steps for episode: 100" \
                 "\nSeed states range [0, 0]" \
                 "\nAcceptance value: none" \
                 "\nThreshold value: 0.1" \
                 "\nReward scale: 100\n"
    run_experiment(experiment,
                   training_episodes,
                   max_training_episodes, episode_length_max,
                   validation_episodes,
                   testing_episodes, test_cycles,
                   render_during_training, render_during_validation, render_during_test,
                   workspace_path, __file__,
                   logger, None, experiment_iterations_number,
                   intro, plot_sample_density)

