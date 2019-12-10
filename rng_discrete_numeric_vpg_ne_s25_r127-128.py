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

# Import usienarl

from usienarl import Config, LayerType, run_experiment, command_line_parse
from usienarl.po_models import VanillaPolicyGradient


# Import required src

from src.rng_discrete_environment_numeric import RNGDiscreteNumericEnvironment
from src.vpg_rng_discrete_agent import VPGRNGDiscreteAgent
from src.rng_discrete_pass_through_interface import RNGDiscretePassThroughInterface
from src.rng_discrete_experiment_notepisodic import RNGDiscreteNotEpisodicExperiment


# Define utility functions to run the experiment

def _define_vpg_model(config: Config) -> VanillaPolicyGradient:
    # Define attributes
    learning_rate_policy: float = 0.00003
    learning_rate_advantage: float = 0.0001
    discount_factor: float = 0.60
    value_steps_per_update: int = 80
    lambda_parameter: float = 0.95
    # Return the _model
    return VanillaPolicyGradient("model", discount_factor,
                                 learning_rate_policy, learning_rate_advantage,
                                 value_steps_per_update, config, lambda_parameter)


def _define_agent(model: VanillaPolicyGradient) -> VPGRNGDiscreteAgent:
    # Define attributes
    updates_per_training_volley: int = 20
    # Return the agent
    return VPGRNGDiscreteAgent("vpg_agent", model, updates_per_training_volley)


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
    nn_config.add_hidden_layer(LayerType.dense, [4096, tensorflow.nn.relu, True, tensorflow.contrib.layers.xavier_initializer()])
    nn_config.add_hidden_layer(LayerType.dense, [4096, tensorflow.nn.relu, True, tensorflow.contrib.layers.xavier_initializer()])
    nn_config.add_hidden_layer(LayerType.dense, [4096, tensorflow.nn.relu, True, tensorflow.contrib.layers.xavier_initializer()])
    nn_config.add_hidden_layer(LayerType.dense, [4096, tensorflow.nn.relu, True, tensorflow.contrib.layers.xavier_initializer()])
    nn_config.add_hidden_layer(LayerType.dense, [4096, tensorflow.nn.relu, True, tensorflow.contrib.layers.xavier_initializer()])
    nn_config.add_hidden_layer(LayerType.dense, [4096, tensorflow.nn.relu, True, tensorflow.contrib.layers.xavier_initializer()])
    # Define model
    inner_model: VanillaPolicyGradient = _define_vpg_model(nn_config)
    # Define agent
    vpg_agent: VPGRNGDiscreteAgent = _define_agent(inner_model)
    environment_name: str = 'RNGDiscreteNumeric'
    # Generate RNG discrete environment
    sequence_size: int = 25
    range_min: int = -128
    range_max: int = 127
    seed_range_min: int = -128
    seed_range_max: int = 127
    environment: RNGDiscreteNumericEnvironment = RNGDiscreteNumericEnvironment(environment_name, sequence_size,
                                                                               range_min, range_max,
                                                                               seed_range_min, seed_range_max)

    # Define interfaces
    interface: RNGDiscretePassThroughInterface = RNGDiscretePassThroughInterface(environment)
    # Define experiments
    success_threshold: float = 0.25
    last_steps_number: int = 20
    experiment: RNGDiscreteNotEpisodicExperiment = RNGDiscreteNotEpisodicExperiment("experiment",
                                                                                    success_threshold, last_steps_number,
                                                                                    environment,
                                                                                    vpg_agent, interface)
    # Define experiments data
    testing_episodes: int = 100
    test_cycles: int = 10
    training_episodes: int = 1000
    validation_episodes: int = 1000
    max_training_episodes: int = 100000
    episode_length_max: int = 150
    plot_sample_density: int = 100
    # Run experiment
    intro: str = "Data:\n" \
                 "\nVanilla Policy Gradient with GAE buffer" \
                 "\nSix dense layer with 4096 neurons each using xavier initialization" \
                 "\nLearning rate policy: 0.00003" \
                 "\nLearning rate advantage: 0.0001" \
                 "\nDiscount factor: 0.60" \
                 "\nValue steps per update: 80" \
                 "\nLambda parameter: 0.95" \
                 "\nUpdates per training volley: 20" \
                 "\nSuccess threshold: 0.25 scaled reward on average over last 20 steps among all the validation volleys episodes" \
                 "\nEpisodic: no" \
                 "\nMax allowed steps for episode: 150" \
                 "\nSeed states range [-128, 127]" \
                 "\nAcceptance value: none\n"
    run_experiment(experiment,
                   training_episodes,
                   max_training_episodes, episode_length_max,
                   validation_episodes,
                   testing_episodes, test_cycles,
                   render_during_training, render_during_validation, render_during_test,
                   workspace_path, __file__,
                   logger, None, experiment_iterations_number,
                   intro, plot_sample_density)

