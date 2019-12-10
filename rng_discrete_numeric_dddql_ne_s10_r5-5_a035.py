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
from usienarl.td_models import DuelingDeepQLearning
from usienarl.exploration_policies import BoltzmannExplorationPolicy


# Import required src

from src.rng_discrete_environment_numeric import RNGDiscreteNumericEnvironment
from src.dddql_rng_discrete_agent import DDDQLRNGDiscreteAgent
from src.rng_discrete_pass_through_interface import RNGDiscretePassThroughInterface
from src.rng_discrete_experiment_notepisodic import RNGDiscreteNotEpisodicExperiment


# Define utility functions to run the experiment

def _define_dddqn_model(config: Config) -> DuelingDeepQLearning:
    # Define attributes
    learning_rate: float = 0.000001
    discount_factor: float = 0.60
    buffer_capacity: int = 100000
    minimum_sample_probability: float = 0.01
    random_sample_trade_off: float = 0.6
    importance_sampling_value_increment: float = 0.4
    importance_sampling_value: float = 0.001
    # Return the _model
    return DuelingDeepQLearning("model",
                                learning_rate, discount_factor,
                                buffer_capacity,
                                minimum_sample_probability, random_sample_trade_off,
                                importance_sampling_value, importance_sampling_value_increment,
                                config)


def _define_boltzmann_exploration_policy() -> BoltzmannExplorationPolicy:
    # Define attributes
    temperature_max: float = 1.0
    temperature_min: float = 0.1
    temperature_decay: float = 0.00002
    # Return the explorer
    return BoltzmannExplorationPolicy(temperature_max, temperature_min, temperature_decay)


def _define_boltzmann_agent(model: DuelingDeepQLearning, exploration_policy: BoltzmannExplorationPolicy) -> DDDQLRNGDiscreteAgent:
    # Define attributes
    weight_copy_step_interval: int = 1000
    batch_size: int = 150
    # Return the agent
    return DDDQLRNGDiscreteAgent("dddqn_boltzmann_agent", model, exploration_policy, weight_copy_step_interval, batch_size)


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
    nn_config.add_hidden_layer(LayerType.dense, [2048, tensorflow.nn.relu, True, tensorflow.contrib.layers.xavier_initializer()])
    nn_config.add_hidden_layer(LayerType.dense, [2048, tensorflow.nn.relu, True, tensorflow.contrib.layers.xavier_initializer()])
    nn_config.add_hidden_layer(LayerType.dense, [2048, tensorflow.nn.relu, True, tensorflow.contrib.layers.xavier_initializer()])
    nn_config.add_hidden_layer(LayerType.dense, [2048, tensorflow.nn.relu, True, tensorflow.contrib.layers.xavier_initializer()])
    nn_config.add_hidden_layer(LayerType.dense, [2048, tensorflow.nn.relu, True, tensorflow.contrib.layers.xavier_initializer()])
    nn_config.add_hidden_layer(LayerType.dense, [2048, tensorflow.nn.relu, True, tensorflow.contrib.layers.xavier_initializer()])
    # Define model
    inner_model: DuelingDeepQLearning = _define_dddqn_model(nn_config)
    # Define exploration policy
    boltzmann_exploration_policy: BoltzmannExplorationPolicy = _define_boltzmann_exploration_policy()
    # Define agent
    dddqn_boltzmann_agent: DDDQLRNGDiscreteAgent = _define_boltzmann_agent(inner_model, boltzmann_exploration_policy)
    environment_name: str = 'RNGDiscreteNumeric'
    # Generate RNG discrete environment
    sequence_size: int = 10
    range_min: int = -5
    range_max: int = 5
    seed_range_min: int = -5
    seed_range_max: int = 5
    acceptance_value: float = 0.35
    environment: RNGDiscreteNumericEnvironment = RNGDiscreteNumericEnvironment(environment_name, sequence_size,
                                                                               range_min, range_max,
                                                                               seed_range_min, seed_range_max,
                                                                               acceptance_value)

    # Define interfaces
    interface: RNGDiscretePassThroughInterface = RNGDiscretePassThroughInterface(environment)
    # Define experiments
    success_threshold: float = 0.25
    last_steps_number: int = 20
    experiment_boltzmann: RNGDiscreteNotEpisodicExperiment = RNGDiscreteNotEpisodicExperiment("b_experiment",
                                                                                              success_threshold, last_steps_number,
                                                                                              environment,
                                                                                              dddqn_boltzmann_agent, interface)
    # Define experiments data
    testing_episodes: int = 100
    test_cycles: int = 10
    training_episodes: int = 1000
    validation_episodes: int = 1000
    max_training_episodes: int = 75000
    episode_length_max: int = 30
    plot_sample_density: int = 1
    # Run experiment
    intro: str = "Data:\n" \
                 "\nDueling Double DQN with Prioritized Experience Replay (100000 buffer capacity)" \
                 "\nSix dense layer with 2048 neurons each using xavier initialization" \
                 "\nLearning rate: 0.000001" \
                 "\nDiscount factor: 0.60" \
                 "\nWeight copy step interval: 1000" \
                 "\nBatch size: 150" \
                 "\nBoltzmann exploration policy with temperature decay: 0.00002 per episode" \
                 "\nSuccess threshold: 0.25 scaled reward on average over last 20 steps among all the validation volleys episodes" \
                 "\nEpisodic: no" \
                 "\nMax allowed steps for episode: 30" \
                 "\nSeed states range [-5, 5]" \
                 "\nAcceptance value: 0.35 as a reward in a single step\n"
    run_experiment(experiment_boltzmann,
                   training_episodes,
                   max_training_episodes, episode_length_max,
                   validation_episodes,
                   testing_episodes, test_cycles,
                   render_during_training, render_during_validation, render_during_test,
                   workspace_path, __file__,
                   logger, None, experiment_iterations_number,
                   intro, plot_sample_density)

