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

import logging

# Import usienarl

from usienarl import Experiment, Agent, Interface

# Import required src

from src.rng_discrete_environment import RNGDiscreteEnvironment


class RNGDiscreteEpisodicExperiment(Experiment):
    """
    RNG Discrete Experiment which is validated when the validation average total reward is above the given threshold.
    It is always passed since it's hard to define a proper pass criterion.
    """

    def __init__(self,
                 name: str,
                 validation_threshold: float,
                 environment: RNGDiscreteEnvironment,
                 agent: Agent,
                 interface: Interface):
        # Define RNG discrete episodic experiment attributes
        self._validation_threshold: float = validation_threshold
        # Generate the base experiment
        super(RNGDiscreteEpisodicExperiment, self).__init__(name, environment, agent, interface)

    def initialize(self):
        pass

    def _is_validated(self, logger: logging.Logger, last_average_validation_total_reward: float,
                      last_average_validation_scaled_reward: float, last_average_training_total_reward: float,
                      last_average_training_scaled_reward: float, last_validation_volley_rewards: [],
                      last_training_volley_rewards: [],
                      plot_sample_density: int = 1) -> bool:
        # Check if the average total reward has reached the given threshold
        if last_average_validation_total_reward > self._validation_threshold:
            return True
        return False

    def _display_test_cycle_metrics(self,
                                    logger: logging.Logger,
                                    last_test_cycle_average_total_reward: float,
                                    last_test_cycle_average_scaled_reward: float,
                                    last_test_cycle_rewards: [],
                                    plot_sample_density: int = 1):
        pass

    def _is_successful(self, logger: logging.Logger, average_test_total_reward: float,
                       average_test_scaled_reward: float, max_test_total_reward: float, max_test_scaled_reward: float,
                       last_average_validation_total_reward: float, last_average_validation_scaled_reward: float,
                       last_average_training_total_reward: float, last_average_training_scaled_reward: float,
                       test_cycles_rewards: [], last_validation_volley_rewards: [],
                       last_training_volley_rewards: [],
                       plot_sample_density: int = 1) -> bool:
        # Always terminate successfully
        return True
