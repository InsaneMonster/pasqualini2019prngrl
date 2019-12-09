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
import numpy
import matplotlib.pyplot as plot

# Import usienarl

from usienarl import Experiment, Agent, Interface

# Import required src

from src.rng_discrete_environment import RNGDiscreteEnvironment


class RNGDiscreteNotEpisodicExperiment(Experiment):
    """
    RNG Discrete Experiment which is validated when the validation average scaled reward is above the given threshold
    over the given number of last steps per episode. It is always passed since it's hard to define a proper pass
    criterion.
    """

    def __init__(self,
                 name: str,
                 validation_threshold: float,
                 last_steps_number: int,
                 environment: RNGDiscreteEnvironment,
                 agent: Agent,
                 interface: Interface):
        # Define RNG discrete not episodic experiment attributes
        self._validation_threshold: float = validation_threshold
        self._last_steps_number: int = last_steps_number if last_steps_number > 0 else 1
        self._all_scaled_training_episodes_rewards: [] = []
        self._all_scaled_validation_episodes_rewards: [] = []
        self._all_scaled_training_volleys_rewards: [] = []
        self._all_scaled_validation_volleys_rewards: [] = []
        # Generate the base experiment
        super(RNGDiscreteNotEpisodicExperiment, self).__init__(name, environment, agent, interface)

    def initialize(self):
        # Reset all stored rewards from last experiment
        self._all_scaled_training_episodes_rewards = []
        self._all_scaled_validation_episodes_rewards = []
        self._all_scaled_training_volleys_rewards = []
        self._all_scaled_validation_volleys_rewards = []

    def _is_validated(self, logger: logging.Logger, last_average_validation_total_reward: float,
                      last_average_validation_scaled_reward: float, last_average_training_total_reward: float,
                      last_average_training_scaled_reward: float, last_validation_volley_rewards: [],
                      last_training_volley_rewards: [],
                      plot_sample_density: int = 1) -> bool:
        # Add training and validation episodes scaled rewards over last required episodes to the relative attributes for plot purposes
        scaled_training_episodes_rewards: numpy.ndarray = self._get_scaled_reward_over_last_steps(last_validation_volley_rewards, self._last_steps_number)
        scaled_validation_episodes_rewards: numpy.ndarray = self._get_scaled_reward_over_last_steps(last_validation_volley_rewards, self._last_steps_number)
        average_scaled_training_episodes_reward: float = numpy.round(numpy.average(scaled_training_episodes_rewards), 3)
        average_scaled_validation_episodes_reward: float = numpy.round(numpy.average(scaled_validation_episodes_rewards), 3)
        self._all_scaled_training_volleys_rewards.append(average_scaled_training_episodes_reward)
        self._all_scaled_validation_volleys_rewards.append(average_scaled_validation_episodes_reward)
        for episode_reward in scaled_training_episodes_rewards:
            self._all_scaled_training_episodes_rewards.append(episode_reward)
        for episode_reward in scaled_validation_episodes_rewards:
            self._all_scaled_validation_episodes_rewards.append(episode_reward)
        # Print info about validation
        logger.info("Average scaled reward of the last " + str(self._last_steps_number) + " steps over " + str(scaled_validation_episodes_rewards.size) + " validation episodes: " + str(average_scaled_validation_episodes_reward))
        # Check if the average scaled reward over the last required steps for all the validation episodes is above the threshold
        if average_scaled_validation_episodes_reward >= self._validation_threshold:
            return True
        return False

    def _display_test_cycle_metrics(self,
                                    logger: logging.Logger,
                                    last_test_cycle_average_total_reward: float,
                                    last_test_cycle_average_scaled_reward: float,
                                    last_test_cycle_rewards: [],
                                    plot_sample_density: int = 1):
        # Get average scaled reward over last required steps in the last test cycle episodes
        scaled_cycle_episodes_rewards: numpy.ndarray = self._get_scaled_reward_over_last_steps(last_test_cycle_rewards, self._last_steps_number)
        average_scaled_cycle_episodes_reward: float = numpy.round(numpy.average(scaled_cycle_episodes_rewards), 3)
        # Print info
        logger.info("Average scaled reward of the last " + str(self._last_steps_number) + " steps over last test cycles " + str(average_scaled_cycle_episodes_reward))

    def _is_successful(self, logger: logging.Logger, average_test_total_reward: float,
                       average_test_scaled_reward: float, max_test_total_reward: float, max_test_scaled_reward: float,
                       last_average_validation_total_reward: float, last_average_validation_scaled_reward: float,
                       last_average_training_total_reward: float, last_average_training_scaled_reward: float,
                       test_cycles_rewards: [], last_validation_volley_rewards: [],
                       last_training_volley_rewards: [],
                       plot_sample_density: int = 1) -> bool:
        # Save the plot of the average scaled rewards over the last required steps for all the validation volleys
        logger.info("Saving RNG experiment specific plots...")
        plot.plot(list(range(len(self._all_scaled_training_episodes_rewards)))[::plot_sample_density], self._all_scaled_training_episodes_rewards[::plot_sample_density])
        plot.xlabel('Training episode')
        plot.ylabel('Scaled reward over last ' + str(self._last_steps_number) + " steps")
        plot.savefig(self._plots_path + "/training_scaled_rewards_over_last_" + str(self._last_steps_number) + "_steps.png", dpi=300, transparent=True)
        plot.clf()
        plot.plot(list(range(len(self._all_scaled_validation_episodes_rewards)))[::plot_sample_density], self._all_scaled_validation_episodes_rewards[::plot_sample_density])
        plot.xlabel('Validation episode')
        plot.ylabel('Scaled reward over last ' + str(self._last_steps_number) + " steps")
        plot.savefig(self._plots_path + "/validation_scaled_rewards_over_last_" + str(self._last_steps_number) + "_steps.png", dpi=300, transparent=True)
        plot.clf()
        plot.plot(list(range(len(self._all_scaled_training_volleys_rewards))), self._all_scaled_training_volleys_rewards)
        plot.xlabel('Training volley')
        plot.ylabel('Scaled reward over last ' + str(self._last_steps_number) + " steps")
        plot.savefig(self._plots_path + "/training_volleys_scaled_rewards_over_last_" + str(self._last_steps_number) + "_steps.png", dpi=300, transparent=True)
        plot.clf()
        plot.plot(list(range(len(self._all_scaled_validation_volleys_rewards))), self._all_scaled_validation_volleys_rewards)
        plot.xlabel('Validation volley')
        plot.ylabel('Scaled reward over last ' + str(self._last_steps_number) + " steps")
        plot.savefig(self._plots_path + "/validation_volleys_scaled_rewards_over_last_" + str(self._last_steps_number) + "_steps.png", dpi=300, transparent=True)
        plot.clf()
        logger.info("RNG experiment specific plots saved successfully")
        # Always terminate successfully
        return True

    @staticmethod
    def _get_scaled_reward_over_last_steps(volley_rewards: [], last_steps_number: int) -> numpy.ndarray:
        # Check if steps number is a valid number
        steps_number = last_steps_number if last_steps_number > 0 else 1
        # Get the last steps number rewards in each episode in the volley
        average_episode_rewards: numpy.ndarray = numpy.zeros(len(volley_rewards), dtype=float)
        for index, episode_rewards in enumerate(volley_rewards):
            # Get the average of the requires steps or the entire episode if not enough steps are defined
            if len(episode_rewards) > steps_number:
                average_episode_rewards[index] = numpy.average(numpy.array(episode_rewards[-steps_number:]))
            else:
                average_episode_rewards[index] = numpy.average(numpy.array(episode_rewards))
        # Return all the averages wrapped in a numpy array
        return average_episode_rewards
