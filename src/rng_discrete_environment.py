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
import random
import math

# Import usienarl packages

from usienarl import Environment, SpaceType

# Import nistrng packages

from nistrng import run_in_order_battery, run_all_battery, check_eligibility_all_battery, SP800_22R1A_BATTERY, pack_sequence


class RNGDiscreteEnvironment(Environment):
    """
    Abstract class for all environments simulating a random number generator for integers number.

    Note: an acceptance value can be used only if the episode length in infinite, i.e. the task is not episodic.
    The seed states, if not defined inside the range, are always just a sequence of zeros, i.e. a single seed state.
    """

    def __init__(self,
                 name: str,
                 sequence_size: int,
                 range_min: int, range_max: int,
                 seed_range_min: int = 0, seed_range_max: int = 0,
                 acceptance_value: float = math.inf,
                 episode_length: int = math.inf,
                 threshold_value: float = -1,
                 reward_scale: float = 1):
        # Define RNG discrete environment attributes
        self._sequence_size: int = sequence_size
        self._range_min: int = range_min
        self._range_max: int = range_max
        self._acceptance_value: float = acceptance_value
        self._episode_length: int = episode_length
        self._threshold_value: float = threshold_value
        self._reward_scale: float = reward_scale
        # Define environment state attributes
        # Note: states are represented as a list of numeric state and binary encoded state
        self._state_last: [] = None
        self._state_current: [] = None
        self._score_state_last: float = 0.0
        self._score_state_current: float = 0.0
        # Define environment internal attributes
        self._episode_done: bool = False
        self._step_count: int = 0
        self._reward: float = 0.0
        self._eligible_battery: dict = None
        # Generate the seed states on which to train the agent
        self._seed_states: dict = {}
        for seed in range(seed_range_min, seed_range_max + 1, 1):
            seed_state_numeric: numpy.ndarray = seed * numpy.ones(sequence_size, dtype=int)
            seed_state_binary: numpy.ndarray = pack_sequence(seed_state_numeric)
            self._seed_states[seed] = [seed_state_numeric, seed_state_binary]
            # Generate the eligible battery of tests for the first seed state (the sequence length does not vary)
            if self._eligible_battery is None:
                self._eligible_battery = check_eligibility_all_battery(seed_state_binary, SP800_22R1A_BATTERY)
            # Add both states representation to the seed if score is 0.0
            # if self._compute_score(seed_state_binary) <= 0.0:
            #    self._seed_states[seed] = [seed_state_numeric, seed_state_binary]
        # Init the base environment
        super().__init__(name)

    def setup(self,
              logger: logging.Logger) -> bool:
        return True

    def initialize(self,
                   logger: logging.Logger,
                   session):
        pass

    def close(self,
              logger: logging.Logger,
              session):
        pass

    def reset(self,
              logger: logging.Logger,
              session):
        # Reset environment attributes
        self._score_state_last = 0.0
        self._state_last = None
        # Choose a random seed
        seed: int = random.choice(list(self._seed_states))
        # Reset the environment to a starting state according to the chosen seed
        self._state_current = [self._seed_states[seed][0].copy(), self._seed_states[seed][1].copy()]
        # Compute the score on the current starting state
        if self._episode_length == math.inf:
            self._score_state_current = self._compute_score_in_order(self._state_current[1])
        else:
            self._score_state_current = self._compute_score_all(self._state_current[1])
        # Reset internal attributes
        self._episode_done = False
        self._step_count = 0
        self._reward = 0.0
        # Return the current state in binary format
        return self._state_current[1]

    def step(self,
             logger: logging.Logger,
             action,
             session):
        # Save the current state in the last state
        self._state_last = [self._state_current[0].copy(), self._state_current[1].copy()]
        # Save the current state score in the last state score and reset the current state score
        self._score_state_last = self._score_state_current
        self._score_state_current = 0.0
        # Change the current state by the given action
        self._update_state(action)
        # Increment step count
        self._step_count += 1
        # Assign the reward depending on the nature of the environment (episodic/not episodic)
        if self._episode_length == math.inf:
            # Compute the score at each state
            self._score_state_current = self._compute_score_in_order(self._state_current[1])
            # Set the reward at the current score
            self._reward = self._score_state_current * self._reward_scale
            # If the reward is above the acceptance value, assign a high reward to this terminal state and terminate
            if self._reward >= self._acceptance_value:
                self._reward = 1.0 * self._reward_scale
                self._episode_done = True
        else:
            if self._step_count >= self._episode_length:
                # Compute the score at the last state
                self._score_state_current = self._compute_score_all(self._state_current[1])
                if self._score_state_current >= self._threshold_value:
                    # Set the reward at the current score
                    self._reward = self._score_state_current * self._reward_scale
                # End the episode
                self._episode_done = True
        # Return the result
        return self._state_current[1], self._reward, self._episode_done

    def render(self,
               logger: logging.Logger,
               session):
        print(str(self._state_last[0]) + " (" + str(self._score_state_last) + ") => " + str(self._state_current[0]) + " (" + str(self._score_state_current) + ") r = " + str(self._reward))
        if self._episode_done:
            print("____________________________________________________________")

    def get_random_action(self,
                          logger: logging.Logger,
                          session):
        return numpy.random.randint(0, 2 * self._sequence_size + 1)

    @property
    def state_space_type(self):
        # Return the continuous observation space type
        return SpaceType.continuous

    @property
    def state_space_shape(self):
        # Get the observation space shape of the state block in binary format
        # Note: it is computed to avoid requiring a setup to properly set before reset
        dummy_state: numpy.ndarray = numpy.zeros(self._sequence_size, dtype=int)
        dummy_state_binary: numpy.ndarray = pack_sequence(dummy_state)
        return dummy_state_binary.shape

    @property
    def action_space_type(self) -> SpaceType:
        # Return the discrete action space
        return SpaceType.discrete

    @property
    def action_space_shape(self):
        # Get the action space size depending on the child environment type
        return [self._get_action_space_size()]

    def get_action_mask(self,
                        logger: logging.Logger,
                        session) -> numpy.ndarray:
        """
        Return all the possible action at the current state in the environment wrapped in a numpy array mask.

        :param logger: the logger used to print the environment information, warnings and errors
        :param session: the session of tensorflow currently running, if any
        :return: an array of -infinity (for unavailable actions) and 0.0 (for available actions)
        """
        raise NotImplementedError()

    def get_possible_actions(self,
                             logger: logging.Logger,
                             session) -> []:
        """
        Return a list of the indices of all the possible actions at the current state of the environment.

        :param logger: the logger used to print the environment information, warnings and errors
        :param session: the session of tensorflow currently running, if any
        :return: a list of indices containing the possible actions
        """
        # Get the mask
        mask: numpy.ndarray = self.get_action_mask(logger, session)
        # Return the list of possible actions from the mask indices
        return numpy.where(mask >= 0.0)[0].tolist()

    def _compute_score_all(self,
                           sequence_binary: numpy.ndarray) -> float:
        """
        Compute the reward on the given state by running the NIST test battery.

        :return: the float reward of computed according to the test battery
        """
        # Execute the NIST test battery (only eligible tests) on the current sequence encoded in binary format
        results: [] = run_all_battery(sequence_binary, self._eligible_battery, False)
        # Generate a score list from the results
        scores: [] = []
        for result, _ in results:
            # Make sure there is no computational error inside the test battery wreaking havoc inside the loss
            if not numpy.isnan(result.score):
                scores.append(int(result.passed) * result.score)
            else:
                scores.append(0.0)
        # Compute the reward averaging the elements in the score list
        return numpy.round(numpy.average(numpy.array(scores)), 2)

    def _compute_score_in_order(self,
                                sequence_binary: numpy.ndarray) -> float:
        """
        Compute the reward on the given state by running the NIST test battery.
        The battery is run in order to save computational time (if a previous test is not passed the following is not
        checked).

        :return: the float reward of computed according to the test battery
        """
        # Execute the NIST test battery (only eligible tests and in order) on the current sequence encoded in binary format
        results: [] = run_in_order_battery(sequence_binary, self._eligible_battery, False)
        # Generate a score list from the results
        scores: [] = []
        for result, _ in results:
            # Make sure there is no computational error inside the test battery wreaking havoc inside the loss
            if not numpy.isnan(result.score):
                scores.append(int(result.passed) * result.score)
            else:
                scores.append(0.0)
        # Compute the reward averaging the elements in the score list (using also the eligible but not run tests)
        return numpy.round(numpy.sum(numpy.array(scores)) / len(self._eligible_battery), 2)

    def _update_state(self,
                      action):
        """
        Abstract method to delegate the update of the state with the current action.

        :param action: the action at this step used to update the state
        """
        raise NotImplementedError()

    def _get_action_space_size(self) -> int:
        """
        Abstract method to delegate the returning of the size of the action space.

        :return: the integer size of the action space
        """
        raise NotImplementedError()
