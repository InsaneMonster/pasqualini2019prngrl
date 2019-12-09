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
import math

# Import nistrng package

from nistrng import pack_sequence

# Import required src

from src.rng_discrete_environment import RNGDiscreteEnvironment


class RNGDiscreteNumericEnvironment(RNGDiscreteEnvironment):
    """
    Environment simulating a random number generator for integers number.

    The state is represented by a sequence of integers represented with signed 8-bit binary format [-128, 128] and it
    is fully observable.
    The actions are:
        - do nothing
        - add 1 to a certain position represented in decimal integer value
        - add -1 to a certain position represented in decimal integer value
    Actions are then codified with the following rule:
        - 0 -> do nothing
        - 2n -> add 1 to n-1
        - 2n+1 -> add -1 to n-1

    At each step the NIST test battery for random number generators is used. The framework used is
    https://github.com/InsaneMonster/NistRng.
    Depending on the sequence size, the eligible tests are computed and the battery is then sequentially run on the sequence
    represented in binary. When a certain test is failed, further test are not carried out in the same step to improve performance.

    The reward is computed with the average of the eligible tests (passed with value and failed with 0.0) and given at
    each step.

    An optional acceptance value can be used to better guide the learning, awarding a reward of 1.0 and terminating the episode
    when that value of reward is reached.
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
        # Generate the base RNG discrete environment
        super(RNGDiscreteNumericEnvironment, self).__init__(name, sequence_size,
                                                            range_min, range_max,
                                                            seed_range_min, seed_range_max,
                                                            acceptance_value,
                                                            episode_length,
                                                            threshold_value,
                                                            reward_scale)

    def _update_state(self,
                      action):
        # Update the numeric current state
        if action == 0:
            pass
        elif action % 2 == 1:
            self._state_current[0][(action - 1) // 2] += -1
        else:
            self._state_current[0][(action - 2) // 2] += 1
        # Update the binary representation
        self._state_current[1] = pack_sequence(self._state_current[0])

    def _get_action_space_size(self) -> int:
        # Return the size of the sequence multiplied by 2 (two actions for each position) plus 1 (the null-action)
        return self._sequence_size * 2 + 1

    def get_action_mask(self,
                        logger: logging.Logger,
                        session) -> numpy.ndarray:
        # Get all the possible action mask according to current sequence state
        mask: numpy.ndarray = -math.inf * numpy.ones(self.action_space_shape, dtype=float)
        for action in range(*self.action_space_shape):
            if action == 0:
                mask[action] = 0.0
            elif action % 2 == 1:
                if self._state_current[0][(action - 1) // 2] - 1 >= self._range_min:
                    mask[action] = 0.0
            else:
                if self._state_current[0][(action - 2) // 2] + 1 <= self._range_max:
                    mask[action] = 0.0
        return mask
