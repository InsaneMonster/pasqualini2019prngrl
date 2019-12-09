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


class RNGDiscreteBinaryEpisodicEnvironment(RNGDiscreteEnvironment):
    """
    Environment simulating a random number generator for integers number.

    The state is represented by a sequence of integers represented with signed 8-bit binary format [-127, 127] and it
    is fully observable.
    The actions are:
        - set 1 to a certain position represented in binary value
        - set 0 to a certain position represented in binary value
    Actions are then codified with the following rule:
        - 2n -> set 1 to n
        - 2n+1 -> set 0 to n

    Note: do nothing action is not required since it is equivalent to set a certain bit to itself

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
        super(RNGDiscreteBinaryEpisodicEnvironment, self).__init__(name, sequence_size,
                                                                   range_min, range_max,
                                                                   seed_range_min, seed_range_max,
                                                                   acceptance_value,
                                                                   episode_length,
                                                                   threshold_value,
                                                                   reward_scale)

    def _update_state(self,
                      action):
        # Update the binary representation
        if action % 2 == 1:
            self._state_current[1][action // 2] = 0
        else:
            self._state_current[1][action // 2] = 1
        # Update the numeric representation
        self._state_current[0] = numpy.packbits(numpy.array(self._state_current[1])).astype(numpy.int8)

    def _get_action_space_size(self) -> int:
        # Return the size of the binary sequence multiplied by 2 (two actions for each position)
        # Note: we use a dummy state to avoid to compute the real state before initialization
        dummy_state: numpy.ndarray = numpy.zeros(self._sequence_size, dtype=int)
        dummy_state_binary: numpy.ndarray = pack_sequence(dummy_state)
        return dummy_state_binary.size * 2

    def get_action_mask(self,
                        logger: logging.Logger,
                        session) -> numpy.ndarray:
        # Get all the possible action mask according to current sequence state
        mask: numpy.ndarray = -math.inf * numpy.ones(self.action_space_shape, dtype=float)
        for action in range(*self.action_space_shape):
            # Cannot set something whose integer will exceed range in their numeric form
            temporary_sequence_binary: numpy.ndarray = self._state_current[1].copy()
            # Update the temporary binary representation
            if action % 2 == 1:
                temporary_sequence_binary[action // 2] = 0
            else:
                temporary_sequence_binary[action // 2] = 1
            temporary_sequence_numeric: numpy.ndarray = numpy.packbits(numpy.array(temporary_sequence_binary)).astype(numpy.int8)
            if all(temporary_sequence_numeric <= self._range_max) and all(temporary_sequence_numeric >= self._range_min):
                mask[action] = 0.0
        return mask

