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

# Import usienarl

from usienarl import Interface, SpaceType

# Import required src

from src.rng_discrete_environment import RNGDiscreteEnvironment


class RNGDiscretePassThroughInterface(Interface):
    """
    Default pass-through interface for all Tic Tac Toe environments.
    """

    def __init__(self,
                 environment: RNGDiscreteEnvironment):
        # Define specific rng discrete environment variable
        self._rng_discrete_environment: RNGDiscreteEnvironment = environment
        # Generate the base interface
        super(RNGDiscretePassThroughInterface, self).__init__(environment)

    def agent_action_to_environment_action(self,
                                           logger: logging.Logger,
                                           session,
                                           agent_action):
        # Just return the agent action
        return agent_action

    def environment_action_to_agent_action(self,
                                           logger: logging.Logger,
                                           session,
                                           environment_action):
        # Just return the environment action
        return environment_action

    def environment_state_to_observation(self,
                                         logger: logging.Logger,
                                         session,
                                         environment_state):
        # Just return the environment state
        return environment_state

    def get_possible_actions(self,
                             logger: logging.Logger,
                             session) -> []:
        """
        Get the possible agent actions from the environment current state available actions.

        :param logger: the logger used to print the interface information, warnings and errors
        :param session: the session of tensorflow currently running, if any
        :return: a list of agent actions which the agent can execute
        """
        environment_actions: [] = self._rng_discrete_environment.get_possible_actions(logger, session)
        agent_actions: [] = []
        for environment_action in environment_actions:
            agent_actions.append(self.environment_action_to_agent_action(logger, session, environment_action))
        return agent_actions

    def get_action_mask(self,
                        logger: logging.Logger,
                        session) -> numpy.ndarray:
        """
        Get an array representing the agent action mask (-infinity masked out actions, 1.0 available actions).

        :param logger: the logger used to print the interface information, warnings and errors
        :param session: the session of tensorflow currently running, if any
        :return: an array of values where 0.0 means available action at that index, -infinity means instead not available
        """
        # Get the possible actions
        possible_actions: [] = self.get_possible_actions(logger, session)
        # Generate a numpy array of -infinity
        mask: numpy.ndarray = -math.inf * numpy.ones(self.agent_action_space_shape, dtype=float)
        # Set to one all indexes contained in the possible action list
        for action_index in possible_actions:
            mask[action_index] = 0.0
        # Return the mask
        return mask

    @property
    def observation_space_type(self) -> SpaceType:
        # Just return the environment state space type
        return self._environment.state_space_type

    @property
    def observation_space_shape(self):
        # Just return the environment state space shape
        return self._environment.state_space_shape

    @property
    def agent_action_space_type(self) -> SpaceType:
        # Just return the environment action space type
        return self._environment.action_space_type

    @property
    def agent_action_space_shape(self):
        # Just return the environment action space shape
        return self._environment.action_space_shape
