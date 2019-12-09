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

# Import usienarl package

from usienarl import Agent, ExplorationPolicy, SpaceType
from usienarl.td_models import DuelingDeepQLearning

# Import required src

from src.rng_discrete_pass_through_interface import RNGDiscretePassThroughInterface


class DDDQLRNGDiscreteAgent(Agent):
    """
    Dueling Double Deep Q-Learning agent for random number generation discrete environment.
    """

    def __init__(self,
                 name: str,
                 model: DuelingDeepQLearning,
                 exploration_policy: ExplorationPolicy,
                 weight_copy_step_interval: int,
                 batch_size: int = 1,
                 reward_scale: bool = False):
        # Define agent attributes
        self.model: DuelingDeepQLearning = model
        self.exploration_policy: ExplorationPolicy = exploration_policy
        # Define internal agent attributes
        self._weight_copy_step_interval: int = weight_copy_step_interval
        self._batch_size: int = batch_size
        self._current_absolute_errors = None
        self._current_loss = None
        self._reward_scale: bool = reward_scale
        # Generate base agent
        super(DDDQLRNGDiscreteAgent, self).__init__(name)

    def _generate(self,
                  logger: logging.Logger,
                  observation_space_type: SpaceType, observation_space_shape,
                  agent_action_space_type: SpaceType, agent_action_space_shape) -> bool:
        # Generate the exploration policy and check if it's successful, stop if not successful
        if self.exploration_policy.generate(logger, agent_action_space_type, agent_action_space_shape):
            # Generate the _model and return a flag stating if generation was successful
            return self.model.generate(logger, self._scope + "/" + self._name,
                                       observation_space_type, observation_space_shape,
                                       agent_action_space_type, agent_action_space_shape)
        return False

    def initialize(self,
                   logger: logging.Logger,
                   session):
        # Reset internal agent attributes
        self._current_absolute_errors = None
        self._current_loss = None
        # Initialize the model
        self.model.initialize(logger, session)
        # Initialize the exploration policy
        self.exploration_policy.initialize(logger, session)
        # Run the weight copy operation to uniform main and target networks
        self.model.copy_weight(session)

    def act_warmup(self,
                   logger: logging.Logger,
                   session,
                   interface: RNGDiscretePassThroughInterface,
                   agent_observation_current):
        # Act randomly
        action = interface.get_random_agent_action(logger, session)
        # Return the random action
        return action

    def act_train(self,
                  logger: logging.Logger,
                  session,
                  interface: RNGDiscretePassThroughInterface,
                  agent_observation_current):
        # Get the best action and all actions q-values predicted by the model
        best_action, all_actions = self.model.get_best_action_and_all_action_values(session, agent_observation_current, interface.get_action_mask(logger, session))
        # Act according to the exploration policy
        action = self.exploration_policy.act(logger, session, interface, all_actions, best_action)
        # Return the chosen action
        return action

    def act_inference(self,
                      logger: logging.Logger,
                      session,
                      interface: RNGDiscretePassThroughInterface,
                      agent_observation_current):
        # Return the best action predicted by the model
        return self.model.get_best_action(session, agent_observation_current, interface.get_action_mask(logger, session))

    def complete_step_warmup(self,
                             logger: logging.Logger,
                             session,
                             interface: RNGDiscretePassThroughInterface,
                             agent_observation_current,
                             agent_action, reward: float,
                             agent_observation_next,
                             warmup_step_current: int,
                             warmup_episode_current: int,
                             warmup_episode_volley: int):
        # Adjust the next observation if None (final step)
        last_step: bool = False
        if agent_observation_next is None:
            last_step = True
            if self._observation_space_type == SpaceType.discrete:
                agent_observation_next = 0
            else:
                agent_observation_next = numpy.zeros(self._observation_space_shape, dtype=float)
        # Scale the reward if required
        # Note: scaling function uses the following formula: (reward * 10) ^ 2
        if self._reward_scale:
            reward = math.pow(reward * 10, 2)
        # Save the current step in the buffer
        self.model.buffer.store(agent_observation_current, agent_action, reward, agent_observation_next, last_step)

    def complete_step_train(self,
                            logger: logging.Logger,
                            session,
                            interface: RNGDiscretePassThroughInterface,
                            agent_observation_current,
                            agent_action,
                            reward: float,
                            agent_observation_next,
                            train_step_current: int, train_step_absolute: int,
                            train_episode_current: int, train_episode_absolute: int,
                            train_episode_volley: int, train_episode_total: int):
        # Adjust the next observation if None (final step)
        last_step: bool = False
        if agent_observation_next is None:
            last_step = True
            if self._observation_space_type == SpaceType.discrete:
                agent_observation_next = 0
            else:
                agent_observation_next = numpy.zeros(self._observation_space_shape, dtype=float)
        # After each weight step interval update the target network weights with the main network weights
        if train_step_absolute % self._weight_copy_step_interval == 0:
            self.model.copy_weight(session)
        # Scale the reward if required
        # Note: scaling function uses the following formula: (reward * 10) ^ 2
        if self._reward_scale:
            reward = math.pow(reward * 10, 2)
        # Save the current step in the buffer
        self.model.buffer.store(agent_observation_current, agent_action, reward, agent_observation_next, last_step)
        # Update the model and save current loss and absolute errors
        summary, self._current_loss, self._current_absolute_errors = self.model.update(session, self.model.buffer.get(self._batch_size))
        # Update the buffer with the computed absolute error
        self.model.buffer.update(self._current_absolute_errors)
        # Update the summary at the absolute current step
        self._summary_writer.add_summary(summary, train_step_absolute)

    def complete_step_inference(self,
                                logger: logging.Logger,
                                session,
                                interface: RNGDiscretePassThroughInterface,
                                agent_observation_current,
                                agent_action,
                                reward: float,
                                agent_observation_next,
                                inference_step_current: int,
                                inference_episode_current: int,
                                inference_episode_volley: int):
        pass

    def complete_episode_warmup(self,
                                logger: logging.Logger,
                                session,
                                interface: RNGDiscretePassThroughInterface,
                                last_step_reward: float,
                                episode_total_reward: float,
                                warmup_episode_current: int,
                                warmup_episode_volley: int):
        pass

    def complete_episode_train(self,
                               logger: logging.Logger,
                               session,
                               interface: RNGDiscretePassThroughInterface,
                               last_step_reward: float,
                               episode_total_reward: float,
                               train_step_absolute: int,
                               train_episode_current: int, train_episode_absolute: int,
                               train_episode_volley: int, train_episode_total: int):
        # Update the exploration policy
        self.exploration_policy.update(logger, session)

    def complete_episode_inference(self,
                                   logger: logging.Logger,
                                   session,
                                   interface: RNGDiscretePassThroughInterface,
                                   last_step_reward: float,
                                   episode_total_reward: float,
                                   inference_episode_current: int,
                                   inference_episode_volley: int):
        pass

    @property
    def trainable_variables(self):
        # Return the trainable variables of the agent model in experiment/agent _scope
        return self.model.trainable_variables

    @property
    def warmup_episodes(self) -> int:
        # Return the amount of warmup episodes required by the model
        return self.model.warmup_episodes
