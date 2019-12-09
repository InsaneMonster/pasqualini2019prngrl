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
import math

# Import usienarl package

from usienarl import Agent, SpaceType
from usienarl.po_models import ProximalPolicyOptimization

# Import required src

from src.rng_discrete_pass_through_interface import RNGDiscretePassThroughInterface


class PPORNGDiscreteAgent(Agent):
    """
    Proximal Policy Optimization agent for random number generation discrete environment.
    """

    def __init__(self,
                 name: str,
                 model: ProximalPolicyOptimization,
                 updates_per_training_volley: int,
                 reward_scale: bool = False):
        # Define PPO agent attributes
        self.updates_per_training_volley: int = updates_per_training_volley
        # Define tensorflow model
        self._model: ProximalPolicyOptimization = model
        # Define internal agent attributes
        self._current_value_estimate = None
        self._current_log_likelihood = None
        self._current_policy_loss = None
        self._current_value_loss = None
        self._reward_scale: bool = reward_scale
        # Generate base agent
        super(PPORNGDiscreteAgent, self).__init__(name)

    def _generate(self,
                  logger: logging.Logger,
                  observation_space_type: SpaceType, observation_space_shape,
                  agent_action_space_type: SpaceType, agent_action_space_shape) -> bool:
        # Generate the _model and return a flag stating if generation was successful
        return self._model.generate(logger, self._scope + "/" + self._name,
                                    observation_space_type, observation_space_shape,
                                    agent_action_space_type, agent_action_space_shape)

    def initialize(self,
                   logger: logging.Logger,
                   session):
        # Reset internal agent attributes
        self._current_value_estimate = None
        self._current_log_likelihood = None
        self._current_policy_loss = None
        self._current_value_loss = None
        # Initialize the model
        self._model.initialize(logger, session)

    def act_warmup(self,
                   logger: logging.Logger,
                   session,
                   interface: RNGDiscretePassThroughInterface,
                   agent_observation_current):
        pass

    def act_train(self,
                  logger: logging.Logger,
                  session,
                  interface: RNGDiscretePassThroughInterface,
                  agent_observation_current):
        # If there is no exploration policy just use the model best prediction (a sample from the probability distribution)
        # Note: it is still inherently exploring
        best_action, self._current_value_estimate, self._current_log_likelihood = self._model.sample_action(session, agent_observation_current, interface.get_action_mask(logger, session))
        # Return the exploration action
        return best_action

    def act_inference(self,
                      logger: logging.Logger,
                      session,
                      interface: RNGDiscretePassThroughInterface,
                      agent_observation_current):
        # Predict the action with the model by sampling from its probability distribution
        action, _, _ = self._model.sample_action(session, agent_observation_current, interface.get_action_mask(logger, session))
        # Return the predicted action
        return action

    def complete_step_warmup(self,
                             logger: logging.Logger,
                             session,
                             interface: RNGDiscretePassThroughInterface,
                             agent_observation_current,
                             agent_action,
                             reward: float,
                             agent_observation_next,
                             warmup_step_current: int,
                             warmup_episode_current: int,
                             warmup_episode_volley: int):
        pass

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
        # Scale the reward if required
        # Note: scaling function uses the following formula: (reward * 10) ^ 2
        if self._reward_scale:
            reward = math.pow(reward * 10, 2)
        # Save the current step in the buffer together with the current value estimate
        self._model.buffer.store(agent_observation_current, agent_action, reward, self._current_value_estimate, self._current_log_likelihood)

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
        # Update the buffer at the end of trajectory
        self._model.buffer.finish_path(last_step_reward)
        # Update exploration policy, if any
        # Execute update only after a certain number of trajectories each time
        if train_episode_current % (train_episode_volley / self.updates_per_training_volley) == 0 and train_episode_current > 0:
            # Execute the update and store policy and value loss
            summary, self._current_policy_loss, self._current_value_loss = self._model.update(session, self._model.buffer.get())
            # Update the summary at the absolute current step
            self._summary_writer.add_summary(summary, train_step_absolute)

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
        return self._model.trainable_variables

    @property
    def warmup_episodes(self) -> int:
        # Return the amount of warmup episodes required by the model
        return self._model.warmup_episodes
