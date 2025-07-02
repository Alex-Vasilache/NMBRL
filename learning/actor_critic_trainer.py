# This file will implement the training loop for the Actor-Critic agent.
# It will manage the agent's interaction with the world model (real or learned),
# and apply the learning updates (e.g., using sparse RTRL) to the agent's networks.

from Neuromorphic_MBRL.world_models.ini_cartpole_wrapper import INICartPoleWrapper
from Neuromorphic_MBRL.agents.snn_actor_critic_agent import SnnActorCriticAgent


class ActorCriticTrainer:
    """
    Trainer for the Actor-Critic agent.
    Manages the training loop, including agent-environment interaction and learning updates.
    """

    def __init__(self, config):
        """
        Initializes the trainer.

        :param config: A dictionary containing training parameters.
        """
        self.config = config
        self.world_model = INICartPoleWrapper()

        # The action space needs to be passed to the agent
        action_space = self.world_model.env.action_space
        self.agent = SnnActorCriticAgent(action_space=action_space)

    def train(self):
        """
        Runs the main training loop.
        """
        num_episodes = self.config.get("num_episodes", 10)

        for episode in range(num_episodes):
            state = self.world_model.reset()
            terminated = False
            total_reward = 0
            step_count = 0

            while not terminated:
                action = self.agent.get_action(state)
                next_state, reward, terminated, info = self.world_model.step(action)

                # In a real scenario, we would store this transition (s, a, r, s', d)
                # in a replay buffer and perform a learning step.

                print(
                    f"Episode: {episode+1}, Step: {step_count+1}, Reward: {reward:.2f}, Terminated: {terminated}"
                )

                state = next_state
                total_reward += reward
                step_count += 1

            print(f"Episode {episode+1} finished. Total Reward: {total_reward:.2f}")


if __name__ == "__main__":
    # Example configuration
    training_config = {"num_episodes": 5}

    trainer = ActorCriticTrainer(config=training_config)
    trainer.train()
