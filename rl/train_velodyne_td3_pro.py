import numpy as np
import torch

from .td3_agent import TD3
from .velodyne_env import VelodyneEnv


def main(episodes: int = 10):
    env = VelodyneEnv()
    agent = TD3(state_dim=env.state_dim, action_dim=env.action_dim)

    for ep in range(episodes):
        state = env.reset()
        episode_reward = 0.0
        for step in range(200):
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.replay_buffer.add(state, action, reward, float(done), next_state)
            state = next_state
            episode_reward += reward
            if agent.replay_buffer.size > 1000:
                agent.train(1)
            if done:
                break
        print(f"Episode {ep}: reward {episode_reward:.2f}")


if __name__ == "__main__":
    main()

