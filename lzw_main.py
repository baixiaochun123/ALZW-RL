# lzw_main.py

import numpy as np
import random
from collections import deque
from lzw_agent import RLAgent
from lzw_env import LZWEnv
import matplotlib.pyplot as plt
import os

if __name__ == "__main__":
    # Create models directory
    if not os.path.exists('models'):
        os.makedirs('models')

    # Read data (in bytes)
    with open('data/input.txt', 'rb') as f:
        data = f.read()

    # Create environment and agent
    env = LZWEnv(data)
    state_size = env.get_state().shape[0]
    action_size = env.action_space.n
    rl_agent = RLAgent(state_size, action_size)

    num_episodes = 10
    batch_size = 32
    total_rewards = []

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            # Agent selects an action
            action = rl_agent.get_action(state)

            # Environment executes action, returns next state and reward
            next_state, reward, done, _ = env.step(action)

            # Agent stores experience
            rl_agent.remember(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward

            # Agent training
            if len(rl_agent.memory) > batch_size:
                rl_agent.replay(batch_size)

        total_rewards.append(total_reward)
        print(f"Episode {episode+1}/{num_episodes}, Total Reward: {total_reward:.2f}, Epsilon: {rl_agent.epsilon:.2f}")

        # Save model periodically
        if (episode + 1) % 10 == 0:
            rl_agent.save('models/rl_agent_model.pth')

    # Plot reward curve
    plt.plot(range(1, num_episodes + 1), total_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Reward over Episodes')
    plt.show()

    # Test agent
    rl_agent.load('models/rl_agent_model.pth')
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = rl_agent.get_action(state)
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        state = next_state

    print(f"Test completed, Total Reward: {total_reward:.2f}")
    print(f"Final Compression Ratio: {env.get_compression_ratio():.2f}")