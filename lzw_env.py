# lzw_env.py

import numpy as np
import gym
from gym import spaces
from collections import defaultdict
from lzw_agent import RLAgent

class LZWEnv(gym.Env):
    """
    Custom LZW Compression RL Environment
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, data, state_size=5, action_size=4):
        super(LZWEnv, self).__init__()

        # Convert data to a list of integers if it's in bytes format
        if isinstance(data, bytes):
            self.data = list(data)
        else:
            self.data = data  # Assume it's already a list of integers

        self.state_size = state_size  # Size of the state window
        self.agent = RLAgent(state_size, action_size)  # Initialize the RL agent

        # Define action and observation spaces
        self.action_space = spaces.Discrete(action_size)
        self.observation_space = spaces.Box(low=0, high=255, shape=(state_size,), dtype=np.uint8)

        # Initialize environment
        self.reset()

    def reset(self):
        """
        Reset the environment and return the initial state
        """
        self.pointer = 0  # Data pointer indicating the current processing position
        self.dictionary = dict()  # Initialize dictionary
        self.total_encoded_length = 0  # Total length of encoded data
        self.total_raw_length = len(self.data)  # Length of original data
        self.done = False  # Completion flag

        # Initialize agent and get initial state
        self.state = self._get_state()

        return self.state

    def _get_state(self):
        """
        Get the current state
        """
        state = self.data[self.pointer:self.pointer + self.state_size]
        if len(state) < self.state_size:
            state = np.pad(state, (0, self.state_size - len(state)), 'constant', constant_values=0)
        return np.array(state, dtype=np.uint8)

    def step(self, action):
        """
        Take an action, return the next state, reward, done, and extra info
        """
        if self.done:
            return self.state, 0, self.done, {}

        encode_length = action + 1
        end_pointer = min(self.pointer + encode_length, len(self.data))
        sequence = self.data[self.pointer:end_pointer]
        sequence_key = tuple(sequence)

        # Update dictionary
        if sequence_key in self.dictionary:
            encoded_length = 1
        else:
            self.dictionary[sequence_key] = len(self.dictionary) + 1
            encoded_length = len(sequence)

        self.total_encoded_length += encoded_length
        self.pointer = end_pointer

        # Calculate reward using the agent's method
        compression_ratio = self.total_raw_length / self.total_encoded_length if self.total_encoded_length else 0
        processing_time = end_pointer / len(self.data)  # Simple proxy for processing time
        reward = self.agent.calculate_reward(compression_ratio, processing_time)

        # Check if done
        if self.pointer >= len(self.data):
            self.done = True

        # Get next state
        self.state = self._get_state()

        return self.state, reward, self.done, {}

    def get_state(self):
        """
        Public method to get the current state
        """
        return self._get_state()

    def get_compression_ratio(self):
        """
        Calculate and return the compression ratio
        """
        if self.total_encoded_length == 0:
            return 0
        return self.total_raw_length / self.total_encoded_length
        
    def render(self, mode='human'):
        pass

    def close(self):
        pass