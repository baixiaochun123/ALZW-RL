# lzw_agent.py

import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim

class RLAgent(nn.Module):
    def __init__(self, state_size, action_size):
        super(RLAgent, self).__init__()
        self.state_size = state_size  # 状态维度
        self.action_size = action_size  # 动作空间
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # 折扣因子
        self.epsilon = 1.0   # 探索率
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._build_model().to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
    def _build_model(self):
        # 构建神经网络模型
        model = nn.Sequential(
            nn.Linear(self.state_size, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, self.action_size)
        )
        return model
        
    def initial_dictionary_size(self):
        # 返回初始最大字典大小
        return 4096
        
    def get_action(self, state):
        # 根据当前状态选择动作
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            act_values = self.model(state)
        return torch.argmax(act_values[0]).item()
        
    def remember(self, state, action, reward, next_state, done):
        # 存储经验
        self.memory.append((state, action, reward, next_state, done))
        
    def replay(self, batch_size):
        # 经验回放，训练模型
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            next_state = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
            reward = torch.tensor([reward]).to(self.device)
            target = reward
            if not done:
                with torch.no_grad():
                    target = reward + self.gamma * torch.max(self.model(next_state)[0])
            output = self.model(state)[0]
            target_f = output.clone()
            target_f[action] = target
            self.optimizer.zero_grad()
            loss = self.criterion(output, target_f)
            loss.backward()
            self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    def calculate_reward(self, compression_ratio, processing_time):
        # 计算奖励
        alpha = 1.0
        beta = 0.1
        reward = (alpha * compression_ratio) - (beta * processing_time)
        return reward
            
    def load(self, name):
        # 加载模型
        self.model.load_state_dict(torch.load(name, map_location=self.device))
        
    def save(self, name):
        # 保存模型
        torch.save(self.model.state_dict(), name)
        
    def compress(self, data):
        """
        使用 RL 模型进行无损压缩，并记录动作序列和每个动作对应的数据长度。
        结果格式：
        - 前4个字节：动作序列长度（大端整数）
        - 接下来的 N 个字节：动作序列
        - 接下来的 N 个字节：每个动作对应的数据长度（与动作序列一一对应）
        - 其余字节：压缩后的数据
        """
        try:
            print("RLAgent 开始压缩数据...")
            data_int = list(data)
            states = [data_int[i:i+self.state_size] for i in range(0, len(data_int), self.state_size)]
            actions = []
            lengths = []
            compressed_data = bytearray()
            
            for idx, state in enumerate(states):
                if len(state) < self.state_size:
                    state += [0] * (self.state_size - len(state))
                action = self.get_action(state)
                actions.append(action)
                print(f"状态 {idx}: {state}, 选择的动作: {action}")
                
                # 根据动作决定如何压缩
                if action == 0:
                    # 动作0: 添加整个状态（5字节）
                    byte_vals = state
                    compressed_data.extend(byte_vals)
                    lengths.append(len(byte_vals))
                    print(f"动作0: 添加 {len(byte_vals)} 个字节: {byte_vals}")
                elif action == 1:
                    # 动作1: 使用 zlib 压缩当前状态
                    compressed_state = zlib.compress(bytes(state))
                    compressed_data.extend(compressed_state)
                    lengths.append(len(compressed_state))
                    print(f"动作1: 使用 zlib 压缩，添加 {len(compressed_state)} 个字节")
                elif action == 2:
                    # 动作2: 使用 bz2 压缩当前状态
                    compressed_state = bz2.compress(bytes(state))
                    compressed_data.extend(compressed_state)
                    lengths.append(len(compressed_state))
                    print(f"动作2: 使用 bz2 压缩，添加 {len(compressed_state)} 个字节")
                elif action == 3:
                    # 动作3: 不存储任何数据（仅作为示例，不建议在无损压缩中使用）
                    lengths.append(0)
                    print(f"动作3: 不添加任何数据")
                # 更多动作可以继续添加
                else:
                    # 默认动作，直接添加整个状态
                    byte_vals = state
                    compressed_data.extend(byte_vals)
                    lengths.append(len(byte_vals))
                    print(f"默认动作: 添加 {len(byte_vals)} 个字节: {byte_vals}")
                    
            # 将动作序列长度和动作序列添加到压缩数据的开头
            actions_length = len(actions)
            actions_bytes = bytes(actions)
            lengths_bytes = b''.join(length.to_bytes(4, byteorder='big') for length in lengths)
            actions_length_bytes = actions_length.to_bytes(4, byteorder='big')  # 4字节表示动作序列长度
            final_compressed_data = actions_length_bytes + actions_bytes + lengths_bytes + compressed_data
            
            print("RLAgent 压缩完成。")
            print(f"压缩后数据大小: {len(final_compressed_data)} 字节")
            return bytes(final_compressed_data)
        except Exception as e:
            print(f"RLAgent 压缩时出错: {e}")
            return b''