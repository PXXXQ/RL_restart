import torch
import random
import torch.nn as nn
import numpy as np
import multiprocessing as mp
import gym
import redis
from copy import deepcopy

from common.common import serialize, deserialize


class DQN_net(nn.Module):
    def __init__(self, state_input_dim, hidden_size, action_dim):
        super().__init__()
        self.layer_0 = nn.Linear(state_input_dim, hidden_size)
        self.layer_1 = nn.Linear(hidden_size, hidden_size)
        self.q_head = nn.Linear(hidden_size, action_dim)
        self.action_dim = action_dim
        self.env = gym.make("CartPole-v1")
        self.redis = redis.Redis(host="0.0.0.0", port=6379)
        self.redis.delete("sample")
        self.memory_pool = dict(state=[], action=[], reward=[], next_state=[], done=[])
        self.epsilon = 0.2

    def forward(self, x):
        x = torch.relu(self.layer_0(x))
        x = torch.relu(self.layer_1(x))
        q = self.q_head(x)

        return q

    def select_action(self, x):
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
        q = self.forward(x).tolist()[0]
        if np.random.rand() < self.epsilon:
            action = np.random.randint(0, self.action_dim - 1)
        else:
            action = np.argsort(q).tolist().index(self.action_dim - 1)

        return action

    def roll_out(self, game_iter):
        batch = dict(state=[], action=[], reward=[], next_state=[], done=[])
        for _ in range(game_iter):
            state = self.env.reset()
            for _ in range(500):
                action = self.select_action(state)
                next_state, reward, done, _ = self.env.step(action)

                batch["state"].append(state.tolist())
                batch["action"].append(action)
                batch["reward"].append(reward - 10 if done else reward)
                batch["next_state"].append(next_state.tolist())
                batch["done"].append(0 if done else 1)

                state = next_state

                if done:
                    break

        # return batch
        self.redis.lpush("sample", serialize(batch))

    def sample(self, game_iter):
        process = []
        for _ in range(mp.cpu_count()):
            process.append(mp.Process(target=self.roll_out, args=(game_iter,)))
            process[-1].start()

        for item in process:
            item.join()

        result = self.redis.lrange("sample", 0, -1)
        self.redis.delete("sample")

        for single_result in result:
            single_result = deserialize(single_result)
            for key in self.memory_pool:
                for item in single_result[key]:
                    self.memory_pool[key].append(item)

    def clear_memory(self):
        self.memory_pool = dict(state=[], action=[], reward=[], next_state=[], done=[])


def train_net(net, memory_pool):

    dtype = torch.float32
    device = torch.device('cuda', 0)
    net.to(device)

    state = torch.tensor(memory_pool["state"], dtype=dtype).to(device)
    next_state = torch.tensor(memory_pool["next_state"], dtype=dtype).to(device)
    reward = torch.tensor(memory_pool["reward"], dtype=dtype).to(device)
    done = torch.tensor(memory_pool["done"], dtype=dtype).to(device)
    action = torch.tensor(memory_pool["action"], dtype=dtype).to(device)

    q_state = net.forward(state)
    q_action = q_state.gather(-1, action.long().unsqueeze(-1))
    with torch.no_grad():
        q_next_state = net.forward(next_state)
        q_next_max = torch.max(q_next_state, -1, keepdim=True)[0]

    print(len(memory_pool["state"]))

    # print(q_next_max.shape)
    # print(done.shape)
    # print(reward.shape)
    # print(q_action.shape)

    q_dev = q_next_max * done.unsqueeze(-1) + reward.unsqueeze(-1) - q_action

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
    loss = q_dev.pow(2).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    net.to(torch.device("cpu"))


if __name__ == "__main__":
    agent = DQN_net(4, 32, 2)

    for _ in range(10):

        agent.clear_memory()
        agent.sample(10)

        print(agent.memory_pool)
        print("agent_length", len(agent.memory_pool["state"]))

        train_net(agent, deepcopy(agent.memory_pool))









