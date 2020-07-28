import torch
from copy import deepcopy
import torch.nn as nn
import math
import numpy as np
import matplotlib.pyplot as plt


def generate_track():
    sin_value = []
    cos_value = []
    for i in range(1000):
        angle = i / 1000 * 4 * np.pi
        sin_value.append(math.sin(angle))
        cos_value.append(math.cos(angle))

    return sin_value, cos_value


def create_sample(sin_value, cos_value, seq_len):
    input_value = []
    output_value = []
    sample = []
    for i in range(len(sin_value) - seq_len):
        cur_input = []
        for j in range(seq_len):
            cur_input.append(sin_value[i + j])
        cur_output = cos_value[i + seq_len]
        input_value.append([cur_input])
        output_value.append(cur_output)
        sample.append([cur_input, cur_output])

    return input_value, output_value, sample


class Fit_Net(nn.Module):
    def __init__(self, input_size, hidden_size, LSTM_hidden_size):
        super().__init__()
        self.hidden_layer = nn.Linear(input_size, hidden_size)
        self.rnn_layer = nn.LSTM(hidden_size, LSTM_hidden_size, 1, batch_first=True)
        self.value_out = nn.Linear(LSTM_hidden_size, 1)

    def forward(self, x):
        x = torch.relu(self.hidden_layer(x))
        # print(x.shape)
        x, h_1 = self.rnn_layer(x, None)
        # print(x.shape)
        x_out = x[:, -1, :]
        # print(x_out.shape)
        y = self.value_out(x_out)
        return y, h_1


if __name__ == "__main__":
    sin_track, cos_track = generate_track()
    plt.plot(sin_track)
    plt.plot(cos_track)
    # print(len(sin_track))
    # print(len(cos_track))
    input_value, output_value, sample = create_sample(sin_track, cos_track, 5)
    # print(len(input_value))
    # print(len(output_value))

    lstm_net = Fit_Net(1, 32, 16)

    # # single out
    # input_value_0 = input_value[0]
    # output_value_0 = output_value[0]
    #
    # out, _ = lstm_net.forward(torch.tensor(input_value_0).unsqueeze(-1).unsqueeze(0),
    #                           torch.zeros(16).unsqueeze(0).unsqueeze(0),
    #                           torch.zeros(16).unsqueeze(0).unsqueeze(0))

    # print(len(input_value), len(input_value[0]), len(input_value[0][0]))
    input_value_copy = deepcopy(input_value)

    input_value = torch.tensor(input_value).transpose(-1, -2)
    output_value = torch.tensor(output_value).unsqueeze(-1)

    print(input_value.shape)

    for i in range(5000):

        perm = np.arange(input_value.shape[0])
        np.random.shuffle(perm)
        perm = torch.LongTensor(perm)

        # perm = torch.LongTensor(perm).to(device)
        input_value = input_value[perm].clone()
        output_value = output_value[perm].clone()

        out, _ = lstm_net.forward(input_value)
        # print(out.shape)

        optimizer = torch.optim.Adam(lstm_net.parameters(), lr=5e-3)
        optimizer.zero_grad()
        loss = (out - output_value).pow(2).mean()
        print(loss)
        loss.backward()
        optimizer.step()

    # draw map #
    t = np.arange(3, 1000, 1)
    # plt.plot(output_value)

    # print(out)

    input_value_copy = torch.tensor(input_value_copy).transpose(-1, -2)
    out, _ = lstm_net.forward(input_value_copy)
    print(out.shape)
    out = out.squeeze(-1).tolist()
    print(type(out))

    plt.plot(out)
    plt.show()

