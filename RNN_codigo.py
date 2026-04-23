import numpy as np

class SimpleRNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.W_xh = np.random.randn(hidden_size, input_size) * 0.01
        self.W_hh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.W_hy = np.random.randn(output_size, hidden_size) * 0.01
        self.b_h = np.zeros((hidden_size, 1))
        self.b_y = np.zeros((output_size, 1))
        self.hidden_size = hidden_size

    def forward(self, x_sequence):
        h_states = []
        y_outputs = []
        h_prev = np.zeros((self.hidden_size, 1))

        for x_t in x_sequence:
            if x_t.ndim == 1:
                x_t = x_t.reshape(-1, 1)

            h_t = np.tanh(
                self.W_xh @ x_t +
                self.W_hh @ h_prev +
                self.b_h
            )

            y_t = self.W_hy @ h_t + self.b_y

            h_states.append(h_t)
            y_outputs.append(y_t)
            h_prev = h_t

        return h_states, y_outputs

rnn = SimpleRNN(input_size=2, hidden_size=3, output_size=1)
x_seq = [np.array([2, 3]), np.array([4, 5]), np.array([6, 7])]
h_states, y_outputs = rnn.forward(x_seq)