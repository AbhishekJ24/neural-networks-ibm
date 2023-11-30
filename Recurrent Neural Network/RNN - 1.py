import numpy as np

input_size=3
hidden_size=4
sequence_length=5
wxh=np.random.randn(hidden_size, input_size)
whh = np.random.randn(hidden_size, hidden_size)
bh=np.zeros((hidden_size,1))
h_prev=np.zeros((hidden_size,1))

def rnn_forward(x, h_prev, Wxh, Whh, bh):
    h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h_prev) + bh)
    return h

x = np.random.randn(input_size, sequence_length)
hidden_states = []
for t in range(sequence_length):
    h_prev = rnn_forward(x[:, t:t+1], h_prev, wxh, whh, bh)
    hidden_states.append(h_prev)
for t, h in enumerate(hidden_states):
    print(f"\nEpoch {t+1}:")
    print(h)