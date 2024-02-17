import itertools
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def softmax(q_values, temperature):
    """
    Apply softmax function with temperature to a set of Q-values.

    :param q_values: A tensor of Q-values for each action.
    :param temperature: The temperature parameter for softmax.
                        Higher values increase exploration.
    :return: The probabilities for each action.
    """
    q_values_temp = q_values / temperature
    exp_q_values = torch.exp(q_values_temp - torch.max(q_values_temp))
    probabilities = exp_q_values / torch.sum(exp_q_values)

    return probabilities


class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.hidden_size = hidden_size
        self.attention_weights = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states):
        # hidden_states shape: (batch_size, sequence_length, hidden_size)
        attention_scores = self.attention_weights(hidden_states)
        attention_scores = F.softmax(attention_scores, dim=1)
        context_vector = torch.cumsum(attention_scores * hidden_states, dim=1)
        return context_vector, attention_scores


def MLP(input_dim, hidden_dim, activation, n_layers):
    layers = [nn.Linear(input_dim, hidden_dim), activation()]
    for _ in range(n_layers - 1):
        layers += [nn.Linear(hidden_dim, hidden_dim), activation()]
    return layers


class DRRN(torch.nn.Module):
    def __init__(
            self,
            mem_type="sum",
            embedding_dim=300,
            n_layers_action=1,
            n_layers_state=1,
            n_layers_scorer=1,
            n_layers_lstm=1,
            hidden_dim_action=128,
            hidden_dim_state=128,
            hidden_dim_scorer=128,
            hidden_lstm=128,
            activation="relu",
            state_dim=300,
            llm="fasttext",
            use_attention=False,
    ):
        super(DRRN, self).__init__()
        self.activation = None
        self.llm = llm
        self.use_attention = use_attention
        if activation == "relu":
            self.activation = nn.ReLU
        self.mem_type = mem_type
        self.action = nn.ModuleList(
            MLP(embedding_dim, hidden_dim_action, self.activation, n_layers_action))
        self.input_state = state_dim
        self.state = nn.ModuleList(
            MLP(hidden_lstm if mem_type == "lstm" else self.input_state, hidden_dim_state, self.activation,
                n_layers_state))
        self.scorer = nn.ModuleList(
            MLP(hidden_dim_action + hidden_dim_state,
                hidden_dim_scorer, self.activation, n_layers_scorer, )
            + [nn.Linear(hidden_dim_scorer, 1)])  # output layer
        if mem_type == "lstm":
            self.lstm = nn.LSTM(
                input_size=embedding_dim, hidden_size=hidden_lstm, num_layers=n_layers_lstm, batch_first=True)
        if self.use_attention:
            self.attention = AttentionLayer(hidden_lstm if mem_type == "lstm" else self.input_state)

    def forward(self, state_batch, act_batch):
        if self.mem_type == "sum" or self.mem_type == "None":
            state_batch = torch.from_numpy(np.concatenate(
                state_batch, axis=0)).float().to(device)
            new_state = None
        elif self.mem_type == "lstm":
            # Pad the sequences to a common length
            if isinstance(state_batch, tuple):
                state_batch = list(state_batch)
                state_batch[0] = torch.from_numpy(state_batch[0]).float().to(
                    device)

                output, (hn, cn) = self.lstm(state_batch[0],
                                             state_batch[1][0] if self.use_attention else state_batch[1])
                output = output.view(-1, output.size(-1))
                new_state = [hn, cn]
                if self.use_attention:
                    context_vector, attention_scores = self.attention(output)
                    output = context_vector + state_batch[1][1]
                    new_state = [new_state, output]
                state_batch = output
            elif isinstance(state_batch[0], np.ndarray):
                state_batch = [torch.from_numpy(x).float().to(
                    device) for x in state_batch]
                batch_len = len(state_batch)
                # print(batch_len)
                padded_sequences = rnn_utils.pad_sequence(
                    state_batch, batch_first=True, padding_value=0.0)
                # Create a list of sequence lengths
                seq_lengths = [len(seq) for seq in state_batch]
                seq_lengths = torch.LongTensor(seq_lengths).to(device)
                # Pack the padded sequences to handle variable lengths
                packed_sequences = rnn_utils.pack_padded_sequence(
                    padded_sequences, seq_lengths, batch_first=True, enforce_sorted=False)
                # Pass the packed sequences through the LSTM layer to get the packed hidden states
                packed_hidden_states, (h, c) = self.lstm(packed_sequences)

                # Unpack the hidden states and reshape
                unpacked_hidden_states, _ = rnn_utils.pad_packed_sequence(
                    packed_hidden_states, batch_first=True)
                state_batch_encoded = []
                for i in range(len(seq_lengths)):
                    state_batch_encoded.append(unpacked_hidden_states[i][
                                               :seq_lengths[i]])
                state_batch = torch.cat(state_batch_encoded, dim=0)
                if self.use_attention:
                    state_batch = state_batch.view(batch_len, state_batch.size(0), state_batch.size(1))
                    # print(state_batch.shape)
                    context_vector, attention_scores = self.attention(state_batch)
                    output = context_vector
                    state_batch = output.view(-1, output.size(-1))
                new_state = [[h[:, -1, :], c[:, -1, :]], output] if self.use_attention else [h[:, -1, :], c[:, -1, :]]
            else:
                raise ("state_batch type not supported")
        else:
            raise ("not implemented")
        act_sizes = [len(a) for a in act_batch]
        # Combine next actions into one long list
        act_batch = list(
            itertools.chain.from_iterable(act_batch)
        )  # [["1", "2"], ["3"]]->['1', '2', '3']
        act_batch = np.concatenate(act_batch, axis=0)
        act_batch = torch.from_numpy(act_batch).float().to(device)
        for l in self.action:
            act_batch = l(act_batch)
        for l in self.state:
            state_batch = l(state_batch)
        state_batch = torch.cat(
            [state_batch[i].repeat(j, 1) for i, j in enumerate(act_sizes)], dim=0
        )
        # Concat along hidden_dim
        # print(state_batch.shape, act_batch.shape)
        z = torch.cat((state_batch, act_batch), dim=1)
        for l in self.scorer:
            z = l(z)
        act_values = z.squeeze(-1)
        return act_values.split(act_sizes), new_state

    @torch.no_grad()
    def act(self, states, act_ids, policy="softmax", epsilon=1, temperature=1):
        """Returns an action-string, optionally sampling from the distribution
        of Q-Values.
        """
        act_values, new_state = self.forward(states, act_ids)
        act_probs = None
        # print(len(act_ids[0]))
        if policy == "softmax":
            vals = act_values[0]
            act_probs = softmax(vals, temperature=temperature)
            act_idxs = [torch.multinomial(act_probs, num_samples=1).item()]

        elif policy == "epsilon_greedy":
            act_opts = [vals.argmax(dim=0).item() for vals in act_values]
            if epsilon > random.uniform(0, 1):
                act_idxs = [np.random.choice(len(vals), 1)[0]
                            for vals in act_values]
            else:
                act_idxs = act_opts

        else:
            act_idxs = [vals.argmax(dim=0).item() for vals in act_values]
        return act_idxs, act_values, new_state,act_probs


def main():
    m = DRRN(embedding_dim=300,
             n_layers_action=1,
             n_layers_state=1,
             n_layers_scorer=1,
             hidden_dim_action=32,
             hidden_dim_state=64,
             hidden_dim_scorer=128,
             activation="relu").to(device)
    s = [np.random.rand(1, 300)]
    a = [[np.random.rand(1, 300), np.random.rand(1, 300)]]
    print(m(s, a))


if __name__ == "__main__":
    main()
