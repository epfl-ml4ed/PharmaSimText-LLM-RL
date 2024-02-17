import os
import pickle
from os.path import join as pjoin

import numpy as np
import torch
import torch.nn as nn
from scipy.special import softmax
import wandb
from encoder import Encoder
from memory import ReplayMemory, Transition
from model import DRRN
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


def join_statistics(stats, unq=False):
    output = stats["unq_interactions" if unq else "interactions"]
    for c in stats.keys():
        if c not in ["unq_interactions", "interactions"]:
            output = np.concatenate([output, stats[c]])
    return output


class DRRNAgent:
    def __init__(self, args, state_dim):
        self.train_type = args["train_type"]
        self.llm_assisted = args["llm_assisted"]
        self.pretrained_explore = args["pretrained_explore"]
        self.state_encoder = Encoder(encoder_type=args["encoder_type"], emb_dim=args["embedding_dim"],
                                     memory_size=args["encoder_memory_size"], save_memory=args["save_memory"],
                                     memory_path=pjoin(args["memory_path"],
                                                       args["encoder_type"] + "_" + "state_memory.json"))
        self.action_encoder = Encoder(encoder_type=args["encoder_type"], emb_dim=args["embedding_dim"],
                                      memory_size=args["encoder_memory_size"], save_memory=args["save_memory"],
                                      memory_path=pjoin(args["memory_path"],
                                                        args["encoder_type"] + "_""action_memory.json"))
        self.embedding_dim = args["embedding_dim"]
        self.emb = args["emb"]
        self.hc = args["hc"]
        self.unq = args["unq"]
        self.save_path = args["save_path"]
        self.a2e = dict()
        self.pretrained_network = None
        if self.llm_assisted:
            print("loading the pretrained model...")
            model_name = "pretrained_" + args.encoder_type + "_" + args.emb + (
                "_attention" if args.use_attention else "")

            print(model_name)
            path = wandb.use_artifact(f"xxxx/model-registry/{model_name}:latest").download()
            self.pretrained_network = torch.load(pjoin(path, "best_model.pt"))
            print("pretrained model loaded!")
        self.policy_network = DRRN(embedding_dim=args["embedding_dim"],
                                   state_dim=state_dim,
                                   mem_type=args["emb"],
                                   n_layers_action=args["n_layers_action"],
                                   n_layers_state=args["n_layers_state"],
                                   n_layers_scorer=args["n_layers_scorer"],
                                   n_layers_lstm=args["n_layers_lstm"],
                                   hidden_dim_action=args["hidden_dim_action"],
                                   hidden_dim_state=args["hidden_dim_state"],
                                   hidden_dim_scorer=args["hidden_dim_scorer"],
                                   hidden_lstm=args["hidden_lstm"],
                                   activation=args["activation"], llm=args["encoder_type"],
                                   use_attention=args["use_attention"]).to(device)
        self.target_network = DRRN(embedding_dim=args["embedding_dim"],
                                   state_dim=state_dim,
                                   mem_type=args["emb"],
                                   n_layers_action=args["n_layers_action"],
                                   n_layers_state=args["n_layers_state"],
                                   n_layers_scorer=args["n_layers_scorer"],
                                   n_layers_lstm=args["n_layers_lstm"],
                                   hidden_dim_action=args["hidden_dim_action"],
                                   hidden_dim_state=args["hidden_dim_state"],
                                   hidden_dim_scorer=args["hidden_dim_scorer"],
                                   hidden_lstm=args["hidden_lstm"],
                                   activation=args["activation"], llm=args["encoder_type"],
                                   use_attention=args["use_attention"]).to(device)
        self.criterion = nn.SmoothL1Loss()
        self.memory = ReplayMemory(args['memory_size'])
        self.clip = args["clip"]
        self.TAU = args["TAU"]
        self.gamma = args["gamma"]
        self.batch_size = args["batch_size"]
        self.optimizer = torch.optim.Adam(
            self.policy_network.parameters(), lr=args["learning_rate"]
        )

        if ("policy_model.pt" in os.listdir(self.save_path)) and ("target_model.pt" in os.listdir(self.save_path)):
            print("loading the model...")
            self.load_recent()

    def watch(self):
        wandb.watch(self.policy_network, self.criterion, log="all", log_freq=1)

    def observe(self, inp, ):
        self.memory.push(*inp)

    def reset_dictionaries(self):
        self.a2e = dict()

    def encode_actions(self, actions):
        kind = actions[0]["type"]
        if kind not in self.a2e.keys():
            l = []
            for a in actions:
                l.append(self.action_encoder.encode([a["sentence"]])[0])
            self.a2e[kind] = l
        elif len(self.a2e[kind]) != len(actions):
            l = []
            for a in actions:
                l.append(self.action_encoder.encode([a["sentence"]])[0])
            self.a2e[kind] = l
        if len(actions) == 1:
            return [self.action_encoder.encode([actions[0]["sentence"]])[0]]
        return self.a2e[kind]

    def encode_state(self, state):
        s = state[1]
        e = self.state_encoder.encode([s])[0]
        return e.reshape(-1)

    def create_state(self, update_sentence, hc, previous_state=None):
        state = None
        if self.emb is None:
            if self.hc is None:
                raise "At least one of the features should be added to the state!"
            else:
                state = np.concatenate(
                    [hc["posttest_indicator"], hc["posttest"], join_statistics(hc["statistics"], unq=self.unq)])
                if self.hc == "bq":
                    state = np.concatenate([state, hc["binary_qs"]])
                elif self.hc == "kw":
                    state = np.concatenate([state, hc["keywords"]])
                elif self.hc == "both":
                    state = np.concatenate(
                        [state, hc["binary_qs"], hc["keywords"]])
                else:
                    pass
        else:
            emb = self.encode_state(update_sentence)
            if previous_state is not None:
                if self.emb == "sum":
                    emb += previous_state[0, -self.embedding_dim:]
                elif self.emb == "max":
                    emb = np.maximum(
                        emb, previous_state[0, -self.embedding_dim:])
                elif self.emb == "avg":
                    emb = emb + (previous_state[0, -self.embedding_dim:] - emb) / (
                            hc["statistics"]["interactions"].index(1) + 1)
                elif self.emb == "lstm":
                    emb = emb
                elif self.emb == "None":
                    emb = emb
                else:
                    raise "Unknown embedding type!"
            if self.hc is None:
                state = emb
            else:
                if self.hc == "bq":
                    state = np.concatenate(
                        [join_statistics(hc["statistics"], unq=self.unq), hc["binary_qs"], emb])
                elif self.hc == "kw":
                    state = np.concatenate(
                        [join_statistics(hc["statistics"], unq=self.unq), hc["keywords"], emb])
                elif self.hc == "both":
                    state = np.concatenate(
                        [join_statistics(hc["statistics"], unq=self.unq), hc["binary_qs"], hc["keywords"], emb])
                else:
                    state = np.concatenate(
                        [join_statistics(hc["statistics"], unq=self.unq), emb])
        return state.reshape(1, -1)

    def act(self, states, poss_acts, policy="softmax", epsilon=1, eval_mode=False, action_strs=None, temperature=0.001):
        """Returns a string action from poss_acts."""
        if self.llm_assisted and not eval_mode:
            # random number between 0 and 1
            if np.random.rand() > self.pretrained_explore:
                idxs, values, next_state,act_probs = self.policy_network.act(
                    states, poss_acts, policy=policy, epsilon=epsilon,temperature=temperature)
            else:
                idxs, values, next_state,act_probs = self.pretrained_network.act(
                    states, poss_acts, policy=policy, epsilon=epsilon,temperature=temperature)
        else:
            idxs, values, next_state,act_probs = self.policy_network.act(
                    states, poss_acts, policy=policy, epsilon=epsilon,temperature=temperature)
        act_ids = [poss_acts[batch][idx] for batch, idx in enumerate(idxs)]
        sorted_values = np.sort(values[0].detach().cpu().numpy())
        sorted_idxs = np.argsort(values[0].detach().cpu().numpy())
        act_probs = act_probs.detach().cpu().numpy()
        act_probs = act_probs[sorted_idxs]
        if eval_mode:
            print("....................................")
            s = action_strs[sorted_idxs[-1]]["sentence"]
            print(f"1st={s},{np.sort(sorted_values)[-1]:.3f}({act_probs[-1]:.3f}%)")
            if len(sorted_values) > 1:
                s = action_strs[sorted_idxs[-2]]["sentence"]
                print(f"2nd={s},{np.sort(sorted_values)[-2]:.3f}({act_probs[-2]:.3f}%)")
        return act_ids, idxs, values, next_state

    def update(self):
        if self.train_type == "normal":
            if len(self.memory) < self.batch_size:
                return
            transitions = self.memory.sample(self.batch_size)
        elif self.train_type == "episode_unbatched":
            transitions = self.memory.pull_all()
        elif self.train_type == "episode_batched":
            raise "Not implemented!"
        else:
            raise "Unknown training type!"
        batch = Transition(*zip(*transitions))
        next_state = [np.concatenate(batch.next_state, axis=0)]
        state = [np.concatenate(batch.state, axis=0)]
        # Compute Q(s', a') for all a'
        with torch.no_grad():
            next_qvals, _ = self.target_network(
                next_state if self.train_type == "episode_unbatched" else batch.next_state, batch.next_acts)
        # Take the max over next q-values
        next_qvals = torch.tensor([vals.max()
                                   for vals in next_qvals], device=device)
        # Zero all the next_qvals that are done
        next_qvals = next_qvals * (
                1 - torch.tensor(batch.done, dtype=torch.float, device=device)
        )
        # print(sum(batch.done))
        targets = (
                torch.tensor(batch.reward, dtype=torch.float, device=device)
                + self.gamma * next_qvals
        )
        # Next compute Q(s, a)
        # Nest each action in a list - so that it becomes the only admissible cmd
        nested_acts = tuple([[a] for a in batch.act])
        qvals, next_state = self.policy_network(
            state if self.train_type == "episode_unbatched" else batch.state, nested_acts)
        # print(qvals)
        # print(next_qvals)
        # Combine the qvals: Maybe just do a greedy max for generality
        qvals = torch.cat(qvals)
        # Compute Huber loss
        loss = self.criterion(qvals, targets.detach())
        self.optimizer.zero_grad()
        loss.backward()
        if self.clip is not None:
            nn.utils.clip_grad_norm_(
                self.policy_network.parameters(), self.clip)
        self.optimizer.step()
        target_net_state_dict = self.target_network.state_dict()
        policy_net_state_dict = self.policy_network.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * \
                                         self.TAU + target_net_state_dict[key] * (1 - self.TAU)
        self.target_network.load_state_dict(target_net_state_dict)
        return loss.item()

    def load_recent(self):
        try:
            self.memory = pickle.load(
                open(pjoin(self.save_path, "memory.pkl"), "rb"))
            self.policy_network = torch.load(
                pjoin(self.save_path, "policy_model.pt"))
            self.target_network = torch.load(
                pjoin(self.save_path, "target_model.pt"))
        except Exception as e:
            print("Error loading model.")
        return

    def save_recent(self):
        torch.save(self.policy_network, pjoin(
            self.save_path, "policy_model.pt"))
        torch.save(self.target_network, pjoin(
            self.save_path, "target_model.pt"))
        pickle.dump(self.memory, open(
            pjoin(self.save_path, "memory.pkl"), "wb"))
        return

    def load_best(self):
        self.save_recent()
        self.policy_network = torch.load(
            pjoin(self.save_path, "best_model.pt"))

    def save_best(self):
        print("saving the model...")
        torch.save(self.policy_network, pjoin(
            self.save_path, f"best_model.pt"))
        pickle.dump(self.memory, open(
            pjoin(self.save_path, f"memory_best.pkl"), "wb"))
