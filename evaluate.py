import argparse
import os
import random

import numpy as np
import torch
import tqdm

import game
import wandb
from drrn import DRRNAgent

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False




def evaluate_episode(agent, env, eval, policy):
    episode = []
    step = 0
    score = 0
    done = False
    agent.reset_dictionaries()
    ob, valid_acts, hc = env.reset()
    scenario_name = env.scenario["name"]
    state = agent.create_state(update_sentence=ob, hc=hc)
    while not done:
        transition = [env.scenario["name"], step, ob[1], ]
        valid_ids = agent.encode_actions(valid_acts)
        _, action_idx, action_values, _ = agent.act(
            [state], [valid_ids], policy=policy, eval_mode=True)
        action_values = action_values[0]
        action_values = action_values.detach().cpu().numpy()
        for a,v in zip(valid_acts, list(action_values)):
            print(a["sentence"], v)
        action_idx = action_idx[0]


        action_str = valid_acts[action_idx]
        state_update, rew, done, valid_acts, hc, traj_score = env.step(ob, action_str)
        if not done:
            trace = env.trace
        ob = state_update
        score += rew
        step += 1
        transition += [action_str, rew, score]
        episode.append(transition)
        state = agent.create_state(
            update_sentence=ob, hc=hc, previous_state=state)

    traj_score = sum(
        a in trace for a in env.scenario["present_actions"]) / len(env.scenario["present_actions"])
    agent.reset_dictionaries()
    return score, episode, traj_score, scenario_name



def run(args):
    config = vars(args)
    ##################set random seed##################
    random_seed = config["seed"]
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    ########start evaluation############
    env = game.Game(path=os.path.join(config["game_path"], "val"), env_step_limit=config["env_step_limit"],
                    wrong_answer=config["wrong_answer"], emb=config["emb"],
                    hc=config["hc"],
                    embedding_dim=config["embedding_dim"],
                    wording=config["wording"], evaluation=config["evaluation"],
                    random_scenarios=False,
                    reward_scale=config["reward_scale"], reduced=config["reduced"])
    state_dim = env.get_state_len()
    total_num = env.get_num_of_scenarios()
    agent = DRRNAgent(config, state_dim)
    agent.load_best()

    score, episode, traj_score, scenario_name = evaluate_episode(agent, env, 0, policy="greedy")
    print(score, traj_score, scenario_name)
    print(episode)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Your program description here")

    # Add arguments with default values
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--embedding_dim", type=int, default=300)
    parser.add_argument("--n_layers_action", type=int, default=1)
    parser.add_argument("--n_layers_state", type=int, default=1)
    parser.add_argument("--n_layers_scorer", type=int, default=1)
    parser.add_argument("--n_layers_lstm", type=int, default=1)
    parser.add_argument("--hidden_dim_action", type=int, default=64)
    parser.add_argument("--hidden_dim_state", type=int, default=512)
    parser.add_argument("--hidden_dim_scorer", type=int, default=512)
    parser.add_argument("--hidden_lstm", type=int, default=128)
    parser.add_argument("--activation", type=str, default="relu")
    parser.add_argument("--emb", type=str, default="lstm")
    parser.add_argument("--hc", type=str, default=None)
    parser.add_argument("--unq", type=bool, default=False)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--env_step_limit", type=int, default=20)
    parser.add_argument("--seed", type=int, default=256)
    parser.add_argument("--max_steps", type=int, default=100000)
    parser.add_argument("--update_freq", type=int, default=1)
    parser.add_argument("--log_freq", type=int, default=500)
    parser.add_argument("--eval_freq", type=int, default=1000)
    parser.add_argument("--memory_size", type=int, default=500000)
    parser.add_argument("--encoder_memory_size", type=int, default=10)
    parser.add_argument("--save_memory", type=float, default=0.5)
    parser.add_argument("--memory_path", type=str, default="./encoder_memory/")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--clip", type=int, default=100)
    parser.add_argument("--game_path", type=str, default="./scenarios")
    parser.add_argument("--wrong_answer", type=bool, default=True)
    parser.add_argument("--soft_reward", type=bool, default=False)
    parser.add_argument("--reward_scale", type=int, default=1)
    parser.add_argument("--wording", type=bool, default=True)
    parser.add_argument("--evaluation", type=str, default="cause")
    parser.add_argument("--document", type=bool, default=False)
    parser.add_argument("--reduced", type=bool, default=False)
    parser.add_argument("--encoder_type", type=str, default="fasttext")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--test_mode", type=str, default="subtask")
    parser.add_argument("--save_path", type=str, default="./artifacts/")
    parser.add_argument("--train_type", type=str, default="episode_unbatched")
    parser.add_argument("--TAU", type=float, default=0.005)
    parser.add_argument("--pretrain", type=bool, default=False)
    parser.add_argument("--llm_assisted", type=bool, default=False)
    parser.add_argument("--use_attention", type=bool, default=False)
    parser.add_argument("--pretrained_explore", type=float, default=0.3)
    parser.add_argument("--reduce_scenarios", type=bool, default=False)
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    print(args)
    run(args)
if __name__ == "__main__":
    main()