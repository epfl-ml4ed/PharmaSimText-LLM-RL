import argparse
from train import run as agg_train
from scenario_helper import split_single_scenario
from extract_scenarios import scenario_extractor
import os
import shutil
import subprocess
# from gpt.train import train as gpt_train
# from gpt_assisted.train import train as gpt_assisted_train
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
    parser.add_argument("--emb", type=str, default="sum")
    parser.add_argument("--hc", type=str, default=None)
    parser.add_argument("--unq", type=bool, default=False)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--env_step_limit", type=int, default=20)
    parser.add_argument("--seed", type=int, default=256)
    parser.add_argument("--max_steps", type=int, default=500000)
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
    parser.add_argument("--save_path", type=str, default="./models/")
    parser.add_argument("--train_type", type=str, default="normal")
    parser.add_argument("--TAU", type=float, default=0.5)
    parser.add_argument("--pretrain", type=bool, default=False)
    parser.add_argument("--llm_assisted", type=bool, default=False)
    parser.add_argument("--use_attention", type=bool, default=False)
    parser.add_argument("--pretrained_explore", type=float, default=0.3)
    parser.add_argument("--reduce_scenarios", type=bool, default=False)
    parser.add_argument("--patient", type=str, default="baby")
    parser.add_argument("--penalty", type=float, default=-0.01)
    parser.add_argument("--modenum", type=int, default=1)
    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()
    print(args)
    print(os.getcwd())
    path = "./scenarios"

    if args.patient in os.listdir(os.path.join(path, "patients")):
        if args.test_mode in os.listdir(os.path.join(path, "patients", args.patient)):
            for d in ["train", "test","val"]:
                if d in os.listdir(os.path.join(path, "patients", args.patient, args.test_mode)):
                    shutil.copytree(os.path.join(path, "patients", args.patient, args.test_mode, d,str(args.modenum)), os.path.join(path, d))
    agg_train(args)

if __name__ == "__main__":
    main()
