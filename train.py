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


class TemperatureScheduler:
    def __init__(self, initial_temp, min_temp, total_decay_steps, hold_steps):
        self.initial_temp = initial_temp
        self.current_temp = initial_temp
        self.min_temp = min_temp
        self.total_decay_steps = total_decay_steps
        self.hold_steps = hold_steps
        self.step_count = 0

    def get_temperature(self):
        """
        Return the current temperature and update it for the next step.
        """
        if self.step_count > self.hold_steps:
            decay_step = self.step_count - self.hold_steps
            decay_amount = (self.initial_temp - self.min_temp) / self.total_decay_steps
            self.current_temp = max(self.min_temp, self.initial_temp - decay_amount * decay_step)

        self.step_count += 1
        return self.current_temp


def save_and_log_artifact(agent, artifact_name, file_name, run):
    full_path = os.path.join(wandb.config.save_path, file_name)
    name = artifact_name.split("_")[0].split("-")[0]
    name = "target" if name == "target" else "policy"
    if file_name not in os.listdir(wandb.config.save_path):
        torch.save(getattr(agent, name + "_network").state_dict(), full_path)
    artifact = wandb.Artifact(name=artifact_name, type="model")
    artifact.add_file(full_path)
    run.log_artifact(artifact)
    return


def update_and_log_model_artifact(run, artifact_name, model_filename):
    saved_artifact = run.use_artifact(f"{artifact_name}:latest")
    draft_artifact = saved_artifact.new_draft()
    draft_artifact.remove(model_filename)
    model_full_path = os.path.join(wandb.config.save_path, model_filename)
    draft_artifact.add_file(model_full_path)
    wandb.log_artifact(draft_artifact)
    return


# running average
def delete_files_in_directory(directory_path):
    # Get a list of all files in the directory
    file_list = os.listdir(directory_path)

    # Iterate over the list and delete each file
    for file_name in file_list:
        file_path = os.path.join(directory_path, file_name)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                os.rmdir(file_path)
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")


def moving_average(eval, x, window=10):
    eval.append(x)
    if len(eval) > window:
        eval = eval[1:]
    return np.mean(eval), eval


def evaluate(agent, env, eval, eval_list, window=10, nb_episodes=4, policy="greedy", mode="train", evaluation="cause",
             scaled=False):
    cols = ["scenario", "step", "obs_update", "action", "reward", "total_reward"]
    with torch.no_grad():
        total_score = 0
        total_traj_score = 0
        episodes = []
        correct_scenarios = []
        for ep in range(nb_episodes):
            score, episode, traj_score, scenario_name = evaluate_episode(
                agent, env, eval, policy=policy)
            if evaluation == "old":
                if not scaled:
                    score = score == 5
                else:
                    score = score == 1
            else:
                score = score == 1
            if score:
                correct_scenarios.append(scenario_name)
            total_score += score
            total_traj_score += traj_score
            episodes += episode
            env.increase_episodes()
        avg_score = total_score / nb_episodes
        avg_traj_score = total_traj_score / nb_episodes
        ma_score, eval_list = moving_average(eval_list, avg_score, window)
        t = wandb.Table(columns=cols, data=episodes)
        cs = wandb.Table(columns=["evaluation_step", "score", "correct_scenarios"],
                         data=[[eval, avg_score, ", ".join(correct_scenarios)]])
        wandb.log({
            f"evaluation_{mode}": t,
            f"correct_scenarios_{mode}": cs,
            f"Eval_Score_{mode}": avg_score,
            f"Traj_Eval_Score_{mode}": avg_traj_score,
            f"MA_Eval_Score_{mode}": ma_score}, )

        return avg_score, eval_list


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
            [state], [valid_ids], policy=policy, eval_mode=True, action_strs=valid_acts)
        action_idx = action_idx[0]
        action_values = action_values[0]
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


def train(
        agent, env_train, env_train_eval, env_val, env_test, max_steps, update_freq, eval_freq, log_freq, total_num,
        train_type="episode_unbatched", window=10, evaluation="cause", scaled=False, run=None
):
    initial_temp = 1.0
    min_temp = 0.01
    total_decay_steps = int(max_steps*0.8)# Temperature will decay linearly for the next 800 steps
    hold_steps = int(max_steps*0.1)  # Temperature will stay at maximum for the first 100 steps

    temp_scheduler = TemperatureScheduler(initial_temp, min_temp, total_decay_steps, hold_steps)

    # save_and_log_artifact(agent, "target-model", "target_model.pt", run)
    # save_and_log_artifact(agent, "policy-model", "policy_model.pt", run)
    save_and_log_artifact(agent, "best-model", "best_model.pt", run)
    eval_train, eval_val, eval_test = [], [], []
    max_score_val = -100
    obs, actions_list, hc = env_train.reset()
    score = 0
    state = agent.create_state(update_sentence=obs, hc=hc)
    done = False
    if agent.emb == "lstm":
        state = state.reshape(1, -1)
    valid_actions = agent.encode_actions(actions_list)
    for step in tqdm.tqdm(range(1, max_steps + 1)):
        temperature = temp_scheduler.get_temperature()

        if train_type == "normal":
            action_id, action_idx, values, next_state = agent.act(
                [state], [valid_actions], policy="softmax", eval_mode=False, action_strs=actions_list,
                temperature=temperature)

            action_str = actions_list[action_idx[0]]

            state_update, reward, done, actions_list, hc, traj_score = env_train.step(
                obs, action_str)

            obs = state_update
            score += reward

            next_state = agent.create_state(
                update_sentence=obs, hc=hc, previous_state=state)
            next_valid = agent.encode_actions(actions_list)
            agent.observe([state, action_id[0],
                           reward, next_state, next_valid, done])

            if done:
                agent.reset_dictionaries()
                wandb.log({"Train_ep_score": score})
                score = 0
                obs, actions_list, _ = env_train.reset()
                next_valid = agent.encode_actions(actions_list)
                next_state = agent.create_state(update_sentence=obs, hc=hc, previous_state=None)
            state = next_state
            valid_actions = next_valid
            if step % update_freq == 0:
                loss = agent.update()
                if loss is not None:
                    wandb.log({"Loss": loss})
        elif train_type == "episode_unbatched":
            #####use one episode to train
            while not done:
                action_id, action_idx, _, hidden = agent.act(
                    [state] if isinstance(state, np.ndarray) else state, [valid_actions], policy="softmax",
                    temperature=temperature, eval_mode=False, action_strs=actions_list)
                action_str = actions_list[action_idx[0]]
                state_update, reward, done, actions_list, hc, traj_score = env_train.step(
                    obs, action_str)

                obs = state_update
                score += reward
                next_state = agent.create_state(
                    update_sentence=obs, hc=hc, previous_state=state)
                next_valid = agent.encode_actions(actions_list)
                agent.observe([state if isinstance(state, np.ndarray) else state[0], action_id[0],
                               reward, next_state, next_valid, done])
                if done:
                    agent.reset_dictionaries()
                    wandb.log({"Train_ep_score": score})
                    score = 0
                    state = None
                    obs, actions_list, _ = env_train.reset()
                    next_valid = agent.encode_actions(actions_list)
                    next_state = agent.create_state(update_sentence=obs, hc=hc, previous_state=None)
                valid_actions = next_valid
                if state is None:
                    state = next_state
                else:
                    state = tuple([next_state, hidden])

            loss = agent.update()
            if loss is not None:
                wandb.log({"Loss": loss})
            done = False
        else:
            raise NotImplementedError
        if step % eval_freq == 0:
            _, eval_train = evaluate(agent, env_train_eval, step, eval_train, policy="softmax", mode="train",
                                     window=window, nb_episodes=env_train_eval.get_num_of_scenarios(),
                                     evaluation=evaluation, scaled=scaled)
            avg_score_val, eval_val = evaluate(agent, env_val, step, eval_val, policy="softmax", mode="val",
                                               window=window, nb_episodes=env_val.get_num_of_scenarios(),
                                               evaluation=evaluation, scaled=scaled)
            if avg_score_val >= max_score_val:
                max_score_val = avg_score_val
                agent.save_best()
                update_and_log_model_artifact(run, "best-model", "best_model.pt")
        # if step % log_freq == 0:
        #     agent.save_recent()
        #     update_and_log_model_artifact(run, "policy-model", "policy_model.pt")
        #     update_and_log_model_artifact(run, "target-model", "target_model.pt")
    agent.load_best()
    _, test_eval = evaluate(agent, env_test, step, eval_test, policy="softmax", window=window, mode="test",
                            nb_episodes=env_test.get_num_of_scenarios(), evaluation=evaluation, scaled=scaled)


def run(args):
    print("training started")
    with wandb.init(config=args,
                    project=(str(args.modenum)+args.patient + "_" + args.test_mode + "_") + args.encoder_type + "_" + args.evaluation + "_" + args.emb + (
                            "_attention" if args.use_attention else "") + (
                            "_reduce_scenarios" if args.reduce_scenarios else "")) as run:
        delete_files_in_directory(wandb.config.memory_path)
        print(os.listdir(wandb.config.memory_path))
        delete_files_in_directory(wandb.config.save_path)
        print(os.listdir(wandb.config.save_path))
        random_seed = wandb.config["seed"]
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        spec = ','.join([wandb.config["evaluation"],
                         str(wandb.config["hc"]
                             ) if wandb.config["hc"] is not None else "",
                         str(wandb.config["emb"]
                             ) if wandb.config["emb"] is not None else "",
                         wandb.config["encoder_type"],
                         "reduced=" +
                         str(wandb.config["reduced"]
                             ) if wandb.config["encoder_type"] == "fasttext" else "",
                         "" if wandb.config["evaluation"] == "cause" else (
                                 "scaled=" + str(wandb.config["reward_scale"] != 0)),
                         "wording=" + str(wandb.config["wording"]
                                          ) if wandb.config["emb"] is not None else "",
                         "hidden_dim_action=" +
                         str(wandb.config["hidden_dim_action"]),
                         "hidden_dim_state=" + str(wandb.config["hidden_dim_state"]),
                         "hidden_dim_scorer=" +
                         str(wandb.config["hidden_dim_scorer"]),
                         "clip=" + str(wandb.config["clip"])])
        wandb.config.update({"spec": spec}, allow_val_change=True)
        if wandb.config["encoder_type"] == "fasttext":
            wandb.config.update(
                {"embedding_dim": 100 if (wandb.config["reduced"] and wandb.config["emb"] is not None) else 300},
                allow_val_change=True)
        elif wandb.config["encoder_type"] == "bert":
            wandb.config.update(
                {"embedding_dim": 384}, allow_val_change=True)
        else:
            wandb.config.update(
                {"embedding_dim": 768}, allow_val_change=True)
        config = wandb.config
        # split_train_val_test(path=config["game_path"], train_ratio=config["train_ratio"],test_ratio=config["test_ratio"], test_mode=config["test_mode"])
        env_train = game.Game(path=os.path.join(config["game_path"], "train"), env_step_limit=config["env_step_limit"],
                              wrong_answer=config["wrong_answer"],
                              emb=config["emb"], hc=config["hc"],
                              embedding_dim=config["embedding_dim"],
                              wording=config["wording"], evaluation=config["evaluation"],
                              random_scenarios=True,
                              reward_scale=config["reward_scale"], reduced=config["reduced"],penalty=config["penalty"],training=True)
        env_train_eval = game.Game(path=os.path.join(config["game_path"], "train"),
                                   env_step_limit=config["env_step_limit"],
                                   wrong_answer=config["wrong_answer"], emb=config["emb"],
                                   hc=config["hc"],
                                   embedding_dim=config["embedding_dim"],
                                   wording=config["wording"], evaluation=config["evaluation"],
                                   random_scenarios=False,
                                   reward_scale=config["reward_scale"], reduced=config["reduced"])
        env_val = game.Game(path=os.path.join(config["game_path"], "val"), env_step_limit=config["env_step_limit"],
                            wrong_answer=config["wrong_answer"], emb=config["emb"],
                            hc=config["hc"],
                            embedding_dim=config["embedding_dim"],
                            wording=config["wording"], evaluation=config["evaluation"],
                            random_scenarios=False,
                            reward_scale=config["reward_scale"], reduced=config["reduced"])
        env_test = game.Game(path=os.path.join(config["game_path"], "test"), env_step_limit=config["env_step_limit"],
                             wrong_answer=config["wrong_answer"],
                             emb=config["emb"],
                             hc=config["hc"],
                             embedding_dim=config["embedding_dim"],
                             wording=config["wording"], evaluation=config["evaluation"],
                             random_scenarios=False,
                             reward_scale=config["reward_scale"], reduced=config["reduced"])
        state_dim = env_train.get_state_len()
        total_num = env_train.get_num_of_scenarios()
        agent = DRRNAgent(config, state_dim)
        train(
            agent,
            env_train,
            env_train_eval,
            env_val,
            env_test,
            config["max_steps"],
            config["update_freq"],
            config["eval_freq"],
            config["log_freq"],
            total_num,
            evaluation=config["evaluation"],
            scaled=config["reward_scale"] != 0,
            window=config["window"],
            train_type=config["train_type"],
            run=run
        )
