import json
import os
import pathlib

import openai
from tenacity import retry, wait_random_exponential, stop_after_attempt

import game
import llm_helper as llm_helper
from llm_helper import format_actions
from scenario_helper import extract_all_scenarios
from test import summarize_ep

PROMPT = 2


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(10))
def summarize_with_retry(summarizer, pa, customer, summarize=True):
    return summarizer.summarize(pa, customer, summarize)


if not os.getenv("OPENAI_API_KEY"):
    # TODO: Annonymize
    os.environ["OPENAI_API_KEY"] = "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"


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(10))
def act_with_retry(agent, summary, valid_actions):
    return agent.act(summary, valid_actions)


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(1))
def act2_with_retry(agent, obs_history, action_history, current_obs, valid_actions, summary=""):
    return agent.act2(obs_history, action_history, current_obs, valid_actions, summary=summary)


class GPTAgent:
    def __init__(self, prompt_format=llm_helper.AgentPrompt(), model="gpt-3.5-turbo", temperature=0,
                 max_tokens=100) -> None:
        super().__init__()
        self.prompt_format = prompt_format
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def act(self, summary, valid_actions):
        prompt = self.prompt_format.format_prompt(summary, valid_actions)
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model=self.model,
            messages=prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return self.prompt_format.parse_response(response)

    def act2(self, obs_history, action_history, current_obs, valid_actions, summary=""):
        prompt = self.prompt_format.format_prompt(obs_history, action_history, current_obs, valid_actions,
                                                  summary=summary)
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model=self.model,
            messages=prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return self.prompt_format.parse_response(response)


def main():
    if PROMPT == 1:
        agent = GPTAgent(prompt_format=llm_helper.AgentPrompt(oneshot=False), temperature=0.3, max_tokens=500,
                         model="gpt-4"
                         )
        num_of_trials = 1
        if len(os.listdir("./scenarios/all_scenarios")) == 0:
            extract_all_scenarios()
        env = game.Game(path="./scenarios/all_scenarios")
        prompt_format = llm_helper.SummarizerPrompt(oneshot=False)
        summarizer = llm_helper.Summarizer(
            prompt_format, temperature=0.3, max_tokens=500, model="gpt-4")
        files = env.get_scenario_files()
        if not os.path.exists("./results/gpt_result.json"):
            result = {}
        else:
            result = json.load(open("./results/gpt_result.json", "r"))
        print(files)
        files = files[:2]
        for file in files:
            if file in result.keys():
                if len(result[file]["rewards"]) >= num_of_trials:
                    continue
                else:
                    result[file]["rewards"] = []
            initial_state, actions, _ = env.load_scenario(file, gpt=True)
            actions = format_actions(actions)
            initial_actions = actions
            pa = []
            customer = [initial_state[1]]
            state = initial_state
            score = 0
            pas = []
            customers = []
            rewards = []
            for i in range(num_of_trials):
                print(i)
                terminal = False
                while not terminal:
                    summary = summarize_with_retry(
                        summarizer, pa, customer, summarize=False)
                    # print(summary)
                    action = act_with_retry(
                        agent=agent, summary=summary, valid_actions=actions)
                    pa.append(action)
                    state, reward, terminal, actions, hc, traj_score = env.scenario_step(
                        state, action.split("\n")[0], i)
                    score += reward
                    if state[1] == "done":
                        print(summary)
                        print(action)
                        break
                    actions = format_actions(actions)
                    # print((state[1].split(".")))
                    if state[0] == "interaction":
                        customer.append(".".join(state[1].split(".")[2:]))
                    else:
                        # print(".".join(state[1].split(".")[-1:]))
                        customer.append(".".join(state[1].split(".")[-1:]))
                env.reset()
                rewards.append(reward)
                pas.append(pa)
                customers.append(customer)
                pa = []
                customer = [initial_state[1]]
                state = initial_state
                actions = initial_actions
                print(score / (i + 1))

                result[file] = {"states": customers, "actions": pas,
                                "rewards": rewards, "avg_score": score / (i + 1)}
                json.dump(result, open("./results/gpt_result.json", "w"))

            print(score / num_of_trials)

            result[file] = {"states": customers, "actions": pas,
                            "rewards": rewards, "avg_score": score / num_of_trials}
            json.dump(result, open("./results/gpt_result.json", "w"))
    else:
        agent = GPTAgent(prompt_format=llm_helper.AgentPrompt2(), temperature=0.3, max_tokens=1024, model="gpt-4"
                         )
        if len(os.listdir("./scenarios/all_scenarios")) == 0:
            extract_all_scenarios()
        env = game.Game(path="./scenarios/all_scenarios")
        files = env.get_scenario_files()

        print(files)

        files = files
        for ep in range(5):
            result = {}
            # if not os.path.exists(f"./results/clin_result_{episodeIdx}.json"):
            #     result = {}
            # else:
            # result = json.load(open("./results/gpt_result2.json", "r"))
            for file in files:
                task, sub_task = file.split(".")[0].split("_")
                save_path = f"./results/memory/{task}/{sub_task}"
                if not os.path.exists(save_path):
                    pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)
                    episodeIdx = 0
                else:
                    episodeIdx = len(os.listdir(save_path))
                episodeIdx = max(episodeIdx, ep)
                if episodeIdx > 0:
                    summary = json.load(open(f"{save_path}/{episodeIdx - 1}.json", "r"))["summary"]
                else:
                    summary = ""
                file_name = f"{save_path}/{episodeIdx}.json"

                if file in result.keys():
                    if len(result[file]["rewards"]) > 0:
                        continue
                    else:
                        result[file]["rewards"] = []
                initial_state, actions, _ = env.load_scenario(file, gpt=True)

                history = []
                taskDescription = f"You see a customer saying{initial_state[1]}"
                print("###################################")
                print(env.scenario["name"])
                print("###################################")
                valid_actions = format_actions(actions)
                action_history = []
                obs_history = []
                current_obs = initial_state[1]
                state = initial_state
                score = 0
                final_traj_score = 0
                rewards = []
                learning_ids = []
                reasonings = []
                terminal = False
                while not terminal:
                    invalid_action = True
                    i = 0
                    while invalid_action and i < 5:
                        i += 1
                        action = act2_with_retry(
                            agent=agent, obs_history=obs_history, action_history=action_history,
                            current_obs=current_obs,
                            valid_actions=valid_actions, summary=summary)
                        action = action.split("$$$")
                        learning_id = (action[0].replace("\n", ""))
                        if len(action) > 1:
                            action = action[1]
                            action = action.split("###")
                            if len(action) > 1:
                                reasoning = (action[0].replace("\n", ""))
                                action = action[1].replace("\n", "")
                                invalid_action = False
                    try:
                        state, reward, terminal, actions, hc, traj_score = env.scenario_step(
                            state, action, i)

                    except Exception as e:
                        print(e)
                        invalid_action = True
                        i = 0
                        while invalid_action and i < 5:
                            i += 1
                            action = act2_with_retry(
                                agent=agent, obs_history=obs_history, action_history=action_history,
                                current_obs=current_obs,
                                valid_actions=valid_actions)
                            action = action.split("$$$")
                            learning_id = (action[0].replace("\n", ""))
                            if len(action) > 1:
                                action = action[1]
                                action = action.split("###")
                                if len(action) > 1:
                                    reasoning = (action[0].replace("\n", ""))
                                    action = action[1].replace("\n", "")
                                    invalid_action = False
                    action_history.append(action)
                    obs_history.append(current_obs)
                    learning_ids.append(learning_id)
                    reasonings.append(reasoning)
                    print(reasoning)
                    print(action)
                    history.append({"observation": current_obs, "action": action, "rationale": reasoning, })
                    score += reward
                    final_traj_score += traj_score
                    rewards.append(reward)
                    if state[1] == "done":
                        break
                    valid_actions = format_actions(actions)
                    if state[0] == "interaction":
                        current_obs = (".".join(state[1].split(".")[2:]))
                    else:
                        current_obs = (".".join(state[1].split(".")[-1:]))
                env.reset()
                print(score)

                result[file] = {"observations": obs_history, "actions": action_history,
                                "rewards": rewards, "score": score}
                json.dump(result, open(f"./results/clin_result_{episodeIdx}.json", "w"))
                data = dict()
                data["taskDescription"] = taskDescription
                data["episodeIdx"] = episodeIdx
                data["history"] = history
                data["finalScore"] = score
                data["finalTrajScore"] = final_traj_score
                json.dump(data, open(file_name, "w"))
                summarize_ep(task, sub_task)


if __name__ == "__main__":
    main()
