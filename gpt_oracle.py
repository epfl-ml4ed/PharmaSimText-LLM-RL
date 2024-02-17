import openai
import os
import llm_helper
from tenacity import retry, stop_after_attempt, wait_random_exponential
from scenario_helper import extract_all_scenarios
from llm_helper import format_actions
import json
import re
import game
if not os.getenv("OPENAI_API_KEY"):
    # TODO: Annonymize
    os.environ["OPENAI_API_KEY"] = "sk-XXXXX"


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(1))
def summarize_with_retry(summarizer, pa, customer, summarize=True):
    return summarizer.summarize(pa, customer, summarize)


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(1))
def suggest_with_retry(oracle, summary, valid_actions):
    return oracle.suggest(summary, valid_actions)


class GPTOracle:
    def __init__(self, prompt_format=llm_helper.AgentPrompt(), model="gpt-3.5-turbo", temperature=0,
                 max_tokens=100) -> None:
        super().__init__()
        self.prompt_format = prompt_format
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def suggest(self, summary, valid_actions):
        prompt = self.prompt_format.format_prompt(summary, valid_actions)
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model=self.model,
            messages=prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return self.prompt_format.parse_response(response)


def main():
    env = game.Game(path="./scenarios/all_scenarios")
    oracle = GPTOracle(prompt_format=llm_helper.OraclePrompt(oneshot=False, game=env), temperature=0.3, max_tokens=500, model="gpt-3.5-turbo"
                       )
    num_of_trials = 1
    if len(os.listdir("./scenarios/all_scenarios")) == 0:
        extract_all_scenarios()

    prompt_format = llm_helper.SummarizerPrompt(oneshot=False)
    summarizer = llm_helper.Summarizer(
        prompt_format, temperature=0.3, max_tokens=500, model="gpt-3.5-turbo")
    files = env.get_scenario_files()
    file = files[2]
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
            action = suggest_with_retry(
                oracle=oracle, summary=summary, valid_actions=actions)
            print(action)
            action = action[0]
            pa.append(action)
            state, reward, terminal, actions, hc, traj_score = env.scenario_step(
                state, action, i)
            score += reward
            if state[1] == "done":
                print(summary)
                print(action)
                break
            actions = format_actions(actions)
            if state[0] == "interaction":
                customer.append(".".join(state[1].split(".")[2:]))
            else:
                customer.append(".".join(state[1].split(".")[-1:]))
        env.reset()
        rewards.append(reward)
        pas.append(pa)
        customers.append(customer)
        pa = []
        customer = [initial_state[1]]
        state = initial_state
        actions = initial_actions

    print(score/num_of_trials)


if __name__ == "__main__":
    main()
