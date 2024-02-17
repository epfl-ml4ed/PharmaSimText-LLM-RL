import json
import os

# import tiktoken             # pip install tiktoken
# tokenizer = tiktoken.get_encoding("cl100k_base")
MAXTOKENSINHISTORY = 2500
import time
import openai

if not os.getenv("OPENAI_API_KEY"):

    os.environ["OPENAI_API_KEY"] = "sk-XXXX"


# Get the number of tokens for a string, measured using tiktoken

# TODO: add relevant questions
def success_map(metric, score):
    feedback = ''
    if metric == 'reward':
        if score <= 0:
            feedback += "The agent performed  poorly and could not make the right diagnosis."
        if score > 0:
            feedback += "The agent performed well and made the correct diagnosis"
    elif metric == 'kq':
        if score == 0:
            feedback += "The agent did not ask any relevant questions."
        elif score <= 0.33:
            feedback += "The agent did not ask many relevant questions."
        elif score <= 0.66:
            feedback += "The agent asked some relevant questions."
        elif score < 1:
            feedback += "The agent asked most of the relevant questions."
        elif score == 1:
            feedback += "The agent asked all relevant questions."
    return feedback


def get_trace(data, truncate=False):
    trace = "\n\nCURRENT TRACE\n\n"
    trace += "Task: {}\n\n".format(data["taskDescription"])
    if data['history']:
        for item in data['history']:
            if truncate:
                trace += "Observation: {}\n".format(item['observation'].split('.')[0])
            else:
                trace += "Observation: {}\n".format(item['observation'])
            trace += "Rationale: {}\n".format(item.get('rationale', ""))
            trace += "Action: {}\n\n".format(item['action'])

        trace += "\n\nEVALUATION REPORT:\n"
        trace += "REWARD_FINAL: {}. This means: {}\n".format(data['finalScore'],
                                                             success_map('reward', data['finalScore']))
        trace += "KEY_QUESTIONS_FINAL: {}. This means: {}\n".format(data['finalTrajScore'],
                                                                            success_map('kq', data['finalTrajScore']))

    return trace


def format_memory(memories):
    # memories list of last-k jsons
    memory_string = "\n\nPREVIOUS LEARNINGS\n\n"
    for m in memories:
        if m['summary']:
            memory_string += "TASK: {}\n".format(m['taskDescription'])
            memory_string += "EPISODE: {}\n".format(m['episodeIdx'])
            memory_string += "LEARNINGS: {}\n".format(m['summary'])

            memory_string += "\nEVALUATION REPORT (for the attempt associated with the learning):\n"
            final_score = m['finalScore']
            final_traj_score = m['finalTrajScore']
            memory_string += "REWARD_FINAL: {}. This means: {}\n".format(final_score,
                                                                         success_map('reward', final_score))
            memory_string += "KEY_QUESTIONS_FINAL: {}. This means: {}\n".format(final_traj_score,
                                                                                success_map('kq', final_traj_score))
            memory_string += '\n'

    return memory_string


def summarize(trace, summary_prompt, system_prompt, demo_examples="", prev_memories="", model="gpt4", temp=0.7,
              tokens=1000):
    print(f"trace:{trace}")
    print(f"summary_prompt:{summary_prompt}")
    print(f"system_prompt:{system_prompt}")
    print(f"prev_memories:{prev_memories}")
    print("######################################")
    response = None
    while response is None:
        try:
            client = openai.OpenAI()
            response = openai.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system_prompt},
                          {"role": "user", "content": summary_prompt},
                          {"role": "user", "content": demo_examples},
                          {"role": "user", "content": prev_memories},
                          {"role": "user", "content": trace}],
                temperature=temp,
                max_tokens=tokens,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )
        except Exception as e:
            print(e)
            print("GPT3 error. Retrying in 10 seconds...")
            time.sleep(2)

    output_summary = response.choices[0].message.content
    return output_summary


def summarize_trace_for_preconditions_sTsW(current_run,
                                           prev_runs_list=None,
                                           model="gpt-4-0613",
                                           temp=0,
                                           quadrant=0,
                                           use_last_k_memories=3):
    system_prompt = "You are an expert assistant."
    summary_prompt = "You are given CURRENT TRACE, a sequence of actions that an agent made in a world to accomplish a task." \
                     "Task is detailed at the beginning." \
                     "For each action, there is a rationale why the agent made that action.\n" \
                     "There is an observation that provide details about the response of the customer after each action was executed." \
                     "The CURRENT TRACE is accompanied by an EVALUATION REPORT indicating the success of the attempt to the task.\n\n" \
                     "You can also be provided with PREVIOUS LEARNINGS which are learnings from the previous attempts by the agent for the same task in the same environment/world. TASK indicates the task description. EPISODE indicates the number of previous attempts of the task.\n" \
                     "PREVIOUS LEARNINGS also have EVALUATION REPORTs indicated how sucessful the respective attempt was for solving the task."

    task_prompt = "Generate a numbered list of general advice based on your previous attempt, that will help the agent to successfully accomplish the SAME task with the SAME customer in future.\n" \
                  "Each numbered item in the summary can ONLY be of the form:\n" \
                  "X MAY BE NECCESSARY to Y.\n" \
                  "X SHOULD BE NECCESSARY to Y.\n" \
                  "X MAY CONTRIBUTE to Y.\n" \
                  "X DOES NOT CONTRIBUTE to Y.\n\n" \
                  "Do not write X and Y in all UPPERCASE.\n" \
                  "Summary of learning as a numbered list:"

    final_prompt = summary_prompt + '\n\n' + task_prompt
    trace = get_trace(current_run, truncate=False)
    memories = {'prev_memory': prev_runs_list, 'trace': trace}

    final_trace = memories['trace']
    final_trace += "\n\nList of advice:" if quadrant == 0 else ""
    final_prev_memories = ""
    if memories['prev_memory']:
        final_prev_memories = format_memory(memories['prev_memory'])
    summary = summarize(trace=final_trace, summary_prompt=final_prompt,
                        system_prompt=system_prompt, prev_memories=final_prev_memories,
                        model=model, temp=temp, tokens=1000)

    print("SUMMARY:\n {}".format(summary))
    return summary


def summarize_ep(task, subtask):
    total_ep = 0
    prev_runs_list = []
    for e in os.listdir(f"./results/memory/{task}/{subtask}"):
        total_ep = max(total_ep, int(e.split(".")[0]))
    for e in os.listdir(f"./results/memory/{task}/{subtask}"):
        if e.split(".")[0] != str(total_ep):
            prev_runs_list.append(json.load(open(f"./results/memory/{task}/{subtask}/{e}", "r")))
    current_run = json.load(open(f"./results/memory/{task}/{subtask}/{total_ep}.json", "r"))

    summary = summarize_trace_for_preconditions_sTsW(current_run,
                                                     prev_runs_list=prev_runs_list,
                                                     model="gpt-4-0613",
                                                     temp=0,
                                                     quadrant=1,
                                                     use_last_k_memories=3)
    current_run['summary'] = summary
    json.dump(current_run, open(f"./results/memory/{task}/{subtask}/{total_ep}.json", "w"))


def main():
    task = "baby"
    subtask = 1
    summarize_ep(task, subtask)


if __name__ == '__main__':
    main()
