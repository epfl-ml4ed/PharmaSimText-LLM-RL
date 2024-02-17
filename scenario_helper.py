import os
import json

import numpy as np

from extract_scenarios import scenario_extractor
import shutil
import random
import argparse
import openai
api_key = "sk-xxxxx"
def rewrite_sentence(sentence):
    client = openai.OpenAI(api_key=api_key)
    msg = client.chat.completions.create(
        model="gpt-3.5-turbo",
        # max_tokens=max_tokens,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "You are a customer consulting with a pharmacist about your urinary problems.",
            },
            {
                "role": "user",
                "content": """
                        task: try to articulate the sentences I give you in 9 different ways.
                        sentence: "I\'ve been going to the bathroom a lot more than usual, both during the day and at night. Sometimes, I feel like I need to go urgently, but then I have trouble starting."
                    """,
            },
            {
                "role": "user",
                "content": """
                        task: try to articulate the sentences I give you in 9 different ways. make sure you are using this tempelate:
                        ["s1","s2","s3","s4","s5","s6","s7","s8","s9"]
                        sentence: “he has been having a diarrhea. he doesn’t have a fever. he isn’t vomiting either and his urine seems normal.”
                    """,
            },
            {
                "role": "assistant",
                "content": """[
                "I've noticed an increased frequency of bathroom visits, both in the daytime and at night. Occasionally, I experience a strong urge to go, but initiating the process becomes challenging.",
                "Lately, I've been using the restroom more frequently, not only during the day but also at night. There are moments when I feel a sudden need to go, yet I encounter difficulties in commencing the process.",
                "I've found myself visiting the bathroom more often than usual, both in the daytime and throughout the night. On occasion, I experience a sense of urgency, but initiating the process becomes troublesome.",
                "My bathroom trips have become more frequent recently, occurring not only during the day but also during the night. Sometimes, I feel an urgent need to go, but initiating the process becomes problematic.",
                "I've been making more trips to the bathroom than usual, both during the day and at night. There are instances when I feel an immediate need to go, but initiating the process becomes troublesome.",
                "Recently, I've had to use the restroom more often, both during the day and at night. There are moments when I feel an urgent need to go, but commencing the process becomes challenging.",
                "I've observed a heightened frequency of bathroom visits, both during the day and at night. Occasionally, I experience a strong urge to go, but initiating the process becomes difficult.",
                "Lately, I've been visiting the bathroom more frequently, not only during the day but also at night. There are times when I feel a sudden need to go, but initiating the process becomes problematic.",
                "My restroom trips have become more frequent recently, taking place not only during the day but also at night. Sometimes, I feel an immediate need to go, but starting the process becomes challenging."
            ]""",
            },
            {"role": "user", "content": "sentence: “" + sentence + "”"},
        ],
    )
    sentences = msg.choices[0].message.content
    y = [sentence]
    for s in sentences.split('"'):
        if len(s) > 50:
            y.append(s)
    return y
def parse_arguments():
    parser = argparse.ArgumentParser(description="Your program description here")

    # Add arguments with default values
    parser.add_argument("--path", type=str, default="./scenarios")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--test_mode", type=str, default="wording")
    args = parser.parse_args()
    return args

def extract_all_scenarios(path="./scenarios"):
    dirs = ["all_scenarios"]
    for d in dirs:
        if os.path.exists(os.path.join(path, d)):
            try:
                shutil.rmtree(os.path.join(path, d))
            except OSError as e:
                print(f"Error: {e}")
        os.makedirs(os.path.join(path, d))
    subjects, num_of_subtasks = scenario_extractor(path)
    for s in subjects:
        scenario_path = os.path.join(path, s)
        shutil.copytree(scenario_path,
                        os.path.join(path, dirs[0]), dirs_exist_ok=True)
    return
def split_single_scenario(path="./scenarios/", patient = "baby", train_ratio=0.8, test_ratio=0.1, test_mode="wording",mode_nums=3):
    dirs = ["train", "val", "test"]

    val_ratio = 1 - train_ratio - test_ratio
    if test_mode == "subtask":
        scenario_path = os.path.join(path, "patients", patient)
        for d in dirs:
            for i in range(mode_nums):
                if not os.path.exists(os.path.join(scenario_path, test_mode, d,str(i))):
                    os.makedirs(os.path.join(scenario_path, test_mode, d,str(i)))
        available_scenarios = [f for f in os.listdir(
            scenario_path) if f.endswith('.json')]
        test_splits = []
        val_splits = []
        while len(test_splits) < mode_nums:
            test, val = random.sample(list(np.arange(len(available_scenarios))), k=2)
            if test not in test_splits and val not in val_splits:
                test_splits.append(test)
                val_splits.append(val)
        i = 0
        for test_idx, val_idx in zip(test_splits, val_splits):
            test_scenarios = [available_scenarios[test_idx]]
            val_scenarios = [available_scenarios[val_idx]]
            train_scenarios = [s for s in available_scenarios if s not in [test_scenarios[0], val_scenarios[0]]]
            print(train_scenarios, val_scenarios, test_scenarios)
            sets = [train_scenarios, val_scenarios, test_scenarios]
            for sset in sets:
                for t in sset:
                    print(os.path.join(scenario_path, t))
                    print(os.path.join(scenario_path,test_mode, dirs[sets.index(sset)],str(i), t))
                    shutil.copy(os.path.join(scenario_path, t),
                                os.path.join(scenario_path,test_mode, dirs[sets.index(sset)],str(i), t))
            i += 1
    elif test_mode == "wording" or ("simple" in test_mode):
        scenario_path = os.path.join(path, "patients", patient)
        available_scenarios = [f for f in os.listdir(
            scenario_path) if f.endswith('.json')]
        #########################simplify one cause#########################
        if test_mode == "simple1":
            available_scenarios = available_scenarios[:1]
        for subtask in available_scenarios:

            with open(os.path.join(scenario_path, subtask), 'r') as f:
                scenario = json.load(f)
            for c in scenario["characters"][:-1]:
                for q in scenario["question_answers"][c].keys():
                    wordings = scenario["question_answers"][c][q]
                    random.shuffle(wordings)
                    num_of_wordings = len(wordings)

                    if num_of_wordings > 3:
                        num_of_test_wordings = int(
                            num_of_wordings * test_ratio) if int(num_of_wordings * test_ratio) > 0 else 1
                        num_of_val_wordings = int(
                            num_of_wordings * val_ratio) if int(num_of_wordings * val_ratio) > 0 else 1
                        num_of_train_wordings = num_of_wordings - \
                            num_of_test_wordings - num_of_val_wordings

                        test_wordings = wordings[:num_of_test_wordings]
                        val_wordings = wordings[num_of_test_wordings:
                                                num_of_test_wordings + num_of_val_wordings]
                        train_wordings = wordings[num_of_test_wordings +
                                                  num_of_val_wordings:]
                    elif num_of_wordings == 3:
                        test_wordings = [wordings[0]]
                        val_wordings = [wordings[1]]
                        train_wordings = [wordings[2]]
                    else:
                        test_wordings = [wordings[0]]
                        val_wordings = [wordings[0]]
                        train_wordings = [wordings[0]]
                    #########################simplify one wording#########################
                    if "simple" in test_mode:
                        train_wordings = [train_wordings[0]]
                        val_wordings = [val_wordings[0]]
                        test_wordings = [test_wordings[0]]
                    for (d, w) in zip(dirs, [train_wordings, val_wordings, test_wordings]):
                        if not os.path.exists(os.path.join(scenario_path, test_mode, d)):
                            os.makedirs(os.path.join(scenario_path, test_mode, d))
                        if os.path.exists(os.path.join(path, d, subtask)):
                            data = json.load(open(os.path.join(path, d, subtask), 'r'))
                        else:
                            data = json.load(open(os.path.join(scenario_path, subtask), 'r'))
                        data["question_answers"][c][q] = w
                        print(d, w)
                        json.dump(data, open(os.path.join(scenario_path, test_mode, d, subtask), 'w'), indent=4)
    else:
        raise Exception("Invalid test mode.")
    return

def split_train_val_test(path="../scenarios", train_ratio=0.8, test_ratio=0.1, test_mode="wording"):
    print(os.path.abspath(path))
    dirs = ["train", "val", "test"]
    for d in dirs:
        if os.path.exists(os.path.join(path, d)):
            try:
                shutil.rmtree(os.path.join(path, d))
            except OSError as e:
                print(f"Error: {e}")
        os.makedirs(os.path.join(path, d))
    val_ratio = 1 - train_ratio - test_ratio
    subjects, num_of_subtasks = scenario_extractor(path)
    print(subjects)
    if test_mode == "task":
        if len(subjects) < 3:
            raise Exception("Not enough scenarios for this test mode.")
        else:
            random.shuffle(subjects)
            num_of_tasks = len(subjects)
            num_of_test_tasks = int(
                num_of_tasks * test_ratio) if int(num_of_tasks * test_ratio) > 0 else 1
            num_of_val_tasks = int(
                num_of_tasks * val_ratio) if int(num_of_tasks * val_ratio) > 0 else 1
            num_of_train_tasks = num_of_tasks - num_of_test_tasks - num_of_val_tasks
            test_tasks = subjects[:num_of_test_tasks]
            val_tasks = subjects[num_of_test_tasks:num_of_test_tasks +
                                 num_of_val_tasks]
            train_tasks = subjects[num_of_test_tasks + num_of_val_tasks:]
            sets = [train_tasks, val_tasks, test_tasks]
            for sset in sets:
                for t in sset:
                    shutil.copytree(os.path.join(path, t),
                                    os.path.join(path, dirs[sets.index(sset)]))
    elif test_mode == "subtask":
        for s in subjects:
            scenario_path = os.path.join(path, s)
            available_scenarios = [f for f in os.listdir(
                scenario_path) if f.endswith('.json')]
            random.shuffle(available_scenarios)
            num_of_scenarios = len(available_scenarios)
            num_of_test_scenarios = int(
                num_of_scenarios * test_ratio) if int(num_of_scenarios * test_ratio) > 0 else 1
            num_of_val_scenarios = int(
                num_of_scenarios * val_ratio) if int(num_of_scenarios * val_ratio) > 0 else 1
            num_of_train_scenarios = num_of_scenarios - \
                num_of_test_scenarios - num_of_val_scenarios
            test_scenarios = available_scenarios[:num_of_test_scenarios]
            val_scenarios = available_scenarios[num_of_test_scenarios:
                                                num_of_test_scenarios + num_of_val_scenarios]
            train_scenarios = available_scenarios[num_of_test_scenarios +
                                                  num_of_val_scenarios:]
            sets = [train_scenarios, val_scenarios, test_scenarios]
            for sset in sets:
                for t in sset:
                    shutil.copy(os.path.join(scenario_path, t),
                                os.path.join(path, dirs[sets.index(sset)], t))
    elif test_mode == "wording":
        for s in subjects:
            scenario_path = os.path.join(path, s)
            available_scenarios = [f for f in os.listdir(
                scenario_path) if f.endswith('.json')]
            for subtask in available_scenarios:
                with open(os.path.join(scenario_path, subtask), 'r') as f:
                    scenario = json.load(f)
                for c in scenario["characters"][:-1]:
                    for q in scenario["question_answers"][c].keys():
                        wordings = scenario["question_answers"][c][q]
                        # random.shuffle(wordings)
                        num_of_wordings = len(wordings)

                        if num_of_wordings > 3:
                            num_of_test_wordings = int(
                                num_of_wordings * test_ratio) if int(num_of_wordings * test_ratio) > 0 else 1
                            num_of_val_wordings = int(
                                num_of_wordings * val_ratio) if int(num_of_wordings * val_ratio) > 0 else 1
                            num_of_train_wordings = num_of_wordings - \
                                num_of_test_wordings - num_of_val_wordings

                            test_wordings = wordings[:num_of_test_wordings]
                            val_wordings = wordings[num_of_test_wordings:
                                                    num_of_test_wordings + num_of_val_wordings]
                            train_wordings = wordings[num_of_test_wordings +
                                                      num_of_val_wordings:]
                        elif num_of_wordings == 3:
                            test_wordings = [wordings[0]]
                            val_wordings = [wordings[1]]
                            train_wordings = [wordings[2]]
                        else:
                            test_wordings = [wordings[0]]
                            val_wordings = [wordings[0]]
                            train_wordings = [wordings[0]]
                        for (d, w) in zip(dirs, [train_wordings, val_wordings, test_wordings]):
                            print(d,w)
                            if os.path.exists(os.path.join(path, d, subtask)):
                                data = json.load(open(os.path.join(path, d, subtask), 'r'))
                            else:
                                data = json.load(open(os.path.join(path, s, subtask), 'r'))
                            data["question_answers"][c][q] = w
                            with open(os.path.join(path, d, subtask), 'w') as f:
                                json.dump(data, f)
                            print(d)
                            te = json.load(open(os.path.join(path, "train", subtask), 'r'))
                            print(te["question_answers"][c][q])
    else:
        raise Exception("Invalid test mode.")
    return
def main():
    args = parse_arguments()
    split_train_val_test(path=args.path, train_ratio=args.train_ratio,
                         test_ratio=args.test_ratio, test_mode=args.test_mode)
    for d in ["train", "val", "test"]:
        os.listdir(os.path.join(args.path, d))
        for f in os.listdir(os.path.join(args.path, d))[:1]:
            if f.endswith(".json"):
                with open(os.path.join(args.path, d, f), "r") as json_file:
                    data = json.load(json_file)
                print(data["question_answers"])
    print(f"Done!{args.path}")

if __name__ == '__main__':
    main()
