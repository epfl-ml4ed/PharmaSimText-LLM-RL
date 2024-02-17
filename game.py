import json
import os
import random

import numpy as np
import sentence_transformers

from encoder import Encoder


def add_onehot(l):
    l = list(l)
    i = l.index(1) if 1 in l else -1
    l[i + 1] = 1
    if i > -1:
        l[i] = 0
    return np.array(l)


class Game:
    def __init__(self, name="Pharmasim", path=r".\scenarios", env_step_limit=20,
                 wrong_answer=False, reward_scale=1, penalty=0, emb="sum", hc="bq", embedding_dim=300,
                 wording=True, evaluation="cause", scenario_name=False,
                 random_scenarios=False, reduced=True, training = False) -> None:
        self.reduced = "reduced" if reduced else "not_reduced"
        self.evaluation = evaluation
        self.wording = wording
        self.path = path
        self.training = training
        # maximum number of allowed interactions per episode
        self.max_steps_per_episode = env_step_limit
        self.wrong_answer = wrong_answer
        self.penalty = penalty
        self.reward_scale = reward_scale
        self.subjects = list(set(
            [f.split('_')[0]
             for f in os.listdir(self.path) if f.endswith('.json')]
        ))
        self.num_of_scenarios = dict(
            zip(self.subjects, [sum([f.split('_')[0] == s for f in os.listdir(self.path)]) for s in self.subjects]))
        self.available_scenarios = \
            [f for f in os.listdir(self.path) if f.endswith('.json')]
        self.total_num = len(self.available_scenarios)
        print(f"Total number of scenarios{self.path}: ", self.total_num)
        if self.total_num == 0:
            raise ("No scenarios found!")
        self.trace = []
        self.scenario = None
        self.hc_state = dict()
        self.hc = hc
        self.emb = emb
        self.embedding_dim = embedding_dim
        self.scenario_name = scenario_name
        self.number_of_episodes = 0
        self.random_scenarios = random_scenarios
        self.num_of_trials = 100
        self.question_answers = {}
        self.gpt = False
        self.single_scenario = False

    def get_num_of_scenarios(self):
        return self.total_num

    def increase_episodes(self):
        self.number_of_episodes += 1

    def get_scenario_files(self):
        return self.available_scenarios

    def load_scenario(self, file_name, num_of_trials=100, gpt=False):
        self.single_scenario = True
        self.gpt = gpt
        self.scenario = json.load(open(os.path.join(self.path, file_name)))
        self.num_of_trials = num_of_trials
        # choose wording for each question
        for k1 in self.scenario["question_answers"]:
            self.question_answers[k1] = dict()
            for k2 in self.scenario["question_answers"][k1]:
                b = len(self.scenario["question_answers"][k1][k2])
                if b <= num_of_trials:
                    self.scenario["question_answers"][k1][k2] = np.repeat(
                        self.scenario["question_answers"][k1][k2], num_of_trials // b).tolist()
                    if len(self.scenario["question_answers"][k1][k2]) != num_of_trials:
                        self.scenario["question_answers"][k1][k2] += np.random.choice(
                            self.scenario["question_answers"][k1][k2],
                            num_of_trials - len(self.scenario["question_answers"][k1][k2])).tolist()
                else:
                    self.scenario["question_answers"][k1][k2] = np.random.choice(
                        ["question_answers"][k1][k2], size=num_of_trials).tolist()
                assert len(self.scenario["question_answers"]
                           [k1][k2]) == num_of_trials
                self.question_answers[k1][k2] = random.sample(
                    self.scenario["question_answers"][k1][k2], len(self.scenario["question_answers"][k1][k2]))
        # hc features
        self.hc_state["posttest_indicator"] = np.zeros(1)
        # self.hc_state["posttest"] = np.zeros(len(self.scenario["posttest_qs"]))
        self.hc_state["binary_qs"] = np.zeros(len(self.scenario["relevant_actions"]))
        # hc statistics features
        l = []
        for c in self.scenario["characters"]:
            if c == "others":
                l.append(np.zeros(self.max_steps_per_episode))
            elif c in self.scenario["question_answers"].keys():
                l.append(
                    np.zeros(len(self.scenario["question_answers"][c].keys())))
            else:
                raise ("Character is not valid")
        self.hc_state["statistics"] = dict(zip(self.scenario["characters"], l))
        self.hc_state["statistics"]["interactions"] = np.zeros(
            self.max_steps_per_episode)
        self.hc_state["statistics"]["unq_interactions"] = np.zeros(
            self.max_steps_per_episode)
        actions = self.get_gpt_actions(
            "interaction") if self.gpt else self.get_actions("interaction")
        return self.get_initial_state(), actions, self.hc_state

    def scenario_step(self, previous_update, action, trial_num, ):
        assert trial_num < self.num_of_trials
        for k1 in self.scenario["question_answers"]:
            for k2 in self.scenario["question_answers"][k1]:
                self.scenario["question_answers"][k1][k2] = self.question_answers[k1][k2][trial_num]
        if self.gpt:
            if "i want" in action.lower():
                if "diagnosis" in action.lower() or "solution" in action.lower():
                    action = {
                        "type": "interaction",
                        "part": "solution",
                        "detail": "",
                        "sentence": "i want to suggest a solution."
                    }
                elif "know" in action.lower() or "ask" in action.lower():
                    subject = None
                    topic = None
                    for s in self.scenario["subjects"]:
                        if s.lower() in action.lower():
                            subject = s

                    for t in self.scenario["topics"]:
                        if t.lower() in action.lower():
                            topic = t

                    if subject is not None and topic is not None:
                        action = {
                            "type": "interaction",
                            "part": "discuss",
                            "detail": f"{subject},{topic}",
                            "sentence": f"i want to know about the {subject} 's {topic}."
                        }
                    else:
                        raise ValueError("Not Implemented")
                else:
                    raise ValueError("Not Implemented")
            elif "i think" in action.lower():
                for c in self.scenario["causes"]:
                    if c.lower() in action.lower():
                        action = {
                            "type": "posttest",
                            "part": "",
                            "detail": "",
                            "sentence": c
                        }
                        print(action)
                        break
                if isinstance(action, str):
                    raise ValueError("Not Implemented")
            else:
                raise ValueError("Not Implemented")
        state_update, reward, terminal, actions, hc, traj_score = self.step(
            previous_update, action)
        if self.gpt:
            actions = self.get_gpt_actions(state_update[0])
        return state_update, reward, terminal, actions, hc, traj_score

    def reset(self):
        self.trace = []
        if not self.single_scenario:
            # choose a scenario
            if self.random_scenarios:
                s = np.random.choice(self.available_scenarios)
            else:
                i = self.number_of_episodes % self.total_num
                s = self.available_scenarios[i]
            self.scenario = json.load(open(os.path.join(self.path, s)))
            # choose wording for each question
            for k1 in self.scenario["question_answers"]:
                for k2 in self.scenario["question_answers"][k1]:
                    self.scenario["question_answers"][k1][k2] = np.random.choice(
                        self.scenario["question_answers"][k1][k2]) if self.training else \
                        self.scenario["question_answers"][k1][k2][0]
        # hc features
        self.hc_state["posttest_indicator"] = np.zeros(1)
        # self.hc_state["posttest"] = np.zeros(len(self.scenario["posttest_qs"]))
        self.hc_state["binary_qs"] = np.zeros(len(self.scenario["relevant_actions"]))
        # hc statistics features
        l = []
        for c in self.scenario["characters"]:
            if c == "others":
                l.append(np.zeros(self.max_steps_per_episode))
            elif c in self.scenario["question_answers"].keys():
                l.append(
                    np.zeros(len(self.scenario["question_answers"][c].keys())))
            else:
                raise ("Character is not valid")
        self.hc_state["statistics"] = dict(zip(self.scenario["characters"], l))
        self.hc_state["statistics"]["interactions"] = np.zeros(
            self.max_steps_per_episode)
        self.hc_state["statistics"]["unq_interactions"] = np.zeros(
            self.max_steps_per_episode)
        actions = self.get_gpt_actions(
            "interaction") if self.gpt else self.get_actions("interaction")
        return self.get_initial_state(), actions, self.hc_state

    def get_initial_state(self):
        return self.scenario["initial_state"]

    def get_state_len(self):
        self.reset()
        l = 0
        if isinstance(self.emb, str):
            if self.emb in ["sum", "avg", "max", "lstm"]:
                l += self.embedding_dim
            else:
                raise ("Not Implemented")
        else:
            if isinstance(self.hc_state, str):
                if self.hc_state == "bq":
                    l += len(self.hc_state["binary_qs"])
                # elif self.hc_state == "kw":
                #     l += len(self.hc_state["keywords"])
                # elif self.hc_state == "both":
                #     # l += len(self.hc_state["keywords"])
                #     l += len(self.hc_state["binary_qs"])
                else:
                    raise ("Not Implemented")
            else:
                "At least one of the features should be added to the state!"
        return l

    def get_actions(self, kind):
        return self.scenario["actions"][kind]

    def get_gpt_actions(self, kind):
        if kind == "interaction":
            subjects = self.scenario["subjects"]
            topics = self.scenario["topics"]
            valid_actions = self.scenario["valid_actions"][kind]
            return valid_actions, subjects, topics
        else:
            causes = self.scenario["causes"]
            valid_actions = self.scenario["valid_actions"][kind]
            return valid_actions, causes

    def get_reward(self, state, action):
        reward = 0
        traj_score = 0
        if state[0] == "posttest":
            traj_score = sum(
                a in self.trace for a in self.scenario["present_actions"]) / len(self.scenario["present_actions"])
            if self.evaluation == "binary":
                if state[1] != "done":
                    sentence = state[1].split(".")[-1]
                    i = self.scenario["posttest_qs"].index(sentence)
                    if action == self.scenario["posttest_as"][i]:
                        reward += self.reward_scale * \
                                  (self.scenario["posttest_as"]
                                   [i]["sentence"] == "yes")
                    else:
                        reward += self.reward_scale * (
                            -1 if (self.scenario["posttest_as"][i]["sentence"] == "no") else 0)
            elif self.evaluation == "cause":
                if state[1] != "done":
                    sentence = state[1].split(".")[-1]
                    i = self.scenario["posttest_qs"].index(sentence)
                    # print("answer is " +
                    #       self.scenario["posttest_as"][i]["sentence"])
                    if action == self.scenario["posttest_as"][i]:
                        reward += self.reward_scale
                    else:
                        reward += self.reward_scale * \
                                  (-1 if self.wrong_answer else 0)
            elif self.evaluation == "rel":
                if state[1] != "done":
                    traj_score = sum(
                        a in self.trace for a in self.scenario["present_actions"]) / len(
                        self.scenario["present_actions"])
                    reward += traj_score
            elif self.evaluation == "relcause1":
                if state[1] != "done":
                    traj_score = sum(
                        a in self.trace for a in self.scenario["present_actions"]) / len(
                        self.scenario["present_actions"])
                    sentence = state[1].split(".")[-1]
                    i = self.scenario["posttest_qs"].index(sentence)
                    if action == self.scenario["posttest_as"][i]:
                        reward += (traj_score + self.reward_scale)
                    else:
                        reward += (self.reward_scale * \
                                   (-1 if self.wrong_answer else 0) + traj_score)
            elif self.evaluation == "relcause2":
                if state[1] != "done":
                    traj_score = sum(
                        a in self.trace for a in self.scenario["present_actions"]) / len(
                        self.scenario["present_actions"])
                    sentence = state[1].split(".")[-1]
                    i = self.scenario["posttest_qs"].index(sentence)
                    if action == self.scenario["posttest_as"][i]:
                        reward += (traj_score * self.reward_scale)
                    else:
                        reward += (self.reward_scale * \
                                   (-1 if self.wrong_answer else 0))
            elif self.evaluation == "relcause3":
                if state[1] != "done":
                    traj_score = sum(
                        a in self.trace for a in self.scenario["present_actions"]) / len(
                        self.scenario["present_actions"])
                    sentence = state[1].split(".")[-1]
                    i = self.scenario["posttest_qs"].index(sentence)
                    if action == self.scenario["posttest_as"][i]:
                        reward += (traj_score * self.reward_scale)
                    else:
                        reward += (self.reward_scale * \
                                   (-1 if self.wrong_answer else 0) + traj_score)
            else:
                if state[1] != "done":
                    sentence = state[1].split(".")[-1]
                    i = self.scenario["posttest_qs"].index(sentence)
                    if action == self.scenario["posttest_as"][i][0]:
                        item = self.scenario["posttest_as"][i][1][0]
                        c = ((2 * (item in self.trace) - 1)
                             if action["sentence"] == "yes" else 1)
                        r = self.reward_scale * c
                        reward += r
                    else:
                        reward += self.reward_scale * \
                                  (-1 if self.wrong_answer else 0)
        else:
            reward += self.penalty
        return reward, traj_score

    def step(self, previous_update, action):
        traj_score = 0
        state = previous_update
        self.trace.append(action)
        if (len(self.trace) < self.max_steps_per_episode) or (action["type"] != "interaction"):
            reward, traj_score = self.get_reward(state, action)
            terminal = 0
            if action["type"] == state[0]:
                part = action["part"]
                detail = action["detail"]
                if action["type"] == "interaction":
                    if action in self.scenario["relevant_actions"]:
                        ind = self.scenario["relevant_actions"].index(action)
                        self.hc_state["binary_qs"][ind] = 1
                    self.hc_state["statistics"]["interactions"] = add_onehot(
                        self.hc_state["statistics"]["interactions"])  # add one to the interactions count
                    if action not in self.trace[:-1]:
                        self.hc_state["statistics"]["unq_interactions"] = add_onehot(
                            self.hc_state["statistics"]["unq_interactions"])
                    state_update = ".".join([part, action["sentence"]])
                    if part == "solution":
                        state_update += self.scenario["posttest_qs"][0]
                        state_update = ("posttest", state_update)
                        # change the indicator to posttest mode
                        self.hc_state["posttest_indicator"][0] = 1
                    elif part == "discuss":

                        subject = detail.split(",")[0]
                        topic = detail.split(",")[1]
                        # print(subject, topic)
                        if "others" in self.scenario["question_answers"].keys():
                            subject = (
                                subject
                                if subject in self.scenario["question_answers"].keys()
                                else "others"
                            )
                            topic = (
                                topic
                                if topic in self.scenario["question_answers"][subject].keys()
                                else "all"
                            )
                        else:
                            if subject not in self.scenario["question_answers"].keys():
                                print(self.scenario["name"])
                                print(subject, topic)
                                self.scenario["question_answers"][subject] = dict(
                                )
                                self.scenario["question_answers"][subject][
                                    topic] = "I don't understand how this is relevant."
                        # print(subject, topic)
                        if isinstance(self.scenario["question_answers"][subject][topic], str):
                            state_update += self.scenario["question_answers"][subject][topic]
                        else:
                            print(self.scenario["name"])
                            print(subject, topic)
                            raise ("Not Implemented")
                        subject = subject if subject in self.hc_state[
                            "statistics"].keys() else "others"  # to filter characters that are unrelevant
                        if action not in self.trace[:-1]:
                            self.hc_state["statistics"][subject] = add_onehot(
                                self.hc_state["statistics"][subject])
                    elif part == "document":
                        state_update += self.scenario["document"][detail]
                    else:
                        raise ("Not Implemented")
                    if isinstance(state_update, str):
                        state_update = ("interaction", state_update)
                else:
                    # if state[1] != "done":
                    sentence = state[1].split(".")[-1]
                    i = self.scenario["posttest_qs"].index(sentence)
                    len_qs = len(self.scenario["posttest_qs"])
                    # print(i, len_qs)
                    if i < len_qs - 1:
                        state_update = (
                            "posttest", self.scenario["posttest_qs"][i + 1])
                        # self.hc_state["posttest"] = add_onehot(self.hc_state["posttest"])
                    else:
                        state_update = (
                            "posttest",
                            "done",
                        )
                        terminal = 1
                        ############################################################TODO
                    # else:
                    #     state_update, actions, self.hc_state = self.reset()
                    #
                    #     terminal = 1
            else:
                state_update = (
                    state[0], "you can not do this  try another action ")
                terminal = 0
            actions = self.get_actions(state_update[0])
        else:
            state_update = "solution.i want to suggest a solution.." + \
                           self.scenario["posttest_qs"][0]
            # change the indicator to posttest mode
            self.hc_state["posttest_indicator"][0] = 1
            # self.hc_state["posttest"] = add_onehot(self.hc_state["posttest"])
            state_update = ("posttest", state_update)
            actions = self.get_actions(state_update[0])
            reward = 0
            terminal = 0
        if ((len(self.trace) + 1) == self.max_steps_per_episode) and (action["type"] == "interaction"):
            actions = [{
                "type": "interaction",
                "part": "solution",
                "detail": "",
                "sentence": "i want to suggest a solution."
            }]
        return state_update, reward, terminal, actions, self.hc_state, traj_score


class LLMGame:
    def __init__(self, path=r".\scenarios", env_step_limit=20, reward_scale=1, emb="sum", hc="bq", embedding_dim=300,
                 wording=True, evaluation="cause", scenario_name=False,
                 random_scenarios=False, reduced=True) -> None:
        self.reduced = "reduced" if reduced else "not_reduced"
        self.evaluation = evaluation
        self.wording = wording
        self.path = path
        # maximum number of allowed interactions per episode
        self.max_steps_per_episode = env_step_limit
        self.reward_scale = reward_scale
        self.subjects = list(set(
            [f.split('_')[0]
             for f in os.listdir(self.path) if f.endswith('.json')]
        ))
        self.num_of_scenarios = dict(
            zip(self.subjects, [sum([f.split('_')[0] == s for f in os.listdir(self.path)]) for s in self.subjects]))
        self.available_scenarios = \
            [f for f in os.listdir(self.path) if f.endswith('.json')]
        self.total_num = len(self.available_scenarios)
        print(f"Total number of scenarios{self.path}: ", self.total_num)
        if self.total_num == 0:
            raise ("No scenarios found!")
        self.trace = []
        self.achieved_goals = []
        self.scenario = None
        self.hc_state = dict()
        self.hc = hc
        self.emb = emb
        self.embedding_dim = embedding_dim
        self.scenario_name = scenario_name
        self.number_of_episodes = 0
        self.random_scenarios = random_scenarios
        self.encoder = Encoder(memory_path="./encoder_memory/goals_memory.json", memory_size=100, save_memory=0.1)

    def get_num_of_scenarios(self):
        return self.total_num

    def increase_episodes(self):
        self.number_of_episodes += 1

    def get_scenario_files(self):
        return self.available_scenarios

    def reset(self):
        self.trace = []
        self.achieved_goals = []
        # choose a scenario
        if self.random_scenarios:
            s = np.random.choice(self.available_scenarios)
        else:
            i = self.number_of_episodes % self.total_num
            s = self.available_scenarios[i]
        self.scenario = json.load(open(os.path.join(self.path, s)))
        # choose wording for each question
        for k1 in self.scenario["question_answers"]:
            for k2 in self.scenario["question_answers"][k1]:
                self.scenario["question_answers"][k1][k2] = np.random.choice(
                    self.scenario["question_answers"][k1][k2]) if self.wording else \
                    self.scenario["question_answers"][k1][k2][0]
        # hc_state features
        self.hc_state["posttest_indicator"] = np.zeros(1)
        # self.hc_state["posttest"] = np.zeros(len(self.scenario["posttest_qs"]))
        self.hc_state["binary_qs"] = np.zeros(len(self.scenario["relevant_actions"]))
        # hc_state statistics features
        l = []
        for c in self.scenario["characters"]:
            if c == "others":
                l.append(np.zeros(self.max_steps_per_episode))
            elif c in self.scenario["question_answers"].keys():
                l.append(
                    np.zeros(len(self.scenario["question_answers"][c].keys())))
            else:
                raise ("Character is not valid")
        self.hc_state["statistics"] = dict(zip(self.scenario["characters"], l))
        self.hc_state["statistics"]["interactions"] = np.zeros(
            self.max_steps_per_episode)
        self.hc_state["statistics"]["unq_interactions"] = np.zeros(
            self.max_steps_per_episode)
        return self.get_initial_state(), self.get_actions("interaction"), self.hc_state, self.get_gpt_actions(
            "interaction")

    def get_initial_state(self):
        return self.scenario["initial_state"]

    def get_state_len(self):
        self.reset()
        l = 0
        if isinstance(self.emb, str):
            if self.emb in ["sum", "avg", "max", "lstm"]:
                l += self.embedding_dim
            else:
                raise ("Not Implemented")
        else:
            if isinstance(self.hc, str):
                if self.hc == "bq":
                    l += len(self.hc_state["binary_qs"])
                else:
                    raise ("Not Implemented")
            else:
                "At least one of the features should be added to the state!"
        return l

    def get_actions(self, kind):
        return self.scenario["actions"][kind]

    def get_gpt_actions(self, kind):
        if kind == "interaction":
            subjects = self.scenario["subjects"]
            topics = self.scenario["topics"]
            valid_actions = self.scenario["valid_actions"][kind]
            return valid_actions, subjects, topics
        else:
            causes = self.scenario["causes"]
            valid_actions = self.scenario["valid_actions"][kind]
            return valid_actions, causes

    def get_intrinsic_reward(self, state, goals, action):

        # obs = ".".join([state[0], state[1], action["sentence"]])
        obs = action["detail"] if action["type"] == "interaction" else action["sentence"]
        if len(obs) < 2:
            obs = action["sentence"]
        if action["type"] == "posttest":
            obs = ",".join(["diagnosis", obs])
        for (i, g) in enumerate(goals):
            if "I want to know about the ".lower() in g.lower():
                goals[i] = g.lower().replace("I want to know about the ".lower(), "").replace(".", "").replace(" 's ",
                                                                                                               ",")
        print(goals)
        encoded_obs = self.encoder.encode([obs])[0]
        encoded_goals = np.concatenate(self.encoder.encode(goals), axis=0)

        # print(encoded_obs.shape, encoded_goals.shape)
        scores = sentence_transformers.util.cos_sim(encoded_obs, encoded_goals).numpy()
        max_score = scores.max()
        if goals[scores.argmax()] in self.achieved_goals:
            max_score = 0
        if max_score > 0.6:
            print("score:", max_score)
            print("agent's actions:", obs)
            print("aligned goal:", goals[scores.argmax()])
            self.achieved_goals.append(goals[scores.argmax()])
        else:
            max_score = 0

        return max_score

    def step(self, previous_update, action, goals=None):
        # print(self.scenario["name"])
        # print(action)
        state = previous_update
        self.trace.append(action)
        if (len(self.trace) < self.max_steps_per_episode) or (action["type"] != "interaction"):
            reward = self.get_intrinsic_reward(state, goals, action)
            terminal = 0
            if action["type"] == state[0]:
                part = action["part"]
                detail = action["detail"]
                if action["type"] == "interaction":
                    if action in self.scenario["relevant_actions"]:
                        ind = self.scenario["relevant_actions"].index(action)
                        self.hc_state["binary_qs"][ind] = 1
                    self.hc_state["statistics"]["interactions"] = add_onehot(
                        self.hc_state["statistics"]["interactions"])  # add one to the interactions count
                    if action not in self.trace[:-1]:
                        self.hc_state["statistics"]["unq_interactions"] = add_onehot(
                            self.hc_state["statistics"]["unq_interactions"])
                    state_update = ".".join([part, action["sentence"]])
                    if part == "solution":
                        state_update += self.scenario["posttest_qs"][0]
                        state_update = ("posttest", state_update)
                        # change the indicator to posttest mode
                        self.hc_state["posttest_indicator"][0] = 1
                    elif part == "discuss":
                        subject = detail.split(",")[0]
                        topic = detail.split(",")[1]
                        if "others" in self.scenario["question_answers"].keys():
                            subject = (
                                subject
                                if subject in self.scenario["question_answers"].keys()
                                else "others"
                            )
                            topic = (
                                topic
                                if topic in self.scenario["question_answers"][subject].keys()
                                else "all"
                            )
                        else:
                            if subject not in self.scenario["question_answers"].keys():
                                print(self.scenario["name"])
                                print(subject, topic)
                                self.scenario["question_answers"][subject] = dict(
                                )
                                self.scenario["question_answers"][subject][
                                    topic] = "I don't understand how this is relevant."
                        if isinstance(self.scenario["question_answers"][subject][topic], str):
                            state_update += self.scenario["question_answers"][subject][topic]
                        else:
                            print(self.scenario["name"])
                            print(subject, topic)
                            raise ("Not Implemented")
                        subject = subject if subject in self.hc_state[
                            "statistics"].keys() else "others"  # to filter characters that are unrelevant
                        if action not in self.trace[:-1]:
                            self.hc_state["statistics"][subject] = add_onehot(
                                self.hc_state["statistics"][subject])
                    elif part == "document":
                        state_update += self.scenario["document"][detail]
                    else:
                        raise ("Not Implemented")
                    if isinstance(state_update, str):
                        state_update = ("interaction", state_update)
                else:
                    if state[1] != "done":
                        sentence = state[1].split(".")[-1]
                        i = self.scenario["posttest_qs"].index(sentence)
                        len_qs = len(self.scenario["posttest_qs"])
                        if i < len_qs - 1:
                            state_update = (
                                "posttest", self.scenario["posttest_qs"][i + 1])
                        else:
                            state_update = (
                                "posttest",
                                "done",
                            )
                    else:
                        state_update, actions, self.hc_state, gpt_actions = self.reset()

                        terminal = 1
            else:
                state_update = (
                    state[0], "you can not do this  try another action ")
                terminal = 0
            actions = self.get_actions(state_update[0])
        else:
            state_update = "solution.i want to suggest a solution.." + \
                           self.scenario["posttest_qs"][0]
            # change the indicator to posttest mode
            self.hc_state["posttest_indicator"][0] = 1
            # self.hc_state["posttest"] = add_onehot(self.hc_state["posttest"])
            state_update = ("posttest", state_update)
            actions = self.get_actions(state_update[0])
            reward = 0
            terminal = 0
        gpt_actions = self.get_gpt_actions(state_update[0])
        return state_update, reward, terminal, actions, self.hc_state, gpt_actions


def main():
    path = "./scenarios/test"
    env = Game(path=path)
    env.reset()
    print(env.get_state_len())
    for i in range(10000000):
        state_update, reward, terminal, actions, hc, traj_score = env.step(
            env.get_initial_state(), env.get_actions("interaction")[0])
        print(i)
        while not terminal:
            state_update, reward, terminal, actions, hc, traj_score = env.step(
                state_update, np.random.choice(actions))


if __name__ == "__main__":
    main()
