import os
import json

def scenario_extractor(path="./scenarios", e="cause"):
    subjects = list(set(
        [f.split('.')[0] for f in os.listdir(path) if f.endswith('.json')]
    ))
    num_of_subtasks = []
    for s in subjects:
        scenario_path = os.path.join(path, s)
        scenario = json.load(open(os.path.join(path, s + ".json"),encoding="utf-8"))
        num_of_scenarios = len(scenario["scenarios"])
        num_of_subtasks.append(num_of_scenarios)
        for i in range(num_of_scenarios):
            scenario = json.load(open(os.path.join(path, s + ".json"),encoding="utf-8"))
            scenario.update(scenario["scenarios"][i])
            del scenario["scenarios"]
            if len(scenario["present_actions"]) == 0:
                scenario["present_actions"] = scenario["relevant_actions"]
            scenario["actions"] = dict()
            scenario["actions"]["posttest"] = scenario["posttest_actions"][e]
            if e in scenario["posttest_qs"]:
                scenario["posttest_qs"] = scenario["posttest_qs"][e]
            else:
                scenario["posttest_qs"] = scenario[e]
                del scenario[e]
            scenario["posttest_as"] = scenario["posttest_as"][e]
            del scenario["posttest_actions"]
            scenario["actions"]["interaction"] = scenario["interaction"]
            if not os.path.exists(scenario_path):
                os.makedirs(scenario_path)
            with open(os.path.join(scenario_path, s + "_" + str(i) + ".json"), 'w') as f:
                json.dump(scenario, f)
    return subjects, num_of_subtasks
