import re

import openai
from openai import OpenAI
import os
if not os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = "sk-xxxxxx"

# this functions takes valid subjects, topics, and causes and formats them into a string


def format_actions(actions):
    actions = list(actions)
    assert len(actions) == 2 or len(actions) == 3
    for i in range(len(actions)):
        actions[i] = ['"' + a + '"' for a in actions[i]]
    actions = ["\n".join(a) for a in actions]
    if len(actions) == 3:
        actions = f"""{actions[0]}
        Valid Subjects: {actions[1]}
        Valid Topics: {actions[2]}"""
    elif len(actions) == 2:
        actions = f"""{actions[0]}
        Valid Causes: {actions[1]}"""
    else:
        raise NotImplementedError
    return actions


class PromptFormat:
    def format_prompt(self, input):
        raise NotImplementedError

    def parse_response(self, response):
        raise NotImplementedError

######################################################################without summarizer
class GPTChooses_or_Recs(PromptFormat):
    def __init__(self, topk=5):
        self.topk = topk

    def format_prompt(self, history, subject, problem, valid_subjects, valid_topics, valid_causes, choices, posttest=False,prev_exp=None,summarizer=None):
        formated_history = ""
        for (i, s) in enumerate(history):
            prefix = "Student: " if i % 2 == 1 else "Customer: "
            formated_history += (prefix + s + "\n")
        formated_prev_exp = ""
        if prev_exp is not None:
            for (j,exp) in enumerate(prev_exp):
                formated_prev_exp += f"Experience {j+1}:\n"
                for (i, s) in enumerate(exp):
                    prefix = "Student: " if i % 2 == 1 else "Customer: "
                    formated_prev_exp += (prefix + s + "\n")
        else:
            formated_prev_exp = "No previous experience\n"
        formated_choices = ""

        for (i, c) in enumerate(choices[:min(self.topk,len(choices))]):
            formated_choices += (f"{i + 1}. " + c + "\n")
        formated_valid_topics = "[\n"
        formated_valid_subjects = "[\n"
        formated_valid_causes = "[\n"
        for s in valid_subjects:
            formated_valid_subjects += s + ",\n"
        for t in valid_topics:
            formated_valid_topics += t + ",\n"
        for t in valid_causes:
            formated_valid_causes += t + ",\n"
        formated_valid_topics += "\n]"
        formated_valid_subjects += "\n]"
        formated_valid_causes += "\n]"
        output = "1. choose(x)\n2. choose(y)" if posttest else "1. ask(x,y)\n2. answer()"
        available_actions = "1.choose(cause)[The most probable reason is cause]" if posttest else " 1. ask(subject,topic)[I want to ask about subject’s topic.] 2. answer()[I want to suggest the cause behind the customer’s problem.]\n"
        available_options = f"valid causes: {formated_valid_causes}" if posttest else f"valid subjects: {formated_valid_subjects}\nvalid topics: {formated_valid_topics}\n"
        question = "What should the student answer?" if posttest else "What should the student choose to do?"
        return [
            {
                "role": "system",
                "content": "You are a pharmacist helping a pharmacy student go through an educational game. You will receive a description of the customer and you will help the student ask a series of questions needed to help the customer. Make sure you are keeping the discussion short with the customer and offer a diagnosis once you are sure."
            },
            {
                "role": "user",
                "content": f"Task: Find the cause behind the {subject}'s {problem}\nYour previous experience with similar tasks:\n{formated_prev_exp}Interaction History:\n{formated_history}Student top choices:\n{formated_choices} available actions:{available_actions}\n{available_options} {question} if there is a suitable action between the student choice answer with the choice number. you can first reason about your choice in less than 50 words. Don't forget to put ### after your reasoning finishes.Then write your chosen action.Remember that asking the same question again will not give you a different answer. If you do not find any suitable actions between the student's top choices recommend {5 if not posttest else 2} actions from the list to the student. Write whether you are doing choose or recommend followes by $$$.\nexample output1:\nchoose\n$$$\nreason: we need to explore the baby's symptoms more.\n###\n1\nexample output2:\nrecommend\n$$$\nreason: we need to explore the baby's symptoms more but the student have not included it in their choice.\n###\n{output}"
            }
        ]

    def parse_response(self, response):
        return response.choices[0].message.content
class GPTChooses(PromptFormat):
    def __init__(self, topk=5):
        self.topk = topk

    def format_prompt(self, history, subject, problem, choices, posttest=False,prev_exp=None,summarizer=None):
        formated_history = ""
        for (i, s) in enumerate(history):
            prefix = "Student: " if i % 2 == 1 else "Customer: "
            formated_history += (prefix + s + "\n")
        formated_prev_exp = ""
        if prev_exp is not None:
            for (j,exp) in enumerate(prev_exp):
                formated_prev_exp += f"Experience {j+1}:\n"
                for (i, s) in enumerate(exp):
                    prefix = "Student: " if i % 2 == 1 else "Customer: "
                    formated_prev_exp += (prefix + s + "\n")
        else:
            formated_prev_exp = "No previous experience\n"
        formated_choices = ""

        for (i, c) in enumerate(choices[:min(self.topk,len(choices))]):
            formated_choices += (f"{i + 1}. " + c + "\n")
        question = "What should the student answer?" if posttest else "What should the student choose to do?"
        return [
            {
                "role": "system",
                "content": "You are a pharmacist helping a pharmacy student go through an educational game. You will receive a description of the customer and you will help the student ask a series of questions needed to help the customer. Make sure you are keeping the discussion short with the customer and offer a diagnosis once you are sure."
            },
            {
                "role": "user",
                "content": f"Task: Find the cause behind the {subject}'s {problem}\nYour previous experience with similar tasks:\n{formated_prev_exp}Interaction History:\n{formated_history}Student top choices:\n{formated_choices} {question} answer with the choice number you can first reason about your choice in less than 50 words. Don't forget to put ### after your reasoning finishes.Then write your chosen action.Remember that asking the same question again will not give you a different answer.\nexample output:\nreason: we need to explore the baby's symptoms more.\n###\n1"
            }
        ]

    def parse_response(self, response):
        return response.choices[0].message.content
class GPTRecs(PromptFormat):
    def __init__(self, num_of_recs=5):
        self.num_of_recs = num_of_recs

    def format_prompt(self, history, subject, problem, valid_subjects, valid_topics, valid_causes, posttest=False,prev_exp=None,summarizer=None):
        formated_history = ""
        for (i, s) in enumerate(history):
            prefix = "Student: " if i % 2 == 1 else "Customer: "
            formated_history += (prefix + s + "\n")
        formated_prev_exp = ""
        if prev_exp is not None:
            for (j,exp) in enumerate(prev_exp):
                formated_prev_exp += f"Experience {j+1}:\n"
                for (i, s) in enumerate(exp):
                    prefix = "Student: " if i % 2 == 1 else "Customer: "
                    formated_prev_exp += (prefix + s + "\n")
        else:
            formated_prev_exp = "No previous experience\n"
        formated_valid_topics = "[\n"
        formated_valid_subjects ="[\n"
        formated_valid_causes ="[\n"
        for s in valid_subjects:
            formated_valid_subjects += s+",\n"
        for t in valid_topics:
            formated_valid_topics += t + ",\n"
        for t in valid_causes:
            formated_valid_causes += t + ",\n"
        formated_valid_topics += "\n]"
        formated_valid_subjects += "\n]"
        formated_valid_causes += "\n]"
        available_actions ="1.choose(cause)[The most probable reason is cause]" if posttest else " 1. ask(subject,topic)[I want to ask about subject’s topic.] 2. answer()[I want to suggest the cause behind the customer’s problem.]\n"
        available_options = f"valid causes: {formated_valid_causes}" if posttest else f"valid subjects: {formated_valid_subjects}\nvalid topics: {formated_valid_topics}\n"
        output = "1. choose(x)\n2. choose(y)" if posttest else "1. ask(x,y)\n2. answer()"

        return \
            [
                {
                    "role": "system",
                    "content": "You are a pharmacist helping a pharmacy student go through an educational game. You will receive a description of the customer and you will help the student ask a series of questions needed to help the customer. Make sure you are keeping the discussion short with the customer and offer a diagnosis once you are sure. At each step, you will receive the list of available actions for the student and you will recommend 5 actions for the student to do."
                },
                {
                    "role": "user",
                    "content": f"Task: Find the cause behind the {subject}'s {problem}\nYour previous experience with similar tasks:\n{formated_prev_exp}Interaction History:\n{formated_history}available actions:{available_actions}\n{available_options}What are your top {2 if posttest else self.num_of_recs} recommended actions from the action list? you can first reason about your recommendation in less than 50 words. Don't forget to put ### after your reasoning finishes. Then write your suggested actions. \noutput format:\nreason: r\n\n###\n{output}"
                }
            ]


    def parse_response(self, response):
        return response.choices[0].message.content
class GPTPlays(PromptFormat):
    def __init__(self):
        super().__init__()
    def format_prompt(self, history, subject, problem, valid_subjects, valid_topics, valid_causes, posttest=False,prev_exp=None,summarizer=None):
        formated_history = ""
        for (i, s) in enumerate(history):
            prefix = "Student: " if i % 2 == 1 else "Customer: "
            formated_history += (prefix + s + "\n")
        formated_prev_exp = ""
        if prev_exp is not None:
            for (j,exp) in enumerate(prev_exp):
                formated_prev_exp += f"Experience {j+1}:\n"
                for (i, s) in enumerate(exp):
                    prefix = "Student: " if i % 2 == 1 else "Customer: "
                    formated_prev_exp += (prefix + s + "\n")
        else:
            formated_prev_exp = "No previous experience\n"
        formated_valid_topics = "[\n"
        formated_valid_subjects = "[\n"
        formated_valid_causes = "[\n"
        for s in valid_subjects:
            formated_valid_subjects += s + ",\n"
        for t in valid_topics:
            formated_valid_topics += t + ",\n"
        for t in valid_causes:
            formated_valid_causes += t + ",\n"
        formated_valid_topics += "\n]"
        formated_valid_subjects += "\n]"
        formated_valid_causes += "\n]"
        available_actions = "1.choose(cause)[The most probable reason is cause]" if posttest else " 1. ask(subject,topic)[I want to ask about subject’s topic.] 2. answer()[I want to suggest the cause behind the customer’s problem.]\n"
        available_options = f"valid causes: {formated_valid_causes}" if posttest else f"valid subjects: {formated_valid_subjects}\nvalid topics: {formated_valid_topics}\n"
        output = "choose(x)" if posttest else "ask(x,y)"

        return \
            [
                {
                    "role": "system",
                    "content": "You are a pharmacist helping a pharmacy student go through an educational game. You will receive a description of the customer and you will help the student ask a series of questions needed to help the customer. Make sure you are keeping the discussion short with the customer and offer a diagnosis once you are sure. At each step, you will receive the list of available actions for the student and you will choose the best action for the student to do."
                },
                {
                    "role": "user",
                    "content": f"Task: Find the cause behind the {subject}'s {problem}\nYour previous experience with similar tasks:\n{formated_prev_exp}Interaction History:\n{formated_history}available actions:{available_actions}\n{available_options}What action would you take next? you can first reason about your choice in less than 50 words. Don't forget to put ### after your reasoning finishes. Then write your chosen action. Remember that asking the same question again will not give you a different answer. \noutput format:\nreason: r\n\n###\n{output}"
                }
            ]
    def parse_response(self, response):
        return response.choices[0].message.content

class CLINChooses(PromptFormat):
    def __init__(self, topk=5):
        self.topk = topk

    def format_prompt(self, history, subject, problem, choices,summary, posttest=False,prev_exp=None,summarizer=None):
        formated_history = ""
        for (i, s) in enumerate(history):
            prefix = "Student: " if i % 2 == 1 else "Customer: "
            formated_history += (prefix + s + "\n")
        formated_prev_exp = ""
        if prev_exp is not None:
            for (j,exp) in enumerate(prev_exp):
                formated_prev_exp += f"Experience {j+1}:\n"
                for (i, s) in enumerate(exp):
                    prefix = "Student: " if i % 2 == 1 else "Customer: "
                    formated_prev_exp += (prefix + s + "\n")
        formated_choices = ""

        for (i, c) in enumerate(choices[:min(self.topk,len(choices))]):
            formated_choices += (f"{i + 1}. " + c + "\n")
        question = "What should the student answer?" if posttest else "What should the student choose to do?"
        summary_prompt = f"Here is a summary of learnings based on your previous attempts on this task." \
                         f"These learnings capture important pre-conditions: X MAY BE NECESSARY to Y, X SHOULD BE NECESSARY to Y, and mistakes: X MAY NOT CONTRIBUTE to Y, X DOES NOT CONTRIBUTE to Y. These can be useful for completing your task:\n"
        user_em = """First, scan the (unordered) list of learnings, if provided. Decide if any of the learnings are applicable given the last observation to make progress in this task. Then only use selected learnings, if any, to construct a rationale for picking the next action. If no Learning is selected, construct the rationale based on the observation history. Format your response as follows:
                        Write the selected learning_ids as a comma separated list; the list can be empty if no learnings selected. Then, write $$$ followed by the rationale of at most 50 words. Finally, write ### followed by the single next action you would like to take.
                        """
        return [
            {
                "role": "system",
                "content": "You are a pharmacist helping a pharmacy student go through an educational game. You will receive a description of the customer and you will help the student ask a series of questions needed to help the customer. Make sure you are keeping the discussion short with the customer and offer a diagnosis once you are sure."
            },
            {
                "role": "user",
                "content": f"Task: Find the cause behind the {subject}'s {problem}\nYour previous experience with similar tasks:\n{formated_prev_exp}{summary_prompt}\n{summary}\nObservation History:\n{formated_history}Student top choices:\n{formated_choices} {question} answer with the choice number.{user_em}\nexample output:\n1, 3\n$$$\nreason: r\n\n###\n1"
            }
        ]

    def parse_response(self, response):
        return response.choices[0].message.content
class CLINRecs(PromptFormat):
    def __init__(self, num_of_recs=5):
        self.num_of_recs = num_of_recs

    def format_prompt(self, history, subject, problem, valid_subjects, valid_topics, valid_causes,summary, posttest=False,prev_exp=None,summarizer=None):
        formated_history = ""
        for (i, s) in enumerate(history):
            prefix = "Student: " if i % 2 == 1 else "Customer: "
            formated_history += (prefix + s + "\n")
        formated_prev_exp = ""
        if prev_exp is not None:
            for (j,exp) in enumerate(prev_exp):
                formated_prev_exp += f"Experience {j+1}:\n"
                for (i, s) in enumerate(exp):
                    prefix = "Student: " if i % 2 == 1 else "Customer: "
                    formated_prev_exp += (prefix + s + "\n")
        formated_valid_topics = "[\n"
        formated_valid_subjects ="[\n"
        formated_valid_causes ="[\n"
        for s in valid_subjects:
            formated_valid_subjects += s+",\n"
        for t in valid_topics:
            formated_valid_topics += t + ",\n"
        for t in valid_causes:
            formated_valid_causes += t + ",\n"
        formated_valid_topics += "\n]"
        formated_valid_subjects += "\n]"
        formated_valid_causes += "\n]"
        available_actions ="1.choose(cause)[The most probable reason is cause]" if posttest else " 1. ask(subject,topic)[I want to ask about subject’s topic.] 2. answer()[I want to suggest the cause behind the customer’s problem.]\n"
        available_options = f"valid causes: {formated_valid_causes}" if posttest else f"valid subjects: {formated_valid_subjects}\nvalid topics: {formated_valid_topics}\n"
        summary_prompt = f"Here is a summary of learnings based on your previous attempts on this task." \
                         f"These learnings capture important pre-conditions: X MAY BE NECESSARY to Y, X SHOULD BE NECESSARY to Y, and mistakes: X MAY NOT CONTRIBUTE to Y, X DOES NOT CONTRIBUTE to Y. These can be useful for completing your task:\n"
        user_em = """First, scan the (unordered) list of learnings, if provided. Decide if any of the learnings are applicable given the last observation to make progress in this task. Then only use selected learnings, if any, to construct a rationale for picking the next action. If no Learning is selected, construct the rationale based on the observation history. Format your response as follows:
                        Write the selected learning_ids as a comma separated list; the list can be empty if no learnings selected. Then, write $$$ followed by the rationale of at most 50 words. Finally, write ### followed by your list of recommended actions.
                        """
        output = "1. choose(x)\n2. choose(y)" if posttest else "1. ask(x,y)\n2. answer()"

        return \
            [
                {
                    "role": "system",
                    "content": "You are a pharmacist helping a pharmacy student go through an educational game. You will receive a description of the customer and you will help the student ask a series of questions needed to help the customer. Make sure you are keeping the discussion short with the customer and offer a diagnosis once you are sure. At each step, you will receive the list of available actions for the student and you will recommend 5 actions for the student to do."
                },
                {
                    "role": "user",
                    "content": f"Task: Find the cause behind the {subject}'s {problem}\nYour previous experience with similar tasks:\n{formated_prev_exp}{summary_prompt}\n{summary}\nObservation History:\n{formated_history}available actions:{available_actions}\n{available_options}What are your top {2 if posttest else self.num_of_recs} recommended actions from the action list? {user_em}\noutput format:\n1, 3\n$$$\nreason: r\n\n###\n{output}"
                }
            ]


    def parse_response(self, response):
        return response.choices[0].message.content
class CLINPlays(PromptFormat):
    def __init__(self):
        super().__init__()

    def format_prompt(self, history, subject, problem, valid_subjects, valid_topics, valid_causes, summary, posttest=False,prev_exp=None,summarizer=None):
        formated_history = ""
        for (i, s) in enumerate(history):
            prefix = "Student: " if i % 2 == 1 else "Customer: "
            formated_history += (prefix + s + "\n")
        formated_prev_exp = ""
        if prev_exp is not None:
            for (j,exp) in enumerate(prev_exp):
                formated_prev_exp += f"Experience {j+1}:\n"
                for (i, s) in enumerate(exp):
                    prefix = "Student: " if i % 2 == 1 else "Customer: "
                    formated_prev_exp += (prefix + s + "\n")
        formated_prev_exp = ""
        if prev_exp is not None:
            for (j,exp) in enumerate(prev_exp):
                formated_prev_exp += f"Experience {j+1}:\n"
                for (i, s) in enumerate(exp):
                    prefix = "Student: " if i % 2 == 1 else "Customer: "
                    formated_prev_exp += (prefix + s + "\n")
        formated_valid_topics = "[\n"
        formated_valid_subjects = "[\n"
        formated_valid_causes = "[\n"
        for s in valid_subjects:
            formated_valid_subjects += s + ",\n"
        for t in valid_topics:
            formated_valid_topics += t + ",\n"
        for t in valid_causes:
            formated_valid_causes += t + ",\n"
        formated_valid_topics += "\n]"
        formated_valid_subjects += "\n]"
        formated_valid_causes += "\n]"
        available_actions = "1.choose(cause)[The most probable reason is cause]" if posttest else " 1. ask(subject,topic)[I want to ask about subject’s topic.] 2. answer()[I want to suggest the cause behind the customer’s problem.]\n"
        available_options = f"valid causes: {formated_valid_causes}" if posttest else f"valid subjects: {formated_valid_subjects}\nvalid topics: {formated_valid_topics}\n"
        summary_prompt = f"Here is a summary of learnings based on your previous attempts on this task." \
                  f"These learnings capture important pre-conditions: X MAY BE NECESSARY to Y, X SHOULD BE NECESSARY to Y, and mistakes: X MAY NOT CONTRIBUTE to Y, X DOES NOT CONTRIBUTE to Y. These can be useful for predicting your next action:\n"
        user_em = """First, scan the (unordered) list of learnings, if provided. Decide if any of the learnings are applicable given the last observation to make progress in this task. Then only use selected learnings, if any, to construct a rationale for picking the next action. If no Learning is selected, construct the rationale based on the observation history. Format your response as follows:
                Write the selected learning_ids as a comma separated list; the list can be empty if no learnings selected. Then, write $$$ followed by the rationale of at most 50 words. Finally, write ### followed by the single next action you would like to take.
                """
        output = "choose(x)" if posttest else "ask(x,y)"

        return \
            [
                {
                    "role": "system",
                    "content": "You are a pharmacist helping a pharmacy student go through an educational game. You will receive a description of the customer and you will help the student ask a series of questions needed to help the customer. Make sure you are keeping the discussion short with the customer and offer a diagnosis once you are sure. At each step, you will receive the list of available actions for the student and you will choose the best action for the student to do."
                },
                {
                    "role": "user",
                    "content": f"Task: Find the cause behind the {subject}'s {problem}\nYour previous experience with similar tasks:\n{formated_prev_exp}{summary_prompt}\n{summary}\nObservation History:\n{formated_history}available actions:{available_actions}\n{available_options}What action would you take next? {user_em}\noutput format:\n1, 3\n$$$\nreason: r\n\n###\n{output}"
                }
            ]
    def parse_response(self, response):
        return response.choices[0].message.content
######################################################################with summarizer
class GPTPlays2(PromptFormat):
    def __init__(self):
        super().__init__()
    def format_prompt(self, history, subject, problem, valid_subjects, valid_topics, valid_causes,summarizer, posttest=False,prev_exp=None):
        formated_history = summarizer.summarize(history)
        if prev_exp is not None:
            formated_prev_exp = ""
            for (j,exp) in enumerate(prev_exp):
                formated_prev_exp += f"Experience {j+1}:\n"
                formated_prev_exp += summarizer.summarize(exp)
        formated_prev_actions = ""
        for (i, h) in enumerate(history):
            if i % 2 == 1:
                formated_prev_actions += (h.split("the ")[-1].split('.')[0] + ",")
        formated_valid_topics = "[\n"
        formated_valid_subjects = "[\n"
        formated_valid_causes = "[\n"
        for s in valid_subjects:
            formated_valid_subjects += s + ",\n"
        for t in valid_topics:
            formated_valid_topics += t + ",\n"
        for t in valid_causes:
            formated_valid_causes += t + ",\n"
        formated_valid_topics += "\n]"
        formated_valid_subjects += "\n]"
        formated_valid_causes += "\n]"
        available_actions = "1.choose(cause)[The most probable reason is cause]" if posttest else " 1. ask(subject,topic)[I want to ask about subject’s topic.] 2. answer()[I want to suggest the cause behind the customer’s problem.]\n"
        available_options = f"valid causes: {formated_valid_causes}" if posttest else f"valid subjects: {formated_valid_subjects}\nvalid topics: {formated_valid_topics}\n"
        output = "choose(x)" if posttest else "ask(x,y)"

        return \
            [
                {
                    "role": "system",
                    "content": "You are a pharmacist helping a pharmacy student go through an educational game. You will receive a description of the customer and you will help the student ask a series of questions needed to help the customer. Make sure you are keeping the discussion short with the customer and offer a diagnosis once you are sure. At each step, you will receive the list of available actions for the student and you will choose the best action for the student to do."
                },
                {
                    "role": "user",
                    "content": f"Task: Find the cause behind the {subject}'s {problem}\nPrevious actions(Remember repeated actions does not give you additional information):{formated_prev_actions}\nSummary of Interaction:\n{formated_history}\navailable actions:{available_actions}\n{available_options}What action would you take next? Only write your chosen action.\nexample output:{output}"
                }
            ]
    def parse_response(self, response):
        return response.choices[0].message.content
class GPTRecs2(PromptFormat):
    def __init__(self, num_of_recs=5):
        self.num_of_recs = num_of_recs

    def format_prompt(self, history, subject, problem, valid_subjects, valid_topics, valid_causes, posttest=False,prev_exp=None,summarizer=None):
        formated_history = summarizer.summarize(history)
        if prev_exp is not None:
            formated_prev_exp = ""
            for (j, exp) in enumerate(prev_exp):
                formated_prev_exp += f"Experience {j + 1}:\n"
                formated_prev_exp += summarizer.summarize(exp)
        formated_prev_actions = ""
        for (i, h) in enumerate(history):
            if i % 2 == 1:
                formated_prev_actions += (h.split("the ")[-1].split('.')[0] + ",")
        formated_valid_topics = "[\n"
        formated_valid_subjects ="[\n"
        formated_valid_causes ="[\n"
        for s in valid_subjects:
            formated_valid_subjects += s+",\n"
        for t in valid_topics:
            formated_valid_topics += t + ",\n"
        for t in valid_causes:
            formated_valid_causes += t + ",\n"
        formated_valid_topics += "\n]"
        formated_valid_subjects += "\n]"
        formated_valid_causes += "\n]"
        available_actions ="1.choose(cause)[The most probable reason is cause]" if posttest else " 1. ask(subject,topic)[I want to ask about subject’s topic.] 2. answer()[I want to suggest the cause behind the customer’s problem.]\n"
        available_options = f"valid causes: {formated_valid_causes}" if posttest else f"valid subjects: {formated_valid_subjects}\nvalid topics: {formated_valid_topics}\n"
        output = "1. choose(x)\n2. choose(y)" if posttest else "1. ask(x,y)\n2. answer()"

        return \
            [
                {
                    "role": "system",
                    "content": "You are a pharmacist helping a pharmacy student go through an educational game. You will receive a description of the customer and you will help the student ask a series of questions needed to help the customer. Make sure you are keeping the discussion short with the customer and offer a diagnosis once you are sure. At each step, you will receive the list of available actions for the student and you will recommend 5 actions for the student to do."
                },
                {
                    "role": "user",
                    "content": f"Task: Find the cause behind the {subject}'s {problem}\nPrevious actions(Remember repeated actions does not give you additional information):{formated_prev_actions}\nSummary of Interaction:\n{formated_history}\navailable actions:{available_actions}\n{available_options}What are your top {2 if posttest else self.num_of_recs} recommended actions from the action list? Only write your recommended actions.\noutput format:{output}"
                }
            ]


    def parse_response(self, response):
        return response.choices[0].message.content
class GPTChooses2(PromptFormat):
    def __init__(self, topk=5):
        self.topk = topk

    def format_prompt(self, history, subject, problem, choices, posttest=False,prev_exp=None,summarizer=None):
        formated_history = summarizer.summarize(history)
        if prev_exp is not None:
            formated_prev_exp = ""
            for (j, exp) in enumerate(prev_exp):
                formated_prev_exp += f"Experience {j + 1}:\n"
                formated_prev_exp += summarizer.summarize(exp)
        formated_prev_actions = ""
        for (i, h) in enumerate(history):
            if i % 2 == 1:
                formated_prev_actions += (h.split("the ")[-1].split('.')[0] + ",")
        formated_choices = ""

        for (i, c) in enumerate(choices[:min(self.topk,len(choices))]):
            formated_choices += (f"{i + 1}. " + c + "\n")
        question = "What should the student answer?" if posttest else "What should the student choose to do?"
        return [
            {
                "role": "system",
                "content": "You are a pharmacist helping a pharmacy student go through an educational game. You will receive a description of the customer and you will help the student ask a series of questions needed to help the customer. Make sure you are keeping the discussion short with the customer and offer a diagnosis once you are sure."
            },
            {
                "role": "user",
                "content": f"Task: Find the cause behind the {subject}'s {problem}\nPrevious actions(Remember repeated actions does not give you additional information):{formated_prev_actions}\nSummary of Interaction:\n{formated_history}\nStudent top choices:\n{formated_choices} {question} answer with the choice number.Only write your chosen action.\nexample output:1"
            }
        ]

    def parse_response(self, response):
        return response.choices[0].message.content

class CLINPlays2(PromptFormat):
    def __init__(self):
        super().__init__()

    def format_prompt(self, history, subject, problem, valid_subjects, valid_topics, valid_causes, summary, posttest=False,prev_exp=None,summarizer=None):
        formated_history = summarizer.summarize(history)
        if prev_exp is not None:
            formated_prev_exp = ""
            for (j, exp) in enumerate(prev_exp):
                formated_prev_exp += f"Experience {j + 1}:\n"
                formated_prev_exp += summarizer.summarize(exp)
        formated_prev_actions = ""
        for (i, h) in enumerate(history):
            if i % 2 == 1:
                formated_prev_actions += (h.split("the ")[-1].split('.')[0] + ",")
        formated_valid_topics = "[\n"
        formated_valid_subjects = "[\n"
        formated_valid_causes = "[\n"
        for s in valid_subjects:
            formated_valid_subjects += s + ",\n"
        for t in valid_topics:
            formated_valid_topics += t + ",\n"
        for t in valid_causes:
            formated_valid_causes += t + ",\n"
        formated_valid_topics += "\n]"
        formated_valid_subjects += "\n]"
        formated_valid_causes += "\n]"
        available_actions = "1.choose(cause)[The most probable reason is cause]" if posttest else " 1. ask(subject,topic)[I want to ask about subject’s topic.] 2. answer()[I want to suggest the cause behind the customer’s problem.]\n"
        available_options = f"valid causes: {formated_valid_causes}" if posttest else f"valid subjects: {formated_valid_subjects}\nvalid topics: {formated_valid_topics}\n"
        summary_prompt = f"Here is a summary of learnings based on your previous attempts on this task." \
                  f"These learnings capture important pre-conditions: X MAY BE NECESSARY to Y, X SHOULD BE NECESSARY to Y, and mistakes: X MAY NOT CONTRIBUTE to Y, X DOES NOT CONTRIBUTE to Y. These can be useful for predicting your next action:\n"
        user_em = """First, scan the (unordered) list of learnings, if provided. Decide if any of the learnings are applicable given the last observation to make progress in this task. Then only use selected learnings, if any, to construct a rationale for picking the next action. If no Learning is selected, construct the rationale based on the observation history. Format your response as follows:
                Write the selected learning_ids as a comma separated list; the list can be empty if no learnings selected. Then, write $$$ followed by the single next action you would like to take.
                """
        output = "choose(x)" if posttest else "ask(x,y)"

        return \
            [
                {
                    "role": "system",
                    "content": "You are a pharmacist helping a pharmacy student go through an educational game. You will receive a description of the customer and you will help the student ask a series of questions needed to help the customer. Make sure you are keeping the discussion short with the customer and offer a diagnosis once you are sure. At each step, you will receive the list of available actions for the student and you will choose the best action for the student to do."
                },
                {
                    "role": "user",
                    "content": f"Task: Find the cause behind the {subject}'s {problem}\n{summary_prompt}\n{summary}\nPrevious actions(Remember repeated actions does not give you additional information):{formated_prev_actions}\nSummary of Interaction:\n{formated_history}\navailable actions:{available_actions}\n{available_options}What action would you take next? {user_em}\noutput format:\n1, 3\n$$$\n{output}"
                }
            ]
    def parse_response(self, response):
        return response.choices[0].message.content

    class CLINRecs2(PromptFormat):
        def __init__(self, num_of_recs=5):
            self.num_of_recs = num_of_recs

        def format_prompt(self, history, subject, problem, valid_subjects, valid_topics, valid_causes, summary,
                          posttest=False, prev_exp=None, summarizer=None):
            formated_history = summarizer.summarize(history)
            if prev_exp is not None:
                formated_prev_exp = ""
                for (j, exp) in enumerate(prev_exp):
                    formated_prev_exp += f"Experience {j + 1}:\n"
                    formated_prev_exp += summarizer.summarize(exp)
            formated_prev_actions = ""
            for (i, h) in enumerate(history):
                if i % 2 == 1:
                    formated_prev_actions += (h.split("the ")[-1].split('.')[0] + ",")
            formated_valid_topics = "[\n"
            formated_valid_subjects = "[\n"
            formated_valid_causes = "[\n"
            for s in valid_subjects:
                formated_valid_subjects += s + ",\n"
            for t in valid_topics:
                formated_valid_topics += t + ",\n"
            for t in valid_causes:
                formated_valid_causes += t + ",\n"
            formated_valid_topics += "\n]"
            formated_valid_subjects += "\n]"
            formated_valid_causes += "\n]"
            available_actions = "1.choose(cause)[The most probable reason is cause]" if posttest else " 1. ask(subject,topic)[I want to ask about subject’s topic.] 2. answer()[I want to suggest the cause behind the customer’s problem.]\n"
            available_options = f"valid causes: {formated_valid_causes}" if posttest else f"valid subjects: {formated_valid_subjects}\nvalid topics: {formated_valid_topics}\n"
            summary_prompt = f"Here is a summary of learnings based on your previous attempts on this task." \
                             f"These learnings capture important pre-conditions: X MAY BE NECESSARY to Y, X SHOULD BE NECESSARY to Y, and mistakes: X MAY NOT CONTRIBUTE to Y, X DOES NOT CONTRIBUTE to Y. These can be useful for completing your task:\n"
            user_em = """First, scan the (unordered) list of learnings, if provided. Decide if any of the learnings are applicable given the last observation to make progress in this task. Then only use selected learnings, if any, to construct a rationale for picking the next action. If no Learning is selected, construct the rationale based on the observation history. Format your response as follows:
                            Write the selected learning_ids as a comma separated list; the list can be empty if no learnings selected. Then, write $$$ followed by your list of recommended actions.
                            """
            output = "1. choose(x)\n2. choose(y)" if posttest else "1. ask(x,y)\n2. answer()"

            return \
                [
                    {
                        "role": "system",
                        "content": "You are a pharmacist helping a pharmacy student go through an educational game. You will receive a description of the customer and you will help the student ask a series of questions needed to help the customer. Make sure you are keeping the discussion short with the customer and offer a diagnosis once you are sure. At each step, you will receive the list of available actions for the student and you will recommend 5 actions for the student to do."
                    },
                    {
                        "role": "user",
                        "content": f"Task: Find the cause behind the {subject}'s {problem}\n{summary_prompt}\n{summary}\nPrevious actions(Remember repeated actions does not give you additional information):{formated_prev_actions}\nSummary of Interaction:\n{formated_history}\navailable actions:{available_actions}\n{available_options}What are your top {2 if posttest else self.num_of_recs} recommended actions from the action list? {user_em}\noutput format:\n1, 3\n$$$\n{output}"
                    }
                ]

        def parse_response(self, response):
            return response.choices[0].message.content
class CLINChooses2(PromptFormat):
    def __init__(self, topk=5):
        self.topk = topk

    def format_prompt(self, history, subject, problem, choices,summary, posttest=False,prev_exp=None,summarizer=None):
        formated_history = summarizer.summarize(history)
        if prev_exp is not None:
            formated_prev_exp = ""
            for (j, exp) in enumerate(prev_exp):
                formated_prev_exp += f"Experience {j + 1}:\n"
                formated_prev_exp += summarizer.summarize(exp)
        formated_prev_actions = ""
        for (i, h) in enumerate(history):
            if i % 2 == 1:
                formated_prev_actions += (h.split("the ")[-1].split('.')[0] + ",")
        formated_choices = ""
        for (i, c) in enumerate(choices[:min(self.topk,len(choices))]):
            formated_choices += (f"{i + 1}. " + c + "\n")
        question = "What should the student answer?" if posttest else "What should the student choose to do?"
        summary_prompt = f"Here is a summary of learnings based on your previous attempts on this task." \
                         f"These learnings capture important pre-conditions: X MAY BE NECESSARY to Y, X SHOULD BE NECESSARY to Y, and mistakes: X MAY NOT CONTRIBUTE to Y, X DOES NOT CONTRIBUTE to Y. These can be useful for completing your task:\n"
        user_em = """First, scan the (unordered) list of learnings, if provided. Decide if any of the learnings are applicable given the last observation to make progress in this task. Then only use selected learnings, if any, to construct a rationale for picking the next action. If no Learning is selected, construct the rationale based on the observation history. Format your response as follows:
                        Write the selected learning_ids as a comma separated list; the list can be empty if no learnings selected. Then, write $$$ followed by the single next action you would like to take.
                        """
        return [
            {
                "role": "system",
                "content": "You are a pharmacist helping a pharmacy student go through an educational game. You will receive a description of the customer and you will help the student ask a series of questions needed to help the customer. Make sure you are keeping the discussion short with the customer and offer a diagnosis once you are sure."
            },
            {
                "role": "user",
                "content": f"Task: Find the cause behind the {subject}'s {problem}\n{summary_prompt}\n{summary}\n\nPrevious actions(Remember repeated actions does not give you additional information):{formated_prev_actions}\nSummary of Interaction:\n{formated_history}\nStudent top choices:\n{formated_choices} {question} answer with the choice number.{user_em}\nexample output:\n1, 3\n$$$\n1"
            }
        ]

    def parse_response(self, response):
        return response.choices[0].message.content
class Chooser_or_Recommender():
    def __init__(self, prompt_format, model="gpt-3.5-turbo", temperature=0,
                 max_tokens=200) -> None:
        self.prompt_format = prompt_format
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def cor(self, history, subject, problem, valid_subjects, valid_topics, valid_causes, choices,summary=None, posttest=False,prev_exp=None,summarizer=None):
        client = OpenAI()
        if summary is None:
            prompt = self.prompt_format.format_prompt(history, subject, problem, valid_subjects, valid_topics, valid_causes, choices, posttest=posttest,prev_exp=prev_exp,summarizer=summarizer)
        else:
            prompt = self.prompt_format.format_prompt(history, subject, problem, valid_subjects, valid_topics, valid_causes, choices, summary, posttest=posttest,prev_exp=prev_exp,summarizer=summarizer)
        response = client.chat.completions.create(
            model=self.model,
            messages=prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        return self.prompt_format.parse_response(response)
class Chooser():
    def __init__(self, prompt_format, model="gpt-3.5-turbo", temperature=0,
                 max_tokens=200) -> None:
        self.prompt_format = prompt_format
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def choose(self, history, subject, problem, choices,summary=None, posttest=False,prev_exp=None,summarizer=None):
        client = OpenAI()
        if summary is None:
            prompt = self.prompt_format.format_prompt(history, subject, problem, choices, posttest=posttest,prev_exp=prev_exp,summarizer=summarizer)
        else:
            prompt = self.prompt_format.format_prompt(history, subject, problem, choices, summary, posttest=posttest,prev_exp=prev_exp,summarizer=summarizer)
        response = client.chat.completions.create(
            model=self.model,
            messages=prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        return self.prompt_format.parse_response(response)
class Recommender():
    def __init__(self, prompt_format, model="gpt-3.5-turbo", temperature=0,
                 max_tokens=200) -> None:
        self.prompt_format = prompt_format
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def rec(self, history, subject, problem, valid_subjects, valid_topics, valid_causes, summary=None, posttest=False,prev_exp=None,summarizer=None):
        client = OpenAI()
        if summary is None:
            prompt = self.prompt_format.format_prompt(history, subject, problem, valid_subjects, valid_topics, valid_causes, posttest=posttest,prev_exp=prev_exp,summarizer=summarizer)
        else:
            prompt = self.prompt_format.format_prompt(history, subject, problem, valid_subjects, valid_topics, valid_causes, summary, posttest=posttest,prev_exp=prev_exp,summarizer=summarizer)
        response = client.chat.completions.create(
            model=self.model,
            messages=prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        return self.prompt_format.parse_response(response)
class Player():
    def __init__(self, prompt_format, model="gpt-3.5-turbo", temperature=0,
                 max_tokens=200) -> None:
        self.prompt_format = prompt_format
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def play(self, history, subject, problem, valid_subjects, valid_topics, valid_causes,summary=None, posttest=False,prev_exp=None,summarizer=None):

        if summary is None:
            prompt = self.prompt_format.format_prompt(history, subject, problem, valid_subjects, valid_topics, valid_causes,summarizer=summarizer, posttest=posttest,prev_exp=prev_exp)
        else:
            prompt = self.prompt_format.format_prompt(history, subject, problem, valid_subjects, valid_topics, valid_causes, summary,summarizer=summarizer, posttest=posttest,prev_exp=prev_exp)
        print(prompt)
        client = OpenAI()
        response = client.chat.completions.create(
            model=self.model,
            messages=prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        response = self.prompt_format.parse_response(response)
        print(response)
        return response

class OraclePrompt(PromptFormat):
    def __init__(self, game, oneshot=False) -> None:
        super().__init__()
        self.oneshot = oneshot
        self.game = game
        self.system = """You are a pharmacy assistant student. You are going to play a game based on scenarios of customers coming in to a simulated pharmacy for help. You will interact with the customer and whenever you are ready you will suggest a solution to the game. The solution is the diagnosis you have for the customer’s problem. Remember suggesting a diagnosis without enough interactions is not a good strategy. I'll tell you a summary of which interactions were done, and how the customer responded, and you suggest me what to do next based on the list of valid actions I give you."""
        self.task = "Task: Based on the interactions so far, please suggest me which actions are more helpful to do next from the list below. Suggest 3-5 actions. Follow the template I gave you when choosing the action."
        self.separator = "\n###\n"
        self.ex_summary = """Summary: 
- Increased urinary frequency during the day and at night.
- Customer experiences urgency to urinate but has difficulty starting the stream.
- Asked about urinary frequency specifically.
"""
        self.ex_valid_actions = """List of the actions:[
            "i want to suggest a solution",
            "i want to know about the “subject” 's “topic”."
        ]
Valid Subjects: ["customer", "parents", "dog", "cutomer's child", "house"]
Valid Topics: ['Urinary Frequency', 'Urinary Sensation', 'Urine Characteristics', 'Sexual Function', 'Leakage', 'Overal Symptoms', 'Symptoms Localization', 'Symptoms Intensity', 'Duration of Symptoms', 'Medication History', 'Allergies', 'Underlying Medical Conditions', 'Current Medications', 'Surgical History', 'Pregnancy and Breastfeeding', 'Age', 'Sleep', 'Diet', 'Exercise', 'Teeth']
"""
        self.ex_action = "I want to know about the customer's Sexual Function."

    def format_prompt(self, summary, valid_actions):
        return [{"role": "system", "content": self.system},
                {"role": "user", "content": self.separator + self.ex_summary + self.separator + self.task +
                                            self.separator + self.ex_valid_actions},
                {"role": "assistant", "content": self.ex_action},
                {"role": "user", "content": self.separator + "Summary:\n" + summary + self.separator +
                                            self.task + self.separator + "List of the actions:\n" + valid_actions},
                ] if self.oneshot else [{"role": "system", "content": self.system},
                                        {"role": "user",
                                         "content": self.separator + "Summary:\n" + summary + self.separator +
                                                    self.task + self.separator + "List of the actions:\n" + valid_actions},
                                        ]

    def parse_response(self, response):
        response = response.choices[0].message.content
        print(response)
        suggestions = re.findall(
            r'\d+\.\s+"([^"]*)"', response)
        if len(suggestions) == 0:
            suggestions = re.findall(
                r'\d+\.\s+(.*)', response)
        if len(suggestions) == 0:
            suggestions = re.findall(
                f'"([^"]*)"', response)
        mapped_suggestions = []
        if len(suggestions) > 0:
            print(suggestions)
            for suggestion in suggestions:
                action = None
                if "i want" in suggestion.lower():
                    if "diagnosis" in suggestion.lower():
                        action = "i want to suggest a solution."
                    elif "know" in suggestion.lower():
                        subject = None
                        topic = None
                        for s in self.game.scenario["subjects"]:
                            if s.lower() in suggestion.lower():
                                subject = s

                        for t in self.game.scenario["topics"]:
                            if t.lower() in suggestion.lower():
                                topic = t

                        if subject is not None and topic is not None:
                            action = f"i want to know about the {subject} 's {topic}."
                elif "i think" in suggestion.lower():
                    for c in self.game.scenario["causes"]:
                        if c.lower() in suggestion.lower():
                            action = f"I think the most probable cause behind the problem is {c}."
                            break
                if action is None:
                    action = suggestion
                mapped_suggestions.append(action)
        else:
            mapped_suggestions = []
        return mapped_suggestions


class AgentPrompt(PromptFormat):
    def __init__(self, oneshot=False) -> None:
        super().__init__()
        self.oneshot = oneshot
        self.system = """Pretend you're a pharmacy assistant student, you are going to play a game based on scenarios of customers coming in to a simulated pharmacy for help. You will interact with the customer and whenever you are ready you will suggest a solution to the game. The solution is the diagnosis you have for the customer’s problem. Remember suggesting a diagnosis without enough interactions is not a good strategy. I'll tell you a summary of which interactions were done, and how the customer responded, and you tell me what to do next based on the list of valid actions I give you."""
        self.task = "Task: Please tell me which action to choose from the list below. Choose only one action. Follow the template I gave you when choosing the action."
        self.separator = "\n###\n"
        self.ex_summary = """Summary: 
- Increased urinary frequency during the day and at night.
- Customer experiences urgency to urinate but has difficulty starting the stream.
- Asked about urinary frequency specifically.
"""
        self.ex_valid_actions = """List of the actions:[
            "i want to suggest a solution",
            "i want to know about the “subject” 's “topic”."
        ]
Valid Subjects: ["customer", "parents", "dog", "cutomer's child", "house"]
Valid Topics: ['Urinary Frequency', 'Urinary Sensation', 'Urine Characteristics', 'Sexual Function', 'Leakage', 'Overal Symptoms', 'Symptoms Localization', 'Symptoms Intensity', 'Duration of Symptoms', 'Medication History', 'Allergies', 'Underlying Medical Conditions', 'Current Medications', 'Surgical History', 'Pregnancy and Breastfeeding', 'Age', 'Sleep', 'Diet', 'Exercise', 'Teeth']
"""
        self.ex_action = "I want to know about the customer's Sexual Function."

    def format_prompt(self, summary, valid_actions):
        return [{"role": "system", "content": self.system},
                {"role": "user", "content": self.separator + self.ex_summary + self.separator + self.task +
                                            self.separator + self.ex_valid_actions},
                {"role": "assistant", "content": self.ex_action},
                {"role": "user", "content": self.separator + "Summary:\n" + summary + self.separator +
                                            self.task + self.separator + "List of the actions:\n" + valid_actions},
                ] if self.oneshot else [{"role": "system", "content": self.system},
                                        {"role": "user",
                                         "content": self.separator + "Summary:\n" + summary + self.separator +
                                                    self.task + self.separator + "List of the actions:\n" + valid_actions},
                                        ]

    def parse_response(self, response):
        return response.choices[0].message.content


class AgentPrompt2(PromptFormat):
    def __init__(self) -> None:
        super().__init__()
        self.system = """You are an AI agent helping diagnose the most probable cause behind a customer's problem in a pharmacy simulation with a limited number of questions to be asked from the customer available at each step."""
        ###################################
        self.user_sm1 = """I'd like you to work your way through a simulation of a real scenario in a pharmacy to help a customer with a certain problem. Your goal is to ask enough questions from the customer to be able to diagnose the most probable cause behind their problem. Once you have asked enough questions, you must move to the next step of the simulation by saying "i want to propose the most probable diagnosis." 
        At each step, tell me which action you want to take, e.g., I want to ask about the baby's diet, I want to ask about the mother's symptoms, etc. and I will tell you the result. Then tell me the next action you want to do, until you complete the task."""
        self.user_sm2 = "Below you can see the most recent history of your actions to help you decide your next action."
        self.task = "Task: you see a customer saying"
        #################################################
        self.obs_history = "What action would you like to do next?"
        self.action_history = "Selected action:"
        #####################################################
        self.obs = "Here is what you currently see:\n"
        self.action = "Possible actions:\n"
        self.amb = """If I say "Ambiguous request", your action might mean multiple things. In that case, respond with the number corresponding to the action you want to take in each list of possibilities."""
        self.summary = f"Here is a summary of learnings based on your previous attempts on this task." \
                       f"These learnings capture important pre-conditions: X MAY BE NECESSARY to Y, X SHOULD BE NECESSARY to Y, and mistakes: X MAY NOT CONTRIBUTE to Y, X DOES NOT CONTRIBUTE to Y. These can be useful for predicting your next action:\n"
        self.user_em = """First, scan the (unordered) list of learnings, if provided. Decide if any of the learnings are applicable given the last observation to make progress in this task. Then only use selected learnings, if any, to construct a rationale for picking the next action. If no Learning is selected, construct the rationale based on the last observation. Format your response as follows:
Write 'I used learning id(s):' as a comma separated list; the list can be empty if no learnings selected. Then, write $$$ followed by the rationale. Finally, write ### followed by the single next action you would like to take.
If you think you have asked enough questions, please write "i want to propose the most probable diagnosis." as the next action. Remember that asking the same question again will not give you a different answer."""

    def format_prompt(self, obs_history, action_history, current_obs, valid_actions, summary=""):
        history = ""
        if len(action_history) > 0:
            history += "History of your actions:\n"
            for (o, a) in zip(obs_history, action_history):
                history += "The customer said:\n" + o + "\n" + self.obs_history + "\n\n" + \
                           self.action_history + a
        initial_obs = current_obs if len(
            action_history) == 0 else obs_history[0]
        return [{"role": "system", "content": self.system},
                {"role": "user",
                 "content": "\n\n".join(
                     [self.user_sm1, self.task + f"\"{initial_obs}\"", self.user_sm2,
                      self.summary + summary if len(summary) > 0 else ""])},
                {"role": "user",
                 "content": "\n\n".join(
                     [history, self.obs + current_obs + self.action + valid_actions, self.amb,
                      self.obs_history, self.user_em])}, ]

    def parse_response(self, response):
        return response.choices[0].message.content


class SummarizerPrompt(PromptFormat):
    def __init__(self, oneshot=False) -> None:
        super().__init__()
        self.oneshot = oneshot
        self.system = """You are a pharmacy assistant trying to interact with a customer. You need to memorize the keypoints of your discussion with the customer so that you know what you did till now so you can decide what to do next after reading them."""
        self.task = "Task: Write concise and understandable notes based on the customer responses given to you. Write your note based on the text given to you. Your notes should only include the information you received. Do not include any speculations or thoughts on the customer's diagnosis in them. Use sentences and phrases in your notes. Include clues about what you already asked. Try to not exceed 100 words in your notes."
        self.separator = "\n###\n"
        self.ex_text = """Text: 
        Customer: I have urinary problems. can you help?
        I: I want to know about the customer's overall symptoms. 
        Customer: I've been having trouble starting to pee and my stream is pretty weak. I also find myself needing to go a lot, even at night. Sometimes, I can't hold it in and I end up wetting myself. 
        I: I want to know about the customer's urinary frequency. 
        Customer: I've been going to the bathroom a lot more than usual, both during the day and at night. Sometimes, I feel like I need to go urgently, but then I have trouble starting the stream."""
        self.ex_summary = """* Customer experiencing difficulty initiating urination and weak urine stream.
* Increased urinary frequency and urgency, both day and night.
* Instances of incontinence (wetting themselves).
* Asked about overall symptoms and urinary frequency specifically."""

    def format_prompt(self, pa, customer):
        assert len(pa) + 1 == len(customer)
        customer = ["Customer: " + c + "\n" for c in customer]
        pa = ["Pharmacy Assistant: " + p + "\n" for p in pa]
        input = ""
        input += customer[0]
        customer = customer[1:]
        if len(pa) > 0:
            for (p, c) in zip(pa, customer):
                input += (p + c)
        return [{"role": "system", "content": self.system},
                {"role": "user", "content": self.task +
                                            self.separator + self.ex_text},
                {"role": "assistant", "content": self.ex_summary},
                {"role": "user", "content": self.task +
                                            self.separator + "Text:\n" + input},
                ] if self.oneshot else [{"role": "system", "content": self.system},
                                        {"role": "user", "content": self.task +
                                                                    self.separator + "Text:\n" + input},
                                        ]

    def parse_response(self, response):
        return response.choices[0].message.content
class SummarizerPrompt2(PromptFormat):
    def __init__(self, oneshot=False) -> None:
        super().__init__()
        self.oneshot = oneshot
        self.system = """You are a pharmacy assistant trying to interact with a customer. You need to memorize the keypoints of your discussion with the customer so that you know what you did till now so you can decide what to do next after reading them."""
        self.task = "Task: Write concise and understandable notes based on the customer responses given to you. Write your note based on the text given to you. Your notes should only include the information you received. Do not include any speculations or thoughts on the customer's diagnosis in them. Use sentences and phrases in your notes. Include clues about what you already asked. Try to not exceed 100 words in your notes."
        self.separator = "\n###\n"

    def format_prompt(self, history):
        formated_history = ""
        for (i, s) in enumerate(history):
            prefix = "Assistant: " if i % 2 == 1 else "Customer: "
            formated_history += (prefix + s + "\n")
        return [{"role": "system", "content": self.system},
                {"role": "user", "content": self.task +
                                            self.separator + "Interaction History:\n" + formated_history}]


    def parse_response(self, response):
        return response.choices[0].message.content


class Summarizer():
    def __init__(self, prompt_format, model="gpt-3.5-turbo", temperature=0,
                 max_tokens=100) -> None:
        super().__init__()
        self.prompt_format = prompt_format
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def summarize(self, history):
        prompt = self.prompt_format.format_prompt(history)
        client = OpenAI()
        response = client.chat.completions.create(
            model=self.model,
            messages=prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        return self.prompt_format.parse_response(response)


def main():
    pa = ["I want to know about the customer's urinary frequency."]
    customer = ["I have urinary problems. can you help?",
                "I've been going to the bathroom a lot more than usual, both during the day and at night. Sometimes, I feel like I need to go urgently, but then I have trouble starting the stream."]
    prompt_format = SummarizerPrompt()
    summarizer = Summarizer(prompt_format)
    print(summarizer.summarize(pa, customer))


if __name__ == "__main__":
    main()
