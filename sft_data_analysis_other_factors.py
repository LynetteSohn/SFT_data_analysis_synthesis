#!/usr/bin/env python3
#
import sys, os, pdb
import json, csv
import random, argparse, time
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from collections import OrderedDict
from nltk.stem import WordNetLemmatizer
from nltk.corpus import words
#from fuzzywuzzy import fuzz
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
from openai_api import openai_api_chat

random.seed(42)
np.random.seed(42)
WORD_SET = set(words.words())
# update vocab
WORD_SET = WORD_SET.union({"numpy", "targeting", "templating", "talor", "nesting", "continuation", "writing", "naming", "keywords"})

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:\n"
    ),
}

class GPTGenerator(object):
    def __init__(self, args) -> None:
        self.args = args
        self.data_path = os.path.join("./", self.args.data_name, self.args.file_name)
        self.save_path = os.path.join("./", self.args.data_name, self.args.save_file_name)

    def _load_json(self, path):
        if path is None or not os.path.exists(path):
            raise IOError(f"File doe snot exists: {path}")
        print(f"Loading data from {path} ...")
        with open(path) as df:
            data = json.loads(df.read())
        return data
    
    def _save_json(self, data, file_path):
        dir_path = "/".join(file_path.split("/")[:-1])
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path)
        # print(f"Saving file {file_path} ...")
        with open(file_path, "w") as tf:
            json.dump(data, tf, indent=2)
    
    def normalize_model_output(self, seq):
        seq = seq.strip()
        seq = seq.strip("\"")
        return seq

    def get_length_distribution(self, slice_size=50):
        """
        use the first 50 """
        length_distribution_source_path = "./result/length/length_count.csv"
        with open(length_distribution_source_path, "r") as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if row[0].startswith("sharegpt"):
                    num_list = [int(num) for num in row[1:]]
                    total_num = sum(num_list)
                    return [num/total_num for num in num_list[:slice_size]]
                
    def get_task_distribution(self, slice_size=100):
        task_distribution_source_path = "./result/task/action_sharegpt_xc_20000.csv"
        action_dict = defaultdict(int)
        with open(task_distribution_source_path, "r") as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if not row[0].isnumeric(): continue
                action_dict[f"{row[1]} {row[2]}"] += 1
        sort_action_dict = sorted(action_dict.items(), key=lambda x:x[1], reverse=True)
        with open("./result/task/count_sharegpt_xc_20000.csv", "w") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["action", "count"])
            for row in sort_action_dict:
                writer.writerow(list(row))
        action_dict = dict(sort_action_dict)
        return list(action_dict.keys()), list(action_dict.values())
    
    def get_domain_distribution(self):
        """
        distribution over summarized topics:
        {
            domain : num,
            ... ,
        }"""
        # # # use un-summarized topics
        sample_num = 10000
        distribution_data_path = f"./result/topic/count_sharegpt_xc_{sample_num}.csv"
        if os.path.exists(distribution_data_path):
            sort_topic_dict = []
            with open(distribution_data_path, "r") as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    if row[-1] == "count": continue
                    sort_topic_dict.append((row[0], int(row[-1])))
        else:
            origin_data_path = f"./data/sharegpt_xc/data_sample_{sample_num}.json"
            # domain_distribution_source_path = "./result/topic/action_sharegpt_xc_20000.csv"
            topics = [turn["topic"] for turn in self._load_json(origin_data_path)]
            topic_dict=defaultdict(int)
            for topic in sorted(topics):
                topic = self.normalize(topic, processed_topics=list(topic_dict.keys()))
                if topic is not None:
                    topic_dict[topic] += 1
            sort_topic_dict = sorted(topic_dict.items(), key=lambda x:x[1], reverse=True)
            with open(distribution_data_path, "w") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["topic", "count"])
                for row in sort_topic_dict:
                    writer.writerow(list(row))
        topic_dict = dict(sort_topic_dict)
        return list(topic_dict.keys()), list(topic_dict.values())

    

    def create_prompt(self, domain, length, task):
        """ Create a prompt for the given domain and task """
        prompt = f"Please write one prompt suitable for conducting a TASK: {task}" + \
                 f", focused on the DOMAIN: {domain}" + \
                 f". Each prompt should be of LENGTH: {length}" + \
                 ".\nOnly return the prompt, don't respond to the instruction or add anything in the front or at the end."
                # ". Besides instruction, please also provide the output of the instruction"
        return prompt


    def generate_output(self, ori_version=3):
        data = self._load_json(self.data_path)
        progress_bar = tqdm(range(len(data)))
        gen_idx = 0 
        ori_output_key = f"output_ori"
        for turn in data:
            if ori_output_key in turn:
                    # already generated
                    progress_bar.update(1)
            else:
                if "output" in turn:
                    turn[ori_output_key] = turn["output"]
                # if turn["input"]:
                #     input_seq = PROMPT_DICT["prompt_input"].format_map(turn)
                # else:
                    # input_seq = PROMPT_DICT["prompt_no_input"].format_map(turn)
                if "input" in turn:
                    input_seq = turn["instruction"] + "\n" + turn["input"]
                else:
                    input_seq = turn["instruction"]
                output = openai_api_chat(self.args, input_seq=input_seq, system_prompt="You are a helpful agent.")
                turn["output"] = output
                progress_bar.update(1)
                gen_idx += 1
                if gen_idx % 10 == 0:
                    # print("Starting saving ...... ")
                    self._save_json(data, self.save_path)
                    # print("Finishing saving ...... ")

                if gen_idx % 1000 == 0 or gen_idx < 5:
                    print("#Instruction: " , turn["instruction"])
                    print("#Output: " , turn["output"])
            
        self._save_json(data, self.save_path.replace(".json", "_done.json"))






def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_name",
        type=str,
        default="generate",
        help="The name of the dataset to generate or augment",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="gpt-4", # "gpt-3.5-turbo", # 
        help="The name of the model for generation",
    )
    parser.add_argument(
        "--file_name",
        type=str,
        default="data_test.json",
        help="The name of the input data for topic extraction ",
    )
    parser.add_argument(
        "--file_path",
        type=str,
        default="",
        help="The path to the input data for topic extraction / summarization / ppl computation, usually not used together with --file_name",
    )
    parser.add_argument(
        "--save_file_name",
        type=str,
        default="data_test.json",
        help="The path to save the data with extracted topic",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="generate_instruction_from_scratch",
        choices=["generate_instruction_from_scratch", "get_rewriting", "generate_output"],
        help="Choose which task to conduct",
    )
    parser.add_argument(
        "--sample_num",
        type=int,
        default=500,
        help="The number of sampled datapoint",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature for decoding",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=1024,
        help="Maximum decoding length",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="Maximum decoding length",
    )
    parser.add_argument(
        "--frequency_penalty",
        type=float,
        default=0.0,
        help="Penalty for token frequency",
    )
    parser.add_argument(
        "--presence_penalty",
        type=float,
        default=0.0,
        help="Penalty for token presence",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--prompt_style",
        type=str,
        default="instruction",
        choices=["instruction", "instruction_input", "instruction_input_output"],
        help="Choose what is contained within the prompt",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    gen = GPTGenerator(args)
    
    function = getattr(gen, args.task, None)
    if callable(function):
        function()
    else:
        raise ValueError("Choose a pre-defined task to execute ... ")

if __name__ == "__main__":
    main()