import random
import json
from datasets import load_dataset
from tqdm import tqdm
from collections import defaultdict 


def load_sampled_ultrachat(in_addr):
    conversation_list = json.load(open(in_addr))
    for conversation in conversation_list:
        print("======================================")
        print(conversation["instruction"])
        print("~~~~~~~~~~~~~~~~~~~~~~~")
        print(conversation["output"])

def load_generated_prompts(in_addr):
    conversation_list = json.load(open(in_addr))
    for conversation in conversation_list:
        print("======================================")
        #print(conversation["input_to_gpt"])
        #print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print(conversation["instruction"])

def load_dtanalysis_and_generated_prompts(dt_addr, prompt_addr):
    conversation_list = json.load(open(prompt_addr))
    dt_list = json.load(open(dt_addr))
    for (dt_dict, conversation_dict) in zip(dt_list, conversation_list):
        print("======================================")
        print(dt_dict["input_to_gpt"])
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print(dt_dict["analysis_result"])
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print(conversation_dict["instruction"])

def domain_pair_analysis(file_addr):
    domain_pair_analysis_result_list = json.load(open(file_addr))
    domain_dict = defaultdict(int)
    task_dict = defaultdict(int)
    pair_dict = defaultdict(int)
    for analysis_result in domain_pair_analysis_result_list:
        #print(analysis_result["analysis_result"].replace("'",'"'))
        try:
            result_dict = json.loads(analysis_result["analysis_result"].replace("'",'"'))
        except:
            continue
        task = result_dict["task"].lower()
        if "domain" in result_dict:
            domain = result_dict["domain"].lower()
            domain_dict[domain] += 1
        task_dict[task] += 1
        pair = domain + "_" + task
        pair_dict[pair] += 1
        if task == 'explanation': #"information retrieval": #'recipe modification', explanation, providing information
            print(analysis_result)
    sorted_domain = sorted(domain_dict.items(), key=lambda x:x[1], reverse=True)
    sorted_task = sorted(task_dict.items(), key=lambda x:x[1], reverse=True)
    sorted_pair = sorted(pair_dict.items(), key=lambda x:x[1], reverse=True)
    #print(sorted_task)
    # for t in sorted_pair:
    #     print(t)
    for t in sorted_task:
        print(t)
    # for t in sorted_domain:
    #     print(t)
    task_statistics(sorted_task, "tasks")
    task_statistics(sorted_domain, "domains")
    task_statistics(sorted_pair, "pairs")
    #print(sorted_domain)
    return sorted_task, sorted_domain

def task_statistics(sorted_task, meta):
    print("total number of distinct " + meta +": " + str(len(sorted_task)))
    single_instance_tasks = 0
    for (k, v) in sorted_task:
        if v ==1:
            single_instance_tasks += 1
    print("total number of distinct " + meta +": " + str(single_instance_tasks))
