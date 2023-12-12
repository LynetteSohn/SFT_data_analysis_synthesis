import random
import json
from datasets import load_dataset
from tqdm import tqdm
from openai_api import openai_api_chat
import argparse
from collections import defaultdict 

def load_ultrachat():
    dataset = load_dataset("stingning/ultrachat", split="train", cache_dir="/export/share/cxing/hf_cache")
    print(dataset['data'][0])
    print(dataset['id'][0])
    print(dataset)
    conversation_list_for_analysis = []
    for data_id, data in tqdm(zip(dataset['id'], dataset['data'])):
        # data is a list that has multiple turns of conversations.
        turn_list = data[:2]
        conversation_list_for_analysis.append(turn_list)
    sampled_conversation_list_for_analysis = random.sample(conversation_list_for_analysis, 2000)
    return sampled_conversation_list_for_analysis

def load_alpaca():
    conversation_list = json.load(open("/export/home/code/instruction-data-analysis-kun/data/sharegpt_qy_vic/data_in3_out4.json"))
    sampled_conversation_list = random.sample(conversation_list, 2000)
    conversation_list_for_analysis = []
    for conversation in sampled_conversation_list:
        turn_list = [conversation['instruction'],conversation['input'],conversation['output'] ]
        conversation_list_for_analysis.append(turn_list)

    return conversation_list_for_analysis


def load_sharegpt():
    conversation_list = json.load(open("/export/share/cxing/sharegpt/for_chen/split_data/sharegpt.json"))
    print("Total number of conversations: "+ str(len(conversation_list)))
    sampled_conversation_list = random.sample(conversation_list, 2000)
    conversation_list_for_analysis = []
    for conversation in sampled_conversation_list:
        turn_list = [i["value"] for i in conversation["conversations"][:2]]
        conversation_list_for_analysis.append(turn_list)
    return conversation_list_for_analysis


def mix_existing_sources():
    conversation_list = json.load(open("/export/home/code/instruction-data-analysis-kun/data/generated_prompts/ltrachat_analysis_results_detailed_121taskdomain_1thirdspecific.json"))
    sharegpt_conversation_list = json.load(open("/export/home/code/instruction-data-analysis-kun/data/generated_prompts/sharegpt_analysis_results_detailed_121taskdomain_1thirdspecific.json"))
    conversation_list.extend(sharegpt_conversation_list)
    sampled_conversation_list = random.sample(conversation_list, 2000)
    save_json(sampled_conversation_list, "/export/home/code/instruction-data-analysis-kun/data/generated_prompts/mix_ltrachat_shargpt_analysis_results_detailed_121taskdomain_1thirdspecific.json")



def mix_long_context_ultrachat():
    ultra_conversation_list = json.load(open("/export/home/code/instruction-data-analysis-kun/data/ultrachat/ultrachat_2k_firstturn_gpt35_done.json"))
    long_context_instructions = []
    for conversation in ultra_conversation_list:
        instruction = conversation["instruction"]
        if len(instruction)>800:
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print(instruction)
            long_context_instructions.append(conversation)
    print(len(long_context_instructions))
    generated_conversation_list = json.load(open("/export/home/code/instruction-data-analysis-kun/data/generated_prompts/ltrachat_analysis_results_coaesedtask_121taskdomain.json"))
    sampled_generated_conversation_list = random.sample(generated_conversation_list, 2000-len(long_context_instructions))
    long_context_instructions.extend(sampled_generated_conversation_list)
    save_json(sampled_generated_conversation_list, "/export/home/code/instruction-data-analysis-kun/data/generated_prompts/mix_longcontextultrachat_coaesedtask_121taskdomain.json")








def save_json(data, file_path):
    with open(file_path, "w") as tf:
        json.dump(data, tf, indent=2)

def sample_ask_gpt(args, output_addr):
    conversation_list = load_sharegpt()
    # prompt = "You will be given one round of conversation history between a human and an AI agent. \
    # Your job is to tell the task and domain of this conversation.\n Some examples of tasks are, summarization, rewriting, question answering, translation, email generation, etc. \n \
    # Some examples of domains are, Python, Finance, Music, Human History, Quantum Physics, etc. \n \
    # Your answers should not be limited to such examples. The output format should be: {'task': <your-task-name>, 'domain': <your-domain-name>} \n \
    # Here is the conversation: \n\n"
    prompt = "You will be given one round of conversation history between a human and an AI agent. \
    Your job is to tell the task and domain of this conversation.\n Some examples of tasks are, translation, email generation, movie review, recipe modification, etc. \n \
    Some examples of domains are, Python, Quantum Physics, Movies, Air pollution in Delhi, etc. \n \
    Your answers should not be limited to such examples. The output format should be: {'task': <your-task-name>, 'domain': <your-domain-name>} \n \
    Here is the conversation: \n\n"
    analyzed_results = []
    for i, conversation in tqdm(enumerate(conversation_list)):
        log = "\n".join(conversation)
        input_to_gpt = prompt+log
        output = openai_api_chat(args, input_seq=input_to_gpt, system_prompt="You are a helpful agent.")
        print(input_to_gpt)
        print(output)
        print("======================================================")
        analyzed_results.append({"input_to_gpt": input_to_gpt, "analysis_result": output})
        if i % 10 == 0:
            save_json(analyzed_results, output_addr)
    
def postprocess_analysis_result(file_addr):
    analysis_result_list = json.load(open(file_addr))
    domain_dict = defaultdict(int)
    task_dict = defaultdict(int)
    for analysis_result in analysis_result_list:
        try:
            result_dict = json.loads(analysis_result["analysis_result"].replace("'",'"'))
        except:
            continue
        task = result_dict["task"].lower()
        if "domain" in result_dict:
            domain = result_dict["domain"].lower()
            domain_dict[domain] += 1
        task_dict[task] += 1
    sorted_domain = sorted(domain_dict.items(), key=lambda x:x[1], reverse=True)
    sorted_task = sorted(task_dict.items(), key=lambda x:x[1], reverse=True)
    #print(sorted_task)
    for t in sorted_task:
        print(t)
    task_statistics(sorted_task, "tasks")
    task_statistics(sorted_domain, "domains")
    #print(sorted_domain)
    return sorted_task, sorted_domain

def task_statistics(sorted_task, meta):
    print("total number of distinct " + meta +": " + str(len(sorted_task)))
    single_instance_tasks = 0
    for (k, v) in sorted_task:
        if v ==1:
            single_instance_tasks += 1
    print("total number of distinct " + meta +": " + str(single_instance_tasks))

def generate_instruction_with_task_domain_distribution(args, sorted_domain, sorted_task, out_addr):
    domain_distribution_list = []
    task_distribution_list = []
    for (domain, count) in sorted_domain:
        domain_distribution_list.extend([domain for i in range(count)])
    for (task, count) in sorted_task:
        task_distribution_list.extend([task for i in range(count)])
    generation_prompt_list = []
    for i in tqdm(range(2000)):
        task = random.sample(task_distribution_list, 1)[0]
        domain = random.sample(domain_distribution_list, 1)[0]
        # generation_prompt = "Please write a prompt/question that humans would likely want to ask an AI agent in the knowledge domain of " + domain + \
        #                     ". The prompt/question should be of the " + task + " task."
        generation_prompt = "Please write a prompt/question that humans would likely want to ask an AI agent in the knowledge domain of " + domain + \
                            ". The prompt/question should require the AI agent to conduct a " + task + " task. " #+ \
                           # "If the promt/question requires context, please also generate proper and specific context."
        generation_prompt = "Please write a prompt/question that humans would likely want to ask an AI agent in the knowledge domain of " + domain + \
                            ". The prompt/question should require the AI agent to conduct a " + task + " task. Please try to provide detailed and specific context/requirements in the prompt." 
        output = openai_api_chat(args, input_seq=generation_prompt, system_prompt="You are a helpful agent.")
        print("===============================")
        print(generation_prompt)
        print(output)
        generation_prompt_list.append({"instruction": output})
    save_json(generation_prompt_list, out_addr)

def generate_instruction_with_121_task_domain_distribution(args, in_addr, out_addr):
    analysis_result_list = json.load(open(in_addr))
    task_domain_tuple_list = []
    for analysis_result in analysis_result_list:
        try:
            result_dict = json.loads(analysis_result["analysis_result"].replace("'",'"'))
        except:
            continue
        task = result_dict["task"].lower()
        domain = ""
        if "domain" in result_dict:
            domain = result_dict["domain"].lower()
        task_domain_tuple_list.append((task, domain))

    generation_prompt_list = []
    for i in tqdm(range(2000)):
        (task, domain) = random.sample(task_domain_tuple_list, 1)[0]
        # generation_prompt = "Please write a prompt/question that humans would likely want to ask an AI agent in the knowledge domain of " + domain + \
        #                     ". The prompt/question should be of the " + task + " task."
        # generation_prompt = "Please write a prompt/question that humans would likely want to ask an AI agent in the knowledge domain of " + domain + \
        #                     ". The prompt/question should require the AI agent to conduct a " + task + " task. " #+ \
                           # "If the promt/question requires context, please also generate proper and specific context."
        generation_prompt = "Please write a prompt/question that humans would likely want to ask an AI agent in the knowledge domain of " + domain + \
                            ". The prompt/question should require the AI agent to conduct a " + task + " task. Your output should only contain the prompt itself and no other text. "
        auxilary_prompt = "Please try to provide detailed and specific context/requirements in the prompt." 
        auxilary_int = random.randint(0, 2)
        if auxilary_int==0:
            generation_prompt += auxilary_prompt
        generated_prompt = openai_api_chat(args, input_seq=generation_prompt, system_prompt="You are a helpful agent.")
        print("===============================")
        print(generation_prompt)
        print(generated_prompt)
        output = openai_api_chat(args, input_seq=generated_prompt, system_prompt="You are a helpful agent.")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print(output)
        generation_prompt_list.append({"instruction": generated_prompt, "output": output})
    save_json(generation_prompt_list, out_addr)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default= "gpt-3.5-turbo" ,#"gpt-4", # "gpt-3.5-turbo", # 
        help="The name of the model for generation",
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
        "--task",
        type=str,
        default="",
        help="",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if args.task == "ask_gpt":
        sample_ask_gpt(args, "/.../sharegpt_analysis_results_1026_prompt2.json")
    elif args.task == "postprocess":
        sorted_task_distribution, sorted_domain_distribution = postprocess_analysis_result("/.../ultrachat_2k_firstturn_gpt35_done_analysis_results_1023.json")
    elif args.task == "mix":
        mix_existing_sources()
if __name__ == "__main__":
    main()