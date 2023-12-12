import random
import json
from datasets import load_dataset
from tqdm import tqdm
from openai_api import openai_api_chat
import argparse
from collections import defaultdict 

ultrachat_meta_topic = ["Technology", "Health and wellness", "Travel and adventure", "Food and drink", "Art and culture", "Science and innovation", \
    "Fashion and style", "Relationships and dating", "Sports and fitness", "Nature and the environment", "Music and entertainment", "Politics and current events", \
    "Education and learning", "Money and finance", "Work and career", "Philosophy and ethics", "History and nostalgia", "Social media and communication", \
    "Creativity and inspiration", "Personal growth and development", "Spirituality and faith", "Pop culture and trends", "Beauty and self-care", "Family and parenting", \
    "Entrepreneurship and business", "Literature and writing", "Gaming and technology", "Mindfulness and meditation", "Diversity and inclusion", "Travel and culture exchange", \
    ]

gpt_extended_ultrachat_meta_topic = ["Technology and Innovation", "Health and wellness", "Travel and adventure", "Food and drink", "Art and culture", "Science and innovation", \
    "Fashion and style", "Relationships and dating", "Sports and fitness", "Nature and the environment", "Music and entertainment", "Politics and current events", \
    "Education and learning", "Money and finance", "Work and career", "Philosophy and ethics", "History and nostalgia", "Social media and communication", \
    "Creativity and inspiration", "Personal growth and development", "Spirituality and faith", "Pop culture and trends", "Beauty and self-care", "Family and parenting", \
    "Entrepreneurship and business", "Literature and writing", "Gaming and technology", "Mindfulness and meditation", "Diversity and inclusion", "Travel and culture exchange", \
    "Psychology and Mental Health", "Law and Legal Affairs", "Sustainability and Environmental Protection", "Science Fiction and Fantasy", "Home Improvement and DIY", "Film and Cinema", \
    "Photography and Visual Arts", "Medicine and Healthcare", "Career Development and Job Searching", "Sociology and Social Issues", "Automotive and Transportation", "Animals and Wildlife Conservation", \
    "Personal Finance and Investing", "Outdoor Activities and Adventure Sports", "Startups", "Philosophy and Critical Thinking", "Fine Arts and Fine Dining", "Architecture and Urban Planning", \
    "Journalism and Media", "Self-help and Motivation", "Foreign Languages and Linguistics", "Nutrition and Healthy Eating", "Gaming and E-Sports",
    ]
def save_json(data, file_path):
    with open(file_path, "w") as tf:
        json.dump(data, tf, indent=2)


def exhaustive_search_meta_topic(args):
    existing_topic_list = ", ".join(ultrachat_meta_topic)
    prompt = "Now we want to have an exhaustive list of topics/knowledge domains that we humans can encounter in our lives. " + \
        "This list should include not only ordinary people's daily topics, such as 'relationship' and 'parenting', but also specific knowledge domains such as 'math' and 'science'. \n" + \
        "We already have a list like this: \n" + existing_topic_list + " \n\n Can you help me find the topics that this list didn't cover? "
    output = openai_api_chat(args, input_seq=prompt, system_prompt="You are a helpful agent.")
    print(output)
    # 1. Psychology and Mental Health
    # 2. Sports and Recreation
    # 3. Technology and Innovation
    # 4. Law and Legal Affairs
    # 5. Sustainability and Environmental Protection
    # 6. Science Fiction and Fantasy
    # 7. Home Improvement and DIY
    # 8. Film and Cinema
    # 9. Photography and Visual Arts
    # 10. Medicine and Healthcare
    # 11. Career Development and Job Searching
    # 12. Sociology and Social Issues
    # 13. Automotive and Transportation
    # 14. Animals and Wildlife Conservation
    # 15. Personal Finance and Investing
    # 16. Fashion and Design
    # 17. Outdoor Activities and Adventure Sports
    # 18. Entrepreneurship and Startups
    # 19. Philosophy and Critical Thinking
    # 20. Fine Arts and Fine Dining
    # 21. Parenting and Child Development
    # 22. Architecture and Urban Planning
    # 23. Travel and Cultural Exploration
    # 24. Interior Design and Home Decor
    # 25. Journalism and Media
    # 26. Religion and Spirituality
    # 27. Self-help and Motivation
    # 28. Foreign Languages and Linguistics
    # 29. Nutrition and Healthy Eating
    # 30. Gaming and E-Sports


def sub_topic_generation(args, topic_list):
    sub_topic_dict = {}
    for topic in topic_list:
        prompt = 'Please list 50 sub-topics under the following topic. The 50 sub topics should try to cover all the areas and knowledge domains under this give topic. Therefore each sub topic should not be too specific. \n' + \
            'Your output format should be: {"sub_topic_list": ["your_sub_topic_1", "your_sub_topic_2", "your_sub_topic_3", ...]}. \n' + \
            'Here is the given topic: ' + topic
        output = openai_api_chat(args, input_seq=prompt, system_prompt="You are a helpful agent.")
        sub_topic_list = json.loads(output)['sub_topic_list']
        sub_topic_dict[topic] = sub_topic_list
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print(topic)
        print(sub_topic_list)
    
    save_json(sub_topic_dict, "/export/home/code/instruction-data-analysis-kun/prompt_generation_from_scratch_1031/sub_topic_1031.json")


def specific_sub_sub_topic_generation(args):
    sub_topic_dict = json.load(open("/export/home/code/instruction-data-analysis-kun/prompt_generation_from_scratch_1031/sub_topic_1031.json"))
    sub_sub_topic_dict = {}
    for (topic, sub_topic_list) in tqdm(sub_topic_dict.items()):
        for sub_topic in sub_topic_list:
            prompt = 'Please list 10 specific areas under the given "domain -> sub-domain", in which humans are very likely to ask questions to, or seek help from AI assistents. Try to make this 10 specific areas as diverse as possible. \n' + \
                'Your output format should be: {"specific_area_list": ["your_specific_area_1", "your_specific_area_2", "your_specific_area_3", ...]}. \n' + \
                'Here is the given domain -> sub-domain: ' + topic + " -> " + sub_topic
            output = openai_api_chat(args, input_seq=prompt, system_prompt="You are a helpful agent.")
            sub_sub_topic_list = json.loads(output)
            print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            print(prompt)
            print(sub_sub_topic_list)
            sub_sub_topic_dict[sub_topic] = sub_sub_topic_list
        save_json(sub_sub_topic_dict, "/export/home/code/instruction-data-analysis-kun/prompt_generation_from_scratch_1031/sub_sub_topic_1031.json")


def specific_task_generation(args):
    sub_topic_dict = json.load(open("/export/home/code/instruction-data-analysis-kun/prompt_generation_from_scratch_1031/sub_topic_1031.json"))
    sub_sub_topic_dict = json.load(open("/export/home/code/instruction-data-analysis-kun/prompt_generation_from_scratch_1031/sub_sub_topic_1031.json"))
    prompt_list = []
    missing_task = 0
    for (topic, sub_topic_list) in tqdm(sub_topic_dict.items()):
        for sub_topic in sub_topic_list:
            specific_area_list = sub_sub_topic_dict[sub_topic]["specific_area_list"]
            for area in specific_area_list:
                prompt = 'Please give 3 specific tasks under the given "domain -> sub-domain -> specific-area", on which humans would likely to seek help from AI assistents. The tasks you give should be more specific than the given specific areas. Try to make this 3 tasks as diverse as possible. \n' + \
                'Your output format should be: {"task_list": ["task_1", "task_2", "task_3", ...]}. \n' + \
                'Here is the given domain -> sub-domain -> specific-area: ' + topic + " -> " + sub_topic + " -> " + area
                output = openai_api_chat(args, input_seq=prompt, system_prompt="You are a helpful agent.")
                try:
                    task_list = json.loads(output)["task_list"]
                except:
                    missing_task += 1
                    task_list = []
                print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                print(prompt)
                print(task_list)
                if len(task_list)!=0:
                    for task in task_list:
                        prompt_for_prompt_generation = "Please write a prompt/question that humans would likely want to ask an AI agent in this domain: " + topic + " -> " + sub_topic + " -> " + area + \
                            ". The prompt/question should require the AI agent to conduct a '" + task + "' task. Your  prompt/question should be more specific than the given task. Your output should only contain the prompt itself and no other text. "
                        auxilary_prompt = "Please try to provide detailed and specific context/requirements in the prompt." 
                        auxilary_int = random.randint(0, 2)
                        if auxilary_int==0:
                            prompt_for_prompt_generation += auxilary_prompt
                        generated_prompt = openai_api_chat(args, input_seq=prompt_for_prompt_generation, system_prompt="You are a helpful agent.")
                        print(generated_prompt)
                        prompt_list.append({"instruction": generated_prompt, "task": task, "specific_area": area, "sub_topic": sub_topic, "topic": topic})
            save_json(prompt_list, "/export/home/code/instruction-data-analysis-kun/prompt_generation_from_scratch_1031/generated_prompt_1031.json")




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
    #if args.task == "ask_gpt":
    #exhaustive_search_meta_topic(args)
    #sub_topic_generation(args, gpt_extended_ultrachat_meta_topic)
    #specific_sub_sub_topic_generation(args)
    specific_task_generation(args)

if __name__ == "__main__":
    main()