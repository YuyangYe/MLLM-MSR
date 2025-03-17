import torch
import transformers
from multiprocess import set_start_method
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset, load_dataset
import os
import pandas as pd
from torch.cuda.amp import autocast
import logging

os.environ['CURL_CA_BUNDLE'] = ''
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4,5,6,7"

logging.getLogger("transformers").setLevel(logging.ERROR)

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id, token='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')
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", token='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').eval()
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'
model.generation_config.pad_token_id = model.generation_config.eos_token_id

BATCH_SIZE = 12

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

pipelines = {}

def create_prompt(example, title_df, visual_df):
    user, items = example['user'], example['items']
    prompt = f"[INST] Below is a chronological list of videos previously watched by User {user}:\n"
    for i, item in enumerate(items):
        title = title_df.loc[item, 'title']
        visual_desc = visual_df.loc[item, 'summary']
        prompt += f"{i + 1}. {item}: Title - '{title}', Video cover description - {visual_desc}.\n"
    prompt += (
        "Based on the videos listed above, please summarize the user's preferences in terms of both content and visual style in one line. "
        "Only provide information about the user's preferences; do not repeat details about the previously watched videos. "
        "Do not repeat the question in your answer, and keep clear and concise."
        "The answer should start with 'The user appears to have a preference for'."
    )
    return {'prompt': prompt}


def map_prompt(example):
    return create_prompt(example, title_df, visual_df)

ui_pair_path = "../../data/MicroLens-50k/Split/user_items_negs.tsv"
data = []
with open(ui_pair_path, 'r') as file:
    for line in file:
        parts = line.strip().split('\t')
        if len(parts) >= 2:
            user = parts[0]
            items = parts[1].split(', ')[:-1]
            data.append({'user': user, 'items': items})


user_hist_df = pd.DataFrame(data)
user_hist_dataset = Dataset.from_pandas(user_hist_df)

title_df = pd.read_csv("../../data/MicroLens-50k/MicroLens-50k_titles.csv")
visual_df = pd.read_csv("image_summary.csv")

title_df["item"] = title_df["item"].astype(str)
visual_df["item_id"] = visual_df["item_id"].astype(str)
title_df.set_index("item", inplace=True)
visual_df.set_index("item_id", inplace=True)


def build_prompt(user, item_chunk, last_preference):
    prompt = f"[INST] Below is a chronological list of videos previously watched by User {user}:\n"
    for i, item in enumerate(item_chunk):
        title = title_df.loc[item, 'title']
        visual_desc = visual_df.loc[item, 'summary']
        prompt += f"{i + 1}. {item}: Title - '{title}', Video cover description - {visual_desc}.\n"
    if last_preference is None:
        prompt += "Based on the content and visual style of videos listed above, "
    else:
        prompt += f"We also know this user's previous preference.\n {last_preference}\n"
        prompt += "Based on the content and visual style of videos listed above as well as the known aspects of user's preference, "
    prompt += (
        "please summarize the user's preferences in one continuous paragraph. "
        "Only provide information about the user's preferences; do not repeat details about the previously watched videos. "
        "Do not repeat the question in your answer, and keep clear and concise."
        "The answer should start with 'The user appears to have a preference for'."
    )
    return prompt

def infer(user, items, rank):
    last_preference = None

    for i in range(0, len(items), 3):
        prompt = build_prompt(user, items[i:i+3], last_preference)

        messages = [
            {"role": "user", "content": prompt},
        ]

        pipeline = pipelines[rank]

        prompt = pipeline.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
        )
        outputs = pipeline(
            prompt,
            max_new_tokens=1024,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        last_preference = outputs[0]["generated_text"][len(prompt):]
    return last_preference

def gpu_computation(batch, rank):
    device = f"cuda:{rank % torch.cuda.device_count()}"
    # model.to(device)
    if rank not in pipelines:
        pipelines[rank] = transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device=device,
        )
    user, items = batch['user'], batch['items']
    summaries = []
    for i in range(len(user)):
        summaries.append(infer(user[i], items[i], rank))
    print('user', user[0], 'summary', summaries[0])
    return {'user': user, 'summary': summaries}


if __name__ == "__main__":
    set_start_method("spawn")
    num_proc = 6

    chunk_size = 3000

    for i in range(0, len(user_hist_dataset), chunk_size):
        sub_dataset = user_hist_dataset.select(range(i, min(i+chunk_size, len(user_hist_dataset)), 1))
        
        updated_dataset = sub_dataset.map(
            gpu_computation,
            batched=True,
            batch_size=BATCH_SIZE,
            with_rank=True,
            num_proc=num_proc
        )

        user_id = updated_dataset['user']
        summary = updated_dataset['summary']
        df = pd.DataFrame({'user_id': user_id, 'summary': summary})
        if i == 0:
            df.to_csv('user_preference_recurrent.csv', index=False, header=True, mode='w')
        else:
            df.to_csv('user_preference_recurrent.csv', index=False, header=False, mode='a')
