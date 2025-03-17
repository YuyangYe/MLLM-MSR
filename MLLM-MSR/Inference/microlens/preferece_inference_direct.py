import torch
import transformers
from multiprocess import set_start_method
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset, load_dataset
import os
import pandas as pd
from torch.cuda.amp import autocast

os.environ['CURL_CA_BUNDLE'] = ''
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4,5,6,7"

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id, token='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')
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", token='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').eval()
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

pipelines = {}

prompt_template = "[INST] Below is a chronological list of videos previously watched by User {}:\n{}\nBased on the videos listed, please summarize the user's preferences in terms of both content and visual style within one continuous paragraph. Only provide information about the user's preferences; do not repeat details about the previously watched videos. Do not repeat the question in your answer.[/INST]"




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

ui_pair_path = "../../data/MicroLens-50k/sample_subset/user_items_negs.tsv"
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

prompt_dataset = user_hist_dataset.map(map_prompt)
prompt_dataset = prompt_dataset.remove_columns('items')
#prompt_dataset = prompt_dataset.select(range(96))
prompt_dataset.to_csv('user_prompt.csv')

def infer(prompt, rank):
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
    return outputs[0]["generated_text"][len(prompt):]

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
    summaries = [infer(prompt, rank) for prompt in batch['prompt']]
    return {'user': batch['user'], 'summary': summaries}


if __name__ == "__main__":
    set_start_method("spawn")
    num_proc = 6

    # 使用 dataset.map 进行 GPU 推理
    updated_dataset = prompt_dataset.map(
        gpu_computation,
        batched=True,
        batch_size=12,
        with_rank=True,
        num_proc=num_proc
    )

    user_id = updated_dataset['user']
    summary = updated_dataset['summary']
    df = pd.DataFrame({'user_id': user_id, 'summary': summary})
    df.to_csv('user_preference_direct.csv', index=False)