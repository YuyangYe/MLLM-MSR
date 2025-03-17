import torch
from multiprocess import set_start_method
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, BitsAndBytesConfig
from datasets import load_dataset, load_from_disk
from torchvision import transforms
from PIL import ImageOps
from torch.nn.functional import softmax
from torch.cuda.amp import autocast
import os
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score
from peft import PeftModel, PeftConfig



os.environ['CURL_CA_BUNDLE'] = ''
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

base_model_id = "llava-hf/llava-v1.6-mistral-7b-hf"
#peft_model_id = "/data1/share/LLaVA/llava-v1.6-mistral-7b-hf-lora"
#peft_model_id = "/data1/share/LLaVA/llava-v1.6-mistral-7b-hf-lora-distributed"
#peft_model_id = "/data1/share/LLaVA/llava-v1.6-mistral-7b-hf-lora-dist-e12-r16"
peft_model_id = "/data1/share/LLaVA/llava-v1.6-mistral-7b-hf-lora-dist-e8-r32"
#peft_model_id = "/data1/share/LLaVA/llava-v1.6-mistral-7b-hf-lora-recurrent-e4-r16"
config = PeftConfig.from_pretrained(peft_model_id)
model = LlavaNextForConditionalGeneration.from_pretrained(base_model_id, cache_dir='/data1/share/.HF_cache/',
                                                          attn_implementation="flash_attention_2",
                                                          torch_dtype=torch.float16,
                                                          #quantization_config=bnb_config
                                                          #device_map="auto"
                                                          )



#processor_id  = 'llava-hf/llava-v1.6-mistral-7b-hf'
processor = LlavaNextProcessor.from_pretrained(base_model_id, return_tensors=torch.float16)
processor.tokenizer.pad_token = processor.tokenizer.eos_token
#processor.tokenizer.padding_side = "right"

model = PeftModel.from_pretrained(model, peft_model_id).eval()
model.tie_weights()
print(f"PEFT model loaded")
#print(f"Running merge_and_unload")
#model = model.merge_and_unload()

dataset = load_from_disk("MicroLens-50k-test")
dataset = dataset.select(range(2100))
print(dataset)

processor.tokenizer.add_tokens(
    ["<|image|>", "<pad>"], special_tokens=True
)


Yes_id, No_id = processor.tokenizer.convert_tokens_to_ids('Yes'), processor.tokenizer.convert_tokens_to_ids('No')
yes_id, no_id = processor.tokenizer.convert_tokens_to_ids('yes'), processor.tokenizer.convert_tokens_to_ids('no')

def gpu_computation(batch, rank):
    device = f"cuda:{(rank or 0) % torch.cuda.device_count()}"
    model.to(device)
    yes_logits_batch, no_logits_batch = [], []

    batch_images = batch['image']

    max_width = max(img.width for img in batch_images)
    max_height = max(img.height for img in batch_images)

    padded_images = []
    for img in batch_images:
        if img.width == max_width and img.height == max_height:
            padded_images.append(img)
            continue
        else:
            delta_width = max_width - img.width
            delta_height = max_height - img.height

            padding = (
                delta_width // 2, delta_height // 2, delta_width - (delta_width // 2), delta_height - (delta_height // 2))

            new_img = ImageOps.expand(img, border=padding, fill='black')
            padded_images.append(new_img)

    batch['image'] = padded_images

    batch_size = len(batch['image'])
    model_inputs = processor(batch['prompt'], batch['image'], return_tensors="pt", padding=True).to(device)

    with torch.no_grad() and autocast():
        outputs = model.generate(**model_inputs, max_new_tokens=30, return_dict_in_generate=True, output_scores=True)

    scores = outputs['scores'][0]
    yes_probs = scores[:, Yes_id]
    no_probs = scores[:, No_id]


    sequences = outputs['sequences']
    pred_token = sequences[:, -3:]
    batch['output'] = processor.batch_decode(pred_token, skip_special_tokens=True)
    print(batch['output'])

    for i in range(batch_size):
        yes_logits_batch.append(yes_probs[i].item())
        no_logits_batch.append(no_probs[i].item())

    return {"yes_logits": yes_logits_batch, "no_logits": no_logits_batch}



def recall_at_k(y_true, y_prob, k):
    sorted_indices = np.argsort(-y_prob, axis=1)
    sorted_labels = np.take_along_axis(y_true, sorted_indices, axis=1)
    retrieved_positives = np.sum(sorted_labels[:, :k], axis=1)
    total_positives = np.ones_like(retrieved_positives)
    recall_scores = retrieved_positives / total_positives

    return np.mean(recall_scores)


def mrr_at_k(y_true, y_prob, k):
    sorted_indices = np.argsort(-y_prob, axis=1)
    sorted_labels = np.take_along_axis(y_true, sorted_indices, axis=1)
    reciprocal_ranks = np.zeros(y_true.shape[0])

    for i, labels in enumerate(sorted_labels[:, :k]):
        first_pos = np.where(labels == 1)[0]
        if first_pos.size > 0:
            reciprocal_ranks[i] = 1 / (first_pos[0] + 1)

    return np.mean(reciprocal_ranks)



def ndcg_at_k(y_true, y_prob, k):
    def dcg_at_k(scores, k):
        discounts = np.log2(np.arange(2, k + 2))  # 从第二个位置开始计算
        return np.sum((2 ** scores - 1) / discounts, axis=1)
    sorted_indices = np.argsort(-y_prob, axis=1)
    sorted_scores = np.take_along_axis(y_true, sorted_indices, axis=1)[:, :k]

    dcg_scores = dcg_at_k(sorted_scores, k)

    ideal_sorted_scores = np.sort(y_true, axis=1)[:, ::-1][:, :k]
    idcg_scores = dcg_at_k(ideal_sorted_scores, k)

    epsilon = 1e-10
    ndcg_scores = dcg_scores / (idcg_scores + epsilon)

    return np.mean(ndcg_scores)

if __name__ == "__main__":
    set_start_method("spawn")
    torch.cuda.empty_cache()
    updated_dataset = dataset.map(
        gpu_computation,
        batched=True,
        batch_size=6,
        with_rank=True,
        # num_proc=torch.cuda.device_count(),  # one process per GPU
        num_proc=6  # one process per GPU
    )
    updated_dataset = updated_dataset.sort("user")
    yes_logits = torch.tensor(updated_dataset['yes_logits'])
    no_logits = torch.tensor(updated_dataset['no_logits'])
    labels = np.array(updated_dataset['label'])
    yes_prob = torch.stack([no_logits, yes_logits], dim=1)
    yes_probs = F.softmax(yes_prob, dim=1)[:, 1].cpu().numpy()
    print("AUC: ", roc_auc_score(labels, yes_probs))

    #yes_probs = F.sigmoid(yes_prob)[:, 1].cpu().numpy()
    yes_probs = yes_probs.reshape(-1, 21)
    labels = labels.reshape(-1, 21)

    #print(yes_probs)
    #print(labels)

    y_preds = np.argmax(yes_probs, axis=1)

    print("Recall@3: ", recall_at_k(labels, yes_probs, 3))
    print("Recall@5: ", recall_at_k(labels, yes_probs, 5))
    print("Recall@10: ", recall_at_k(labels, yes_probs, 10))
    print("MRR@3: ", mrr_at_k(labels, yes_probs, 3))
    print("MRR@5: ", mrr_at_k(labels, yes_probs, 5))
    print("MRR@10: ", mrr_at_k(labels, yes_probs, 10))
    print("NDCG@3: ", ndcg_at_k(labels, yes_probs, 3))
    print("NDCG@5: ", ndcg_at_k(labels, yes_probs, 5))
    print("NDCG@10: ", ndcg_at_k(labels, yes_probs, 10))

