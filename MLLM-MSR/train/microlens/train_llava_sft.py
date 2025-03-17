MAX_LENGTH = 1024
EPOCH = 4
LORA_R = 16
MODEL_ID = "llava-hf/llava-v1.6-mistral-7b-hf"
REPO_ID = "yeyuyang95/llava-v1.6-mistral-7b-hf-lora"
WANDB_PROJECT = "LLaVaNeXT"
WANDB_NAME = "llava-v1.6-mistral-7b-hf-lora"
SAVE_DIR = f"/data1/share/LLaVA/llava-v1.6-mistral-7b-hf-lora-recurrent-e{EPOCH}-r{LORA_R}"

from transformers import AutoProcessor
from transformers import BitsAndBytesConfig, LlavaNextForConditionalGeneration
import torch
from torch.utils.data import Dataset
from typing import Any, Dict
import random
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from load_llava_dataset import LlavaDataset, LlavaDataset2
from PIL import ImageOps
from lightning.pytorch.loggers import WandbLogger
import lightning as L
from torch.utils.data import DataLoader
import re
import os
from nltk import edit_distance
import numpy as np
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import logging

from huggingface_hub import HfApi

logging.getLogger("transformers").setLevel(logging.ERROR)


processor = AutoProcessor.from_pretrained(MODEL_ID)
processor.tokenizer.padding_side = "right"

USE_LORA = True
USE_QLORA = False

## Load model

# Three options for training, from the lowest precision training to the highest precision training:
# - QLora
# - Standard Lora
# - Full fine-tuning
if USE_QLORA or USE_LORA:
    if USE_QLORA:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
        )

        model = LlavaNextForConditionalGeneration.from_pretrained(
            MODEL_ID,
            cache_dir='/data1/share/.HF_cache/',
            torch_dtype=torch.float16,
            quantization_config=bnb_config,
        )
    else:
        model = LlavaNextForConditionalGeneration.from_pretrained(
            MODEL_ID,
            cache_dir='/data1/share/.HF_cache/',
            torch_dtype=torch.float16,
            _attn_implementation="flash_attention_2",
        )
else:
    # for full fine-tuning, we can speed up the model using Flash Attention
    # only available on certain devices, see https://github.com/Dao-AILab/flash-attention?tab=readme-ov-file#installation-and-features
    model = LlavaNextForConditionalGeneration.from_pretrained(
        MODEL_ID,
        cache_dir='/data1/share/.HF_cache/',
        torch_dtype=torch.float16,
        _attn_implementation="flash_attention_2",
    )


#Apply PEFT
def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['multi_modal_projector', 'vision_model']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

print(find_all_linear_names(model))


lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=find_all_linear_names(model),
    init_lora_weights="gaussian",
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

#Create PyTorch dataset
train_dataset = LlavaDataset2("MicroLens-50k-training-recurrent",  split="train", sort_json_key=False)
val_dataset = LlavaDataset2("MicroLens-50k-training-recurrent", split="validation", sort_json_key=False)

def resize_image(image_list):
    max_width = max(img.width for img in image_list)
    max_height = max(img.height for img in image_list)

    padded_images = []
    for img in image_list:
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

    return padded_images

def train_collate_fn(examples):
    images = []
    texts = []
    for example in examples:
        image, prompt_text, ground_truth = example
        images.append(image)
        prompt = f"[INST] <image>\n{prompt_text} [\INST] {ground_truth}"
        texts.append(prompt)
    images = resize_image(images)

    batch = processor(text=texts, images=images, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")

    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    batch["labels"] = labels

    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    pixel_values = batch["pixel_values"]
    image_sizes = batch["image_sizes"]
    labels = batch["labels"]

    return input_ids, attention_mask, pixel_values, image_sizes, labels

def eval_collate_fn(examples):
    images = []
    texts = []
    answers = []
    for example in examples:
        image, prompt_text, ground_truth = example
        images.append(image)
        prompt = f"[INST] <image>\n{prompt_text} [\INST]"
        texts.append(prompt)
        answers.append(ground_truth)
    images = resize_image(images)

    batch = processor(text=texts, images=images, return_tensors="pt", padding=True)

    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    pixel_values = batch["pixel_values"]
    image_sizes = batch["image_sizes"]

    return input_ids, attention_mask, pixel_values, image_sizes, answers

#Define PyTorch LightningModule
class LlavaModelPLModule(L.LightningModule):
    def __init__(self, config, processor, model):
        super().__init__()
        self.config = config
        self.processor = processor
        # self.model = model.to(self.device)
        self.model = model

        self.batch_size = config.get("batch_size")

    def training_step(self, batch, batch_idx):

        input_ids, attention_mask, pixel_values, image_sizes, labels = batch
        self.model.train()
        outputs = self.model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            pixel_values=pixel_values,
                            image_sizes=image_sizes,
                            labels=labels
                          )
        loss = outputs.loss
        batch_size = input_ids.size(0)

        self.log("train_loss", loss, batch_size=batch_size)

        return loss

    def validation_step(self, batch, batch_idx, dataset_idx=0):

        input_ids, attention_mask, pixel_values, image_sizes, answers = batch
        self.model.eval()
        # autoregressively generate token IDs
        generated_ids = self.model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                       pixel_values=pixel_values, image_sizes=image_sizes, max_new_tokens=MAX_LENGTH)
        # turn them back into text, chopping of the prompt
        # important: we don't skip special tokens here, because we want to see them in the output
        predictions = self.processor.batch_decode(generated_ids[:, input_ids.size(1):], skip_special_tokens=True)

        scores = []
        for pred, answer in zip(predictions, answers):
            pred = re.sub(r"(?:(?<=>) | (?=</s_))", "", pred)
            scores.append(edit_distance(pred, answer) / max(len(pred), len(answer)))

            if self.config.get("verbose", False) and len(scores) == 1:
                print(f"Prediction: {pred}")
                print(f"    Answer: {answer}")
                print(f" Normed ED: {scores[0]}")

        self.log("val_edit_distance", np.mean(scores), sync_dist=True)

        return scores

    def configure_optimizers(self):
        # you could also add a learning rate scheduler if you want
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.get("lr"))

        return optimizer

    def train_dataloader(self):
        return DataLoader(train_dataset, collate_fn=train_collate_fn, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(val_dataset, collate_fn=eval_collate_fn, batch_size=self.batch_size, shuffle=False, num_workers=4)

config = {"max_epochs": EPOCH,
          # "val_check_interval": 0.2, # how many times we want to validate during an epoch
          "check_val_every_n_epoch": 1,
          "gradient_clip_val": 1.0,
          "accumulate_grad_batches": 4,
          "lr": 2e-5,
          "batch_size": 1,
          # "seed":2022,
          "num_nodes": 1,
          "warmup_steps": 50,
          "result_path": "/data1/share/LLaVA",
          "verbose": True,
}

model_module = LlavaModelPLModule(config, processor, model)

#Define callbacks
api = HfApi()

class PushToHubCallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        print(f"Pushing model to the hub, epoch {trainer.current_epoch}")
        pl_module.model.push_to_hub(REPO_ID,
                                    commit_message=f"Training in progress, epoch {trainer.current_epoch}")

    def on_train_end(self, trainer, pl_module):
        print(f"Pushing model to the hub after training")
        pl_module.processor.push_to_hub(REPO_ID,
                                    commit_message=f"Training done")
        pl_module.model.push_to_hub(REPO_ID,
                                    commit_message=f"Training done")

class SaveToDiskCallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.global_rank == 0:
            print(f"Saving model to disk, epoch {trainer.current_epoch}")
            for name, param in pl_module.model.named_parameters():
                if "lora" in name and "layers.31" in name:
                    print(f"LoRA Layer {name}: {param.size()}")
            pl_module.model.save_pretrained(SAVE_DIR)
            pl_module.processor.save_pretrained(SAVE_DIR)

    def on_train_end(self, trainer, pl_module):
        if trainer.global_rank == 0:
            print(f"Saving model to disk after training")
            for name, param in pl_module.model.named_parameters():
                if "lora" in name and "layers.31" in name:
                    print(f"LoRA Layer {name}: {param.size()}")
            pl_module.model.save_pretrained(SAVE_DIR)
            pl_module.processor.save_pretrained(SAVE_DIR)



early_stop_callback = EarlyStopping(monitor="val_edit_distance", patience=3, verbose=False, mode="min")

checkpoint_callback = ModelCheckpoint(
    dirpath='/data1/share/LLaVA/',  # 指定保存路径
    filename='llava-v1.6-mistral-7b-lora-test',  # 指定文件名格式
    save_top_k=1,
    verbose=True,
    #monitor='val_loss',
    mode='min'
)
#Train!
#wandb_logger = WandbLogger(project=WANDB_PROJECT, name=WANDB_NAME)

trainer = L.Trainer(
        accelerator="gpu",
        devices=6,
        strategy='deepspeed_stage_2',
        max_epochs=config.get("max_epochs"),
        accumulate_grad_batches=config.get("accumulate_grad_batches"),
        check_val_every_n_epoch=config.get("check_val_every_n_epoch"),
        gradient_clip_val=config.get("gradient_clip_val"),
        precision="16-mixed",
        log_every_n_steps=10,
        limit_val_batches=5,
        num_sanity_val_steps=0,
        #strategy="ddp_find_unused_parameters_true",
        #logger=wandb_logger,
        #callbacks=[PushToHubCallback(), early_stop_callback],
        callbacks=[SaveToDiskCallback()]
)

trainer.fit(model_module)