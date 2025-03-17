import torch
from multiprocess import set_start_method
from transformers import AutoProcessor, LlavaNextForConditionalGeneration
from datasets import load_dataset
from torchvision import transforms
from PIL import ImageOps
from torch.cuda.amp import autocast
import os
import pandas as pd

os.environ['CURL_CA_BUNDLE'] = ''
os.environ["CUDA_VISIBLE_DEVICES"]="2,3,4,5,6,7"

#model_id  = "lmms-lab/llama3-llava-next-8b"
model_id = "llava-hf/llava-v1.6-mistral-7b-hf"
model = LlavaNextForConditionalGeneration.from_pretrained(model_id,cache_dir = '/data1/share/.HF_cache/',attn_implementation="flash_attention_2", torch_dtype=torch.float16,
                                                        #   device_map="auto"
                                                          ).eval()

prompt = "[INST] <image>\nPlease describe this image, which is a cover of a video." \
         " Provide a detailed description in one continuous paragraph, including content information and visual features such as colors, objects, text," \
         " and any notable elements present in the image.[/INST]"


def add_image_file_path(example):
    file_path = example['image'].filename
    filename = os.path.splitext(os.path.basename(file_path))[0]
    example['item_id'] = filename
    return example

# img_dir = "../../inference_playground/microlens/microlens_50k_subset" #Change this to the real path of the image folder
img_dir = "../../data/MicroLens-50k/MicroLens-50k_covers"
dataset = load_dataset("imagefolder", data_dir=img_dir)
dataset = dataset.map(lambda x: add_image_file_path(x))
print(dataset)

processor = AutoProcessor.from_pretrained(model_id, return_tensors=torch.float16)


def gpu_computation(batch, rank):
    # Move the model on the right GPU if it's not there already
    device = f"cuda:{(rank or 0) % torch.cuda.device_count()}"
    model.to(device)

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

    # Your big GPU call goes here, for example:
    model_inputs = processor([prompt for i in range(len(batch['image']))], batch['image'], return_tensors="pt",padding=True).to(device)

    with torch.no_grad() and autocast():
        outputs = model.generate(**model_inputs, max_new_tokens=200)

    ans = processor.batch_decode(outputs, skip_special_tokens=True)
    ans = [a.split("[/INST]")[1] for a in ans]
    return {"summary": ans}

#f.close()

if __name__ == "__main__":
    set_start_method("spawn")
    updated_dataset = dataset.map(
        gpu_computation,
        batched=True,
        batch_size=8,
        with_rank=True,
        # num_proc=torch.cuda.device_count(),  # one process per GPU
        num_proc = 4
    )

    train_dataset = updated_dataset['train']
    item_id = train_dataset['item_id']
    summary = train_dataset['summary']
    df = pd.DataFrame({'item_id': item_id, 'summary': summary})
    df.to_csv('image_summary.csv', index=False)

