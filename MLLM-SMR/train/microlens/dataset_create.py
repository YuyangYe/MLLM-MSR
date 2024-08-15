from datasets import Dataset, Image, DatasetDict
from pathlib import Path
import pandas as pd
import os

os.environ['CURL_CA_BUNDLE'] = ''
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4,5,6,7"


def get_file_full_paths_and_names(folder_path):
    folder_path = Path(folder_path)
    full_paths = []
    file_names = []
    for file_path in folder_path.glob('*'):
        if file_path.is_file():
            full_paths.append(str(file_path.absolute()))
            file_names.append(file_path.stem)  # 使用.stem获取不带扩展名的文件名
    return full_paths, file_names

train_pair_file_path = "../../data/MicroLens-50k/Split/train_pairs.csv"
df_train = pd.read_csv(train_pair_file_path)
df_train['item'] = df_train['item'].astype(str)
df_train['user'] = df_train['user'].astype(str)

val_pair_file_path = "../../data/MicroLens-50k/Split/val_pairs.csv"
df_val = pd.read_csv(val_pair_file_path)
df_val['item'] = df_val['item'].astype(str)
df_val['user'] = df_val['user'].astype(str)


user_pref_file_path = "../../inference/Microlens/user_preference_recurrent.csv"
user_pref_df = pd.read_csv(user_pref_file_path, header=None, names=["user", "preference"])
user_pref_df['user'] = user_pref_df['user'].astype(str)


item_title_file_path = "../../data/MicroLens-50k/MicroLens-50k_titles.csv"
item_title_df = pd.read_csv(item_title_file_path, header=None, names=["item", "title"])
item_title_df['item'] = item_title_df['item'].astype(str)


folder_path = "../../data/MicroLens-50k/MicroLens-50k_covers"
file_paths, file_names = get_file_full_paths_and_names(folder_path)
image_df = pd.DataFrame({"image": file_paths, "item": file_names})
image_df['item'] = image_df['item'].astype(str)


df_train = pd.merge(df_train, image_df, on="item")
df_train = pd.merge(df_train, item_title_df, on="item")
df_train = pd.merge(df_train, user_pref_df, on="user")

df_val = pd.merge(df_val, image_df, on="item")
df_val = pd.merge(df_val, item_title_df, on="item")
df_val = pd.merge(df_val, user_pref_df, on="user")

prompt_text = "Based on the previous interaction history, the user's preference can be summarized as: {}" \
              "Please predict whether this user would interact with the video at the next opportunity. The video's title is'{}', and the given image is this video's cover? " \
              "Please only response 'yes' or 'no' based on your judgement, do not include any other content including words, space, and punctuations in your response."

#prompt_text = "[INST] As a vision-llm, your task involves analyzing a video's cover image and title, alongside a summary of a user's preferences based on their interaction history. Respond with 'yes' or 'no' to indicate whether the user will interact with the video at their next opportunity. Please limit your response to only 'yes' or 'no', without including any additional content, words, or punctuation.\n" \
#              "<image>\nUser's summarized preferences based on past interactions: {}\n" \
#              "Will the user interact with the video titled '{}' and represented by the above given cover image at the next opportunity? [/INST]"


df_train['prompt'] = df_train.apply(lambda x: prompt_text.format(x['preference'], x['title']), axis=1)
df_train['ground_truth'] = df_train.apply(lambda x: 'Yes' if x['label'] == 1 else 'No', axis=1)
df_train = df_train[['prompt', 'image', 'ground_truth']]

df_val['prompt'] = df_val.apply(lambda x: prompt_text.format(x['preference'], x['title']), axis=1)
df_val['ground_truth'] = df_val.apply(lambda x: 'Yes' if x['label'] == 1 else 'No', axis=1)
df_val = df_val[['prompt', 'image', 'ground_truth']]

train_dataset = Dataset.from_pandas(df_train)
train_dataset = train_dataset.cast_column("image", Image())
train_dataset = train_dataset.select(range(25000))
train_dataset = train_dataset.shuffle(seed=2024)

val_dataset = Dataset.from_pandas(df_val)
val_dataset = val_dataset.cast_column("image", Image())
val_dataset = val_dataset.select(range(1000))

dataset = DatasetDict({"train": train_dataset, "validation": val_dataset})
print(dataset)
dataset.save_to_disk("MicroLens-50k-training-recurrent")
