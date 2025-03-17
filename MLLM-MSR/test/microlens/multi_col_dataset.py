from datasets import Dataset, Image
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

pair_file_path = "../../data/MicroLens-50k/Split/test_pairs.csv"
df = pd.read_csv(pair_file_path)
df['item'] = df['item'].astype(str)
df['user'] = df['user'].astype(str)

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


df = pd.merge(df, image_df, on="item")
df = pd.merge(df, item_title_df, on="item")
df = pd.merge(df, user_pref_df, on="user")

#prompt_text = "[INST]<image>\n As a vision-llm, you will be given the cover image and the title of a video and the summarized preference of a user, and your task is to predict whether the user would interact with the video. Please only response 'yes' or 'no' based on your judgement, do not include any other content including words, space, and punctuations in your response.\n " \
#              "Based on the previous interaction history, the user's preference can be summarized as: {}" \
#         "Please predict whether this user would interact with the video at the next opportunity. The video's title is'{}', and the given image is this video's cover? [/INST]"

prompt_text = "[INST]<image>\nBased on the previous interaction history, the user's preference can be summarized as: {}" \
              "Please predict whether this user would interact with the video at the next opportunity. The video's title is'{}', and the given image is this video's cover? " \
              "Please only response 'yes' or 'no' based on your judgement, do not include any other content including words, space, and punctuations in your response. [/INST]"


#prompt_text = "[INST] As a vision-llm, your task involves analyzing a video's cover image and title, alongside a summary of a user's preferences based on their interaction history. Respond with 'yes' or 'no' to indicate whether the user will interact with the video at their next opportunity. Please limit your response to only 'yes' or 'no', without including any additional content, words, or punctuation.\n" \
#              "<image>\nUser's summarized preferences based on past interactions: {}\n" \
#              "Will the user interact with the video titled '{}' and represented by the above given cover image at the next opportunity? [/INST]"


df['prompt'] = df.apply(lambda x: prompt_text.format(x['preference'], x['title']), axis=1)

df = df[['user', 'prompt', 'image', 'label']]
#print(df.head())

# 创建数据集并指定列类型
#dataset = Dataset.from_dict({"image": file_paths, "item": file_names})
dataset = Dataset.from_pandas(df)
dataset = dataset.cast_column("image", Image())
#dataset = dataset.select(range(2000))

# 检查数据集结构
print(dataset)
dataset.save_to_disk("MicroLens-50k-train-recurrent")
