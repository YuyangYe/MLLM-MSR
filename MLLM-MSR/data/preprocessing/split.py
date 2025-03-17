import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def load_data(file_path):
    df = pd.read_csv(file_path, sep='\t', names=['user_id', 'item_list', 'neg_item_list'])
    df['item_list'] = df['item_list'].apply(lambda x: x.strip().split(','))
    df['neg_item_list'] = df['neg_item_list'].apply(lambda x: x.strip().split(','))
    return df


def generate_pairs(df, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, neg_sampling_train=1, neg_sampling_val=1,
                   neg_sampling_test=20):
    assert train_ratio + val_ratio + test_ratio == 1, "Ratios must sum to 1"

    # Splitting the data
    train_df, temp_df = train_test_split(df, test_size=1 - train_ratio, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=test_ratio / (test_ratio + val_ratio), random_state=42)

    def prepare_data(df, neg_sampling_rate, is_train=True):
        rows = []
        for _, row in df.iterrows():
            pos_item = row['item_list'][-1]
            neg_samples = np.random.choice(row['neg_item_list'], size=neg_sampling_rate, replace=False)

            # Positive sample
            rows.append([row['user_id'], pos_item.strip(), 1])

            # Negative samples
            if is_train:
                neg_samples = neg_samples[:1]  # 1:1 ratio in training
            for item in neg_samples:
                item.strip()
                rows.append([row['user_id'], item, 0])

        return pd.DataFrame(rows, columns=['user', 'item', 'label'])

    train_pairs = prepare_data(train_df, neg_sampling_train, is_train=True)
    val_pairs = prepare_data(val_df, neg_sampling_val, is_train=False)
    test_pairs = prepare_data(test_df, neg_sampling_test, is_train=False)

    return train_pairs, val_pairs, test_pairs

def save_datasets(train, val, test):
    train.to_csv('train_pairs.csv', index=False)
    val.to_csv('val_pairs.csv', index=False)
    test.to_csv('test_pairs.csv', index=False)

file_path = 'user_items_negs.tsv'
data = load_data(file_path)
train_pairs, val_pairs, test_pairs = generate_pairs(data)
save_datasets(train_pairs, val_pairs, test_pairs)