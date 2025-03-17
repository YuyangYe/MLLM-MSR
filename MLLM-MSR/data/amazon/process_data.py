"""process amazon data

Need to download the review and meta for a category, unzip, and put jsonl data
into the `input` folder.
https://amazon-reviews-2023.github.io/#grouped-by-category

Example input and output:
.
├── input
│   ├── Magazine_Subscriptions.jsonl
│   └── meta_Magazine_Subscriptions.jsonl
├── process_data.py
└── processed
    ├── Magazine_Subscriptions_i_map.tsv
    ├── Magazine_Subscriptions_item_desc.tsv
    ├── Magazine_Subscriptions_u_i_pairs.tsv
    ├── Magazine_Subscriptions_u_map.tsv
    ├── Magazine_Subscriptions_user_items_negs.tsv
    ├── Magazine_Subscriptions_user_items_negs_test.csv
    ├── Magazine_Subscriptions_user_items_negs_train.csv
    └── Magazine_Subscriptions_user_items_negs_val.csv
"""

import os, csv
import pandas as pd
import json
import gzip
from collections import Counter
import numpy as np
from sklearn.model_selection import train_test_split

MAGAZINE_DATASET = 'Magazine_Subscriptions'
BEAUTY_DATASET = 'All_Beauty'
OUT_PATH = './processed'
UID, IID = 'user_id', 'item_id'
UMAP_FILE, IMAP_FILE,  = '{dataset}_u_map.tsv', '{dataset}_i_map.tsv'
U_I_PAIR_FILE = '{dataset}_u_i_pairs.tsv'
POS_NEG_FILE = '{dataset}_user_items_negs.tsv'
ITEM_DESC_FILE = '{dataset}_item_desc.tsv'
GZ_MODE = True

if GZ_MODE:
    IN_PATH = './gz'
    IN_SUFFIX = '.jsonl.gz'
else:
    IN_PATH = './input'
    IN_SUFFIX = '.jsonl'


RND_SEED = 2024040331

REVIEWS_JSONL_COLS = ['user_id', 'parent_asin', 'timestamp']
META_JSONL_COLS = ['parent_asin', 'image', 'title', 'summary']

def get_input_file(dataset, meta=False):
    fname = dataset + IN_SUFFIX
    if meta:
        fname = 'meta_' + fname
    return os.path.join(IN_PATH, fname)

def preprocess_review(review):
    return {k:review[k] for k in REVIEWS_JSONL_COLS}


def stream_lines(fname):
    if fname.endswith('.gz'):
        open_ = gzip.open
    else:
        open_ = open
    with open_(fname, mode='rt', encoding="utf8") as f:
        line = f.readline()
        while line:
            yield line
            line = f.readline()


def read_reviews_as_pd(fname):
    collected = []
    for line in stream_lines(fname):
        collected.append(preprocess_review(json.loads(line)))
    df = pd.DataFrame(collected)
    print(f'Before dropped: {df.shape}')
    df.dropna(subset=REVIEWS_JSONL_COLS, inplace=True)
    df.drop_duplicates(subset=REVIEWS_JSONL_COLS, inplace=True)
    print(f'After dropped: {df.shape}')
    df.rename(columns={'parent_asin':'item_id'}, inplace=True)
    return df


def find_invalid_freq_ids(df, field, max_num=np.inf, min_num=-1):
    inter_cnt = Counter(df[field].values)
    blocklist = {x for x,cnt in inter_cnt.items() if not (min_num <= cnt <= max_num)}
    return blocklist


def filter_by_k_core(df, core_req):
    min_u_num, min_i_num = core_req
    iteration = 0
    print(f'Calculating k-core {core_req}...')
    while True:
        ban_users = find_invalid_freq_ids(df, field=UID, min_num=min_u_num)
        ban_items = find_invalid_freq_ids(df, field=IID, min_num=min_i_num)
        if len(ban_users) == 0 and len(ban_items) == 0:
            print(f"{len(df.index)} rows left in (u={min_u_num},i={min_i_num})-core")
            break

        dropped_inter = pd.Series(False, index=df.index)
        dropped_inter |= df[UID].isin(ban_users)
        dropped_inter |= df[IID].isin(ban_items)
        print(f'\titeration {iteration}: {dropped_inter.sum()} dropped interactions',
             f"with {len(ban_users)} users banned and {len(ban_items)} items banned")
        df.drop(df.index[dropped_inter], inplace=True)
        iteration += 1

def reindex(df):
    df.reset_index(drop=True, inplace=True)

    uniq_users = pd.unique(df[UID])
    uniq_items = pd.unique(df[IID])

    # start from 0
    u_map = {k: i for i, k in enumerate(uniq_users)}
    i_map = {k: i for i, k in enumerate(uniq_items)}

    df[UID] = df[UID].map(u_map)
    df[IID] = df[IID].map(i_map)
    df[UID] = df[UID].astype(int)
    df[IID] = df[IID].astype(int)
    df.sort_values(by=[IID, 'timestamp'], inplace=True)
    return df, u_map, i_map

def neg_samples(df, neg=5, neg_multiplier=3):
    rng = np.random.default_rng(seed=202404040331)
    all_items = list(df[IID].unique())
    items_per_user = df.groupby(UID)[IID].unique().reset_index().rename(columns={IID: 'items'})
    items_per_user['samples'] = list(rng.choice(all_items, size=(len(items_per_user.index), neg_multiplier*neg), replace=True))
    user_neg = []
    for user, row in items_per_user.iterrows():
        samples = row['samples']
        items = row['items']
        neg_samples = set(samples) - set(items)
        if len(neg_samples) < neg:
            print(f"Warning: not enough negative samples for user {user}")
            extra_samples = rng.choice(all_items, size=2*neg_multiplier*neg, replace=True)
            neg_samples |= set(extra_samples) - set(items)
        user_neg.append(','.join(str(i) for i in list(neg_samples)[:5]))
    items_per_user['neg'] = user_neg
    items_per_user.drop(columns=['samples', 'items'], inplace=True)
    return items_per_user

def pos_samples(df, pos):
    # TODO(pezhu): decide the policy to select need to sort positive items by timestamp...
    item_frequency = df.groupby(IID).size().reset_index(name='frequency')
    freq = df.merge(item_frequency, on=IID).sort_values(by=[UID, 'frequency', 'timestamp'], ascending=False)
    # For each user, rank items by frequency, and pick the top `pos` items.
    # Break tie by original order.
    pos_df = freq[freq.groupby(UID)['frequency'].rank(method="first", ascending=False) <= pos]
    return pos_df.groupby(UID)[IID].agg(lambda x:','.join(str(i) for i in x)).reset_index().rename(columns={IID: 'pos'})


def user_items_negs(df, pos, neg):
    pos_df = pos_samples(df, pos)
    neg_df = neg_samples(df, neg)
    return pos_df.merge(neg_df, on=UID)


def split_tsv_by_user_id(tsv_file, df=None):
    if df is not None:
        df = pd.read_csv(tsv_file, delimiter='\t')
    users = df[UID].unique()
    train_users, test_users = train_test_split(users, test_size=0.2, random_state=RND_SEED)
    
    train_data = df[df[UID].isin(train_users)]
    test_data = df[df[UID].isin(test_users)]
    for kind, data in {'train':train_data,'test':test_data}.items():
        file_path = tsv_file[:-4] + f'_{kind}.csv'
        data.to_csv(file_path, sep='\t', header=False, index=False)
        print(f"\t{kind} File saved to {file_path}")

def save_reviews_to_csv(dataset, df, u_map, i_map):
    # save interactions
    u_i_pair_path = os.path.join(OUT_PATH, U_I_PAIR_FILE.format(dataset=dataset))
    df.to_csv(u_i_pair_path, sep='\t', index=False)
    print(f"saved file {u_i_pair_path}")
   
    # save u_map
    u_path = os.path.join(OUT_PATH, UMAP_FILE.format(dataset=dataset))
    (pd.DataFrame(list(u_map.items()), columns=['original', UID])
            .to_csv(u_path, sep='\t', index=False))
    print(f"saved file {u_path}")

    # save i_map
    i_path = os.path.join(OUT_PATH, IMAP_FILE.format(dataset=dataset))
    (pd.DataFrame(list(i_map.items()), columns=['original', IID])
            .to_csv(i_path, sep='\t', index=False))
    print(f"saved file {i_path}")
    
    # save user item negs
    pos_neg_path = os.path.join(OUT_PATH, POS_NEG_FILE.format(dataset=dataset))
    df2 = user_items_negs(df, pos=11, neg=5)
    df2.to_csv(pos_neg_path, sep='\t', index=False)
    print(f"saved file {pos_neg_path}")
    
    split_tsv_by_user_id(pos_neg_path, df2)

def process_reviews(dataset, core_req):
    df = read_reviews_as_pd(get_input_file(dataset))
    filter_by_k_core(df, core_req)
    df, u_map, i_map = reindex(df)

    save_reviews_to_csv(dataset, df, u_map, i_map)
    return df, u_map, i_map


def preprocess_meta(payload, i_map):
    parent_asin = payload['parent_asin']
    if parent_asin not in i_map:
        return
    meta = {}
    meta['item_id'] = i_map[parent_asin]
    for image in payload['images']:
        if 'large' in image:
            meta['image'] = image['large']
            break
    if 'description' in payload:
        meta['summary'] = payload['title'] + '. ' + '; '.join(payload['description'])
    return meta

def process_meta(i_map=None, dataset=MAGAZINE_DATASET):
    if i_map is None:
        i_path = os.path.join(OUT_PATH, IMAP_FILE.format(dataset=dataset))
        i_map = pd.read_csv(i_path, sep='\t', header=0)
    item_desc_path = os.path.join(OUT_PATH, ITEM_DESC_FILE.format(dataset=dataset))
    i_file = get_input_file(dataset, meta=True)
    headers = ['item_id', 'image', 'summary']

    with open(item_desc_path, 'w') as f2:
        f2.write('\t'.join(headers))
        f2.write('\n')
        for line in stream_lines(i_file):
            meta = preprocess_meta(json.loads(line), i_map)
            if meta:
                f2.write('\t'.join(str(meta.get(h, '')) for h in headers))
                f2.write('\n')
    print(f"saved file {item_desc_path}")


def process_dataset(dataset, core_req):
    i_map = process_reviews(dataset, core_req)[-1]
    process_meta(i_map, dataset)

# process_dataset(MAGAZINE_DATASET, (3,3))
# process_dataset('Baby_Products', (6,5))
process_dataset('Video_Games', (6,5))
# process_dataset('Sports_and_Outdoors', (6,5))
