from torch.utils.data import Dataset
from typing import Any, Dict
import random
from datasets import load_dataset, load_from_disk
import json

class LlavaDataset(Dataset):
    """
    PyTorch Dataset for LLaVa. This class takes a HuggingFace Dataset as input.

    Each row, consists of image path(png/jpg/jpeg) and ground truth data (json/jsonl/txt).
    """

    def __init__(
        self,
        dataset_name_or_path: str,
        split: str = "train",
        sort_json_key: bool = True,
    ):
        super().__init__()

        self.split = split
        self.sort_json_key = sort_json_key

        self.dataset = load_from_disk(dataset_name_or_path)[split]
        self.dataset_length = len(self.dataset)

        self.pt_token_sequences = []
        self.gt_token_sequences = []
        for sample in self.dataset:
            prompt = sample["prompt"]
            ground_truth = sample["ground_truth"]
            pt_jsons = json.dumps([prompt])
            gt_jsons = json.dumps([ground_truth])

            self.gt_token_sequences.append(
                [
                    self.json2token(
                        gt_json,
                        sort_json_key=self.sort_json_key,
                    )
                    for gt_json in gt_jsons  # load json from list of json
                ]
            )

            self.pt_token_sequences.append(
                [
                    self.json2token(
                        pt_json,
                        sort_json_key=self.sort_json_key,
                    )
                    for pt_json in pt_jsons  # load json from list of json
                ]
            )

    def json2token(self, obj: Any, sort_json_key: bool = True):
        """
        Convert an ordered JSON object into a token sequence
        """
        if type(obj) == dict:
            if len(obj) == 1 and "text_sequence" in obj:
                return obj["text_sequence"]
            else:
                output = ""
                if sort_json_key:
                    keys = sorted(obj.keys(), reverse=True)
                else:
                    keys = obj.keys()
                for k in keys:
                    output += (
                        fr"<s_{k}>"
                        + self.json2token(obj[k], sort_json_key)
                        + fr"</s_{k}>"
                    )
                return output
        elif type(obj) == list:
            return r"<sep/>".join(
                [self.json2token(item, sort_json_key) for item in obj]
            )
        else:
            obj = str(obj)
            return obj

    def __len__(self) -> int:
        return self.dataset_length

    def __getitem__(self, idx: int) -> Dict:
        """
        Returns one item of the dataset.

        Returns:
            image : the original Receipt image
            prompt_sequence : tokenized prompt sequence
            target_sequence : tokenized ground truth sequence
        """
        sample = self.dataset[idx]

        # inputs
        image = sample["image"]
        prompt_sequence = self.pt_token_sequences[idx]
        target_sequence = self.gt_token_sequences[idx]  # can be more than one, e.g., DocVQA Task 1

        return image, prompt_sequence, target_sequence

class LlavaDataset2(Dataset):
    """
    PyTorch Dataset for LLaVa. This class takes a HuggingFace Dataset as input.

    Each row, consists of image path(png/jpg/jpeg) and ground truth data (json/jsonl/txt).
    """

    def __init__(
        self,
        dataset_name_or_path: str,
        split: str = "train",
        sort_json_key: bool = True,
    ):
        super().__init__()

        self.split = split
        self.sort_json_key = sort_json_key

        self.dataset = load_from_disk(dataset_name_or_path)[split]
        self.dataset_length = len(self.dataset)

        self.pt_token_sequences = []
        self.gt_token_sequences = []
        for sample in self.dataset:
            prompt = sample["prompt"].strip()
            ground_truth = sample["ground_truth"].strip()
            self.gt_token_sequences.append(ground_truth)
            self.pt_token_sequences.append(prompt)

    def __len__(self) -> int:
        return self.dataset_length

    def __getitem__(self, idx: int) -> Dict:
        """
        Returns one item of the dataset.

        Returns:
            image : the original Receipt image
            prompt_sequence : tokenized prompt sequence
            target_sequence : tokenized ground truth sequence
        """
        sample = self.dataset[idx]

        # inputs
        image = sample["image"]
        prompt_sequence = self.pt_token_sequences[idx]
        target_sequence = self.gt_token_sequences[idx]  # can be more than one, e.g., DocVQA Task 1

        return image, prompt_sequence, target_sequence



