# MLLM-MSR
![framework (3)](https://github.com/user-attachments/assets/810ac195-3b6e-41a6-9717-f1e8d72b552f)

The code for the paper "Harnessing Multimodal Large Language Models for Multimodal Sequential Recommendation" (Accepted by AAAI-25).

## Dataset
This paper utilizes the following datasets:
- **Microlens Dataset**: [GitHub Repository](https://github.com/westlake-repl/MicroLens)
- **Amazon Review Dataset**: [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/#grouped-by-category)

The data processing scripts have been uploaded for preprocessing and structuring the datasets for model training and inference.

## Steps to Run

### 1. Inference
- First, generate summaries for item images using:
  ```bash
  python Inference/microlens/image_summary.py
  ```
- Next, obtain user preference information using:
  ```bash
  python Inference/microlens/preferece_inference_recurrent.py
  ```

### 2. Dataset Preparation
Before training or testing, datasets must be constructed:
- For training dataset creation:
  ```bash
  python MLLM-MSR/train/dataset_create.py
  ```
- For test dataset creation:
  ```bash
  python MLLM-MSR/test/multi_col_dataset.py
  ```

### 3. Training the Recommender Model
Use the following script to perform supervised fine-tuning (SFT) of the recommender model:
  ```bash
  python MLLM-MSR/train/train_llava_sft.py
  ```

### 4. Testing the Model
To evaluate the trained recommender model:
  ```bash
  python MLLM-MSR/test/test_with_llava_sft.py
  ```

## Citation

If you use the code of this repo, please cite our paper as,

```bibtex
@inproceedings{ye2025harnessing,
  title={Harnessing multimodal large language models for multimodal sequential recommendation},
  author={Ye, Yuyang and Zheng, Zhi and Shen, Yishan and Wang, Tianshu and Zhang, Hengruo and Zhu, Peijun and Yu, Runlong and Zhang, Kai and Xiong, Hui},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={12},
  pages={13069--13077},
  year={2025}
}
