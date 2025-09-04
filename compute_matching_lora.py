import re
import json
from tqdm import tqdm
import collections
import argparse
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from langdetect import DetectorFactory
DetectorFactory.seed = 0
from langdetect import detect_langs

lora_mapping = {
    "shanchen/math-500-jpsft-spanish-lora": ("shanchen/ds-limo-ja-500", "ES"),
    "shanchen/math-500-frsft-spanish-lora": ("shanchen/ds-limo-fr-250", "ES"),
    "shanchen/math-500-base-spanish-lora": ("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", "ES"),
    "shanchen/math-500-jpsft-french-lora": ("shanchen/ds-limo-ja-500", "FR"),
    "shanchen/math-500-sft-french-lora": ("shanchen/ds-limo-fr-250", "FR"),
    "shanchen/math-500-base-french-lora": ("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", "FR"),
    "shanchen/math-500-japanese-lora": ("shanchen/ds-limo-ja-full", "JA"),
}

mnames = [ 
    "shanchen/math-500-jpsft-spanish-lora",
    "shanchen/math-500-frsft-spanish-lora",
    # "shanchen/math-500-base-spanish-lora",
    # "shanchen/math-500-jpsft-french-lora",
    # "shanchen/math-500-sft-french-lora",
    # "shanchen/math-500-base-french-lora",
    # "shanchen/math-500-japanese-lora",
    ]

datasets = [
    # 'aime_combined', 
    'gpqa_diamond_mc_multilingual', 
    # 'mgsm'
    ]


def detect_language(text):
    try:
        text = text.strip()
        res_detect = dict()
        detections = detect_langs(text)
        for i in detections:
            current_lang = i.lang
            if current_lang == 'zh-cn': current_lang = 'zh'
            res_detect[current_lang] = i.prob
        top1 = detections[0].lang
        if top1 == 'zh-cn': top1 = 'zh'
        return res_detect, top1
    except Exception as e:
        return None, None

def compute_matching(output_dir, mname, lang, dataset, lang_think, store_answers, add_scores, reward_model_name):
    # Load reward model if scoring is enabled
    rm = None
    rm_tokenizer = None
    device = "cuda:0" if torch.cuda.is_available() else "cpu" # Use CUDA if available

    matching_rate_norm = 0
    matching_rate_hack = 0
    fpath = f"{output_dir}/{mname.split('/')[1]}_{dataset}_{lang}_think_{lang_think}_1.json"
    with open(fpath, 'r') as f:
        instances = json.load(f)
    f.close()

    for ins in tqdm(instances):
        response = ins['response'][0]
        response_hack = ins['response_hack'][0]
        gold_answer = str(ins['answer']) if 'gpqa' not in dataset else ins['answer'][-2:-1]

        lang_norm_list, lang_norm = detect_language(response.split('</think>')[0])
        lang_hack_list, lang_hack = detect_language(response_hack.split('</think>')[0])

        # Calculate scores if enabled
        matching_rate_norm += (lang_norm == lang_think.lower())
        matching_rate_hack += (lang_hack == lang_think.lower())

    with open('matching_lora.csv', 'a') as f:
        f.write(f"{mname}\t{dataset}\t{lang}\t{lang_think}\t{round(100*matching_rate_norm/len(instances),2)}%\t{round(100*matching_rate_hack/len(instances),2)}%\n")
    f.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Model parameters
    parser.add_argument("--output_dir", type=str, default="outputs_2025", help="Loading from which directory")
    parser.add_argument("--store", action="store_true", help="Store extracted answers in JSON files")
    parser.add_argument("--add_scores", action="store_true", help="Add reward model scores to JSON files")
    parser.add_argument("--reward_model_name", type=str, default="Skywork/Skywork-Reward-Gemma-2-27B-v0.2", help="Reward model name")

    args = parser.parse_args()

    output_dir        = args.output_dir
    store_answers     = args.store
    add_scores        = args.add_scores
    reward_model_name = args.reward_model_name


    for dataset in datasets:
        for mname in mnames:
            _, lang = lora_mapping[mname]
            _, lang_think = lora_mapping[mname]
            compute_matching(output_dir, mname, lang, dataset, lang_think, store_answers, add_scores, reward_model_name)

