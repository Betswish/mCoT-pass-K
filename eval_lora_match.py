import re
import json
from tqdm import tqdm
import argparse
from langdetect import DetectorFactory
DetectorFactory.seed = 0
from langdetect import detect_langs
import os

lora_mapping = {
    "shanchen/math-500-jpsft-spanish-lora": ("shanchen/ds-limo-ja-500", "ES"),
    "shanchen/math-500-frsft-spanish-lora": ("shanchen/ds-limo-fr-250", "ES"),
    "shanchen/math-500-base-spanish-lora": ("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", "ES"),
    "shanchen/math-500-jpsft-french-lora": ("shanchen/ds-limo-ja-500", "FR"),
    "shanchen/math-500-sft-french-lora": ("shanchen/ds-limo-fr-250", "FR"),
    "shanchen/math-500-base-french-lora": ("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", "FR"),
    "shanchen/math-500-japanese-lora": ("shanchen/ds-limo-ja-full", "JA"),
    "shanchen/math-500-base-japanese-lora": ("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", "JA"),
}

mnames = [ 
    "shanchen/math-500-jpsft-spanish-lora",
    "shanchen/math-500-frsft-spanish-lora",
    "shanchen/math-500-base-spanish-lora",
    "shanchen/math-500-jpsft-french-lora",
    "shanchen/math-500-sft-french-lora",
    "shanchen/math-500-base-french-lora",
    "shanchen/math-500-japanese-lora",
    "shanchen/math-500-base-japanese-lora",
    ]

datasets = [
    'aime_combined', 
    'gpqa_diamond_mc_multilingual', 
    'mmmlu'
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

def compute_matching(output_dir, mname, lang, dataset, lang_think, K):
    matching_rate_norm = 0
    fpath = f"{output_dir}/{mname.split('/')[1]}_{dataset}_{lang}_think_{lang_think}_32.json"
    instances = []
    with open(fpath, 'r') as f:
        for line in f:
            if line.strip(): instances.append(json.loads(line)) # skip empty lines
    f.close()
    
    print(f"Eval: {fpath}")
    for ins in tqdm(instances):
        responses = ins['response']
        for response in responses:
            gold_answer = str(ins['answer']) if 'gpqa' not in dataset else ins['answer'][-2:-1]
            lang_norm_list, lang_norm = detect_language(response.split('</think>')[0])

            # Calculate scores if enabled
            matching_rate_norm += (lang_norm == lang_think.lower())

    save_dir = 'results/'
    if not os.path.exists(save_dir): os.makedirs(save_dir, exist_ok=True)
    with open(f'{save_dir}/match.csv', 'a') as f:
        f.write(f"{mname}\t{dataset}\t{lang}\t{lang_think}\t{round(100*matching_rate_norm/(32*len(instances)),2)}%\n")
    f.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Model parameters
    parser.add_argument("--output_dir", type=str, default="outputs_2026", help="Loading from which directory")
    parser.add_argument("--K", type=int, default="1", help="Pass@K (currently not used)")

    args = parser.parse_args()

    output_dir        = args.output_dir


    for dataset in datasets:
        for mname in mnames:
            _, lang = lora_mapping[mname]
            _, lang_think = lora_mapping[mname]
            try:
                compute_matching(output_dir, mname, lang, dataset, lang_think, args.K)
            except Exception as e:
                continue
