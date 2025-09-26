import re
import json
from tqdm import tqdm
import argparse
from pass_at_k import pass_at_k

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
    
def eval_regex(response):
    # Comprehensive pattern list, now capturing any content inside the box
    patterns = [
    # \box{content}
    # r"\\box\{([^}]+?)\}",

    # \box content \box   — require at least one non‐whitespace char
    # r"\\box\s+(\S.+?\S)\s+\\box",

    # \boxed{content}
    r"\\boxed\{([^}]+?)\}",

    # Answer: content  (anchored at line start)
    # r"(?m)^Answer:\s*(\S.+?)\s*$",

    # “xxx is correct”  — non‐greedy and require at least one word char
    # r"(\w.+?)\s+is correct",

    # Final Answer: \box content
    # r"(?m)^Final Answer:\s*\\box\s+(\S.+?)\s*$",
    ]
    for pattern in patterns:
        match = re.findall(pattern, response)
        if match:
            if 'A' in match and 'B' in match and 'C' in match and 'D' in match:
                for i in ['A', 'B', 'C', 'D']:
                    match.remove(i)
            return match
    return None

def eval(output_dir, mname, lang, dataset, lang_think, K):
    # Load reward model if scoring is enabled
    rm = None
    rm_tokenizer = None

    accuracy = 0
    
    # outputs_2025/math-500-jpsft-spanish-lora_aime_combined:problem:answer_ES_think_ES_1.json
    fpath = f"{output_dir}/{mname.split('/')[1]}_{dataset}_{lang}_think_{lang_think}_32.json"
    instances = []
    with open(fpath, 'r') as f:
        for line in f:
            if line.strip(): instances.append(json.loads(line)) # skip empty lines
    f.close()

    print(f"Eval: {fpath}")
    for ins in tqdm(instances):
        gold_answer = str(ins['answer']) if 'gpqa' not in dataset else ins['answer'][-2:-1]
        responses = ins['response'] # length of K
        
        num_total_samples_n = len(responses)
        num_correct_samples_c = 0
        for response in responses:
            # print(ins['index'], '/', len(instances)-1, gold_answer, ':')
            ans = eval_regex(response)
            if ans and gold_answer in ans: num_correct_samples_c += 1
            
        accuracy += pass_at_k(num_total_samples_n, num_correct_samples_c, K)


    with open(f'eval_lora.csv', 'a') as f:
        # f.write(f"{mname}\t{dataset}\t{lang}\t{lang_think}\t{accuracy}/{len(instances)}={round(100*accuracy/len(instances),2)}%, {accuracy_hack}/{len(instances)}={round(100*accuracy_hack/len(instances),2)}%\n")
        f.write(f"Pass@{K}: {mname}\t{dataset}\t{lang}\t{lang_think}\t{round(100*accuracy/len(instances),1)}%\n")
    f.close()
    # print(f"{mname} {dataset} {lang}: {accuracy}/{len(instances)}={round(100*accuracy/len(instances),2)}%, {accuracy_hack}/{len(instances)}={round(100*accuracy_hack/len(instances),2)}%")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Model parameters
    parser.add_argument("--output_dir", type=str, default="outputs_2026", help="Loading from which directory")
    parser.add_argument("--K", type=int, nargs='+', default=[1, 5, 10], help="Pass@K")
    args = parser.parse_args()

    output_dir        = args.output_dir



    for K in args.K:
        for dataset in datasets:
            for mname in mnames:
                _, lang = lora_mapping[mname]
                _, lang_think = lora_mapping[mname]

                try:
                    eval(output_dir, mname, lang, dataset, lang_think, K)
                except Exception as e:
                    continue
