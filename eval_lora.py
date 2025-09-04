import re
import json
from tqdm import tqdm
import collections
import argparse
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

lora_mapping = {
    "shanchen/math-500-jpsft-spanish-lora": ("shanchen/ds-limo-ja-500", "ES"),
    "shanchen/math-500-frsft-spanish-lora": ("shanchen/ds-limo-fr-250", "ES"),
    "shanchen/math-500-base-spanish-lora": ("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", "ES"),
    "shanchen/math-500-jpsft-french-lora": ("shanchen/ds-limo-ja-500", "FR"),
    "shanchen/math-500-sft-french-lora": ("shanchen/ds-limo-fr-250", "FR"),
    "shanchen/math-500-base-french-lora": ("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", "FR"),
    "shanchen/math-500-japanese-lora": ("shanchen/ds-limo-ja-full", "JA"),
}

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

def eval(output_dir, mname, lang, dataset, lang_think, store_answers, add_scores, reward_model_name):
    # Load reward model if scoring is enabled
    rm = None
    rm_tokenizer = None
    device = "cuda:0" if torch.cuda.is_available() else "cpu" # Use CUDA if available
    if add_scores:
        print(f"Loading reward model: {reward_model_name} on device: {device}")
        rm = AutoModelForSequenceClassification.from_pretrained(
            reward_model_name,
            torch_dtype=torch.bfloat16, # Use bfloat16 only on CUDA
            device_map=device, # device_map only works well on CUDA
            attn_implementation="flash_attention_2", # flash attention only on CUDA
            num_labels=1,
        )
        # If not using device_map (i.e., on CPU), explicitly move model to device
        if device != "cuda:0":
            rm.to(device)
        rm_tokenizer = AutoTokenizer.from_pretrained(reward_model_name)
        print("Reward model loaded.")

    accuracy = 0
    accuracy_hack = 0
    
    # outputs_2025/math-500-jpsft-spanish-lora_aime_combined:problem:answer_ES_think_ES_1.json
    fpath = f"{output_dir}/{mname.split('/')[1]}_{dataset}_{lang}_think_{lang_think}_1.json"
    with open(fpath, 'r') as f:
        instances = json.load(f)
    f.close()
    if dataset == "aime_combined": instances = instances[len(instances)//2:] # Only evaluate the second half (AIME25)

    # Track whether we need to update the file
    file_updated = False

    for ins in tqdm(instances):
        response = ins['response'][0]
        response_hack = ins['response_hack'][0]
        print(response)
        gold_answer = str(ins['answer']) if 'gpqa' not in dataset else ins['answer'][-2:-1]

        # print(ins['index'], '/', len(instances)-1, gold_answer, ':')
        ans = eval_regex(response)
        ans_hack = eval_regex(response_hack)
        # ans = eval_regex(response)
        # ans_hack = eval_regex(response_hack)
        # print(ans, ans_hack)

        # Add boolean flags indicating if an answer was extracted
        has_extracted_answer = bool(ans) # True if ans is not None and not empty
        has_extracted_answer_hack = bool(ans_hack) # True if ans_hack is not None and not empty
        ins['has_extracted_answer'] = has_extracted_answer
        ins['has_extracted_answer_hack'] = has_extracted_answer_hack
        # Ensure file is marked for update if we add these fields
        if not file_updated: # Only set if not already set by score/store flags
             file_updated = True # Mark file for update as we added boolean flags

        # Store extracted answers in the JSON if requested
        if store_answers:
            ins['extracted_answer'] = ans
            # Removed duplicate line here
            ins['extracted_answer_hack'] = ans_hack
            file_updated = True

        # Calculate scores if enabled
        if add_scores and rm and rm_tokenizer:
            try:
                # Format for reward model
                prompt = ins.get('prompt', '') # Use get with default to avoid KeyError if field missing
                response = ins.get('response', '')
                prompt_hack = ins.get('prompt_hack', '')
                response_hack = ins.get('response_hack', '')

                conv = [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}]
                conv_hack = [{"role": "user", "content": prompt_hack}, {"role": "assistant", "content": response_hack}]

                # Tokenize
                conv_tokenized = rm_tokenizer.apply_chat_template(conv, tokenize=True, return_tensors="pt").to(device)
                conv_hack_tokenized = rm_tokenizer.apply_chat_template(conv_hack, tokenize=True, return_tensors="pt").to(device)

                # Get scores
                with torch.no_grad():
                    score = rm(conv_tokenized).logits[0][0].item()
                    score_hack = rm(conv_hack_tokenized).logits[0][0].item()

                ins['score'] = score
                ins['score_hack'] = score_hack
                file_updated = True
            except Exception as e:
                print(f"Error scoring instance {ins.get('id', 'N/A')}: {e}")
                ins['score'] = None # Indicate scoring error
                ins['score_hack'] = None

        if ans:
            accuracy += (gold_answer in ans)
        if ans_hack:
            accuracy_hack += (gold_answer in ans_hack)

    # *** Temporarily disabled the following function ***
    file_updated = False
    # Write back the updated JSON if any modifications were made (answers stored, scores added, or boolean flags added)
    if file_updated:
        with open(fpath, 'w') as f:
            json.dump(instances, f, indent=2)
        update_message = []
        if store_answers: # This flag implies extracted answers were stored
            update_message.append("extracted answers")
        if add_scores: # This flag implies scores were added
            update_message.append("reward scores")
        # Always mention boolean flags were added if file was updated,
        # as they are added regardless of store/score flags now.
        update_message.append("extraction status flags")
        # Remove potential duplicates and join
        print(f"Updated {fpath} with {', '.join(sorted(list(set(update_message))))}")   
    # *** Temporarily disabled the above function ***

    with open('eval_lora.csv', 'a') as f:
        # f.write(f"{mname}\t{dataset}\t{lang}\t{lang_think}\t{accuracy}/{len(instances)}={round(100*accuracy/len(instances),2)}%, {accuracy_hack}/{len(instances)}={round(100*accuracy_hack/len(instances),2)}%\n")
        f.write(f"{mname}\t{dataset}\t{lang}\t{lang_think}\t{round(100*accuracy/len(instances),1)}%\t{round(100*accuracy_hack/len(instances),1)}%\n")
    f.close()
    # print(f"{mname} {dataset} {lang}: {accuracy}/{len(instances)}={round(100*accuracy/len(instances),2)}%, {accuracy_hack}/{len(instances)}={round(100*accuracy_hack/len(instances),2)}%")

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

    for dataset in datasets:
        for mname in mnames:
            _, lang = lora_mapping[mname]
            _, lang_think = lora_mapping[mname]

            eval(output_dir, mname, lang, dataset, lang_think, store_answers, add_scores, reward_model_name)
