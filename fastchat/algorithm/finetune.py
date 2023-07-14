import os
import glob
import json
import torch
from fastchat.model.model_adapter import load_model, get_conversation_template
import subprocess
import time

parent_dir = '/nlp/scr/yusun/data'

def load_conversations(prompts_path):
    conversations = []
    with open(prompts_path, 'r+') as f:
        conversations = [json.loads(line) for line in f]
        
    return conversations

def format_parent():
    path = os.path.join(parent_dir, 'data', 'append.jsonl')
    prompts_path = os.path.join(parent_dir, 'data', 'prompts.jsonl')
    convos = []

    with open(prompts_path, 'r+') as f:
        prompts = [json.loads(line)['turns'][0] for line in f]

    with open(path, 'r+') as f:
        responses = [json.loads(line)['choices'][0]['turns'][0] for line in f]

    for prompt, response in zip(prompts, responses):
        convos.append([{'id': "identity_0", "conversations": [{"from": "human", "value": prompt},{"from": "gpt", "value": response}]}])

    for i, convo in enumerate(convos):
        with open(f'{parent_dir}/data/finetune/{i}.json', 'w') as file:
            json.dump(convo, file, indent=4)

def new_chat(model_path):
    return get_conversation_template(model_path)

def finetune(model, tokenizer, prompt, context, data_path, curr_i):
    conv = new_chat('lmsys/vicuna-7b-v1.3')
    responses = [context[0]]

    conv.append_message(conv.roles[0], prompt[0])
    conv.append_message(conv.roles[1], context[0])
    conv.append_message(conv.roles[0], prompt[1])
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
        
    input_ids = tokenizer(prompt).input_ids
    input_ids = [input_ids[-(2048 - 512 - 8):]]  
    output_ids = model.generate(
        torch.as_tensor(input_ids).cuda(),
        do_sample=True,
        temperature=0.8,
        max_new_tokens=512,
    )
    output_ids = output_ids[0][len(input_ids[0]):]
    response = tokenizer.decode(output_ids, skip_special_tokens=True).strip()

    responses.append(response)

    with open(data_path, 'a') as f:
        json.dump({'question_id': curr_i + 1, 'model_id': "vicuna-13b-v1.3", 'choices': [{"index": 0, "turns": responses}]}, f) 
        f.write('\n')

def main():
    model_path = os.path.join(parent_dir, 'model')
    conv_path = os.path.join(parent_dir, 'data', 'prompts.jsonl')
    context_path = os.path.join(parent_dir, 'data', 'append.jsonl')
    data_path = os.path.join(parent_dir, 'data', 'finetune.jsonl')
    num_gpus = 1
    
    if not os.path.exists(data_path):
        open(data_path, 'w').close() 

    prompts = load_conversations(conv_path)
    contexts = load_conversations(context_path)
    
    for i in range(80):
        # Finetune model by calling subprocess
        command = [
            "CUDA_VISIBLE_DEVICES=0,1,2,3",
            "python3",
            "-m",
            "torch.distributed.run",
            "--nproc_per_node=4",
            "--master_port=20001",
            f"{parent_dir}/FastChat/fastchat/train/train_mem.py",
            "--model_name_or_path", "lmsys/vicuna-7b-v1.3",
            "--data_path", f"{parent_dir}/data/finetune/{i}.json",
            "--bf16", "True",
            "--output_dir", model_path,
            "--num_train_epochs", "3",
            "--per_device_train_batch_size", "1",
            "--per_device_eval_batch_size", "1",
            "--gradient_accumulation_steps", "1",
            "--evaluation_strategy", "no",
            "--save_strategy", "steps",
            "--save_steps", "1200",
            "--save_total_limit", "10",
            "--learning_rate", "2e-5",
            "--weight_decay", "0.",
            "--warmup_ratio", "0.03",
            "--lr_scheduler_type", "cosine",
            "--logging_steps", "1",
            "--fsdp", "full_shard auto_wrap",
            "--fsdp_transformer_layer_cls_to_wrap", "'LlamaDecoderLayer'",
            "--tf32", "True",
            "--model_max_length", "2048",
            "--gradient_checkpointing", "True",
            "--lazy_preprocess", "True"
        ]
        subprocess.run(" ".join(command), shell=True, check=True)

        # Wait 10 seconds
        time.sleep(10)

        # Run inference on the follow up question for model
        model, tokenizer = load_model(
            model_path,
            device="cuda",
            num_gpus=num_gpus,
            load_8bit=False,
            cpu_offloading=False,
            debug=False,
        )
        finetune(model, tokenizer, prompts[i]["turns"], contexts[i]["choices"][0]["turns"], data_path, i)

        # Delete the model and clear GPUs
        del model
        del tokenizer
        torch.cuda.empty_cache()
        files = glob.glob(model_path + '/*')
        for f in files:
            os.remove(f)

        # Wait 10 seconds
        time.sleep(10)

if __name__ == "__main__":
    main()