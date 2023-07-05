import json
import openai
import time
import torch
import subprocess
from fastchat.model.model_adapter import load_model, get_conversation_template

# openai.api_key = ''

def load_prompts(prompts_path):
    prompts = []
    with open(prompts_path, "r") as file:
        for line in file:
            data = json.loads(line)
            turns = [turn for turn in data["turns"]]
            prompts.append(turns)

    return prompts

def generate_gpt4(prompts, dump_path):
    for i, prompt in enumerate(prompts):
        messages = []
        responses = []

        for j, turn in enumerate(prompt):
            messages.append({"role": "user", "content": turn})
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=messages,
                max_tokens=512,
                )
            
            response = response['choices'][0]['message']['content']
            messages.append({"role": "assistant", "content": response})
            responses.append(response)
            print(f"Finished turn {j+1} of prompt {i+1}.")
            time.sleep(4)

        with open(dump_path, 'a') as file:
            data = json.dumps({"question_id": i+1, "model": "gpt-4", "turns": responses})
            file.write(data + '\n')
            file.flush()        

def new_chat(model_path):
    return get_conversation_template(model_path)

def get_score(prompt, response):
    

@torch.inference_mode()
def generate_student(model_path, prompts, dump_path):
    model, tokenizer = load_model(
        model_path,
        device="cuda",
        num_gpus=1,
        load_8bit=False,
        cpu_offloading=False,
        debug=False,
    )

    for i, prompt in enumerate(prompts):
        conv = new_chat(model_path)
        responses = []
        
        for j, turn in enumerate(prompt):
            conv.append_message(conv.roles[0], turn)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            input_ids = tokenizer([prompt]).input_ids

            output_ids = model.generate(
                torch.as_tensor(input_ids).cuda(),
                do_sample=True,
                temperature=1.0,
                max_new_tokens=512,
            )
        
            output_ids = output_ids[0][len(input_ids[0]) :]
            output = tokenizer.decode(output_ids, skip_special_tokens=True).strip()

            score = get_score(turn, output)
            
            responses.append(output)
            conv.messages[-1][-1] = output
            print(f"Finished turn {j+1} of prompt {i+1}. Score = {score}")

        with open(dump_path, 'a') as file:
            data = json.dumps({"question_id": i+1, "model": "vicuna-13b", "turns": responses})
            file.write(data + '\n')
            file.flush()    

@torch.inference_mode()
def generate_append_algorithm(model_path, prompts, dump_path):
    model, tokenizer = load_model(
        model_path,
        device="cuda",
        num_gpus=1,
        load_8bit=False,
        cpu_offloading=False,
        debug=False,
    )

    for i, prompt in enumerate(prompts):
        conv = new_chat(model_path)
        responses = []
        
        for j, turn in enumerate(prompt):
            conv.append_message(conv.roles[0], turn)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            input_ids = tokenizer([prompt]).input_ids

            output_ids = model.generate(
                torch.as_tensor(input_ids).cuda(),
                do_sample=True,
                temperature=1.0,
                max_new_tokens=512,
            )
        
            output_ids = output_ids[0][len(input_ids[0]) :]
            output = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
            
            responses.append(output)
            conv.messages[-1][-1] = output
            print(f"Finished turn {j+1} of prompt {i+1}.")

        with open(dump_path, 'a') as file:
            data = json.dumps({"question_id": i+1, "model": "vicuna-13b-append", "turns": responses})
            file.write(data + '\n')
            file.flush()   


def main():
    prompts_path = '/home/yusun/code/karan/data/loop/prompts.jsonl'
    gpt4_dump_path = '/home/yusun/code/karan/data/loop/responses/gpt4.jsonl'
    student_dump_path = '/home/yusun/code/karan/data/loop/responses/base.jsonl'
    model_path = 'lmsys/vicuna-13b-v1.3'

    prompts = load_prompts(prompts_path)

    generate_gpt4(prompts, gpt4_dump_path)
    generate_student(model_path, prompts, student_dump_path)


if __name__ == "__main__":
    main()