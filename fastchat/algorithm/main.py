import os
import json
import openai
import torch
from fastchat.model.model_adapter import load_model, get_conversation_template

# openai.api_key = ''

def load_conversations(prompts_path):
    conversations = []
    with open(prompts_path, 'r+') as f:
        conversations = [json.loads(line) for line in f]
        
    return conversations

def new_chat(model_path):
    return get_conversation_template(model_path)

def algo(conversation, model_path, num_gpus, data_path):
    model, tokenizer = load_model(
        model_path,
        device="cuda",
        num_gpus=num_gpus,
        load_8bit=False,
        cpu_offloading=False,
        debug=False,
    )

    conv = new_chat(model_path)
    responses = []

    for i, turn in enumerate(conversation['turns']):
        conv.append_message(conv.roles[0], turn)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        # Generate student response.
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

        # Determine confidence and generate parent response.
        if i == 0:
            messages = [{"role": "system", "content": conv.system}]
            messages = [{"role": message[0].lower(), "content": message[1]} for message in conv.messages if message[1] is not None]
            messages.append({"role": "user", "content": turn})
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=messages,
                max_tokens=512, ## NOTE: This will not work for future conversations. Need to figure out the token cut off.
            )
            response = response['choices'][0]['message']['content']

            # Finetune on parent response. NOTE: TO DO.
        
        conv.update_last_message(response)
        responses.append(response)

    with open(data_path, 'a') as f:
        json.dump({'question_id': conversation['question_id'], 'model_id': "vicuna-13b-v1.3", 'choices': [{"index": 0, "turns": responses}]}, f) 
        f.write('\n')

def main():
    model_path = 'lmsys/vicuna-7b-v1.3'
    conv_path = '/nlp/scr/yusun/data/data/prompts.jsonl'
    data_path = '/nlp/scr/yusun/data/data/base.jsonl'
    num_gpus = 1

    if not os.path.exists(data_path):
        open(data_path, 'w').close() 

    conversations = load_conversations(conv_path)
    for conversation in conversations:
        algo(conversation, model_path, num_gpus, data_path)

if __name__ == "__main__":
    main()