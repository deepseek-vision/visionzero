import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import json
from tqdm import tqdm
import re
from math_verify import parse, verify
from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images


MODEL_PATH="deepseek-ai/Janus-Pro-1B"
BSZ=16 # reduce it if GPU OOM
OUTPUT_PATH="./logs/geoqa_test_januspro_1b.json"
PROMPT_PATH="./prompts/geoqa_test_prompts.jsonl"


vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(MODEL_PATH)
tokenizer = vl_chat_processor.tokenizer

# load the model
device = "cuda"
vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    # attn_implementation="flash_attn_2",
    device_map=device,
    torch_dtype="bfloat16",
)

data = []
with open(PROMPT_PATH, "r") as f:
    for line in f:
        data.append(json.loads(line))


QUESTION_TEMPLATE = "{Question} Output the thinking process in <think> </think> and final answer (number) in <answer> </answer> tags."


all_outputs = []  # List to store all answers

# Process data in batches
for i in tqdm(range(0, len(data), BSZ)):
    batch_data = data[i:i + BSZ]
    
    prepare_inputs = []
    for s in batch_data:
        question = QUESTION_TEMPLATE.format(Question=s["question"])
        image = s['image_path']
        conversation = [
            {
                "role": "<|User|>",
                "content": f"<image_placeholder>\n{question}",
                "images": [image],
            },
            {"role": "<|Assistant|>", "content": ""},
        ]
        pil_image = load_pil_images(conversation)

        prepare_input = vl_chat_processor(
            conversations=conversation, images=pil_image, force_batchify=False
        )
        prepare_inputs.append(prepare_input)

    # batchify the inputs
    prepare_inputs = vl_chat_processor.batchify(prepare_inputs).to(vl_gpt.device)
    
    # run image encoder to get the image embeddings
    inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

    # run the model to get the response
    outputs = vl_gpt.language_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=256,
        do_sample=False,
        use_cache=True,
    )

    # no need to trim the outputs for janus
    outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    all_outputs.extend(outputs)

    print(f"Processed batch {i//BSZ + 1}/{(len(data) + BSZ - 1)//BSZ}")



final_output = []
correct_number = 0

for input_example, model_output in zip(data,all_outputs):
    original_output = model_output
    ground_truth = input_example['ground_truth']
    model_answer = parse(original_output) 

    # Count correct answers
    if model_answer is not None and float(verify(model_answer,parse(ground_truth)))>0:
        correct_number += 1
        is_correct = True
    else:
        is_correct = False
    
    try:
        result = {
            'question': input_example,
            'ground_truth': ground_truth,
            'model_output': original_output,
            'extracted_answer':str(model_answer[0]) if model_answer is not None else None,
            'is_correct':is_correct
        }

    except Exception as e:
        print("no answer parsed",e,model_answer)
        result = {
            'question': input_example,
            'ground_truth': ground_truth,
            'model_output': original_output,
            'extracted_answer':None,
            'is_correct':is_correct
        }



    final_output.append(result)


# Calculate and print accuracy
accuracy = correct_number / len(data) * 100
print(f"\nAccuracy: {accuracy:.2f}%")

# Save results to a JSON file
output_path = OUTPUT_PATH
with open(output_path, "w") as f:
    json.dump({
        'accuracy': accuracy,
        'results': final_output
    }, f, indent=2, ensure_ascii=False)

print(f"Results saved to {output_path}")





