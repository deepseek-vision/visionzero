from transformers import AutoModel, AutoTokenizer
import torch
import json
from tqdm import tqdm
from math_verify import parse, verify
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from PIL import Image


MODEL_PATH="OpenGVLab/InternVL2_5-2B"
BSZ=16 # reduce it if GPU OOM
OUTPUT_PATH="./logs/geoqa_test_internvl2_5_2b.json"
PROMPT_PATH="./prompts/geoqa_test_prompts.jsonl"


#####################################
#        InternVL preprocessing     #
#####################################
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

model = AutoModel.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True,
    device_map="auto",
).eval()

# default processer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, use_fast=False)

generation_config = dict(max_new_tokens=256, do_sample=False)

data = []
with open(PROMPT_PATH, "r") as f:
    for line in f:
        data.append(json.loads(line))


QUESTION_TEMPLATE = "<image>\n{Question} Output the thinking process in <think> </think> and final answer (number) in <answer> </answer> tags."


all_outputs = []  # List to store all answers

# Process data in batches
for i in tqdm(range(0, len(data), BSZ)):
    batch_data = data[i:i + BSZ]
    
    pixel_values = []
    questions = []
    for s in batch_data:
        pixel_values.append(load_image(s['image_path'], max_num=12).to(torch.bfloat16).to("cuda"))
        questions.append(QUESTION_TEMPLATE.format(Question=s['question']))

    num_patches_list = [x.size(0) for x in pixel_values]
    pixel_values = torch.cat(pixel_values, dim=0)

    responses = model.batch_chat(
        tokenizer, pixel_values,
        num_patches_list=num_patches_list,
        questions=questions,
        generation_config=generation_config
    )
    
    all_outputs.extend(responses)
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





