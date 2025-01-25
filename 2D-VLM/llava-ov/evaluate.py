
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
import torch
from PIL import Image
import json
import os
import re
from word2number import w2n
import argparse
import pandas as pd

def save_json(data, path):
    with open(path, "w") as file:
        json.dump(data, file, indent=4)
def read_csv(path):
    with open(path, "r") as file:
        data = file.readlines()
    return data

# Function to extract all non-NaN values for each scene_id and return as a list
def extract_non_nan_values(df):
    extracted_values = {}

    # Iterate over each row and extract the values
    for index, row in df.iterrows():
        scene_id = row['scene_id']
        if pd.notna(scene_id):
            # Create a dictionary to hold the non-NaN values for the current scene_id
            scene_values = {}

            # Extract non-NaN values for 'Front', 'Back', 'Left', 'Right'
            for direction in ['Front', 'Back', 'Left', 'Right']:
                value = row[direction]
                if pd.notna(value):
                    scene_values[direction] = value

            # Add the non-NaN values for the current scene_id to the overall dictionary
            if scene_values:
                extracted_values[scene_id] = scene_values

    return extracted_values

# Load the Excel file
# file_path = "dataset/ContextQA/Axis Definition.xlsx"
columns_to_load = ["scene_id"]  # Change if needed

# df = pd.read_excel(file_path, sheet_name='Sheet1', engine='openpyxl')


parser = argparse.ArgumentParser()
parser.add_argument("-f", "--data_path", type=str, default="reason3d/recategorize_test_data.json", help="Path to the JSON file containing the test data")
parser.add_argument("-m", "--model_id", type=str, default="llava-hf/llava-onevision-qwen2-7b-ov-hf", help="Model ID of the model to be evaluated")

args = parser.parse_args()

data = json.load(open(args.data_path))
# data = {k: data[k] for k in list(data.keys())[300:400]}

processor = AutoProcessor.from_pretrained(args.model_id)

if'72b' in args.model_id:
    load_in_4bit = True
else:
    load_in_4bit = False

from transformers import BitsAndBytesConfig


quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model = LlavaOnevisionForConditionalGeneration.from_pretrained(
    args.model_id, 
    use_flash_attention_2=True,
    quantization_config=quantization_config, 
    device_map="auto"
)

# model.to("cuda:0")

# Load a local image


images_dir = "dataset/2D_VLM_data/top_view_no_label"

def convert_words_to_digits(text):
    words = text.split()
    converted_words = []
    for word in words:
        try:
            # Attempt to convert the word to a number
            number = w2n.word_to_num(word)
            converted_words.append(str(number))
        except ValueError:
            # If the word is not a number, keep it as is
            converted_words.append(word)
    return ' '.join(converted_words)
    
def normalize_text(text):
    # Convert to lowercase and remove punctuation except digits and letters
    text = text.replace('To the', '').lower()
    if text.startswith('zero') or text.startswith('one') or text.startswith('two') or text.startswith('three') or text.startswith('four') or text.startswith('five') or text.startswith('six') or text.startswith('seven') or text.startswith('eight') or text.startswith('nine'):
        text = text.split(' ')[0]
        
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
    
    # Strip extra whitespace
    text = text.strip()
    
    # Convert words to digits (e.g., "one" to "1")
    text = convert_words_to_digits(text)
    
    # Remove articles (optional, depending on context)
    text = re.sub(r'\b(a|an|the)\b', '', text)
    
    return text
    
def f1_score(predicted, reference):
    pred_tokens = set(predicted.split())
    ref_tokens = set(reference.split())

    common_tokens = pred_tokens.intersection(ref_tokens)
    if len(common_tokens) == 0:
        return 0

    precision = len(common_tokens) / len(pred_tokens)
    recall = len(common_tokens) / len(ref_tokens)
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

def partial_match_score(predicted, reference):
    pred_tokens = predicted.split()
    ref_tokens = reference.split()
    common_tokens = set(pred_tokens).intersection(set(ref_tokens))
    return len(common_tokens) / len(ref_tokens) if len(ref_tokens) > 0 else 0

# Text prompt
# template = '''
# Given a top-view of a 3D scene, mentally rotate the image to align with the specified orientation.

# Scene Orientation: {}

# Now, given a context change, imagine how the scene would look after the change has been applied. Then, answer a question based on the changed scene.

# Context Change: {}
# Question: {}

# The answer should be a single word or short phrase.

# The answer is:
# '''

template = '''
Given a top-view of a 3D scene and a context change, imagine how the scene would look after the change has been applied. Then, answer a question based on the changed scene.

Context Change: {}
Question: {}

The answer should be a single word or short phrase.

The answer is:
'''

# Additional imports for defining sub-question types
def get_sub_question_type(question):
    if question.lower().startswith("what"):
        return "What"
    elif question.lower().startswith("is"):
        return "Is"
    elif question.lower().startswith("how"):
        return "How"
    elif question.lower().startswith("can"):
        return "Can"
    elif question.lower().startswith("which"):
        return "Which"
    elif question.lower().startswith("does"):
        return "Does"
    elif question.lower().startswith("are"):
        return "Are"
    elif question.lower().startswith("where"):
        return "Where"
    else:
        return "Others"
    
# Metrics initialization per question type
total_questions = 0
exact_matches = 0
partial_match_scores = []
total_questions_per_type = {}
exact_matches_per_type = {}
partial_match_scores_per_type = {}

df = pd.read_excel("dataset/Axis Definition.xlsx", sheet_name='Sheet1', engine='openpyxl')

# Main loop
for scene_id, changes_list in list(data.items()):
    image_path = os.path.join(images_dir, f"{scene_id}.png")
    local_image = Image.open(image_path)
            
    scene_orientation = extract_non_nan_values(df[df['scene_id'] == scene_id])
    scene_orientation = " ".join(
        f"The {item} was located at the {direction.lower()} of the scene."
        for scene_id, directions in scene_orientation.items()
        for direction, item in directions.items()
    )
    for i, changes in enumerate(changes_list):
        context_change = changes['context_change']
        question_answers = changes['questions_answers']
        
        for j, qa in enumerate(question_answers):
            question_type = qa['question_type']
            question = qa['question']
            answer = qa['answer']
            
            text_prompt = template.format(context_change, question)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": local_image},
                        {"type": "text", "text": text_prompt},
                    ],
                }
            ]
            
            prompt = processor.apply_chat_template(messages, add_generation_prompt=True)

            inputs = processor(images=local_image, text=prompt, return_tensors="pt").to("cuda:0", torch.float16)

            # Move inputs to GPU
            inputs = {k: v.to("cuda:0") for k, v in inputs.items()}

            # Inference: Generation of the outputs
            generated_ids = model.generate(**inputs, max_new_tokens=128)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
            ]
            
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0].strip()

            qa['predicted_answer'] = output_text
            
            # Metrics calculation
            predicted_answer = normalize_text(output_text)
            reference_answer = normalize_text(answer)
            
            print(f'Processed scene {scene_id}, change {i + 1}, question {j + 1}')
            
            # Initialize metrics for new question types
            if question_type not in total_questions_per_type:
                total_questions_per_type[question_type] = 0
                exact_matches_per_type[question_type] = 0
                partial_match_scores_per_type[question_type] = []

            # Exact Match
            if predicted_answer == reference_answer:
                exact_matches += 1  
                exact_matches_per_type[question_type] += 1

            # Partial Match Score
            partial_match = partial_match_score(predicted_answer, reference_answer)
            partial_match_scores.append(partial_match)
            partial_match_scores_per_type[question_type].append(partial_match)

            total_questions += 1
            total_questions_per_type[question_type] += 1
    print('labelled result without axis')
    save_json(data, f"dataset/contextvqa_{args.model_id.split('/')[1]}_no_label_align.json")
    
# Calculate average metrics for each question type
for question_type in total_questions_per_type:
    exact_match_score_per_type = (exact_matches_per_type[question_type] / total_questions_per_type[question_type]) * 100
    average_partial_match_per_type = sum(partial_match_scores_per_type[question_type]) / len(partial_match_scores_per_type[question_type]) * 100
    
    # Print results for each question type
    print(f"Question Type: {question_type}")
    print(f"  Exact Match Score: {exact_match_score_per_type:.2f}%")
    print(f"  Partial Match Score: {average_partial_match_per_type:.2f}%")


# Calculate overall average metrics
exact_match_score = (exact_matches / total_questions) * 100
average_partial_match_score = sum(partial_match_scores) / len(partial_match_scores) * 100

# Print overall results
print("\nOverall Metrics:")
print(f"Exact Match Score: {exact_match_score:.2f}%")
print(f"Partial Match Score: {average_partial_match_score:.2f}%")
