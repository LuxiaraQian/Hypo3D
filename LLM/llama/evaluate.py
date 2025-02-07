import torch
from transformers import pipeline, AutoTokenizer
import torch
from PIL import Image
import json
import os
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
import re
from word2number import w2n
import argparse
import pandas as pd
import pickle
import random

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

model_id = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
pipe = pipeline(
    "text-generation",
    model=model_id,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    pad_token_id=tokenizer.eos_token_id,  # Set pad_token_id to eos_token_id
)


parser = argparse.ArgumentParser()
parser.add_argument("-f", "--data_path", type=str, default="reason3d/recategorize_test_data.json", help="Path to the JSON file containing the test data")

args = parser.parse_args()

# Load a local image
data = json.load(open(args.data_path))
images_dir = "dataset/2D_VLM_data"

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
    text = text.lower()
    text = text.replace('to the', '')
    text = text.replace('behind', 'back')
    text = text.replace('in front of', 'front')
    text = text.replace('on the left of', 'left')
    text = text.replace('on the right of', 'right')
    text = text.replace('on the left', 'left')
    text = text.replace('on the right', 'right')
    
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

def load_json(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)
    return data

scannet_description = pickle.load(open('dataset/LLM_data/scanrefer_captions_by_scene.pkl', 'rb'))
rscan_description = load_json('dataset/LLM_data/3rscan_scene_cap.json')
    
# Metrics initialization
total_questions = 0
exact_matches = 0
bleu_scores = []
rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
rouge_scores = {'rouge1': [], 'rougeL': []}
f1_scores = []
partial_match_scores = []

# Text prompt
template = '''
Given a 3D scene description, mentally match it with the specified orientation.

Scene Description: {}
Orientation: {}

Next, answer a question based on the aligned scene.

Question: {}

The answer should be a single word or short phrase:

Answer:
'''

df = pd.read_excel("dataset/Axis Definition.xlsx", sheet_name='Sheet1', engine='openpyxl')

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
total_questions_per_type = {}
exact_matches_per_type = {}
bleu_scores_per_type = {}
rouge_scores_per_type = {}
f1_scores_per_type = {}
partial_match_scores_per_type = {}

# Main loop
for scene_id, changes_list in list(data.items()):
    if scene_id.startswith('scene'):
        scene_description = scannet_description[scene_id]
        scene_description = random.sample(scene_description, k=min(len(scene_description), 30))
        scene_description = '. '.join([i[1] for i in scene_description])
        scene_description = scene_description.replace('..', '.')
    else:
        scene_description = rscan_description[scene_id]['captions'][0]

    scene_orientation = extract_non_nan_values(df[df['scene_id'] == scene_id])
    scene_orientation = " ".join(
        f"The {item} was located at the {direction.lower()} of the scene."
        for scene_id, directions in scene_orientation.items()
        for direction, item in directions.items()
    )
    
    for changes in changes_list:
        context_change = changes['context_change']
        question_answers = changes['questions_answers']
        
        for qa in question_answers:
            question_type = qa['question_type']
            question = qa['question']
            answer = qa['answer']
            
            # Prepare the prompt and inputs
            text_prompt = template.format(scene_description, scene_orientation, question)

            messages = [
                {"role": "system", "content": "You are a AI assistant for 3D scene understanding. Answer should be a single word or short phrase."},
                {"role": "user", "content": text_prompt},
            ]
            
            outputs = pipe(
                messages,
                max_new_tokens=32,
            )
            
            # Metrics calculation
            output_text = outputs[0]["generated_text"][-1]['content']
            qa['predicted_answer'] = output_text
            predicted_answer = normalize_text(output_text)
            reference_answer = normalize_text(answer)
            
            print(f'Processed scene {scene_id}')
            
            print('predicted:', predicted_answer)
            
            # Initialize metrics for new question types
            if question_type not in total_questions_per_type:
                total_questions_per_type[question_type] = 0
                exact_matches_per_type[question_type] = 0
                bleu_scores_per_type[question_type] = []
                rouge_scores_per_type[question_type] = {'rouge1': [], 'rougeL': []}
                f1_scores_per_type[question_type] = []
                partial_match_scores_per_type[question_type] = []

            # Exact Match
            if predicted_answer == reference_answer:
                exact_matches += 1
                exact_matches_per_type[question_type] += 1

            # BLEU Score
            reference = [reference_answer.split()]
            hypothesis = predicted_answer.split()
            bleu_score = sentence_bleu(reference, hypothesis)
            bleu_scores.append(bleu_score)
            bleu_scores_per_type[question_type].append(bleu_score)

            # ROUGE Score
            scores = rouge_scorer.score(reference_answer, predicted_answer)
            rouge_scores['rouge1'].append(scores['rouge1'].fmeasure)
            rouge_scores['rougeL'].append(scores['rougeL'].fmeasure)
            rouge_scores_per_type[question_type]['rouge1'].append(scores['rouge1'].fmeasure)
            rouge_scores_per_type[question_type]['rougeL'].append(scores['rougeL'].fmeasure)

            # F1 Score
            f1 = f1_score(predicted_answer, reference_answer)
            f1_scores.append(f1)
            f1_scores_per_type[question_type].append(f1)

            # Partial Match Score
            partial_match = partial_match_score(predicted_answer, reference_answer)
            partial_match_scores.append(partial_match)
            partial_match_scores_per_type[question_type].append(partial_match)

            total_questions += 1
            total_questions_per_type[question_type] += 1

save_json(data, "dataset/context_vqa_noC_without_context_change_llama-3.2-3B_with_axis.json")
# Calculate average metrics for each question type
for question_type in total_questions_per_type:
    exact_match_score_per_type = (exact_matches_per_type[question_type] / total_questions_per_type[question_type]) * 100
    average_bleu_score_per_type = sum(bleu_scores_per_type[question_type]) / len(bleu_scores_per_type[question_type])
    average_rouge1_per_type = sum(rouge_scores_per_type[question_type]['rouge1']) / len(rouge_scores_per_type[question_type]['rouge1'])
    average_rougeL_per_type = sum(rouge_scores_per_type[question_type]['rougeL']) / len(rouge_scores_per_type[question_type]['rougeL'])
    average_f1_score_per_type = sum(f1_scores_per_type[question_type]) / len(f1_scores_per_type[question_type])
    average_partial_match_per_type = sum(partial_match_scores_per_type[question_type]) / len(partial_match_scores_per_type[question_type]) * 100
    
    # Print results for each question type
    print(f"Question Type: {question_type}")
    print(f"  Exact Match Score: {exact_match_score_per_type:.2f}%")
    print(f"  Partial Match Score: {average_partial_match_per_type:.2f}%")
    print(f"  Average BLEU Score: {average_bleu_score_per_type:.4f}")
    print(f"  Average ROUGE-1 Score: {average_rouge1_per_type:.4f}")
    print(f"  Average ROUGE-L Score: {average_rougeL_per_type:.4f}")
    print(f"  Average F1 Score: {average_f1_score_per_type:.4f}")


# Calculate overall average metrics
exact_match_score = (exact_matches / total_questions) * 100
average_bleu_score = sum(bleu_scores) / len(bleu_scores)
average_rouge1 = sum(rouge_scores['rouge1']) / len(rouge_scores['rouge1'])
average_rougeL = sum(rouge_scores['rougeL']) / len(rouge_scores['rougeL'])
average_f1_score = sum(f1_scores) / len(f1_scores)
average_partial_match_score = sum(partial_match_scores) / len(partial_match_scores) * 100

# Print overall results
print("\nOverall Metrics:")
print(f"Exact Match Score: {exact_match_score:.2f}%")
print(f"Partial Match Score: {average_partial_match_score:.2f}%")
print(f"Average BLEU Score: {average_bleu_score:.4f}")
print(f"Average ROUGE-1 Score: {average_rouge1:.4f}")
print(f"Average ROUGE-L Score: {average_rougeL:.4f}")
print(f"Average F1 Score: {average_f1_score:.4f}")