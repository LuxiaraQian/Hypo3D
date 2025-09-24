from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
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
parser.add_argument("-m", "--model_id", type=str, default="Qwen/Qwen2-VL-7B-Instruct", help="Model ID of the model to be evaluated")

args = parser.parse_args()
# Load the model and move it to the GPU
model = Qwen2VLForConditionalGeneration.from_pretrained(
    args.model_id,
    torch_dtype=torch.bfloat16,
    # attn_implementation="flash_attention_2",
    device_map="auto"
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Set min and max pixels
min_pixels = 256 * 28 * 28
max_pixels = 1280 * 28 * 28
processor = AutoProcessor.from_pretrained(args.model_id, min_pixels=min_pixels, max_pixels=max_pixels)

# Load a local image
data = json.load(open(args.data_path))
images_dir = "/mnt/pfs/zitao_team/luxiaoqian/Hypo3D/dataset/top_view_no_label"

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
    """
    Normalize the input text by converting to lowercase, removing certain words/phrases,
    replacing specific terms, removing punctuation, and converting words to digits.
    """
    # Convert to lowercase
    text = text.lower()

    # Define replacements for specific terms
    replacements = {
        'back and right': 'back right',
        'back and left': 'back left',
        'front and right': 'front right',
        'front and left': 'front left',
        'behind and to the right': 'back right',
        'behind and to the left': 'back left',
        'in front and to the right': 'front right',
        'to the': '',
        'by the': '',
        'on the': '',
        'near': '',
        'next': '',
        'corner': '',
        'behind': 'back',
        'bottom': 'back',
        'top': 'front',
        'right side': 'right',
        'left side': 'left',
        'front side': 'front',
        'back side': 'back',
        'in front of': 'front',
        'on the left of': 'left',
        'on the right of': 'right',
        'on the left': 'left',
        'on the right': 'right',
        'north': 'front',
        'south': 'back',
        'east': 'right',
        'west': 'left',
        'northwest': 'front left',
        'northeast': 'front right',
        'southwest': 'back left',
        'southeast': 'back right',
        'forward': 'front',
        'backward': 'back',
        'bottom of': 'back',
        "left of": 'left',
        "right of": 'right',
        "front of": 'front',
        "back of": 'back'
    }
        
    # Use regex for efficient replacements
    sorted_replacements = sorted(replacements.keys(), key=len, reverse=True)
    pattern = re.compile(r'\b(' + '|'.join(map(re.escape, sorted_replacements)) + r')\b')
    text = pattern.sub(lambda match: replacements[match.group(0)], text)
    # Remove articles (e.g., "a", "an", "the")
    text = re.sub(r'\b(?:a|an|the)\b', '', text).strip()

    # Remove punctuation except letters, digits, and spaces
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

    # Convert number words to digits (if applicable)
    text = convert_words_to_digits(text)
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
# Baseline
template0 = '''
Given a top-view of a 3D scene, mentally rotate the image to align with the specified orientation.

Scene Orientation: {}

Now, given a context change, imagine how the scene would look after the change has been applied. Then, answer a question based on the changed scene.

Context Change: {}
Question: {}

The answer should be a single word or short phrase.

The answer is:
'''


# 5-shot
template1 = '''
Given a top-view of a 3D scene, mentally rotate the image to align with the specified orientation.

=== 5-shot Example 1 ===

Scene Orientation: North-facing (top is front)

Context Change: The black backpack has been moved to the laundry basket.
Question: What is the position of the tissue box in comparison to the backpack?

<original_image>
- The backpack is originally located on the couch in the living room (center bottom of the image).
- The laundry basket is at the left side of the bed, near the nightstand.
- The tissue box is on the nightstand at the top side of the bed, near the headboard.
</original_image>

<change>
The black backpack has been moved to the laundry basket.
</change>

<changed_image>
- The backpack is now at the laundry basket, to the left side of the bed.
- The tissue box remains on the nightstand at the top side of the bed.
- Relative to the backpack’s new position, the tissue box is in front of it and slightly to the right.
</changed_image>

<think>
Compare the backpack’s new position (by the laundry basket, left of the bed) to the tissue box (on the top nightstand).
From that reference, the tissue box lies toward the head of the bed (front) and to the right.
</think>

<answer>
front right
</answer>

=== 5-shot Example 2 ===

Scene Orientation: North-facing (top is front)

Context Change: The stool that is closest to the guitar has been moved next to the couch, behind the backpack.
Question: Is the stool placed higher or lower than the tissue box now?

<original_image>
- Multiple stools line the kitchen/table area to the left of the couch.
- The guitar is near the center-left doorway; the closest stool to it is the topmost stool of that row.
- The couch is centered; a backpack rests on its upper side.
- The bed is in the upper center; a tissue box sits on the nightstand at the head (top) of the bed.
</original_image>

<change>
The stool that is closest to the guitar has been moved next to the couch, behind the backpack.
</change>

<changed_image>
- The selected stool is now adjacent to the couch, placed behind (toward the top side relative to the couch) the backpack.
- The tissue box remains at the top side of the bed near the headboard, which is nearer to the very top of the scene than the couch area.
</changed_image>

<think>
“Behind the backpack” puts the stool slightly above the couch but still well below the bed’s head area. The tissue box is at the very top side of the bed, closer to the scene’s top edge than the couch region. Therefore, in vertical placement, the stool is below the tissue box.
</think>

<answer>
lower
</answer>

=== 5-shot Example 3 ===

Scene Orientation: North-facing (top is front)

Context Change: The backpack has been moved onto the bed behind the pillow.
Question: How many objects are now situated on the bed following the backpack’s movement?

<original_image>
- The bed is positioned at the upper center of the room.
- On the bed, there are originally pillows placed near the headboard.
- A tissue box is on the nightstand at the head of the bed (not on the bed itself).
- Other objects (like the backpack) are not on the bed originally, but in the living room area (on the couch).
</original_image>

<change>  
The backpack has been moved onto the bed behind the pillow.  
</change>  

<changed_image>
- After the move, the backpack is now situated on the bed, directly behind the pillow.
- So, the bed now contains two distinct object categories:
  1. Pillows (already there)
  2. Backpack (newly placed)
</changed_image>

<think>  
Originally, only the pillows were on the bed. After the context change, the backpack is added onto the bed behind the pillow. Therefore, the number of objects situated on the bed is now two.  
</think>  

<answer>  
2  
</answer>  

=== 5-shot Example 4 ===

Scene Orientation: North-facing (top is front)

Context Change: The kitchen cabinet on top of the refrigerator has been removed.
Question: Is it spatially possible to place the microwave in the spot where the kitchen cabinet was removed?

<original_image>
- The refrigerator is located in the bottom-left kitchen area.
- Above the refrigerator, there is originally a kitchen cabinet mounted to the wall.
- The microwave is currently placed on top of the counter, beside or integrated with other appliances (like the toaster oven).
</original_image>

<change>  
The kitchen cabinet on top of the refrigerator has been removed.  
</change>  

<changed_image>
- With the kitchen cabinet gone, there is now an open space directly above the refrigerator.
- This location could potentially host another appliance.
- The microwave is smaller in depth and height than a wall cabinet and could fit into the newly cleared space.
</changed_image>

<think>  
The refrigerator’s top is wide and sturdy enough to support appliances, and removing the wall-mounted cabinet above creates sufficient vertical clearance. Since the microwave is typically smaller than the cabinet, it is spatially feasible to place the microwave in that cleared area.  
</think>  

<answer>  
Yes  
</answer>  

=== 5-shot Example 5 ===

Scene Orientation: North-facing (top is front)

Context Change: The trash can near the desk has been removed.
Question: How many trash cans remain in the room after the change?

<original_image>
- There are multiple trash cans in the scene:
- One is located near the desk in the upper-right/bedroom area.
- Another trash can is placed in the kitchen area, close to the refrigerator.
- A smaller bin may also be near the bathroom or doorway area.
</original_image>

<change>  
The trash can near the desk has been removed.  
</change>  

<changed_image>
- After the removal, the desk-side trash can is no longer present.
- Remaining trash cans:
  1. The kitchen trash can beside the refrigerator.
  2. The additional bin near the doorway/bathroom area.
</changed_image>

<think>  
Originally, there were at least three trash cans visible. Removing the one near the desk reduces the count by one, leaving two still in the room.  
</think>  

<answer>  
2  
</answer>  

=== Now answer the new query following the same format. Only fill content inside the tags. ===

Scene Orientation: {}

Now, given a context change, imagine how the scene would look after the change has been applied. Then, answer a question based on the changed scene.

Context Change: {}
Question: {}

<original_image>
Extract spatial and structural information related to the query from the input image:
Include objects, positional relationships, orientation, size, functional attributes, etc.
</original_image>

<change>
{}
</change>

<changed_image>
Update the scene, combining the change operation with the resulting object relations.
</changed_image>

<think>
Reason step by step, explaining how to use the updated scene to answer the query.
</think>

<answer>
(Output the concise final answer here)
</answer>
'''


# 1-shot
template2 = '''
Given a top-view of a 3D scene, mentally rotate the image to align with the specified orientation.

=== One-shot Example ===

Scene Orientation: North-facing (top is front)

Context Change: The black backpack has been moved to the laundry basket.
Question: What is the position of the tissue box in comparison to the backpack?

<original_image>
- The backpack is originally located on the couch in the living room (center bottom of the image).
- The laundry basket is at the left side of the bed, near the nightstand.
- The tissue box is on the nightstand at the top side of the bed, near the headboard.
</original_image>

<change>
The black backpack has been moved to the laundry basket.
</change>

<changed_image>
- The backpack is now at the laundry basket, to the left side of the bed.
- The tissue box remains on the nightstand at the top side of the bed.
- Relative to the backpack’s new position, the tissue box is in front of it and slightly to the right.
</changed_image>

<think>
Compare the backpack’s new position (by the laundry basket, left of the bed) to the tissue box (on the top nightstand).
From that reference, the tissue box lies toward the head of the bed (front) and to the right.
</think>

<answer>
front right
</answer>

=== Now answer the new query following the same format. Only fill content inside the tags. ===

Scene Orientation: {}

Now, given a context change, imagine how the scene would look after the change has been applied. Then, answer a question based on the changed scene.

Context Change: {}
Question: {}

<original_image>
Extract spatial and structural information related to the query from the input image:
Include objects, positional relationships, orientation, size, functional attributes, etc.
</original_image>

<change>
{}
</change>

<changed_image>
Update the scene, combining the change operation with the resulting object relations.
</changed_image>

<think>
Reason step by step, explaining how to use the updated scene to answer the query.
</think>

<answer>
(Output the concise final answer here)
</answer>
'''


# 5-shot v2
template_v3 = '''
Given a top-view of a 3D scene, mentally rotate the image to align with the specified orientation.

=== 5-shot Example 1 ===

Scene Orientation: North-facing (top is front)

Context Change: The black backpack has been moved to the laundry basket.
Question: What is the position of the tissue box in comparison to the backpack?

<original_image>
- The backpack is originally located on the couch in the living room (center bottom of the image).
- The laundry basket is at the left side of the bed, near the nightstand.
- The tissue box is on the nightstand at the top side of the bed, near the headboard.
</original_image>

<change>
The black backpack has been moved to the laundry basket.
</change>

<changed_image>
- The backpack is now at the laundry basket, to the left side of the bed.
- The tissue box remains on the nightstand at the top side of the bed.
- Relative to the backpack’s new position, the tissue box is in front of it and slightly to the right.
</changed_image>

<think>
Compare the backpack’s new position (by the laundry basket, left of the bed) to the tissue box (on the top nightstand).
From that reference, the tissue box lies toward the head of the bed (front) and to the right.
</think>

<answer>
front right
</answer>

=== 5-shot Example 2 ===

Scene Orientation: North-facing (top is front)

Context Change: The stool that is closest to the guitar has been moved next to the couch, behind the backpack.
Question: Is the stool placed higher or lower than the tissue box now?

<original_image>
- Multiple stools line the kitchen/table area to the left of the couch.
- The guitar is near the center-left doorway; the closest stool to it is the topmost stool of that row.
- The couch is centered; a backpack rests on its upper side.
- The bed is in the upper center; a tissue box sits on the nightstand at the head (top) of the bed.
</original_image>

<change>
The stool that is closest to the guitar has been moved next to the couch, behind the backpack.
</change>

<changed_image>
- The selected stool is now adjacent to the couch, placed behind (toward the top side relative to the couch) the backpack.
- The tissue box remains at the top side of the bed near the headboard, which is nearer to the very top of the scene than the couch area.
</changed_image>

<think>
“Behind the backpack” puts the stool slightly above the couch but still well below the bed’s head area. The tissue box is at the very top side of the bed, closer to the scene’s top edge than the couch region. Therefore, in vertical placement, the stool is below the tissue box.
</think>

<answer>
lower
</answer>

=== 5-shot Example 3 ===

Scene Orientation: North-facing (top is front)

Context Change: The backpack has been moved onto the bed behind the pillow.
Question: How many objects are now situated on the bed following the backpack’s movement?

<original_image>
- The bed is positioned at the upper center of the room.
- On the bed, there are originally pillows placed near the headboard.
- A tissue box is on the nightstand at the head of the bed (not on the bed itself).
- Other objects (like the backpack) are not on the bed originally, but in the living room area (on the couch).
</original_image>

<change>  
The backpack has been moved onto the bed behind the pillow.  
</change>  

<changed_image>
- After the move, the backpack is now situated on the bed, directly behind the pillow.
- So, the bed now contains two distinct object categories:
  1. Pillows (already there)
  2. Backpack (newly placed)
</changed_image>

<think>  
Originally, only the pillows were on the bed. After the context change, the backpack is added onto the bed behind the pillow. Therefore, the number of objects situated on the bed is now two.  
</think>  

<answer>  
2  
</answer>  

=== 5-shot Example 4 ===

Scene Orientation: North-facing (top is front)

Context Change: The kitchen cabinet on top of the refrigerator has been removed.
Question: Is it spatially possible to place the microwave in the spot where the kitchen cabinet was removed?

<original_image>
- The refrigerator is located in the bottom-left kitchen area.
- Above the refrigerator, there is originally a kitchen cabinet mounted to the wall.
- The microwave is currently placed on top of the counter, beside or integrated with other appliances (like the toaster oven).
</original_image>

<change>  
The kitchen cabinet on top of the refrigerator has been removed.  
</change>  

<changed_image>
- With the kitchen cabinet gone, there is now an open space directly above the refrigerator.
- This location could potentially host another appliance.
- The microwave is smaller in depth and height than a wall cabinet and could fit into the newly cleared space.
</changed_image>

<think>  
The refrigerator’s top is wide and sturdy enough to support appliances, and removing the wall-mounted cabinet above creates sufficient vertical clearance. Since the microwave is typically smaller than the cabinet, it is spatially feasible to place the microwave in that cleared area.  
</think>  

<answer>  
Yes  
</answer>  

=== 5-shot Example 5 ===

Scene Orientation: North-facing (top is front)

Context Change: The trash can near the desk has been removed.
Question: How many trash cans remain in the room after the change?

<original_image>
- There are multiple trash cans in the scene:
- One is located near the desk in the upper-right/bedroom area.
- Another trash can is placed in the kitchen area, close to the refrigerator.
- A smaller bin may also be near the bathroom or doorway area.
</original_image>

<change>  
The trash can near the desk has been removed.  
</change>  

<changed_image>
- After the removal, the desk-side trash can is no longer present.
- Remaining trash cans:
  1. The kitchen trash can beside the refrigerator.
  2. The additional bin near the doorway/bathroom area.
</changed_image>

<think>  
Originally, there were at least three trash cans visible. Removing the one near the desk reduces the count by one, leaving two still in the room.  
</think>  

<answer>  
2  
</answer>  

=== Now answer the new query following the same format. Only fill content inside the tags. ===

Scene Orientation: {}

Now, given a context change, imagine how the scene would look after the change has been applied. Then, answer a question based on the changed scene. The image itself does not change.

Context Change: {}
Question: {}

<original_image>
Extract spatial and structural information related to the query from the input image:
Include objects, positional relationships, orientation, size, functional attributes, etc.
</original_image>

<change>
{}
</change>

<changed_image>
Update the scene, combining the change operation with the resulting object relations.
</changed_image>

<think>
Reason step by step, explaining how to use the updated scene to answer the query.
</think>

<answer>
If the change is irrelevant to the question, keep the answer unchanged.
Output only a single word/phrase, do not explain.
</answer>
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


# read xlxs file
df = pd.read_excel("dataset/Axis Definition.xlsx", sheet_name='Sheet1', engine='openpyxl')

# Main loop
for scene_id, changes_list in list(data.items())[:50]:
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
            
            # Prepare the prompt and inputs
            text_prompt = template_v3.format(scene_orientation, context_change, question, context_change)
            
            # text_prompt = template.format(context_change, question)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": local_image},
                        {"type": "text", "text": text_prompt},
                    ],
                }
            ]

            # Preparation for inference
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            
            # Move inputs to GPU
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Inference: Generation of the outputs
            generated_ids = model.generate(**inputs, max_new_tokens=2048)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
            ]
            
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0].strip()
         
            def extract_answer(text):
                match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL)
                if match:
                    return match.group(1).strip()
                return text.strip()  # fallback，如果没匹配到就返回原始输出

            # 在主循环里替换：
            raw_output = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0].strip()

            predicted_answer = extract_answer(raw_output)
            qa['predicted_answer'] = predicted_answer
            # qa['predicted_answer'] = output_text
            
            print(f'Processed scene {scene_id}, change {i + 1}, question {j + 1}')
            
            # Metrics calculation
            predicted_answer = normalize_text(output_text)
            reference_answer = normalize_text(answer)
            
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
            # print(f"Prediction: {output_text}")
            # print(f"Reference : {answer}")
save_json(data, f"dataset/contextvqa_Qwen2_VL_7B_no_label_align_1500_prompt_v2.json")

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