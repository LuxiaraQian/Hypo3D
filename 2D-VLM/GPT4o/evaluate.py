import json
import random
import base64
import concurrent.futures
import re
import os
from openai import OpenAI
from PIL import Image
import json
import os
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer

import base64
from io import BytesIO
from PIL import Image
import pandas as pd

def convert_image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


# Initialize OpenAI client
client = OpenAI(api_key="Your Key")
ROOT_DIR = "dataset/3D_VLM_data"

def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

# Load the movement changes once and reuse
def load_data(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def extract_changes_to_list(text):
    """Extract the first sentence from each change."""
    return [change.split('.')[1].strip() for change in text.strip().split("\n\n")]

def load_json(file_path):
    """Loads JSON data from a file."""
    with open(file_path, 'r') as file:
        return json.load(file)

def read_instance_labels(scene_id):
    """Reads instance labels for a given scene."""
    return load_json(f'{ROOT_DIR}/{scene_id}/{scene_id}_id2labels.json')

def encode_image(image_path):
    """Encodes an image to base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def check_existing_requests():
    if not os.path.exists('requests'):
        os.makedirs('requests')
    
    for file in os.listdir('requests'):
        if file.endswith('.jsonl'):
            os.remove(f'requests/{file}')

def append_request_to_jsonl(request, file_path='requests.jsonl'):
    """Appends request as a new line to a .jsonl file."""
    with open(file_path, 'a') as f:
        json.dump(request, f)
        f.write('\n')

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

def collect_requests(filename, system_content, prompt, images_dir, camera_view_dir, split_ratio=5):
    """Collects requests by reading and processing files in the directory."""
    preprocess_data = load_data(filename)
    check_existing_requests()
    axis_definition = pd.read_csv(f'{camera_view_dir}/axis_definition.csv')
    count = 0
    for scene_id, changes_list in preprocess_data.items():
        image_path = os.path.join(images_dir, f"{scene_id}.png")
        local_image = Image.open(image_path)
        camera_view = Image.open(os.path.join(camera_view_dir, f"{scene_id}.png"))
        encoded_image = convert_image_to_base64(local_image)
        encoded_camera_view = convert_image_to_base64(camera_view)
        
        scene_orientation = axis_definition[axis_definition['scene_id'] == scene_id]['orientation'].values[0]
        # scene_orientation = " ".join(
        #     f"The {item} was located at the {direction.lower()} of the scene."
        #     for scene_id, directions in scene_orientation.items()
        #     for direction, item in directions.items()
        # )
        
        for changes in changes_list:
            context_change = changes['context_change']
            question_answers = changes['questions_answers']
            
            for qa in question_answers:
                question_type = qa['question_type']
                question = qa['question']
                answer = qa['answer']
                
                # Prepare the prompt and inputs
                text_prompt = prompt.format(scene_orientation, context_change, question)
                # Assuming `local_image` is a PIL Image object

                request = {
                    "custom_id": f"{scene_id}&{context_change}&{question_type}&{question}&{answer}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": "gpt-4o",
                        "messages": [
                            {"role": "system", "content": system_content},
                            {"role": "user", "content": [
                                {"type": "text", "text": text_prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url":  f"data:image/jpeg;base64,{encoded_image}"
                                    },
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url":  f"data:image/jpeg;base64,{encoded_camera_view}"
                                    },
                                }
                            ]},
                        ]
                    }
                }
    
                request_filename = f'requests/{count//split_ratio}.jsonl'
                count += 1
                append_request_to_jsonl(request, file_path=request_filename)

def show_status(num):
    batch = client.batches.list(limit=num).data
    
    completed_batches = 0
    finialize_batches = 0
    in_progress_batches = 0
    validation_batches = 0
    failure_batches = 0
    
    for i, d in enumerate(batch):
        if d.status == 'completed':
            completed_batches += 1
        elif d.status == 'finalizing':
            finialize_batches += 1
        elif d.status == 'in_progress':
            in_progress_batches += 1
        elif d.status == 'validating':
            validation_batches += 1
        else:
            failure_batches += 1
    return completed_batches, finialize_batches, in_progress_batches, validation_batches, failure_batches

def upload_tasks(request_files):
    for request in request_files:
        batch_input_file = client.files.create(
          file=open(f"requests/{request}", "rb"),
          purpose="batch"
        )

        batch_input_file_id = batch_input_file.id

        client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
              "description": "nightly eval job"
            }
        )


def fetch_file_content(file_id):
    """Fetches file content from OpenAI."""
    try:
        return client.files.content(file_id).read().decode('utf-8')
    except Exception as e:
        print(f"Error fetching file {file_id}: {e}")
        return None

def save_json(data, output_file):   
    with open(output_file, 'w') as file:
        json.dump(data, file, indent=4)
        
def extract_data(total_number_of_files):
    """Extracts data from batches and processes them in parallel."""
    batch = client.batches.list(limit=total_number_of_files).data
    
    data = {}
    # file_ids = [(d.input_file_id, d.output_file_id) for d in batch if d.status != 'completed']
    # with concurrent.futures.ThreadPoolExecutor() as executor:
    #     input_file_contents = list(executor.map(fetch_file_content, [pair[0] for pair in file_ids]))
        
    # for input_content in input_file_contents:
    #     if input_content:
    #         ids = [json.loads(line).get("custom_id") for line in input_content.splitlines() if line]
    #         print(ids)
    # breakpoint()
    
    # print(f"Failed IDs: {faile_ids}")
    file_ids = [(d.input_file_id, d.output_file_id) for d in batch if d.status == 'completed']

    # Fetch files in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        input_file_contents = list(executor.map(fetch_file_content, [pair[0] for pair in file_ids]))
        output_file_contents = list(executor.map(fetch_file_content, [pair[1] for pair in file_ids]))

    # Process fetched data
    for input_content, output_content in zip(input_file_contents, output_file_contents):
        if input_content and output_content:
            ids = [json.loads(line).get("custom_id") for line in input_content.splitlines() if line]

            for count, line in enumerate(output_content.splitlines()):
                try:
                    output_data = json.loads(line)
                    completion_content = output_data["response"]["body"]["choices"][0]["message"]["content"]

                    if count < len(ids):
                        question_id = ids[count]
                        data[question_id] = completion_content
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON: {e}")
                    
    return data


def process_jsonl(file_path):
    with open(file_path, 'r') as file:
        # Read all lines
        lines = file.readlines()
        
    # Clean and wrap each line with proper JSON syntax
    cleaned_lines = []
    for line in lines:
        # Remove trailing newlines and spaces, fix incomplete JSON objects
        line = line.strip()
        cleaned_lines.append(line)
    
    # Wrap the cleaned lines in a list and join with commas
    json_array = f"[{','.join(cleaned_lines)}]"
    
    # Parse the JSON array
    try:
        data = json.loads(json_array)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return []
    
    return data

def split_jsonl(file_path):
    with open(file_path, 'r') as file:
        # Read all lines
        lines = file.readlines()
    
    # Calculate the middle index
    middle_index = len(lines) // 2
    
    # Split the lines into two halves
    first_half = lines[:middle_index]
    second_half = lines[middle_index:]
    
    # Write the first half to a new file
    with open(f"{file_path}_part1.jsonl", 'w') as part1:
        part1.writelines(first_half)
    
    # Write the second half to another new file
    with open(f"{file_path}_part2.jsonl", 'w') as part2:
        part2.writelines(second_half)

    print(f"File has been split into {file_path}_part1.jsonl and {file_path}_part2.jsonl")

    
    
if __name__ == '__main__':
    import argparse
    from time import sleep
    import json
    from nltk.translate.bleu_score import sentence_bleu
    from rouge_score import rouge_scorer
    import re
    from word2number import w2n
    
    
    parser = argparse.ArgumentParser(description="Process a file name.")
    parser.add_argument(
        "-f", "--filename",
        type=str,
        required=True,
        help="The name of the file to process"
    )
    
    
    args = parser.parse_args()
    system_content = "You are a AI assistant for 3D scene understanding. Answer should be a single word or short phrase."
    # prompt = '''
    # Given a top-view of a 3D scene and a context change, imagine how the scene would look after the change has been applied. Then, answer a question based on the changed scene.

    # Context Change: {}
    # Question: {}

    # The answer should be a single word or short phrase.

    # The answer is:
    # '''
    
    prompt = '''
    Given a top-view of a 3D scene and a reference frame image where the items were on the {} side of the room, first mentally align the top-view with the frame image.

    Now, given a context change, imagine how the scene would look after the change has been applied. Then, answer a question based on the changed scene.

    Context Change: {}
    Question: {}

    The answer should be a single word or short phrase.

    The answer is:
    '''
    
    images_dir = "dataset/2D_VLM_data/top_view_with_label_rotated"
    camera_view_dir = "dataset/2D_VLM_data/camera_view"
    collect_requests(args.filename, system_content, prompt, images_dir, camera_view_dir, split_ratio=50)
    
    all_requests = os.listdir('requests')
    for i, request in enumerate(all_requests):
        upload_tasks([request])
        print(f"Generating question for scene {i}")
        completed_batches, finialize_batches, in_progress_batches, validation_batches, failure_batches = show_status((i+1) % 100)
        print(f"Completed: {completed_batches}, Finalizing: {finialize_batches}, In Progress: {in_progress_batches}, Validation: {validation_batches}, Failure: {failure_batches}")
        sleep(3)

    completed_batches, finialize_batches, in_progress_batches, validation_batches, failure_batches = show_status(len(all_requests))
    print(f"Completed: {completed_batches}, Finalizing: {finialize_batches}, In Progress: {in_progress_batches}, Validation: {validation_batches}, Failure: {failure_batches}")
    while completed_batches < len(all_requests):
        print("Waiting for completion...")
        sleep(3)
        completed_batches, finialize_batches, in_progress_batches, validation_batches, failure_batches = show_status(len(all_requests))
        print(f"Completed: {completed_batches}, Finalizing: {finialize_batches}, In Progress: {in_progress_batches}, Validation: {validation_batches}, Failure: {failure_batches}")
    
    output_data = extract_data(len(all_requests))
    original_data = load_json(args.filename)
    # print(output_data)
    for scene_id, changes_list in original_data.items():
        for changes in changes_list:
            question_answers = changes['questions_answers']
            for qa in question_answers:
                id = f"{scene_id}&{changes['context_change']}&{qa['question_type']}&{qa['question']}&{qa['answer']}"
                if id in output_data:
                    qa['predicted_answer'] = output_data[id]
        
    save_json(original_data, f"dataset/{args.filename.split('/')[-1].split('.json')[0]}_camera_view_GPT4o_with_label_rotated.json")