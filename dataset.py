import json
def load_json(json_file_path):
    with open(json_file_path, "r") as json_file:
        data = json.load(json_file)
    return data

def save_json(data, json_file_path):
    with open(json_file_path, "w") as json_file:
        json.dump(data, json_file, indent=4)


from datasets import Dataset

# # Load the dataset from a local file
# dataset = load_dataset("csv", data_files="contextvqa.json")

dataset = Dataset.from_dict(load_json("hypo3d.json"))
dataset.push_to_hub("MatchLab/Hypo3D")




# data = load_json("contextvqa.json")

# reordered_data ={}
# question_id = 0
# for scene_id, scene in data.items():
#     if 'scene' in scene_id:
#         for change in scene:
#             for qa in change['questions_answers']:
#                 # in five digit format
#                 qa['question_id'] = str(question_id).zfill(5)
    
#                 question_id += 1
                
#         reordered_data[scene_id] = scene
        
# for scene_id, scene in data.items():
#     if 'scene' not in scene_id:
#         for change in scene:
#             for qa in change['questions_answers']:
#                 # in five digit format
#                 qa['question_id'] = str(question_id).zfill(5)
    
#                 question_id += 1
#         reordered_data[scene_id] = scene
        
# save_json(reordered_data, "contextvqa_ordered.json")

