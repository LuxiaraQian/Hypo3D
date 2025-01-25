import json
import re
from word2number import w2n
from collections import defaultdict


def convert_words_to_digits(text):
    words = text.split()
    converted_words = []
    is_digit = False
    for word in words:
        try:
            # Attempt to convert the word to a number
            number = w2n.word_to_num(word)
            converted_words.append(str(number))
            is_digit = True
        except ValueError:
            # If the word is not a number, keep it as is
            if not is_digit:
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
    
def partial_match_score(predicted, reference):
    pred_tokens = predicted.split()
    ref_tokens = reference.split()
    common_tokens = set(pred_tokens).intersection(set(ref_tokens))
    return len(common_tokens) / len(ref_tokens) if len(ref_tokens) > 0 else 0

def load_json(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)
    return data

def compute_result_each_change(total_questions_per_type, exact_matches_per_type, partial_match_scores_per_type):
    print(f"{'Change Type':<20}{'EM (%)':<10}{'PM (%)':<10}")
    print("-" * 40)
    for change_type in total_questions_per_type.keys():
        overall_accuracy = (exact_matches_per_type[change_type] / total_questions_per_type[change_type]) * 100
        overall_partial_match_score = (sum(partial_match_scores_per_type[change_type]) / len(partial_match_scores_per_type[change_type])) * 100
        print(f"{change_type.split(' ')[1]:<20} {overall_accuracy:<10.2f}{overall_partial_match_score:<10.2f}")

def compute_result_each_question(total_questions_per_type, exact_matches_per_type, partial_match_scores_per_type):
    question_types = ['Direction', 'Scale', 'Semantic']
    print(f"{'Question Type':<20}{'EM (%)':<10}{'PM (%)':<10}")
    print("-" * 40)
    for question_type in question_types:
        exact_match_score_per_type = (exact_matches_per_type[question_type] / total_questions_per_type[question_type]) * 100
        average_partial_match_per_type = sum(partial_match_scores_per_type[question_type]) / len(partial_match_scores_per_type[question_type]) * 100
        print(f"{question_type:<20} {exact_match_score_per_type:<10.2f}{average_partial_match_per_type:<10.2f}")
        
if __name__ == "__main__":
    total_questions = 0
    exact_matches = 0
    partial_match_scores = []
    total_questions_per_type = {}
    exact_matches_per_type = {}
    partial_match_scores_per_type = {}
    merged_total_matches_per_type = {}
    merged_exact_matches_per_type = {}
    merged_partial_match_scores_per_type = {}

    data = load_json('exp/contextvqa_GPT4o_with_label_rotated.json')

    total_questions_per_type = defaultdict(int)
    exact_matches_per_type = defaultdict(int)
    partial_match_scores_per_type = defaultdict(list)
    question_exact_matches_per_type = defaultdict(int)
    question_partial_match_scores_per_type = defaultdict(list)
    question_total_questions_per_type = defaultdict(int)

    for scene_id, changes_list in data.items():
        for change in changes_list:
            context_change = change['context_change']
            question_answers = change['questions_answers']
            change_type = change['change_type']

            for qa in question_answers:
                question_type = qa['question_type']
                question = qa['question']
                reference_answer = normalize_text(qa['answer'])
                predicted_answer = normalize_text(qa['predicted_answer'])

                # Update total question count
                total_questions_per_type[change_type] += 1

                # Check for exact match
                if predicted_answer == reference_answer:
                    exact_matches_per_type[change_type] += 1

                # Calculate and store partial match score
                partial_match = partial_match_score(predicted_answer, reference_answer)
                partial_match_scores_per_type[change_type].append(partial_match)
                partial_match_scores.append(partial_match)

                for question_type in question_type.split(' '):
    
                    # Initialize metrics for new question types
                    if question_type not in question_total_questions_per_type:
                        question_total_questions_per_type[question_type] = 0
                        question_exact_matches_per_type[question_type] = 0
                        question_partial_match_scores_per_type[question_type] = []
                    
                    # Exact Match
                    if predicted_answer == reference_answer:
                        question_exact_matches_per_type[question_type] += 1

                    question_partial_match_scores_per_type[question_type].append(partial_match_score(predicted_answer, reference_answer))
                    question_total_questions_per_type[question_type] += 1
                    
                # Global metrics
                if predicted_answer == reference_answer:
                    exact_matches += 1
                total_questions += 1

    exact_match_score = (exact_matches / total_questions) * 100
    average_partial_match_score = sum(partial_match_scores) / len(partial_match_scores) * 100

    # Print overall results
    print("\nOverall Metrics:")
    print(f"Exact Match Score: {exact_match_score:.2f}%")
    print(f"Partial Match Score: {average_partial_match_score:.2f}%")
    
    compute_result_each_change(total_questions_per_type, exact_matches_per_type, partial_match_scores_per_type)
    
    compute_result_each_question(question_total_questions_per_type, question_exact_matches_per_type, question_partial_match_scores_per_type)
    

