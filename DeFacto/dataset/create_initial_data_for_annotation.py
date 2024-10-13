import json
import pandas as pd
import difflib

def create_initial_data():
    texts = []
    model_summaries = []
    revised_summaries = []
    extrinsic_errors = []
    intrinsic_errors = []
    instructions = []
    explanations = []
    evidences=[]
    splits = []
    for split in ['train','val','test']:
        with open(f"/data/home/yehonatan-pe/Correction_pipeline/DeFacto/data/{split}.jsonl") as f:
            data = [json.loads(line) for line in f]
            for entry in data:
                if entry['candidate'] == "A man accused of killing two men who tried to stop him abusing two young women wearing hijabs has appeared in court in Portland, Oregon.":
                    print("found")
                if entry['has_error'] == True:
                    texts.append(str(entry['article']))
                    model_summaries.append(str(entry['candidate']))
                    revised_summaries.append(str(entry['feedback']['summary']))
                    instructions.append(entry['feedback']['instruction_list'])
                    explanations.append(str(entry['feedback']['explanation']))
                    evidences.append(str(entry['feedback']['evidence']))
                    extrinsic_errors.append(entry['extrinsic_error'])
                    intrinsic_errors.append(entry['intrinsic_error'])
                    splits.append(split)
    df = pd.DataFrame(
        {'text': texts, 'model_summary': model_summaries, 'revised_summary': revised_summaries,
            'extrinsic_errors': extrinsic_errors, 'intrinsic_error': intrinsic_errors,
            'instruction': instructions, 'explanation': explanations, 'evidence': evidences, 'split': splits})
    return df


def check_any_string_starts_with_words(strings, words):
    # Iterate over each string in the list of strings
    for string in strings:
        # Check if the string starts with any word from the list of words
        if any(string.startswith(word) for word in words):
            return True
    return False

def check_all_string_starts_with_words(strings, words):
    # Iterate over each string in the list of strings
    for string in strings:
        # Check if the string starts with any word from the list of words
        if not any(string.startswith(word) for word in words):
            return False
    return True

def filter_based_on_instructions(df):
    instructions = df['instruction'].tolist()
    mask_all = []
    mask_some = []
    words = ['Remove','Replace','Modify']
    for set_of_instructions in instructions:
        if check_all_string_starts_with_words(set_of_instructions, words):
            mask_all.append(True)
            mask_some.append(True)
        elif check_any_string_starts_with_words(set_of_instructions, words):
            mask_all.append(False)
            mask_some.append(True)
        else:
            mask_all.append(False)
            mask_some.append(False)
    df['all_instructions_usable'] = mask_all

    df = df[mask_some]
    return df

def extract_the_mistakes(instruction: str):
    instruction_splitted = instruction.split()
    if 'Replace' == instruction_splitted[0]:
        instruction = instruction.split('with the information about')[0].replace('Replace the information about', '')
    elif 'Remove' == instruction_splitted[0] or 'remove' == instruction_splitted[0]:
        instruction = instruction.replace('Remove the information about', '').replace('from the summary', '')
    elif 'Modify' == instruction_splitted[0]:
        instruction = instruction.replace('Modify the information about', '').replace('in the summary', '')
    else:
        instruction = None
    return instruction

def extract_information_from_instructions(df):
    instructions = df['instruction'].tolist()
    all_mistakes = []
    for instruction_list in instructions:
        mistakes = []
        for instruction in instruction_list:
            mistake = extract_the_mistakes(instruction)
            if mistake is not None:
                mistakes.append(mistake)
        all_mistakes.append(mistakes)
    df['mistakes_from_instructions'] = all_mistakes
    return df


def clean_word(word):
    word = word.translate(str.maketrans('', '', '.,"\'()'))
    return word.lower()

def get_word_diffs(original, modified):
    # Tokenize the strings into words
    original_words = [clean_word(word) for word in original.split()]
    modified_words = [clean_word(word) for word in modified.split()]
    # Use difflib to find differences
    diff = difflib.ndiff(original_words, modified_words)

    added = []
    deleted = []

    # Temporary variables to group consecutive words
    temp_added = []
    temp_deleted = []

    # Loop through the diff result to classify additions and deletions
    for word in diff:
        if word.startswith('+ '):
            temp_added.append(word[2:])
            if temp_deleted:
                deleted.append(' '.join(temp_deleted))
                temp_deleted = []
        elif word.startswith('- '):
            temp_deleted.append(word[2:])
            if temp_added:
                added.append(' '.join(temp_added))
                temp_added = []
        else:
            # If no addition or deletion, append grouped words
            if temp_added:
                added.append(' '.join(temp_added))
                temp_added = []
            if temp_deleted:
                deleted.append(' '.join(temp_deleted))
                temp_deleted = []

    # In case any grouped words are left unappended
    if temp_added:
        added.append(' '.join(temp_added))
    if temp_deleted:
        deleted.append(' '.join(temp_deleted))

    return added, deleted


def extract_mistakes_via_diff(df):
    model_summaries = df['model_summary'].tolist()
    revised_summaries = df['revised_summary'].tolist()
    all_deleted = []
    for i in range(len(model_summaries)):
        _, deleted = get_word_diffs(model_summaries[i], revised_summaries[i])
        all_deleted.append(deleted)
    df['mistakes_from_difference'] = all_deleted
    return df

df = create_initial_data()
df = filter_based_on_instructions(df)
df = extract_information_from_instructions(df)
df = extract_mistakes_via_diff(df)
df['sample_index'] = df.index
df = df.explode('mistakes_from_instructions').reset_index(drop=True)
c = 1
#df.to_csv("/data/home/yehonatan-pe/Correction_pipeline/DeFacto/data/initial_data_for_annotation.csv")

