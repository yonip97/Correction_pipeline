import pandas as pd

import re
from bertopic import BERTopic




def split_instructions(instruction_string):

    split_strings = ['Remove','Replace','Modify','Add','Rewrite']
    pattern = '(' + '|'.join(map(re.escape, split_strings)) + ')'

    # Split the string by any of the substrings, keeping the split substrings in the result
    result = re.split(pattern, instruction_string)
    result = [x for x in result if x != '']

    # Recombine the split substrings with the rest of the text, making sure the split substring starts each split
    result_with_splits = [result[i] + result[i + 1] for i in range(0, len(result) - 1, 2)]
    result_with_splits = [x.strip() for x in result_with_splits]
    return result_with_splits


def extract_the_mistakes(instruction:str):
    instruction_splitted = instruction.split()
    if 'Replace' == instruction_splitted[0]:
        instruction = instruction.split('with the information about')[0].replace('Replace the information about','')
    elif 'Remove' == instruction_splitted[0]:
        instruction = instruction.replace('Remove the information about','').replace('from the summary','')
    elif 'Modify' == instruction_splitted[0]:
        instruction = instruction.replace('Modify the information about','').replace('in the summary','')
    else:
        instruction = None
    return instruction


def main():
    df = pd.read_csv("/data/home/yehonatan-pe/Correction_pipeline/DeFacto/data/summaries_with_errors.csv",index_col=0)
    instructions = df['instruction'].tolist()
    model_summaries = df['model_summary'].tolist()
    instructions = [split_instructions(instruction) for instruction in instructions]
    wrong_facts =[[extract_the_mistakes(item) for item in instruction] for instruction in instructions]
    flattened_list_wrong_facts = [fact for wrong_fact_per_summary in wrong_facts for fact in wrong_fact_per_summary]
    flattened_list_wrong_facts = [x for x in flattened_list_wrong_facts if x is not None]
    topic_model = BERTopic("english")
    topics, probs = topic_model.fit_transform(flattened_list_wrong_facts,)
    print(topics)
    c= 1


main()