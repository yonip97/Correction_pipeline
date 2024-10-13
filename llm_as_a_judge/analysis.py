import numpy as np
import pandas as pd
from collections import Counter
import re
import evaluate


def split_instructions(instruction_string):
    split_strings = ['Remove', 'Replace', 'Modify', 'Add', 'Rewrite']
    pattern = '(' + '|'.join(map(re.escape, split_strings)) + ')'

    # Split the string by any of the substrings, keeping the split substrings in the result
    result = re.split(pattern, instruction_string)
    result = [x for x in result if x != '']

    # Recombine the split substrings with the rest of the text, making sure the split substring starts each split
    result_with_splits = [result[i] + result[i + 1] for i in range(0, len(result) - 1, 2)]
    result_with_splits = [x.strip() for x in result_with_splits]
    return result_with_splits


def extract_the_mistakes(instruction: str):
    instruction_splitted = instruction.split()
    if 'Replace' == instruction_splitted[0]:
        instruction = instruction.split('with the information about')[0].replace('Replace the information about', '')
    elif 'Remove' == instruction_splitted[0]:
        instruction = instruction.replace('Remove the information about', '').replace('from the summary', '')
    elif 'Modify' == instruction_splitted[0]:
        instruction = instruction.replace('Modify the information about', '').replace('in the summary', '')
    else:
        instruction = None
    return instruction


def remove_until(string, target):
    # Find the index where the target string starts
    index = string.find(target)

    # If the target string is found, return the substring starting from the target
    if index != -1:
        return string[index:]
    else:
        return string

def main():
    df = pd.read_csv(
        'data/mistakes_detection/prompt2/results_llama_3.1_70B_instruct.csv',
        index_col=0)
    original = pd.read_csv("/data/home/yehonatan-pe/Correction_pipeline/DeFacto/data/summaries_with_errors.csv",
                           index_col=0)
    instructions = original['instruction'].tolist()
    instructions = [split_instructions(x) for x in instructions]
    wrong_facts = [[extract_the_mistakes(item) for item in instruction] for instruction in instructions]

    outputs = df['output'].tolist()
    full_outputs = outputs
    outputs = [x.split('assistant\n')[-1].lower().strip() for x in outputs]
    outputs = [remove_until(x, 'fact:') for x in outputs]
    counter = 0
    consistent = []
    facts = []
    dataset_facts = []
    for i in range(len(outputs)):
        x = outputs[i]
        y = full_outputs[i]
        if 'fact:' and 'explanation:' in x:
            facts.append(x)
            dataset_facts.append(wrong_facts[i])
        elif 'consistent' in x or 'inconsistent' in x:
            consistent.append(x)
        else:
            counter += 1
    import re
    final_facts = []
    for x in facts:
        text_batch = []
        matches = re.findall(r'fact:(.*?)explanation:', x, re.DOTALL)

        # Extract and print all matched fact texts
        for i, fact_text in enumerate(matches, 1):
            text_batch.append(fact_text.strip())
        final_facts.append(text_batch)
    rouge = evaluate.load('rouge')
    matches = []
    from tqdm import tqdm
    for possible_mistake, real_mistake in tqdm(zip(final_facts, dataset_facts)):
        if len(possible_mistake) > 0 and len(real_mistake) > 0:
            possible_mistake_taken = []
            for mistake in real_mistake:
                if mistake is not None:
                    match = None
                    match_score = 0.25
                    scores = rouge.compute(predictions=possible_mistake, references=[mistake] * len(possible_mistake),
                                           use_aggregator=False)['rougeL']

                    for i in range(len(possible_mistake)):
                        if scores[i] > match_score and possible_mistake[i] not in possible_mistake_taken:
                            match = possible_mistake[i]
                            match_score = scores[i]
                    if match is not None:
                        possible_mistake_taken.append(match)
                        matches.append({'real': mistake, 'match': match, 'score': match_score})
    scores = [x['score'] for x in matches]
    matches_found = [x['match'] for x in matches]
    print("Num of mistakes found by the model: ", sum([len(x) for x in final_facts if len(x) > 0]))
    print("The amount of matches found is: ", len(matches_found))
    print("The amount of total real facts is: ", sum([len(x) for x in dataset_facts if len(x) > 0]))
    for i in np.linspace(0, 1, 21):
        print(f"The amount of matches with score above {i} and below {i + 0.05} is: ",len([x for x in scores if x >= i and x < i + 0.05]))
    import matplotlib.pyplot as plt
    plt.hist(scores, bins=20)
    plt.show()
    for x in matches:
        if x['score'] >0.25:
            print(x)
            print("---------------------------------------------------------------------------------------")

    consistent = Counter(consistent)
if __name__ == '__main__':
    main()
