import os
import sys

sys.path.append(os.path.dirname(os.getcwd()))
os.chdir('../')
sys.path.append(os.path.dirname(os.getcwd()))
os.chdir('../')
sys.path.append(os.path.dirname(os.getcwd()))
os.chdir('../')
import json

def create_prompt_file(prompt, output_path):
    with open(output_path, 'w') as f:
        f.write(prompt)


def find_largest_numbered_dir(root_dir):
    largest_number = -1  # Initialize the largest number to a very small value
    largest_dir = None

    # Iterate over each entry in the root directory
    for entry in os.listdir(root_dir):
        if os.path.isdir(os.path.join(root_dir, entry)):
            if entry.isdigit():  # Check if entry name is a digit
                entry_number = int(entry)
                if entry_number > largest_number:
                    largest_number = entry_number
                    largest_dir = entry
    if largest_dir is None:
        return -1
    return int(largest_dir)


def main():
    #prompt = "Write a very short, concise and informative summary of the document."
    prompt = "Write a 30 words summary os the text. The summary should be concise and informative."
    abstract = " Be abstract in your summary, and avoid copying the text directly."
    #limit = " Limit your summary to 25 words."
    limit = ""
    with open("/data/home/yehonatan-pe/Correction_pipeline/experiments/ablations/better_summaries/data/examples.json",'r') as f:
        examples = json.load(f)
    few_shot = "\nHere are some examples of summaries: \n"
    for example in examples.values():
        few_shot += f"Text: {example['text']}\nSummary: {example['summary']}\n"
    prompt += abstract + limit + few_shot
    root_dir = "experiments/ablations/better_summaries/data/prompts"
    prompt_num = find_largest_numbered_dir(root_dir) + 1
    os.makedirs(os.path.join(root_dir, str(prompt_num)))
    output_path = os.path.join(root_dir, str(prompt_num), "prompt.txt")
    create_prompt_file(prompt, output_path)


if __name__ == '__main__':
    main()
