import os
import sys

os.chdir('/data/home/yehonatan-pe/Correction_pipeline/')
sys.path.append(os.getcwd())

import pandas as pd

import re
from general.llama import LLama
import torch
from general.utils import iter_list
from tqdm import tqdm
from transformers import BertModel, BertTokenizer
import numpy as np

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


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
    # if instruction is not None and 'and' in instruction.split(' '):
    #     print(instruction)
    return instruction


def get_mistakes():
    df = pd.read_csv("DeFacto/data/summaries_with_errors.csv", index_col=0)
    instructions = df['instruction'].tolist()
    explanations = df['explanation'].tolist()
    extrinsic_errors = df['extrinsic_error'].tolist()
    intrinsic_errors = df['intrinsic_error'].tolist()
    instructions = [split_instructions(instruction) for instruction in instructions]
    wrong_facts = [[extract_the_mistakes(item) for item in instruction if extract_the_mistakes(item) is not None] for
                   instruction in instructions]
    flattened_list_wrong_facts = [fact for wrong_fact_per_summary in wrong_facts for fact in wrong_fact_per_summary]
    # flattened_list_wrong_facts = [x for x in flattened_list_wrong_facts if x is not None]
    max_times = 5
    first_words_in_additional = []
    check = {1: max_times, 2: max_times, 3: max_times}
    possible_delimiters = ['Also', 'In addition', 'and']
    for i in range(len(wrong_facts)):
        if len(wrong_facts[i]) in check and intrinsic_errors[i] == 0 and extrinsic_errors[i] == 1:
            check[len(wrong_facts[i])] -= 1
            print(wrong_facts[i])
            # if '. ' in explanations[i]:
            #     explanations[i] = explanations[i].split('. ')
            #     for j in range(1,len(explanations[i])):
            #         first_word = explanations[i][j].strip().split(' ')[0].strip()
            #         first_words_in_additional.append(first_word)

            print(explanations[i])
            print("----------------------------------------------------------------------------------")
    from collections import Counter
    counter = Counter(first_words_in_additional)
    print(counter)


def main():
    prompt = """I have a text. A part of the information in the text is wrong. I will give you the text and the wrong information. 
    Your task is to categorize this mistake in one or 2 words. Output the category, and then a short explanation. 
    The categories should be things like: name, time, location and so on.
    The output should be in the format of:
    Category: [category]
    Explanation: [explanation]
    """

    df = pd.read_csv("DeFacto/data/summaries_with_errors.csv", index_col=0)
    instructions = df['instruction'].tolist()
    model_summaries = df['model_summary'].tolist()
    instructions = [split_instructions(instruction) for instruction in instructions]
    wrong_facts = [[extract_the_mistakes(item) for item in instruction] for instruction in instructions]
    flattened_list_wrong_facts = [fact for wrong_fact_per_summary in wrong_facts for fact in wrong_fact_per_summary]
    flattened_list_wrong_facts = [x for x in flattened_list_wrong_facts if x is not None]
    model = LLama(model_id='meta-llama/Meta-Llama-3.1-70B-Instruct', device='auto', dtype=torch.bfloat16)
    model_inputs = []
    df_summaries = []
    facts = []
    for i in range(len(model_summaries)):
        summary = model_summaries[i]
        summary_wrong_facts = wrong_facts[i]
        for fact in summary_wrong_facts:
            if fact is not None:
                input = prompt + '\n' + 'Text: \n' + summary + '\n' + 'Mistake: \n' + fact
                model_inputs.append(input)
                df_summaries.append(summary)
                facts.append(fact)
    print("The number of inputs is: ", len(model_inputs))
    outputs = []
    # categories = []
    # explanations = []
    for batch_inputs in tqdm(iter_list(model_inputs, 8)):
        batch_outputs = model.call(inputs=batch_inputs, generation_kwargs={'max_new_tokens': 1000},
                                   tokenizer_kwargs={'truncation': True, 'padding': 'longest', 'max_length': 2048})
        outputs.extend(batch_outputs)
        # categories.extend([output.split('Category:')[1].split('Explanation:')[0].strip() for output in batch_outputs])
        # explanations.extend([output.split('Explanation:')[1].strip() for output in batch_outputs])
        pd.DataFrame.from_dict(
            {'summary': df_summaries[:len(outputs)], 'fact': facts[:len(outputs)], 'output': outputs}).to_csv(
            "DeFacto/data/llama_3.1_70B_instruct_mistake_categorization_temp.csv")

    pd.DataFrame.from_dict({'summary': df_summaries, 'fact': facts, 'output': outputs}).to_csv(
        "DeFacto/data/llama_3.1_70B_instruct_mistake_categorization.csv")


def filter_dict_by_threshold(data_dict, percentage):
    # Initialize a new dictionary to hold the filtered results
    filtered_dict = {}

    # Initialize a variable to keep track of the sum of removed values
    other_sum = 0
    total_sum = sum(data_dict.values())
    threshold = total_sum * percentage
    # Iterate over the items in the original dictionary
    for key, value in data_dict.items():
        # Check if the value is above or equal to the threshold
        if value >= threshold:
            filtered_dict[key] = value
        else:
            # Add the value to the sum of removed values
            other_sum += value

    # If there were any values below the threshold, add them to the 'other' key
    if other_sum > 0:
        filtered_dict['other'] = other_sum

    return filtered_dict


def get_bert_embeddings(texts):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    embeddings = []
    for text in tqdm(texts):
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        print(inputs['input_ids'].shape)
        with torch.no_grad():
            outputs = model(**inputs)
        # Use the mean of token embeddings from the last hidden state
        embedding = outputs.last_hidden_state[:, 1:-1].mean(dim=1).squeeze().numpy()
        embeddings.append(embedding)
    return np.array(embeddings)


import string


def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))


def analysis():
    from collections import Counter
    counter = 0
    df = pd.read_csv(
        "/data/home/yehonatan-pe/Correction_pipeline/DeFacto/data/llama_3.1_70B_instruct_mistake_categorization.csv",
        index_col=0)
    outputs = df['output'].tolist()
    final_outputs = []
    categories = []
    for output in outputs:
        original_output = output
        output = output.split('assistant\n')[-1].lower().strip()
        if 'category:' in output:
            output = remove_punctuation(output)
            category = output.split('category')[-1].strip().split('explanation')[0].strip().lower()
            if '\n' in category:
                print("The category is: ", category)
            if len(category.split()) > 1:
                print("The category is: ", category)
            explanation = output.split('explanation:')[-1].strip()
            categories.append(category)
            final_outputs.append(original_output)
        # elif ':' in output.split(' ')[0]:
        #     category = output.split(':')[0].strip().lower()
        #     if len(category.split())>1:
        #         print("The category is: ", category)
        #     categories.append(category)
        else:
            counter += 1
    # bert = BertModel.from_pretrained('bert-base-uncased')
    # bert.eval()
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    counter = Counter(categories)
    counter = filter_dict_by_threshold(counter, 0.001)
    main_categories = list(Counter(counter).keys())
    main_categories = [x for x in main_categories if x != '']
    embeddings = get_bert_embeddings(main_categories)
    #
    wcss = []
    for i in range(2, 50):
        kmeans = KMeans(n_clusters=i, random_state=0).fit(embeddings)
        wcss.append(kmeans.inertia_)
        clusters = kmeans.labels_
        # pca = PCA(n_components=2)
        # reduced_embeddings = pca.fit_transform(embeddings)
        # plt.figure(figsize=(10, 7))
        # scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=clusters, cmap='viridis')
        #
        # # Add labels
        # plt.title(f'BERT Embeddings Clustering for {i} clusters')
        # plt.colorbar(scatter, label='Cluster')
        # plt.show()
    plt.plot(range(2, 50), wcss)
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()
    # possible_range1 = range(20,30)
    # kmeans = KMeans(n_clusters=10, random_state=0).fit(embeddings)
    # clusters = kmeans.labels_
    # map_cluster_to_categories_list = {j: [main_categories[k] for k in range(len(main_categories)) if clusters[k] ==j] for j in range(10)}
    normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1)[:, None]
    similarity_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)
    c = 1
    # ok_categories = ['name','location','time','date','number','organization','title','event','money','position']
    max_per_category = 5
    appearances = {category: 0 for category in main_categories}
    for i in range(len(final_outputs)):
        if categories[i] in main_categories and appearances[categories[i]] < max_per_category:
            appearances[categories[i]] += 1
            print(categories[i])
            print(final_outputs[i])
            print("----------------------------------------------------------------------------------")

    # for i in range(len(categories)):
    #     if categories[i] in ok_categories:
    #         print(categories[i])
    #         print(outputs[i])
    #         print("----------------------------------------------------------------------------------")

    # print("The number of outputs without category is: ", counter)
    # print("The number of outputs with category is: ", len(categories))
    # categories_counter = Counter(categories)
    # filtered_categories = filter_dict_by_threshold(categories_counter, 0.012)
    #
    # plt.bar(filtered_categories.keys(), filtered_categories.values())
    # plt.show()
    # print(Counter(categories))


def mistakes_with_instructions_that_include_and():
    df = pd.read_csv("DeFacto/data/summaries_with_errors.csv", index_col=0)
    instructions = df['instruction'].tolist()
    explanations = df['explanation'].tolist()
    extrinsic_errors = df['extrinsic_error'].tolist()
    intrinsic_errors = df['intrinsic_error'].tolist()
    summaries = df['model_summary'].tolist()
    instructions = [split_instructions(instruction) for instruction in instructions]
    wrong_facts = [[extract_the_mistakes(item) for item in instruction] for instruction in instructions]
    counter = 0
    for facts, summary in zip(wrong_facts, summaries):
        for item in facts:
            if item is not None and 'and' in item.split():
                print(summary)
                print(item)
                print("----------------------------------------------------------------------------------")
                counter+=1
    print("The number of mistakes with 'and' in the instruction is: ", counter)

if __name__ == '__main__':
    mistakes_with_instructions_that_include_and()
    # get_mistakes()
    # main()
    # analysis()
