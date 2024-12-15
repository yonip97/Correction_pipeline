import pandas as pd
from scipy.optimize import linear_sum_assignment




def read_results():
    df = pd.read_csv("data/2_possible_annotation_almost_final_dataset.csv")
    group_df = df.groupby(['text', 'model_summary'])['minimal text span'].apply(list)
    df = df[['text', 'model_summary']]
    df.drop_duplicates(inplace=True)
    df = df.merge(group_df, on=['text', 'model_summary'], how='left')
    df = df[['text', 'model_summary', 'minimal text span']]
    df2 = pd.read_csv('data/natural_explanation/prompt2/results_llama_3_1_70B_instruct.csv', index_col=0)
    counter = 0
    import textwrap

    for i in range(100):
        print(textwrap.fill(df['text'][i], width=200))
        print(df['model_summary'][i])
        print(df['minimal text span'][i])
        counter += len(df['minimal text span'][i])
        print(textwrap.fill(str(df2['output'][i]), width=200))
        print(counter)
        print('-----------------------------------')


def extract_facts(output):
    if 'Fact:' not in str(output):
        return []
    facts = output.split('Fact:')
    facts = [fact.strip() for fact in facts][1:]
    facts = [fact.split('Explanation:')[0].strip() for fact in facts]
    return facts


def are_in_summary(facts, summary):
    for fact in facts:
        if fact not in summary:
            return False
    return True


def are_equal(facts1, facts2):
    if len(facts1) != len(facts2):
        return False
    for fact in facts1:
        if fact not in facts2:
            return False
    return True


def compare_results():
    df1 = pd.read_csv('data/mistake_detection/prompt2/results_gpt_4_100.csv', index_col=0)
    df2 = pd.read_csv('data/atomic_facts/prompt2/results_gpt_4_100.csv', index_col=0)
    for i in range(100):
        print(i)
        print(df1['model_summary'][i])
        print()
        print()
        print(df1['output'][i])
        print()
        print()
        print(df2['output'][i])
        print()
        # facts1 = extract_facts(df1['output'][i])
        # facts2 = extract_facts(df2['output'][i])
        # print(are_equal(facts1,facts2))
        # print(are_in_summary(facts1,df1['model_summary'][i]))
        # print(are_in_summary(facts2,df2['model_summary'][i]))
        print('------------------------------------------------------------------------------------')


def word_wise_iou(predicted, ground_truth):
    # Tokenize sentences into sets of words
    pred_words = set(predicted.split())
    gt_words = set(ground_truth.split())

    # Compute intersection and union
    intersection = pred_words.intersection(gt_words)
    union = pred_words.union(gt_words)

    # Compute IoU
    iou = len(intersection) / len(union) if union else 0
    return iou


class rougeoncall():
    def __init__(self):
        import evaluate
        self.rouge = evaluate.load('rouge')

    def __call__(self, predicted, ground_truth):
        return self.rouge.compute(predictions=[predicted], references=[ground_truth], use_aggregator=False)['rougeL'][0]


def match(llm_facts, annotated_facts, similarity_metric, threshold=0.1):
    if len(llm_facts) == 0 or len(annotated_facts) == 0:
        return [], 0
    results = []
    for fact in llm_facts:
        fact_results = []
        for annotated_fact in annotated_facts:
            score = similarity_metric(fact, annotated_fact)
            fact_results.append(score * -1)
        results.append(fact_results)
    row_ind, col_ind = linear_sum_assignment(results)
    best_matching = [(llm_facts[row], annotated_facts[col]) for row, col in zip(row_ind, col_ind) if
                     results[row][col] * -1 > threshold]
    best_match_similarity = [results[row][col] * -1 for row, col in zip(row_ind, col_ind) if
                             results[row][col] * -1 > threshold]
    similairty_sum = sum(best_match_similarity)
    return best_matching, similairty_sum


def check_span_metrics():
    df = pd.read_csv("data/mistake_detection/prompt3/results_gpt_4_100.csv", index_col=0)
    annotated_df = pd.read_csv("data/2_possible_annotation_almost_final_dataset.csv")
    texts = df[['text']].drop_duplicates()['text'].tolist()
    sim_sum_minimal = 0
    sim_sum_expanded = 0
    for text in texts:
        temp_annotated_df = annotated_df[annotated_df['text'] == text]
        temp_df = df[df['text'] == text]
        llm_facts = extract_facts(temp_df['output'].tolist()[0])
        print(llm_facts)
        annotated_facts_minimal_span = temp_annotated_df['minimal text span'].tolist()
        print(annotated_facts_minimal_span)
        annotated_facts_expanded_span = temp_annotated_df['expanded text span'].tolist()
        print(annotated_facts_expanded_span)
        matched, sim_sum = match(llm_facts, annotated_facts_minimal_span, rougeoncall())
        sim_sum_minimal += sim_sum
        matched, sim_sum = match(llm_facts, annotated_facts_expanded_span, rougeoncall())
        sim_sum_expanded += sim_sum
        # for fact in llm_facts:
        #     x = rouge.compute(predictions=[fact] * len(annotated_facts_minimal_span),
        #                       references=annotated_facts_minimal_span, use_aggregator=False)
        #     print(x)
        #     x = rouge.compute(predictions=[fact] * len(annotated_facts_expanded_span),
        #                       references=annotated_facts_expanded_span, use_aggregator=False)
        #     print(x)
        print('----------------------------------------------------------')
    print(sim_sum_minimal)
    print(sim_sum_expanded)
