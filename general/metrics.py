import re
from collections import Counter
import evaluate
from tqdm import tqdm

def clean_text(text):
    # Convert to lowercase
    text = text.lower()

    # Remove special characters, numbers, and punctuation
    text = re.sub(r'[^a-z\s]', '', text)

    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def word_wise_f1_score(predicted_text, target_text):
    if predicted_text == '':
        return 0
    gold_toks = clean_text(target_text).split()
    pred_toks = clean_text(predicted_text).split()
    common = Counter(gold_toks) & Counter(pred_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def levenshtein_distance(s1, s2):
    len_s1 = len(s1)
    len_s2 = len(s2)

    # Initialize a 2D matrix to store the distance values
    dp = [[0] * (len_s2 + 1) for _ in range(len_s1 + 1)]

    # Fill in the base cases
    for i in range(len_s1 + 1):
        dp[i][0] = i
    for j in range(len_s2 + 1):
        dp[0][j] = j

    # Calculate the Levenshtein distance using dynamic programming
    for i in range(1, len_s1 + 1):
        for j in range(1, len_s2 + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    # The final Levenshtein distance will be at dp[len_s1][len_s2]
    return dp[len_s1][len_s2]


def preserve_lev(original_summary, new_summary):
    dist = levenshtein_distance(original_summary, new_summary)
    score = max(1 - dist / len(original_summary), 0)
    return score


class Rouge():
    def __init__(self):
        self.rouge = evaluate.load('rouge')

    def __call__(self, predicted_texts: list, target_texts: list):
        return self.rouge.compute(predictions=predicted_texts, references=target_texts, use_aggregator=False)


def add_similarity_metrics_scores(df, col1, col2):
    rouge_metric = Rouge()
    rouge_results = rouge_metric(df[col1].tolist(), df[col2].tolist())
    df['rouge1'] = rouge_results['rouge1']
    df['rouge2'] = rouge_results['rouge2']
    df['rougeL'] = rouge_results['rougeL']
    df['rougeLsum'] = rouge_results['rougeLsum']
    df['preserve_lev'] = df.apply(lambda x: preserve_lev(x['summary'], x['revised_summary']), axis=1)
    f1_scores = []
    for i, row in df.iterrows():
        summary = row['summary']
        revised_summary = row['revised_summary']
        f1_scores.append(word_wise_f1_score(revised_summary, summary))
    df['f1_score'] = f1_scores
    return df
