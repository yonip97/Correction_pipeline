
import math
import os
import pandas as pd
from torch.utils.data import Dataset
from pathlib import Path


def iter_list(list_, batch_size):
    num_of_batches = math.ceil(len(list_) / batch_size)
    for i in range(num_of_batches):
        yield list_[i * batch_size:(i + 1) * batch_size]


def collate_fn(batch):
    datasets = [x[0] for x in batch]
    original_texts = [x[1] for x in batch]
    generated_text = [x[2] for x in batch]
    labels = [x[3] for x in batch]
    return datasets, original_texts, generated_text, labels


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


