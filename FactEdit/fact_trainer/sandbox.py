import json
import os
from tqdm import tqdm

with open('xsum_corr_data/test.json') as data_file:
    x = []
    for line in tqdm(data_file):
        x.append(json.loads(line))