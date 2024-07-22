import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import json
from nltk.tokenize import word_tokenize
from transformers import T5Tokenizer


def main():
    common_args = {}
    for main_dir in range(5, 20):
        print(main_dir)
        path = f"/data/home/yehonatan-pe/Correction_pipeline/experiments/multi_stage_refinement/data/{str(main_dir)}"
        try:
            with open(os.path.join(path, 'args.json'), 'r') as f:
                args = json.load(f)
            if common_args == {}:
                for key in args:
                    common_args[key] = args[key]
            else:
                for key in args:
                    if key in common_args:
                        if common_args[key] != args[key]:
                            common_args.pop(key)
            f.close()
        except:
            continue
    results = {}
    for main_dir in range(5, 20):
        print(main_dir)
        try:
            path = f"/data/home/yehonatan-pe/Correction_pipeline/experiments/multi_stage_refinement/data/{str(main_dir)}"
            with open(os.path.join(path, 'args.json'), 'r') as f:
                args = json.load(f)
            for key in common_args:
                if key in args:
                    args.pop(key)
            f.close()
            dir_paths = os.listdir(path)
            dir_paths = [x for x in dir_paths if 'iter_' in x]
            dir_paths = sorted(dir_paths, key=lambda x: (
                int(x.split('_')[1]), int(x.split('_')[3]) if len(x.split('_')) >= 3 else float('inf')))
            for dir_path in dir_paths:
                df = pd.read_csv(os.path.join(path, dir_path, 'test_results.csv'), index_col=0)
                tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-xl')
                df['model_summary_tokens'] = [len(tokenizer(x)['input_ids']) for x in df['model_summary']]
                df['new_model_summary_tokens'] = [len(tokenizer(x)['input_ids']) for x in df['new_model_summary']]
                df['diffs'] = df['new_model_summary_tokens'] - df['model_summary_tokens']
                df['model_summary_length'] = [len(word_tokenize(x)) for x in df['model_summary']]
                df['new_model_summary_length'] = [len(word_tokenize(x)) for x in df['new_model_summary']]
                df['length_diffs'] = df['new_model_summary_length'] - df['model_summary_length']
                if main_dir not in results:
                    results[main_dir] = {}
                    results[main_dir]['steps'] = []
                    results[main_dir]['mean_diffs_first_500'] = []
                    results[main_dir]['mean_diffs_last_500'] = []
                    results[main_dir]['mean_length_diffs_first_500'] = []
                    results[main_dir]['mean_length_diffs_last_500'] = []
                    results[main_dir]['mean_seahorse_first_500'] = []
                    results[main_dir]['mean_seahorse_last_500'] = []
                    results[main_dir]['mean_density_first_500'] = []
                    results[main_dir]['mean_density_last_500'] = []
                    results[main_dir]['mean_rougeL_first_500'] = []
                    results[main_dir]['mean_rougeL_last_500'] = []
                if 'batch' in dir_path:
                    iteration_docs = common_args['num_of_docs'] * (int(dir_path.split('_')[1]))

                    batch_docs = args['set_size'] * int(dir_path.split('_')[3])
                    results[main_dir]['steps'].append(iteration_docs + batch_docs)
                else:
                    iteration_docs = common_args['num_of_docs'] * (int(dir_path.split('_')[1]) + 1)
                    results[main_dir]['steps'].append(iteration_docs)

                results[main_dir]['mean_diffs_first_500'].append(np.mean(df['diffs'].iloc[:500]))
                results[main_dir]['mean_diffs_last_500'].append(np.mean(df['diffs'].iloc[-500:]))
                results[main_dir]['mean_length_diffs_first_500'].append(np.mean(df['length_diffs'].iloc[:500]))
                results[main_dir]['mean_length_diffs_last_500'].append(np.mean(df['length_diffs'].iloc[-500:]))
                results[main_dir]['mean_seahorse_first_500'].append(
                    np.mean(df['new_model_summary_seahorse'].iloc[:500]))
                results[main_dir]['mean_seahorse_last_500'].append(
                    np.mean(df['new_model_summary_seahorse'].iloc[-500:]))
                results[main_dir]['mean_density_first_500'].append(np.mean(df['new_model_summary_density'].iloc[:500]))
                results[main_dir]['mean_density_last_500'].append(np.mean(df['new_model_summary_density'].iloc[-500:]))
                results[main_dir]['mean_rougeL_first_500'].append(
                    np.mean(df['new_model_summary_rougeL_to_base'].iloc[:500]))
                results[main_dir]['mean_rougeL_last_500'].append(
                    np.mean(df['new_model_summary_rougeL_to_base'].iloc[-500:]))
        except Exception as e:
            print(str(e))

    for key in results[5].keys():
        for main_dir in results.keys():
            plt.plot(results[main_dir]['steps'], results[main_dir][key], label=main_dir)
        plt.legend()
        plt.title(key)
        plt.show()


def check_mid_refinemtns():
    path = "/data/home/yehonatan-pe/Correction_pipeline/experiments/multi_stage_refinement/data"
    for i in [23]:
        new_path = os.path.join(path, str(i))
        import json
        with open(os.path.join(new_path, 'refinement_inputs.json'), 'r') as f:
            data = json.load(f)
        f.close()
        tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-xl')
        original_summaries = data.pop('0')
        original_summaries = original_summaries['original_summaries']
        for key in data:
            all_new_tokens_lengths = [len(tokenizer(x)['input_ids']) for x in data[key]['new_summaries']]
            plt.hist(all_new_tokens_lengths, bins=20, alpha=0.5, label='original',weights=np.ones(len(all_new_tokens_lengths))/len(all_new_tokens_lengths))
            no_revision_tokens_lengths = [len(tokenizer(x)['input_ids']) for x in data[key]['no_revision_summaries']]
            plt.hist(no_revision_tokens_lengths, bins=20, alpha=0.5, label='no_revision_summaries',weights=np.ones(len(no_revision_tokens_lengths))/len(no_revision_tokens_lengths))
            needed_revision_tokens_lengths = [len(tokenizer(x)['input_ids']) for x in data[key]['summaries_which_were_revised']]
            plt.hist(needed_revision_tokens_lengths, bins=20, alpha=0.5, label='to be revised',weights=np.ones(len(needed_revision_tokens_lengths))/len(needed_revision_tokens_lengths))
            successfull_revision_tokens_lengths = [len(tokenizer(x)['input_ids']) for x in data[key]['successfully_revised_summaries']]
            plt.hist(successfull_revision_tokens_lengths, bins=20, alpha=0.5, label='revised',weights=np.ones(len(successfull_revision_tokens_lengths))/len(successfull_revision_tokens_lengths))
            plt.legend()
            plt.show()
            print("The mean and the median of all summaries were: ",np.mean(all_new_tokens_lengths),np.median(all_new_tokens_lengths))
            print("The mean and the median of no revision summaries were: ",np.mean(no_revision_tokens_lengths),np.median(no_revision_tokens_lengths))
            print("The mean and the median of summaries which were revised were: ",np.mean(needed_revision_tokens_lengths),np.median(needed_revision_tokens_lengths))
            print("The mean and the median of successfully revised summaries were: ",np.mean(successfull_revision_tokens_lengths),np.median(successfull_revision_tokens_lengths))


if __name__ == '__main__':
    check_mid_refinemtns()
    #main()
