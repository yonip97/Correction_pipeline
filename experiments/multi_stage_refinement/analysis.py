import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import json
from nltk.tokenize import word_tokenize
from transformers import T5Tokenizer


def main():
    common_args = {}
    for main_dir in range(5, 36):
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
    for main_dir in range(23, 37):
        print(main_dir)
        try:
            path = f"/data/home/yehonatan-pe/Correction_pipeline/experiments/multi_stage_refinement/data/{str(main_dir)}"
            with open(os.path.join(path, 'args.json'), 'r') as f:
                args = json.load(f)
            # for key in common_args:
            #     if key in args:
            #         args.pop(key)
            f.close()
            dir_paths = os.listdir(path)
            dir_paths = [x for x in dir_paths if 'iter_' in x]
            dir_paths = sorted(dir_paths, key=lambda x: (
                int(x.split('_')[1]), int(x.split('_')[3]) if len(x.split('_')) >= 3 else float('inf')))
            for dir_path in dir_paths:
                df = pd.read_csv(os.path.join(path, dir_path, 'test_results.csv'), index_col=0)
                if len(df) != 1000:
                    print("The length of the df is not 1000")
                    continue
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
                    # results[main_dir]['mean_rougeL_first_500'] = []
                    # results[main_dir]['mean_rougeL_last_500'] = []
                if 'batch' in dir_path:
                    iteration_docs = args['num_of_docs'] * (int(dir_path.split('_')[1]))

                    batch_docs = args['set_size'] * int(dir_path.split('_')[3])
                    results[main_dir]['steps'].append(iteration_docs + batch_docs)
                else:
                    iteration_docs = args['num_of_docs'] * (int(dir_path.split('_')[1]) + 1)
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
                # results[main_dir]['mean_rougeL_first_500'].append(
                #     np.mean(df['new_model_summary_rougeL_to_base'].iloc[:500]))
                # results[main_dir]['mean_rougeL_last_500'].append(
                #     np.mean(df['new_model_summary_rougeL_to_base'].iloc[-500:]))
        except Exception as e:
            print(str(e))

    for key in results[29].keys():
        for main_dir in results.keys():
            plt.scatter(results[main_dir]['steps'], results[main_dir][key], label=main_dir)
        plt.legend()
        plt.title(key)
        plt.show()


def check_mid_refinemtns():
    path = "/data/home/yehonatan-pe/Correction_pipeline/experiments/multi_stage_refinement/data"
    for i in range(5, 36):
        new_path = os.path.join(path, str(i))
        import json
        try:
            with open(os.path.join(new_path, 'refinement_inputs.json'), 'r') as f:
                data = json.load(f)
            f.close()
            tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-xl')
            original_summaries = data.pop('0')
            original_summaries = original_summaries['original_summaries']
            for key in data:
                all_new_tokens_lengths = [len(tokenizer(x)['input_ids']) for x in data[key]['new_summaries']]
                # plt.hist(all_new_tokens_lengths, bins=20, alpha=0.5, label='original',
                #          weights=np.ones(len(all_new_tokens_lengths)) / len(all_new_tokens_lengths))
                no_revision_tokens_lengths = [len(tokenizer(x)['input_ids']) for x in
                                              data[key]['no_revision_summaries']]
                # plt.hist(no_revision_tokens_lengths, bins=20, alpha=0.5, label='no_revision_summaries',
                #          weights=np.ones(len(no_revision_tokens_lengths)) / len(no_revision_tokens_lengths))
                needed_revision_tokens_lengths = [len(tokenizer(x)['input_ids']) for x in
                                                  data[key]['summaries_which_were_revised']]
                # plt.hist(needed_revision_tokens_lengths, bins=20, alpha=0.5, label='to be revised',
                #          weights=np.ones(len(needed_revision_tokens_lengths)) / len(needed_revision_tokens_lengths))
                successfull_revision_tokens_lengths = [len(tokenizer(x)['input_ids']) for x in
                                                       data[key]['successfully_revised_summaries']]
                # plt.hist(successfull_revision_tokens_lengths, bins=20, alpha=0.5, label='revised',
                #          weights=np.ones(len(successfull_revision_tokens_lengths)) / len(
                #              successfull_revision_tokens_lengths))
                # plt.legend()
                # plt.show()
                print("The mean and the median of all summaries were: ", np.mean(all_new_tokens_lengths),
                      np.median(all_new_tokens_lengths))
                print("The mean and the median of no revision summaries were: ", np.mean(no_revision_tokens_lengths),
                      np.median(no_revision_tokens_lengths))
                print("The mean and the median of summaries which were revised were: ",
                      np.mean(needed_revision_tokens_lengths), np.median(needed_revision_tokens_lengths))
                print("The mean and the median of successfully revised summaries were: ",
                      np.mean(successfull_revision_tokens_lengths), np.median(successfull_revision_tokens_lengths))
        except Exception as e:
            print(str(e))


def find_different_keys(dict1, dict2):
    different_keys = []

    # Compare keys and values in both dictionaries
    for key in dict1:
        if key in dict2 and dict1[key] != dict2[key]:
            different_keys.append(key)

    # Optionally, include keys that are only in one dictionary
    for key in dict2:
        if key not in dict1:
            different_keys.append(key)

    for key in dict1:
        if key not in dict2:
            different_keys.append(key)

    return different_keys


def check_all_iter_results():
    path = "/data/home/yehonatan-pe/Correction_pipeline/experiments/multi_stage_refinement/data"
    prev_args = None
    for i in range(23):
        new_path = os.path.join(path, str(i))
        for dir in os.listdir(new_path):
            if 'batch' in dir or 'checkpoints' in dir or os.path.isdir(os.path.join(new_path, dir)) is False:
                continue
            try:
                with open(os.path.join(new_path, 'args.json'), 'r') as f:
                    args = json.load(f)
                if prev_args is None:
                    prev_args = args
                else:
                    different_keys = find_different_keys(prev_args, args)
                    diff_args = {key: args[key] for key in different_keys if key in args}
                    print(args)
                    print(diff_args)
                    prev_args = args
                print()

                print(args['num_of_docs'] / args['set_size'])
                print('revision penalty',args['revision_length_penalty'])
                final_path = os.path.join(new_path, dir, 'test_results.csv')
                df = pd.read_csv(final_path, index_col=0)
                df = df[500:]
                print(final_path)
                print(df['new_model_summary_length'].mean())

                print(df['new_model_summary_seahorse'].mean())
                print(df['new_model_trueteacher'].mean())
                print(df['new_model_summary_density'].mean())
                print(df['new_model_summary_coverage'].mean())
                print(df['new_model_summary_rougeL_to_base'].mean())
                # print(df.columns)
            except Exception as e:
                print(str(e))
                final_path = os.path.join(new_path, dir, 'test_results.csv')
                print(final_path)
                df = pd.read_csv(final_path, index_col=0)
                df = df[500:]
                # summaries_length = [len(word_tokenize(x)) for x in df['revised_summary']]
                if 'new_model_summary' in df.columns:
                    summaries_length = [len(word_tokenize(x)) for x in df['new_model_summary']]
                    print(np.mean(summaries_length))
                print(df['revised_summary_seahorse'].mean())
                #print(df['new_model_trueteacher'].mean())
                print(df['revised_summary_density'].mean())
                print(df['revised_summary_coverage'].mean())
                print(df['revised_summary_rougeL_to_base'].mean())

            # print(np.mean(summaries_length))
            # continue
        print("----------------------------------------------------------")


def check_all_iter_results_distillation():
    path = "/data/home/yehonatan-pe/Correction_pipeline/experiments/multi_stage_refinement/distillation_data"
    for i in range(10):
        new_path = os.path.join(path, str(i))
        for dir in os.listdir(new_path):
            if 'batch' in dir or 'checkpoints' in dir or os.path.isdir(os.path.join(new_path, dir)) is False:
                continue
            try:

                with open(os.path.join(new_path, 'args.json'), 'r') as f:
                    args = json.load(f)
                print(args['use_no_revision_needed'])
                print(args['need_revision_factuality_threshold'])
                final_path = os.path.join(new_path, dir, 'test_results.csv')
                df = pd.read_csv(final_path, index_col=0)
                df = df[500:]
                print(df.columns)
                print(final_path)
                print(df['new_model_summary_length'].mean())

                print(df['new_model_summary_seahorse'].mean())
                print(df['new_model_trueteacher'].mean())
                print(df['new_model_summary_density'].mean())
                print(df['new_model_summary_coverage'].mean())
                print(df['new_model_summary_rougeL_to_base'].mean())
            except Exception as e:
                continue
            finally:
                print("----------------------------------------------------------")


def compare_args():
    distillation_path = "/data/home/yehonatan-pe/Correction_pipeline/experiments/multi_stage_refinement/distillation_data"
    path = "/data/home/yehonatan-pe/Correction_pipeline/experiments/multi_stage_refinement/data/22"
    dirs = os.listdir(distillation_path)
    dirs = [int(x) for x in dirs]
    dirs = sorted(dirs)
    for dir in dirs:
        dir = str(dir)
        with open(os.path.join(distillation_path, dir, 'args.json'), 'r') as f:
            args = json.load(f)
        with open(os.path.join(path, 'args.json'), 'r') as f:
            args2 = json.load(f)
        try:
            diff = find_different_keys(args, args2)
            diff_args = {key: args[key] for key in diff if key in args}
            diff_args2 = {key: args2[key] for key in diff if key in args2}
            df_distillation = pd.read_csv(os.path.join(distillation_path, dir, 'iter_0/test_results.csv'), index_col=0)
            df = pd.read_csv(os.path.join(path, 'iter_0/test_results.csv'), index_col=0)
            print('Distillation stats:')
            print(df_distillation['new_model_summary_length'].mean())
            print(df_distillation['new_model_summary_seahorse'].mean())
            print(df_distillation['new_model_summary_density'].mean())
            print('Normal stats:')
            print(df['new_model_summary_length'].mean())
            print(df['new_model_summary_seahorse'].mean())
            print(df['new_model_summary_density'].mean())
            #print(dir2,dir)
            print(dir)
            print(len(diff))
            print(diff_args)
            print(diff_args2)
            print("-----------------------------------------------------")
        except Exception as e:
            print(str(e))
            continue

def find_example_with_all():
    path = "/data/home/yehonatan-pe/Correction_pipeline/experiments/multi_stage_refinement/data"
    prev_args = None
    length_threshold = 10
    density_threshold = 4
    trueteacher_threshold = 0.3
    examples = {}
    for i in range(23):
        new_path = os.path.join(path, str(i))
        for dir in os.listdir(new_path):
            if 'batch' in dir or 'checkpoints' in dir or os.path.isdir(os.path.join(new_path, dir)) is False:
                continue
            try:
                with open(os.path.join(new_path, 'args.json'), 'r') as f:
                    args = json.load(f)
                if prev_args is None:
                    prev_args = args
                else:
                    different_keys = find_different_keys(prev_args, args)
                    diff_args = {key: args[key] for key in different_keys if key in args}
                    print(args)
                    print(diff_args)
                    prev_args = args
                print()

                print(args['num_of_docs'] / args['set_size'])
                print('revision penalty', args['revision_length_penalty'])
                final_path = os.path.join(new_path, dir, 'test_results.csv')
                df = pd.read_csv(final_path, index_col=0)
                df = df[500:]
                df_with_length = df[df['new_model_summary_length'] > length_threshold]
                df_with_density = df[df['new_model_summary_density'] > density_threshold]
                df_with_trueteacher = df[df['new_model_trueteacher'] < trueteacher_threshold]
                examples[i] = {}
                examples[i]['length'] = df_with_length['indices'].tolist()
                examples[i]['density'] = df_with_density['indices'].tolist()
                examples[i]['trueteacher'] = df_with_trueteacher['indices'].tolist()
            except Exception as e:
                print(str(e))
                continue
    for i in range(6,23):
        for j in range(i + 1, 23):
            for l in range(j+1,23):
                length = examples[i]['length']
                density = examples[j]['density']
                trueteacher = examples[l]['trueteacher']
                intersection = list(set(length) & set(density) & set(trueteacher))
                if len(intersection) > 0:
                    print(i,j,l)
                    print(intersection)
                    print(len(intersection))
                    print('---------------------------------------------------')


if __name__ == '__main__':
    #compare_args()
    #check_all_iter_results_distillation()
    check_all_iter_results()
    # check_mid_refinemtns()
    # df= pd.read_csv("/data/home/yehonatan-pe/Correction_pipeline/experiments/multi_stage_refinement/data/19/iter_0/test_results.csv", index_col=0)
    # df2 = pd.read_csv("/data/home/yehonatan-pe/Correction_pipeline/experiments/revision/data/base_model_50000_documents/dev_set/base_model_summaries_500_not_factual_500_factual.csv",index_col=0)
    # df = df[df['indices'].isin(df2['indices'])]
    # print(len(df))
    # print(df.columns)
    # print(df['new_model_summary_seahorse'].mean())
    # print(df['new_model_summary_density'].mean())
    # main()
    find_example_with_all()