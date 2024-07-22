import pandas as pd
path  = "/data/home/yehonatan-pe/Correction_pipeline/experiments/revision/data/base_model_50000_documents/cot_prompts/cot_prompt_with_error_detection_2/100_random_summaries_revised_scored.csv"
df = pd.read_csv(path,index_col=0)


#print(df['revised_summary_density'].mean())
import matplotlib.pyplot as plt
# plt.hist(df['revised_summary_density'])
# plt.show()
# print(sum(df['revised_summary_density']>5))
# print(len(df[df['revised_summary'].isna()]))
# df.loc[df['revised_summary'].isna(),'revised_summary_density'] = df[df['revised_summary'].isna()]['model_summary_density'].tolist()
# df.loc[df['revised_summary'].isna(),'revised_summary_seahorse'] = df[df['revised_summary'].isna()]['model_summary_seahorse'].tolist()
print(df['revised_summary_density'].mean())
print(df['revised_summary_seahorse'].mean())



for i in range(len(df)):
    print(f"Indices: {df['indices'].iloc[i]}")
    print()
    print(df['text'].iloc[i])
    print()
    print(df['model_summary'].iloc[i])
    print()
    print(df['revised_summary_full_text'].iloc[i])
    print("RougeL",df['rougeL_revised_to_base'].iloc[i])
    print("Density:",df['revised_summary_density'].iloc[i])
    print()
    print("---------------------------------------")
path  = "/data/home/yehonatan-pe/Correction_pipeline/experiments/revision/data/base_model_50000_documents/cot_prompts/original/base_model_outputs_below_0.5_text_length_above_65_10000_samples_revised_scored.csv"
df_temp = pd.read_csv(path,index_col=0)
df = df_temp[df_temp['indices'].isin(df['indices'])]

#print(df['revised_summary_density'].mean())
import matplotlib.pyplot as plt
# plt.hist(df['revised_summary_density'])
# plt.show()
# print(sum(df['revised_summary_density']>5))
# print(len(df[df['revised_summary'].isna()]))
# df.loc[df['revised_summary'].isna(),'revised_summary_density'] = df[df['revised_summary'].isna()]['model_summary_density'].tolist()
# df.loc[df['revised_summary'].isna(),'revised_summary_seahorse'] = df[df['revised_summary'].isna()]['model_summary_seahorse'].tolist()
print(df['revised_summary_density'].mean())
print(df['revised_summary_seahorse'].mean())



for i in range(len(df)):
    print(f"Indices: {df['indices'].iloc[i]}")
    print()
    print(df['text'].iloc[i])
    print()
    print(df['model_summary'].iloc[i])
    print()
    print(df['revised_summary_full_text'].iloc[i])
    print("RougeL",df['rougeL_revised_to_base'].iloc[i])
    print("Density:",df['revised_summary_density'].iloc[i])
    print()
    print("---------------------------------------")