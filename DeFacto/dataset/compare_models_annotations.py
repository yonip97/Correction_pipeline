import pandas as pd
import os

#
# # Path to the directory containing the files
data_dir = "/data/home/yehonatan-pe/Correction_pipeline/DeFacto/dataset/data/llm_inference/fine_grain_classification/spans_and_explanations/prompt6"
#
# # Load your dataframes
# claude_df = pd.read_csv(os.path.join(data_dir, "results_claude-3-5-sonnet-20241022_50.csv"))
# gemini_df = pd.read_csv(os.path.join(data_dir, "results_gemini-1.5_50.csv"))
# gpt_df = pd.read_csv(os.path.join(data_dir, "results_gpt_4o_50_.csv"))
#
# # Combine the DataFrames into one for comparison
# data = pd.DataFrame({
#     'claude': claude_df['output'],
#     'gemini': gemini_df['output'],
#     'gpt': gpt_df['output']
# })
#
#
# annotation_file = os.path.join(data_dir, "annotations.csv")
#
# # Initialize annotations if the file doesn't exist
# if not os.path.exists(annotation_file):
#     annotations = pd.DataFrame(columns=['Index', 'Question', 'Answer'])
#     annotations.to_csv(annotation_file, index=False)
# else:
#     annotations = pd.read_csv(annotation_file)
#
# # Questions to answer for each comparison
# questions = ["What did gpt found that was not in the dataset?","What did gemini found that was not in the dataset?","What did claude found that was not in the dataset?"]
#
# # Iterate through each row and collect annotations
# for idx, row in data.iterrows():
#     print(f"\nComparison {idx + 1}:")
#     print(f"claude {row['claude']}")
#     print(f"gemini: {row['gemini']}")
#     print(f"gpt: {row['gpt']}")
#
#     for question in questions:
#         print(f"\nQuestion: {question}")
#         answer = input("Your answer: ")
#
#         # Save the annotation
#         annotations = pd.concat([
#             annotations,
#             pd.DataFrame({
#                 'Index': [idx],
#                 'Question': [question],
#                 'Answer': [answer]
#             })
#         ], ignore_index=True)
#
#     # Save progress after each row
#     annotations.to_csv(annotation_file, index=False)
#     print("Annotations saved.")
#
# print("\nAnnotation process complete. Annotations are saved in 'annotations.csv'.")
results = pd.read_csv(os.path.join(data_dir, 'annotations.csv'))
gpt = []
gemini = []
claude = []
for i in range(50):
    temp = results[results['Index'] == i]
    answers = temp['Answer'].tolist()
    if answers[0] != ' ':
        gpt += answers[0].split(',')
    if answers[1] != ' ':
        gemini += answers[1].split(',')
    if answers[2] != ' ':
        claude += answers[2].split(',')
gpt = [x.strip() for x in gpt]
gemini = [x.strip() for x in gemini]
claude = [x.strip() for x in claude]
from matplotlib_venn import venn3
import matplotlib.pyplot as plt

venn = venn3([set(gpt), set(gemini), set(claude)], ('gpt', 'gemini', 'claude'))

# Customize the diagram (optional)
for subset in venn.subset_labels:
    if subset:  # Only modify non-None labels
        subset.set_fontsize(10)

# Display the plot
plt.title("Venn Diagram of Three Sets")
plt.show()
all = set(gpt).union(set(gemini)).union(set(claude))
print(len(all))
print(sorted(all))
