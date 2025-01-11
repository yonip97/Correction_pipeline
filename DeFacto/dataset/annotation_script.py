import json
import os

import json
import os
def wrap_text(text, width=120):
    """Wrap text to a specified width."""
    return '\n'.join([text[i:i+width] for i in range(0, len(text), width)])

def show_comparison_interface(texts, summaries, llm_explanations, human_explanations, output_file):
    # Load existing results if any
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            results = json.load(f)
    else:
        results = []

    # Determine the starting index
    start_index = len(results)
    texts = texts[start_index:]
    summaries = summaries[start_index:]
    llm_explanations = llm_explanations[start_index:]
    human_explanations = human_explanations[start_index:]

    for i, (text, summary, llm_output, human_output) in enumerate(
        zip(texts, summaries, llm_explanations, human_explanations), start=start_index
    ):
        print("=" * 50)
        print(f"Entry {i + 1}:")
        print("Text:")
        print(wrap_text(text))
        print("\nSummary:")
        print(summary)
        print("\nLLM Explanation:")
        print(llm_output)
        print("\nHuman Explanation:")
        print(human_output)
        print("=" * 50)

        # Questions for manual input
        both_identified = input("1. What facts did the LLM and the human both identify? (Separate with commas): ")
        correct_llm = input("2. What facts did the LLM identify which are correct? (Separate with commas): ")
        wrong_llm = input("3. What facts did the LLM identify which are wrong? (Separate with commas): ")
        maybe_llm = input("4. What facts did the LLM identify which are maybe correct? (Separate with commas): ")
        notes = input("5. Do you have any notes? (Optional, press Enter to skip): ")

        # Validate revisit input
        while True:
            revisit = input("6. Should you come back to this entry? (1 for yes, 0 for no): ").strip()
            if revisit in ["1", "0"]:
                revisit = int(revisit)
                break
            else:
                print("Invalid input. Please enter 1 for yes or 0 for no.")

        # Store the answers
        result = {
            "entry": i + 1,
            "both_identified": [fact.strip() for fact in both_identified.split(",") if fact.strip()],
            "correct_llm": [fact.strip() for fact in correct_llm.split(",") if fact.strip()],
            "wrong_llm": [fact.strip() for fact in wrong_llm.split(",") if fact.strip()],
            "maybe_llm": [fact.strip() for fact in maybe_llm.split(",") if fact.strip()],
            "notes": notes,
            "revisit": revisit,
        }
        results.append(result)

        # Save results after each entry to ensure progress is not lost
        with open(output_file, "w") as f:
            json.dump(results, f, indent=4)

        # Option to stop annotation
        stop = input("Do you want to stop the annotation? (yes to stop, no to continue): ").strip().lower()
        if stop == "yes":
            print("Annotation stopped. You can resume later.")
            break
def annotate():
    import pandas as pd
    df_llm = pd.read_csv("/data/home/yehonatan-pe/Correction_pipeline/DeFacto/dataset/data/llm_inference/fine_grain_classification/spans_and_explanations/prompt4/results_gpt_4o_all_processed_dataset.csv",index_col=0)
    llm_outputs = df_llm['output'].tolist()
    df_human = pd.read_csv("/data/home/yehonatan-pe/Correction_pipeline/DeFacto/dataset/data/2_possible_annotation_almost_final_dataset_fact_span_explanation_format.csv",index_col=0)
    human_annotation = df_human['explanation'].tolist()
    texts = df_human['text'].tolist()
    summaries = df_human['model_summary'].tolist()
    output_file = "/data/home/yehonatan-pe/Correction_pipeline/DeFacto/dataset/data/annotation_of_factually_inconsistent_explanations.json"
    show_comparison_interface(texts, summaries, llm_outputs, human_annotation, output_file)
def create_final_version():
    import pandas as pd
    df = pd.read_csv("/data/home/yehonatan-pe/Correction_pipeline/DeFacto/dataset/data/2_possible_annotation_almost_final_dataset.csv")
    with open("/data/home/yehonatan-pe/Correction_pipeline/DeFacto/dataset/data/annotation_of_factually_inconsistent_explanations.json", "r") as f:
        results = json.load(f)
    grouped_df = (
        df.groupby(["text", "model_summary"], sort=False)["explanation"]
        .apply(list)
        .reset_index()
    )
    full_llm_outputs = pd.read_csv("/data/home/yehonatan-pe/Correction_pipeline/DeFacto/dataset/data/llm_inference/fine_grain_classification/spans_and_explanations/prompt4/results_gpt_4o_all_processed_dataset.csv",index_col=0)
    texts = grouped_df["text"].tolist()
    summaries = grouped_df["model_summary"].tolist()
    explanations = grouped_df["explanation"].tolist()
    final_dataset = []
    for i in range(50):
        text = texts[i]
        summary = summaries[i]
        explanation = explanations[i]
        result = results[i]
        added_explanations = []
        print(full_llm_outputs.loc[i, "output"])
        print("------------------------------------------------------------")
        print(result)
        print("------------------------------------------------------------")
        print(explanation)
        print("------------------------------------------------------------")
        change_all_explanations = input("Do you want to change all explanations? (yes or no): ").strip().lower()
        if change_all_explanations == "yes":
            explanation = []
        while True:
            are_there_more_explanations = input("Are there more explanations for this entry? (yes or no): ").strip().lower()
            if are_there_more_explanations == "yes":
                new_explanation = input("Enter the new explanation: ")
                added_explanations.append(new_explanation)
            else:
                break
        explanation += added_explanations
        final_dataset.append(
            {
                "text": text,
                "model_summary": summary,
                "explanations": explanation,
            }
        )
    final_df = pd.DataFrame(final_dataset)
    final_df.to_csv("/data/home/yehonatan-pe/Correction_pipeline/DeFacto/dataset/data/final_dataset_50_samples.csv", index=False)




create_final_version()
