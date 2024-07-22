
import os
import pandas as pd
def display_csv_row_by_row(df):
    # Load the CSV file
    relevant_columns = ['text', 'model_summary', 'revised_summary', 'indices','model_summary_seahorse','revised_summary_seahorse','revised_summary_rougeL_to_base']
    df = df[relevant_columns]
    for i in range(len(df)):
        # Clear the terminal screen
        # os.system('clear')  # or 'cls' for Windows

        # Get the current row data
        row_data = df.iloc[i]

        # Display the row data
        for column, value in row_data.items():
            print(f"\n{column}: \n{value}")

        # Wait for the user to press Enter to display the next row
        input("Press Enter to see the next row...")

    print("No more rows to display.")


def main():
    path = "/data/home/yehonatan-pe/Correction_pipeline/experiments/revision/inhouse_revision_model_results/500_not_factual_500_factual/flan-t5-xl/7/test_results_2.csv"
    df = pd.read_csv(path,index_col=0)[500:600]
    for col in df.columns:
        if df[col].dtype == 'float64' or df[col].dtype == 'int64':
            print(col, df[col].mean())
    #display_csv_row_by_row(df)
if __name__ == '__main__':
    main()