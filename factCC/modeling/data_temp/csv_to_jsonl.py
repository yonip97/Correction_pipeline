import csv
import json

def csv_to_jsonl(csv_file, jsonl_file):
    with open(csv_file, 'r', newline='', encoding='utf-8') as csvfile, \
            open(jsonl_file, 'w', encoding='utf-8') as jsonlfile:

        # Read the CSV file
        csv_reader = csv.DictReader(csvfile)
        labels_map = {0:"INCORRECT",1:"CORRECT"}
        # Convert and write each row as a JSON object in JSONL format
        counter = 0
        for row in csv_reader:
            new_row = {}
            new_row['id'] = row['id']
            new_row['text'] = row['grounding']
            new_row['claim'] = row['generated_text']
            new_row['label'] = labels_map[int(row['label'])]
            print(new_row['label'])
            print(new_row['claim'] )
            print(new_row['text'])
            print('---------------------------------------------------')
            counter += 1
            if counter > 50:
                break
            # json.dump(new_row, jsonlfile, ensure_ascii=False)
            # jsonlfile.write('\n')

if __name__ == "__main__":
    input_csv_file = '/data/home/yehonatan-pe/correction_pipeline/data/summeval_download.csv'
    output_jsonl_file = "data-dev.jsonl"
    csv_to_jsonl(input_csv_file, output_jsonl_file)