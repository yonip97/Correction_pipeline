import os
import sys
import time

sys.path.append(os.path.dirname(os.getcwd()))
os.chdir('../')
sys.path.append(os.path.dirname(os.getcwd()))
os.chdir('../')
sys.path.append(os.path.dirname(os.getcwd()))
os.chdir('../')

from general.LLMS import SummarizationModel
import argparse
from experiments.xsum_4_sets_experiment.datasets_splits import split_xsum_dataset
import pandas as pd
from tqdm import tqdm


def parseargs():
    parse = argparse.ArgumentParser()
    parse.add_argument('--output_dir', type=str)
    parse.add_argument('--temp_save_dir', type=str)
    parse.add_argument('--past_text_prompt', type=str, default='')
    parse.add_argument('--model', type=str, default='gpt-3.5-turbo')
    parse.add_argument('--API_KEY', type=str, default=None)
    parse.add_argument('--max_generation_length', type=int, default=30)
    args = parse.parse_args()
    return args


def create_prompts():
    prompts = ["Write a very short concise and abstractive summary of the following text in not more than 25 words.",
               "Summarize the main idea of the text in one sentence in not more than 25 words.",
               "Condense the text into a brief, abstract summary in no more than 25 words"]
    summaries = ["Police closed a road in Bracknell following the discovery of a man's body in a park.",
                 "Firmus energy has lost an attempt to overturn a regulator's decision which meant lower gas bills for customers.",
                 "Fast-food chain McDonald's has dropped its appeal against a decision to reject plans for a new restaurant in Newcastle."]
    texts = ["""The body was found in South Hill Park, south of Bracknell town centre, shortly after 06:20 BST.
South Hill Road has been closed in both directions between the A322 Bagshot Road and the A3095 Crowthorne Road, with motorists and members of the public being advised to avoid the area.
Thames Valley Police said the man's death was being treated as unexplained.
The force said inquiries were continuing to identify the man.
""",
             """In September 2016 the Utility Regulator determined how much Northern Ireland's three gas networks could charge over the next six years.
That decision meant Firmus customers could expect their bills to be cut by an average of  £15 a year.
Firmus appealed that decision to the Competition and Markets Authority (CMA) was unsuccessful.
Utility Regulator Chief Executive Jenny Pyper said: "The true winners are local gas consumers."
"Our price control package would have reduced Firmus energy's tariffs by  £15 per annum for the average domestic consumer and by tens of thousands of pounds for the larger industrial consumers.
"The CMA's determination means that at this time, Firmus energy does not receive any increase in its allowances above those identified in our price control and associated licence modifications.
"All utilities are entitled to a return on their investment but this must be commensurate with the risks that the business and its shareholder face and should not expose business and household consumers to further costs that impact on bills."
Firmus said its decision to appeal to the CMA was not taken lightly and "was done with the best long-term interests in mind for the natural gas consumer in Northern Ireland".
The company appealed on a total of 12 issues - nine of which were rejected.
One issue has been referred back to the regulator which may result in future adjustments - upwards or downwards.
The Consumer Council welcomed the appeal's outcome, and urged the CMA to decide who should pay its costs - which it said were likely to be substantial.
"The Consumer Council are seeking assurances from the CMA that the costs of the process will not fall unfairly to consumers and will be proportionately paid by Firmus Energy," said Consumer Council Chief Executive John French.
""",
             """Newcastle City Council refused planning permission to the development near Kenton School in September 2014.
McDonald's had launched an appeal but has now withdrawn it after the council changed its planning policy.
Hundreds opposed the restaurant saying it would encourage children from the school to eat unhealthily.
A McDonald's spokeswoman said the firm was made aware the authority had amended its policy to take into consideration the proximity of certain businesses to schools, meaning the plans no longer adhered to planning guidelines.
"We have withdrawn our appeal as a direct consequence. We are genuinely disappointed and frustrated by this development at such a late stage," she said.
The company claimed the new restaurant would have created about 70 jobs and made a "positive contribution" to the area.
Campaigners said they were delighted the plans were at an end.
Jocasta Williams told BBC Newcastle: "We always thought it was a really long shot, they are a multi-national company, we just had a group of committed people that were prepared to give up their own time and a small amount of money.
"We always doubted we could do it but we have, we kept on fighting."
Newcastle City Council welcomed McDonald's decision and said a Planning Inspection scheduled for Tuesday would not now go ahead.
A city council spokesman said: "Newcastle is a city which welcomes business and investment. We will always work constructively with big business to find solutions that work for them to bring jobs and growth.
"But we must also always strike the right balance to ensure that investments are in the best interests of our local residents."
"""]
    final_prompts = []
    final_prompts += prompts
    for i in range(len(texts)):
        for j in range(len(texts)):
            final_prompts.append(create_one_shot(prompts[i], texts[j], summaries[j]))
        final_prompts.append(create_few_shot(prompts[i], texts, summaries))
    return final_prompts


def create_one_shot(prompt, text, summary):
    final_prompt = prompt + '\n' + "here is an example: " + "\n" + 'text: ' + text + 'summary: ' + summary
    return final_prompt


def create_few_shot(prompt, texts, summaries):
    final_prompt = prompt + '\n' + "here are a few examples " + '\n'
    for i in range(len(texts)):
        final_prompt += "text: " + texts[i] + '\n' + 'summary: ' + summaries[i] + '\n'
    return final_prompt


def main():
    args = parseargs()
    prompt = """Condense the text into a brief, abstract summary in no more than 25 words.
here is an example: 
text: The body was found in South Hill Park, south of Bracknell town centre, shortly after 06:20 BST.
South Hill Road has been closed in both directions between the A322 Bagshot Road and the A3095 Crowthorne Road, with motorists and members of the public being advised to avoid the area.
Thames Valley Police said the man's death was being treated as unexplained.
The force said inquiries were continuing to identify the man.
summary: Police closed a road in Bracknell following the discovery of a man's body in a park."""
    summarization_model = SummarizationModel(temp_save_dir=args.temp_save_dir, prompt=prompt,
                                             past_text_prompt=args.past_text_prompt, model=args.model,
                                             API_KEY=args.API_KEY,azure=True)
    text_dataset = split_xsum_dataset(split='documents_for_summarization',
                                      path_to_documents_for_summarization_indices="experiments/xsum_4_sets_experiment/datasets_splits/xsum_docs_for_summarization_20000_indices_seed_42.pkl",
                                      num_of_documents_for_summarization=20000,
                                      seed=42)
    texts = [text_dataset[i]['text'] for i in range(len(text_dataset))][6750:]

    summaries, errors = summarization_model.summarize(texts=texts, max_generation_length=args.max_generation_length)
    df = pd.DataFrame({'text': texts, 'summary': summaries, 'error': errors, 'prompt': [prompt] * len(texts)})
    df.to_csv(args.output_dir + '/' + 'gpt_3.5_summaries.csv', index=False)


def prompt_searching():
    text_dataset = split_xsum_dataset(split='documents_for_summarization',
                                      path_to_documents_for_summarization_indices="experiments/xsum_4_sets_experiment/datasets_splits/xsum_docs_for_summarization_20000_indices_seed_42.pkl",
                                      num_of_documents_for_summarization=20000,
                                      seed=42)
    texts = [text_dataset[i]['text'] for i in range(len(text_dataset))][::100]
    args = parseargs()

    prompts = create_prompts()
    for index, prompt in enumerate(prompts):
        summarization_model = SummarizationModel(temp_save_dir=args.temp_save_dir, prompt=prompt,
                                                 past_text_prompt="", model=args.model,
                                                 API_KEY=args.API_KEY)
        summaries = []
        errors = []
        for text in tqdm(texts):
            summary, error = summarization_model.summarize_single(text=text,
                                                                  max_generation_length=args.max_generation_length)
            summaries.append(summary)
            errors.append(error)
            time.sleep(3)
        df = pd.DataFrame.from_dict(
            {"prompt": [prompt] * len(summaries), "summary": summaries, "text": texts, 'error': errors})
        df.to_csv(os.path.join(args.output_dir, f'prompt_{index}.csv'), index=False)


def per_prompt_cost():
    from data.cost_etimation import Cost_estimator
    estimator = Cost_estimator('gpt-3.5-turbo', 0.001, 0.002)
    text_dataset = split_xsum_dataset(split='documents_for_summarization',
                                      path_to_documents_for_summarization_indices="experiments/xsum_4_sets_experiment/datasets_splits/xsum_docs_for_summarization_20000_indices_seed_42.pkl",
                                      num_of_documents_for_summarization=20000,
                                      seed=42)
    texts = [text_dataset[i]['text'] for i in range(len(text_dataset))][::100]
    summaries = [text_dataset[i]['summary'] for i in range(len(text_dataset))][::100]
    total = 0
    for prompt in create_prompts():
        print(prompt)
        estimation = estimator.estimate_for_summarization(prompt=prompt, texts=texts, summaries=summaries)
        print(estimation)
        total += estimation[0]
    print(total)


if __name__ == "__main__":
    # prompt_searching()
    # per_prompt_cost()
    main()
