def add_examples_to_prompt(prompt, examples, connector):
    prompt += '\n' + connector + '\n'
    for example in examples:
        prompt += 'document: ' + '\n' + example['document'] + '\n'
        prompt += 'summary: ' + '\n' + example['summary'] + '\n'
        prompt += 'revised summary: ' + example['revised summary'] + '\n'
    return prompt
