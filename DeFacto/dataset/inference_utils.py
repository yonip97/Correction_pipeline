from call_llms import ModelCallerPipeline, OpenAICaller, AnthropicCaller, LlamaApiCaller, GeminiCaller,ModelCallerAutoModel
from constants import MODEL_PRICE_MAP as model_price_map, DTYPE_MAP as dtype_map


def chose_model(model_name, temp_save_dir, llamaapi, azure=False, dtype=None, device_map=None,pipline =True):
    if model_name in model_price_map:
        input_price = model_price_map[model_name]['input']
        output_price = model_price_map[model_name]['output']
    else:
        input_price = 0
        output_price = 0
    if dtype is not None:
        if dtype in dtype_map:
            dtype = dtype_map[dtype]
        else:
            raise ValueError(f"Unknown dtype: {dtype}")
    if 'gpt' in model_name:
        return OpenAICaller(model=model_name, azure=azure, temp_save_dir=temp_save_dir,
                            input_price=input_price, output_price=output_price)
    elif 'gemini' in model_name:
        return GeminiCaller(model=model_name, temp_save_dir=temp_save_dir,
                            input_price=input_price,
                            output_price=output_price)
    elif 'claude' in model_name:
        return AnthropicCaller(model=model_name, temp_save_dir=temp_save_dir,
                               input_price=input_price, output_price=output_price)
    elif llamaapi:
        return LlamaApiCaller(model=model_name, temp_save_dir=temp_save_dir,
                              input_price=input_price, output_price=output_price)
    elif pipline:
        return ModelCallerPipeline(model_id=model_name, device_map=device_map,
                           torch_dtype=dtype)
    else:
        return ModelCallerAutoModel(model_id=model_name, device_map=device_map,
                                    torch_dtype=dtype)


def num_to_uppercase_letter(num):
    if 0 <= num <= 25:
        return chr(num + ord('A'))
    else:
        raise ValueError("Number must be between 0 and 25")


def transform_to_enumerated_descriptions(descriptions):
    samples = []
    for sample_descriptions in descriptions:
        final_sample = ""
        if len(sample_descriptions) == 0:
            samples.append(final_sample)
            continue
        for i, explanation in enumerate(sample_descriptions):
            final_sample += f"{num_to_uppercase_letter(i)}.\nDescription: {explanation}\n"
        samples.append(final_sample)
    return samples
def try_another_option(descriptions):
    samples = []
    for sample_descriptions in descriptions:
        final_sample = ""
        if len(sample_descriptions) == 0:
            samples.append(final_sample)
            continue
        for i, explanation in enumerate(sample_descriptions):
            final_sample += f"{num_to_uppercase_letter(i)}.\nDescription: {explanation}\n"
        if final_sample == "":
            final_sample = "None"
        samples.append(final_sample)
    return samples

