import argparse
import os
from datetime import datetime

import datasets
import torch
from tqdm import tqdm

import common.utils.logging as logging_utils
from utils.script_utils import load_tokenizer_and_model

PADDING_TOKEN = "<|padding|>"
PROMPT_TOKEN = "<|prompter|>"
ASSISTANT_TOKEN = "<|assistant|>"
EOS_TOKEN = "<|endoftext|>"

def __orig_alpacafarm_create_format_input_func():
    def format_input_func(example):
        new_example = {}
        instruction, input_text = example["instruction"], example["input"]
        if input_text:
            query = f"{PROMPT_TOKEN}{instruction}\n{input_text}{ASSISTANT_TOKEN}"
        else:
            query = f"{PROMPT_TOKEN}{instruction}{ASSISTANT_TOKEN}"

        if example["preference"] == 1:
            selected = example["output_1"]
            rejected = example["output_2"]
        else:
            selected = example["output_2"]
            rejected = example["output_1"]

        new_example["query"] = query
        new_example["text_w"] = f"{query}{selected}"
        new_example["text_l"] = f"{query}{rejected}"
        return new_example

    return format_input_func


def __ultrafeedback_create_format_input_func():
    def format_input_func(example):
        new_example = {}
        chosen, rejected = example["chosen"], example["rejected"]
        query = f"{PROMPT_TOKEN}{chosen[0]['content']}{ASSISTANT_TOKEN}"

        new_example["query"] = query
        new_example["text_w"] = f"{query}{chosen[1]['content']}"
        new_example["text_l"] = f"{query}{rejected[1]['content']}"
        return new_example

    return format_input_func


def __create_chat_template_format_input_func(tokenizer, query_field: str = "query", chosen_field: str = "chosen", rejected_field: str = "rejected"):
    def format_input_func(example):
        new_example = {}

        query = [{"role": "user", "content": example[query_field]}]
        query = tokenizer.apply_chat_template(query, tokenize=False, add_generation_prompt=True)
        new_example["query"] = query
        new_example["text_w"] = f"{query}" + example[chosen_field]
        new_example["text_l"] = f"{query}" + example[rejected_field]
        return new_example

    return format_input_func


DATASET_CREATE_FORMAT_INPUT_FUNC = {
    "tatsu-lab/alpaca_farm": __orig_alpacafarm_create_format_input_func,
    "HuggingFaceH4/ultrafeedback_binarized": __ultrafeedback_create_format_input_func
}


def __get_dataset(dataset_name: str, cache_dir: str = None):
    if dataset_name == "tatsu-lab/alpaca_farm":
        url = "https://huggingface.co/datasets/tatsu-lab/alpaca_farm/resolve/main/alpaca_human_preference.json"
        return datasets.load_dataset("json", data_files=url, split="train", cache_dir=cache_dir)
    elif dataset_name == "HuggingFaceH4/ultrafeedback_binarized":
        return datasets.load_dataset(dataset_name, split="train_prefs", cache_dir=cache_dir)
    else:
        # Loads dataset from JSON file for all other datasets
        return datasets.load_dataset("json", data_files=dataset_name, split="train")


def __subsample_dataset(dataset, num_train_samples: int = -1, train_samples_random_seed: int = -1):
    if num_train_samples < 0:
        return torch.arange(len(dataset)), dataset

    if train_samples_random_seed > 0:
        perm = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(train_samples_random_seed))
    else:
        perm = torch.randperm(len(dataset))

    num_samples = min(num_train_samples, len(dataset))
    sample_indices = perm[:num_samples]
    dataset = dataset.select(sample_indices)
    return sample_indices, dataset


def __prepare_and_tokenize_dataset(sample_indices, dataset_name, dataset, tokenizer, max_input_length: int,
                                   chat_query_field: str = "query", chat_chosen_field: str = "chosen", chat_rejected_field: str = "rejected"):
    if not tokenizer.chat_template:
        format_input_func = DATASET_CREATE_FORMAT_INPUT_FUNC[dataset_name]()
    else:
        format_input_func = __create_chat_template_format_input_func(tokenizer, query_field=chat_query_field,
                                                                     chosen_field=chat_chosen_field, rejected_field=chat_rejected_field)

    dataset = dataset.map(format_input_func, batched=False)
    dataset = dataset.select_columns(["query", "text_w", "text_l"])

    max_input_length = max_input_length if max_input_length > 0 else None

    def tokenize_examples(example: dict):
        query_input_ids = tokenizer(example["query"], padding=False, truncation=max_input_length is not None,
                                    max_length=max_input_length, return_tensors="pt", add_special_tokens=not tokenizer.chat_template).input_ids
        text_w_input_ids = tokenizer(example["text_w"], padding=False, truncation=max_input_length is not None,
                                     max_length=max_input_length, return_tensors="pt", add_special_tokens=not tokenizer.chat_template).input_ids
        text_l_input_ids = tokenizer(example["text_l"], padding=False, truncation=max_input_length is not None,
                                     max_length=max_input_length, return_tensors="pt", add_special_tokens=not tokenizer.chat_template).input_ids
        return {
            "query": query_input_ids,
            "text_w": text_w_input_ids,
            "text_l": text_l_input_ids
        }

    dataset = dataset.map(tokenize_examples, batched=False)
    dataset.set_format(type="torch")

    indices_to_include = []
    for i, example in enumerate(dataset):
        query_len = example["query"][0].shape[0]
        preferred_token_ids = example["text_w"][0][query_len:]
        dispreferred_token_ids = example["text_l"][0][query_len:]

        if query_len == 0 or preferred_token_ids.shape[0] == 0 or dispreferred_token_ids.shape[0] == 0:
            continue

        indices_to_include.append(i)

    dataset = dataset.select(indices_to_include)
    return sample_indices[indices_to_include], dataset


def __update_tokenizer_setting_and_chat_tokens(tokenizer):
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "right"

    if not tokenizer.eos_token:
        tokenizer.eos_token = EOS_TOKEN

    if not tokenizer.chat_template:
        if not tokenizer.pad_token:
            tokenizer.add_special_tokens({"pad_token": PADDING_TOKEN})
        tokenizer.add_special_tokens({"additional_special_tokens": [PROMPT_TOKEN, ASSISTANT_TOKEN]})
    else:
        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.unk_token if tokenizer.unk_token else tokenizer.eos_token


def __trim_padding(input_ids, tokenizer):
    return input_ids[torch.argmax((input_ids != tokenizer.vocab[tokenizer.eos_token]).to(torch.int)):]


# Taken from the torchaudio edit_distance function
def __normalized_edit_distance(seq1, seq2):
    len_sent2 = len(seq2)
    dold = list(range(len_sent2 + 1))
    dnew = [0 for _ in range(len_sent2 + 1)]

    for i in range(1, len(seq1) + 1):
        dnew[0] = i
        for j in range(1, len_sent2 + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dnew[j] = dold[j - 1]
            else:
                substitution = dold[j - 1] + 1
                insertion = dnew[j - 1] + 1
                deletion = dold[j] + 1
                dnew[j] = min(substitution, insertion, deletion)

        dnew, dold = dold, dnew

    return int(dold[-1]) / max(len(seq1), len(seq2))


def get_and_log_preferred_dispreferred_normalized_edit_distance(logger, dataset):
    per_example_normalized_edit_distance = []

    logger.info(f"Starting normalized edit distance computation")
    for example in tqdm(dataset):
        query_len = example["query"][0].shape[0]
        preferred_token_ids = example["text_w"][0][query_len:]
        dispreferred_token_ids = example["text_l"][0][query_len:]

        if preferred_token_ids.shape[0] == 0 or dispreferred_token_ids.shape[0] == 0:
            raise ValueError("Found example with preferred or dispreferred outputs have length zero after truncating at maximal allowed length for "
                             "prompt + output.")

        pref_dispref_normalized_edit_distance = __normalized_edit_distance(preferred_token_ids, dispreferred_token_ids)
        per_example_normalized_edit_distance.append(pref_dispref_normalized_edit_distance)

    logger.info("\n===========================================================================================================================\n"
                "Edit distance metrics\n"
                "===========================================================================================================================")

    per_example_normalized_edit_distance = torch.tensor(per_example_normalized_edit_distance).to(torch.float)
    logger.info(f"\n------------------------------------------------------------------------------------------------------------------------------\n"
                f"Normalized edit distance of preferred and dispreferred outputs\n"
                f"Mean: {per_example_normalized_edit_distance.mean()} , Min: {per_example_normalized_edit_distance.min()} , "
                f"25th percentile: {torch.quantile(per_example_normalized_edit_distance, q=0.25)} , "
                f"Median: {per_example_normalized_edit_distance.median()} , "
                f"75th percentile: {torch.quantile(per_example_normalized_edit_distance, q=0.75)} , "
                f"Max: {per_example_normalized_edit_distance.max()}\n"
                f"------------------------------------------------------------------------------------------------------------------------------")

    return per_example_normalized_edit_distance


def log_metric_stats(logger, name, scores):
    """
    Logs the statistical block for a given metric tensor, matching the original file's format.
    """
    # Ensure scores are float for readable logging
    scores = scores.float()
    
    logger.info(f"\n------------------------------------------------------------------------------------------------------------------------------\n"
                f"{name}:\n"
                f"Mean: {scores.mean():.4f} , "
                f"Min: {scores.min():.4f} , "
                f"25th percentile: {torch.quantile(scores, q=0.25):.4f} , "
                f"Median: {scores.median():.4f} , "
                f"75th percentile: {torch.quantile(scores, q=0.75):.4f} , "
                f"Max: {scores.max():.4f}\n"
                f"------------------------------------------------------------------------------------------------------------------------------")


def extract_layer_data(model_outputs, query_len):
    """
    Extracts Summed Embedding and Last Token Embedding for EVERY layer.
    """
    summed_embeddings = []
    last_embeddings = []
    
    first_layer = model_outputs.hidden_states[0]
    full_seq_len = first_layer.shape[1]
    response_len = float(full_seq_len - (query_len - 1))
    
    for hidden_state in model_outputs.hidden_states:
        # hidden_state shape: [1, seq_len, dim]
        
        # only the response tokens
        response_hidden = hidden_state[0, query_len - 1:, :] 
        
        #sum (for ches)
        summed = response_hidden.sum(dim=0)
        summed_embeddings.append(summed)
        
        # last token (for inner product)
        last = response_hidden[-1]
        last_embeddings.append(last)
        
    return summed_embeddings, last_embeddings, response_len


def get_and_log_layer_wise_metrics(logger, dataset, model, device, tokenizer):
    layer_ches_storage = None
    layer_ln_ches_storage = None
    layer_inner_storage = None

    logger.info(f"layer-wise computation for ches, ln_ches, and inner products")
    model.to(device)
    
    for example in tqdm(dataset):

        query_len = __trim_padding(example["query"][0], tokenizer).shape[0]
        preferred = __trim_padding(example["text_w"][0], tokenizer).to(device)
        dispreferred = __trim_padding(example["text_l"][0], tokenizer).to(device)

        if query_len == 0 or query_len >= preferred.shape[0] or query_len >= dispreferred.shape[0]:
            logger.warn("Skipping example due to length issues.")
            continue

        with torch.no_grad():
            out_w = model(input_ids=preferred.unsqueeze(0), output_hidden_states=True)
            out_l = model(input_ids=dispreferred.unsqueeze(0), output_hidden_states=True)

        w_sums, w_lasts, len_w = extract_layer_data(out_w, query_len)
        l_sums, l_lasts, len_l = extract_layer_data(out_l, query_len)
        
        if layer_ches_storage is None:
            num_layers = len(w_sums)
            layer_ches_storage = [[] for _ in range(num_layers)]
            layer_ln_ches_storage = [[] for _ in range(num_layers)]
            layer_inner_storage = [[] for _ in range(num_layers)]

        for i in range(num_layers):
            h_p_sum = w_sums[i].float()
            h_m_sum = l_sums[i].float()
            h_p_last = w_lasts[i].float()
            h_m_last = l_lasts[i].float()

            ches = torch.dot(h_p_sum, h_m_sum) - (torch.norm(h_p_sum) ** 2)
            layer_ches_storage[i].append(ches.cpu())

            term1 = torch.dot(h_p_sum, h_m_sum) / (len_w * len_l)
            term2 = (torch.norm(h_p_sum) ** 2) / (len_w ** 2)
            ln_ches = term1 - term2
            layer_ln_ches_storage[i].append(ln_ches.cpu())

            inner = torch.dot(h_p_last, h_m_last)
            layer_inner_storage[i].append(inner.cpu())

    final_ches_cols = []
    final_ln_ches_cols = []
    final_inner_cols = []
    
    num_layers = len(layer_ches_storage) if layer_ches_storage else 0
    
    logger.info("\n" + "="*50 + " Layer-wise stats " + "="*50)

    for i in range(num_layers):
        c_tensor = torch.tensor(layer_ches_storage[i])
        ln_tensor = torch.tensor(layer_ln_ches_storage[i])
        in_tensor = torch.tensor(layer_inner_storage[i])
        
        final_ches_cols.append(c_tensor)
        final_ln_ches_cols.append(ln_tensor)
        final_inner_cols.append(in_tensor)
        
        logger.info(f"\nStats for each layer {i}")
        log_metric_stats(logger, f"Layer {i} CHES Scores", c_tensor)
        log_metric_stats(logger, f"Layer {i} Length-Normalized CHES", ln_tensor)
        log_metric_stats(logger, f"Layer {i} Last Hidden Inner Products", in_tensor)

    logger.info("="*120 + "\n")

    return (torch.stack(final_ches_cols).T, 
            torch.stack(final_ln_ches_cols).T, 
            torch.stack(final_inner_cols).T)


@torch.no_grad()
def main(config: dict):
    model_name = config["model"]
    dataset_name = config["dataset"]
    num_train_samples = config["num_train_samples"]
    train_samples_random_seed = config["train_samples_random_seed"]
    max_input_length = config["max_input_length"]
    device = torch.device(f"cuda:{config['gpu_id']}" if torch.cuda.is_available() and config["gpu_id"] >= 0 else "cpu")

    dataset_display_name = config["custom_dataset_display_name"] if config["custom_dataset_display_name"] else dataset_name.split("/")[-1]
    subdir_name = model_name.split("/")[-1] + "_" + dataset_display_name
    logger = logging_utils.create_logger(file_logging=not config["dont_save_logs"],
                                         log_dir=os.path.join(config["output_dir"], subdir_name),
                                         log_file_name_prefix=f"log_samples_{num_train_samples}")
    logger.info(f"Config: {config}")

    try:
        start_time = datetime.utcnow()

        logger.info(f"======================================================================================================")
        logger.info(f"Model: '{model_name}', Dataset: '{dataset_name}'")
        logger.info(f"======================================================================================================\n")

        tokenizer, model = load_tokenizer_and_model(model_name, cache_dir=config["cache_dir"], device=device)

        #enabled hidden states output
        model.config.output_hidden_states = True
        model.to(device)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'right'

        __update_tokenizer_setting_and_chat_tokens(tokenizer)
        model.resize_token_embeddings(len(tokenizer))

        dataset = __get_dataset(dataset_name, cache_dir=config["cache_dir"])
        sample_indices, dataset = __subsample_dataset(dataset, num_train_samples, train_samples_random_seed)
        sample_indices, tokenized_dataset = __prepare_and_tokenize_dataset(sample_indices, dataset_name, dataset, tokenizer, max_input_length,
                                                                           chat_chosen_field=config["chat_chosen_field"],
                                                                           chat_rejected_field=config["chat_rejected_field"])
        logger.info(f"Filtered out samples with empty query or outputs\n"
                    f"Original number of samples: {len(dataset)}\n"
                    f"Number of samples after filtering: {len(sample_indices)}")

        normalized_edit_distances = get_and_log_preferred_dispreferred_normalized_edit_distance(logger, tokenized_dataset)

        ches_tensor, ln_ches_tensor, inner_tensor = get_and_log_layer_wise_metrics(
            logger, tokenized_dataset, model, device, tokenizer
        )

        results = {
            "sample_indices": sample_indices,
            "minus_normalized_edit_distances": -normalized_edit_distances,
            
            # trajectories [n_samples, n_layers]
            "ches_layer_trajectories": ches_tensor,
            "ln_ches_layer_trajectories": ln_ches_tensor,
            "inner_product_layer_trajectories": inner_tensor,
            
            "ches_scores": ches_tensor[:, -1],
            "ln_ches_scores": ln_ches_tensor[:, -1],
            "last_hidden_embedding_inner_prods": inner_tensor[:, -1]
        }
        
        torch.save(results, os.path.join(config["output_dir"], subdir_name, f"results_samples.pt"))

        end_time = datetime.utcnow()
        logger.info(f"Finished script, time took: {end_time - start_time}")
    except Exception:
        logger.exception("Exception while running script.")
        raise


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", type=str, default="outputs/pref_similarity", help="Directory to save log file to")
    p.add_argument("--cache_dir", type=str, default=None, help="Directory of cache for HuggingFace models and datasets")
    p.add_argument("--dont_save_logs", action="store_true", help="Only log to console, and not to a file")
    p.add_argument("--model", type=str, default="allenai/OLMo-1B-hf", help="Model to use")
    p.add_argument("--dataset", type=str, default="tatsu-lab/alpaca_farm", help="Dataset to use")
    p.add_argument("--custom_dataset_display_name", type=str, default="", help="Name of dataset to use for creating file name")
    p.add_argument("--num_train_samples", type=int, default=-1,
                   help="Number of training samples to compute preference similarity for (if < 0, all samples are used)")
    p.add_argument("--train_samples_random_seed", type=int, default=-1, help="Random seed to use for selecting train samples")
    p.add_argument("--max_input_length", type=int, default=512,
                   help="Truncate outputs to this maximal length (if < 0, does not truncate)")
    p.add_argument("--chat_chosen_field", type=str, default="chosen", help="Field name for chosen output when using models with chat template")
    p.add_argument("--chat_rejected_field", type=str, default="rejected", help="Field name for rejected output when using models with chat template")
    p.add_argument("--gpu_id", type=int, default=-1, help="GPU id to use (-1 for CPU)")
    args = p.parse_args()

    main(args.__dict__)