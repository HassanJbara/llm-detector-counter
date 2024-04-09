import torch
from transformers import AutoTokenizer
from datasets import load_dataset

def build_dataset_for_gemma(tokenizer, dataset_name="LDJnr/Pure-Dove", max_length=300):
    """
    Build dataset for training.

    Args:
        dataset_name (`str`):
            The name of the dataset to be loaded.

    Returns:
        dataloader (`torch.utils.data.DataLoader`):
            The dataloader for the dataset.
    """

    ds = load_dataset(dataset_name, split="train")
    querys = [ds_item.get('conversation')[0].get('input') for ds_item in ds]
    ds = ds.add_column('query', querys)

    def prepare_dataset(ds_item):
        prompt = [
            {
                "role": "user", 
                "content": ds_item['query']
            },
        ]
        tokens_dict =  tokenizer.apply_chat_template(
            prompt, 
            add_generation_prompt=True, 
            padding='max_length', 
            max_length=max_length, 
            return_tensors='pt', 
            return_dict=True
        )
        ds_item["input_ids"] = tokens_dict["input_ids"][0] # because it returns a list
        ds_item["attention_mask"] = tokens_dict["attention_mask"][0] # because it returns a list
        return ds_item
    
    ds = ds.map(prepare_dataset, batched=False)
    ds = ds.filter(lambda x: len(x["input_ids"]) <= max_length, batched=False)
    ds = ds.remove_columns(['source', 'conversation'])
    ds.set_format(type="torch")
    
    return ds


def build_dataset(tokenizer, dataset_name="LDJnr/Pure-Dove", max_length=300):
    """
    Build dataset for training.

    Args:
        dataset_name (`str`):
            The name of the dataset to be loaded.

    Returns:
        dataloader (`torch.utils.data.DataLoader`):
            The dataloader for the dataset.
    """

    ds = load_dataset(dataset_name, split="train")
    querys = [ds_item.get('conversation')[0].get('input') for ds_item in ds]
    ds = ds.add_column('query', querys)

    def prepare_dataset(ds_item):
        prompt = [
            {
                "role": "system",
                "content": "You are an assistant who gives detailed and long answers",
            },
            {
                "role": "user", 
                "content": ds_item['query']
            },
        ]
        tokens_dict =  tokenizer.apply_chat_template(
            prompt, 
            add_generation_prompt=True, 
            padding='max_length', 
            max_length=max_length, 
            return_tensors='pt', 
            return_dict=True
        )
        ds_item["input_ids"] = tokens_dict["input_ids"][0] # because it returns a list
        ds_item["attention_mask"] = tokens_dict["attention_mask"][0] # because it returns a list
        return ds_item
    
    ds = ds.map(prepare_dataset, batched=False)
    ds = ds.filter(lambda x: len(x["input_ids"]) <= max_length, batched=False)
    ds = ds.remove_columns(['source', 'conversation'])
    ds.set_format(type="torch")
    
    return ds

def build_dataset_example(config, query_dataset, input_min_text_length=2, input_max_text_length=8):
    """
    Build dataset for training. This builds the dataset from `load_dataset`, one should
    customize this function to train the model on its own dataset.

    Args:
        query_dataset (`str`):
            The name of the dataset to be loaded.

    Returns:
        dataloader (`torch.utils.data.DataLoader`):
            The dataloader for the dataset.
    """
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    # load imdb with datasets
    ds = load_dataset(query_dataset, split="train")
    ds = ds.rename_columns({"text": "review"})
    ds = ds.filter(lambda x: len(x["review"]) > 200, batched=False)

    input_size = LengthSampler(input_min_text_length, input_max_text_length)

    def tokenize(sample):
        sample["input_ids"] = tokenizer.encode(sample["review"])[: input_size()]
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch")
    return ds
