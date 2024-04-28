import argparse
import json
import random
import statistics
from pathlib import Path
from utils import build_model_for_benchmark, prepare_classifier_pipe
from dataset import build_dataset
from transformers import pipeline
from tqdm import tqdm


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="LDJnr/Pure-Dove", help="dataset name")
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help="model name")
    parser.add_argument("--classifier", type=str, default="text-classification:Hello-SimpleAI/chatgpt-detector-roberta", help="classifier name with task")
    
    parser.add_argument("--samples", type=int, default=100, help="number of samples to test the model on")
    parser.add_argument("--with_query", type=bool, default=False, help="whether to include the query during evaluation")
    
    parser.add_argument("--quantize", type=bool, default=False, help="whether to load model in 8bit or not")
    parser.add_argument("--flash_attn", type=bool, default=True, help="whether to use flash_attn 2 or not") 
    # parser.add_argument("--use_peft", type=bool, default=False, help="whether to use peft or not")
    
    parser.add_argument("--device", type=str, default="cuda:0", help="which device to load the model to")

    args = parser.parse_args()
    print("input args:\n", json.dumps(vars(args), indent=4, separators=(",", ":")))
    return args

def main(args):
    model, tokenizer = build_model_for_benchmark(args.model_name, args.quantize, args.flash_attn, args.device)
    dataset = build_dataset(tokenizer, args.dataset, sys_prompt=True, padding=False)

    task, model_name = args.classifier.split(":")
    classifier_pipe = pipeline(task, model=model_name, device="cuda:0")
    # Some tokenizers like GPT-2's don't have a padding token by default, so we set one here.
    if classifier_pipe.tokenizer.pad_token_id is None:
        classifier_pipe.tokenizer.pad_token_id = tokenizer.pad_token_id
    
    if classifier_pipe.model.config.pad_token_id is None:
        classifier_pipe.model.config.pad_token_id = tokenizer.pad_token_id
        
    sent_kwargs = {
        "truncation": True, 
        "max_length": 512, # base model context length
        # "return_all_scores": True, # deprecated 
        "top_k": None,
        # "function_to_apply": "none",
    }
    
    # use with llama-3
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    
    # instead of filter for better visibility
    evaluations = []
    for i in tqdm(range(args.samples)):
        sample_index = random.randint(0, len(dataset)-1)
        outputs = model.generate(dataset[sample_index]['input_ids'].unsqueeze(dim=0).to(args.device), 
                                 attention_mask=dataset[sample_index]['attention_mask'].unsqueeze(dim=0).to(args.device), 
                                 max_new_tokens=512, 
                                 eos_token_id=terminators,
                                 pad_token_id=tokenizer.eos_token_id)
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        q_len = len(tokenizer.batch_decode(dataset[sample_index]['input_ids'], skip_special_tokens=True)[0])+1
        
        if args.with_query:
            evaluation = classifier_pipe(text, **sent_kwargs)
        else:
            evaluation = classifier_pipe(text[q_len:], **sent_kwargs) # remove query
            
        evaluations += evaluation

    evaluations = [x['score'] for x in evaluations if x['label'].lower() == 'human']
    avg = statistics.mean(evaluations)
    
    print(avg)
    
if __name__ == "__main__":
    args = parse_arguments()
    main(args)