from typing import Optional
from dataclasses import dataclass, field

from peft import LoraConfig
from accelerate import Accelerator
from utils import prepare_classifier_pipe, train
from transformers import AutoTokenizer, HfArgumentParser
from dataset import build_dataset, build_dataset_for_gemma
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, set_seed

@dataclass
class ScriptArguments:
    trust_remote_code: bool = field(default=True, metadata={"help": "Enable `trust_remote_code`"})
    query_max_length: Optional[int] = field(default=125, metadata={"help": "allowed max length of queries in dataset"}) 

    # LoraConfig
    use_peft: bool = field(default=False, metadata={"help": "whether to use peft"})
    lora_alpha: Optional[float] = field(default=16, metadata={"help": "the lora alpha parameter"})
    lora_r: Optional[int] = field(default=16, metadata={"help": "the lora r parameter"})

def main(args, ppo_config):
    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        ppo_config.model_name, 
        padding_side='left', 
        trust_remote_code=args.trust_remote_code
    )
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # dataset & dataloader
    if 'gemma' in ppo_config.model_name:
        dataset = build_dataset_for_gemma(tokenizer, max_length=args.query_max_length)
    else:
        dataset = build_dataset(tokenizer, max_length=args.query_max_length)
    
    def collator(data):
        return dict((key, [d[key] for d in data]) for key in data[0])
    
    # set seed before initializing value head for deterministic eval
    set_seed(ppo_config.seed)
    
    # build model and ref model
    if not args.use_peft:
        ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(ppo_config.model_name, trust_remote_code=args.trust_remote_code)
        device_map = None
        peft_config = None
    else:
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            bias="none",
            task_type="CAUSAL_LM",
        )
        ref_model = None
        # Copy the model to each device
        device_map = {"": Accelerator().local_process_index}
    
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        ppo_config.model_name,
        trust_remote_code=args.trust_remote_code,
        device_map=device_map,
        peft_config=peft_config,
    )
    
    # PPOTrainer & classifier
    ppo_trainer = PPOTrainer(ppo_config, model, ref_model, tokenizer, dataset=dataset, data_collator=collator)
    
    classifier_pipe = prepare_classifier_pipe(ppo_trainer, ppo_config.reward_model)
    
    # arguments of `generate` function of the PPOTrainer, wrapper around `generate` function of model.
    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        # "pad_token_id": tokenizer.eos_token_id,
        "max_new_tokens": 512,
    }
    
    # classifier pipeline arguments.
    # `return_all_scores` True to get the score for each token.
    sent_kwargs = {
        "truncation": True, 
        "max_length": 512, # base model context length
        "return_all_scores": True, 
        "function_to_apply": "none", 
        "batch_size": ppo_config.mini_batch_size
    }
    
    train(ppo_trainer, classifier_pipe, tokenizer, generation_kwargs, sent_kwargs)

if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, PPOConfig))
    args, ppo_config = parser.parse_args_into_dataclasses()

    main(args, ppo_config)