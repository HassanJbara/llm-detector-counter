from typing import Optional
from dataclasses import dataclass, field

import torch
from trl import PPOConfig, PPOTrainer, set_seed
from transformers import AutoTokenizer, HfArgumentParser
from modules.dataset import build_dataset
from modules.utils import build_classifier, build_model, prepare_optim_and_scheduler
from modules.training import train

@dataclass
class ScriptArguments:
    trust_remote_code: bool = field(default=True, metadata={"help": "Enable `trust_remote_code`"})
    query_max_length: Optional[int] = field(default=125, metadata={"help": "allowed max length of queries in dataset"}) 
    hf_model: Optional[str] = field(default=None, metadata={"help": "model used to rate responses on helpfulness"})
    hf_model_weight: Optional[float] = field(default=0.5, metadata={"help": "weight given to the rewards of the hf_model"})
    normal_training: Optional[bool] = field(default=False, metadata={"help": "run normal training with a human feedback model"})
    use_min: Optional[bool] = field(default=False, metadata={"help": "compute rewards using the min function of human feedback and human score"})
    synced_gpus: Optional[bool] = field(default=False, metadata={"help": "activate this flag when using Deepspeed ZeRO Stage 3"})

    
    # LoraConfig
    use_peft: bool = field(default=False, metadata={"help": "whether to use peft"})
    lora_alpha: Optional[float] = field(default=16, metadata={"help": "the lora alpha parameter"})
    lora_r: Optional[int] = field(default=16, metadata={"help": "the lora r parameter"})

    # model config
    quantize: Optional[bool] = field(default=False, metadata={"help": "load model in 8 bits"}) # currently does not work due to cpu offloading
    flash_attn: Optional[bool] = field(default=True, metadata={"help": "load models with flash attention"})


def main(args, ppo_config):
    assert not (args.quantize and args.flash_attn), "Quantization can not be used with flash attention 2!"
    dataset_name = ppo_config.query_dataset if ppo_config.query_dataset else "LDJnr/Pure-Dove"

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        ppo_config.model_name, 
        padding_side='left', 
        trust_remote_code=args.trust_remote_code
    )
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # dataset & dataloader
    if 'gemma' or 'stablelm' in ppo_config.model_name:
        dataset = build_dataset(tokenizer, dataset_name=dataset_name, max_length=args.query_max_length)
    else:
        dataset = build_dataset_with_system_prompt(tokenizer, dataset_name=dataset_name, max_length=args.query_max_length)
    
    def collator(data):
        return dict((key, [d[key] for d in data]) for key in data[0])
    
    # set seed before initializing value head for deterministic eval
    set_seed(ppo_config.seed)
    
    # build model and ref model
    model, ref_model = build_model(ppo_config.model_name, args)
    
    # PPOTrainer & classifier
    optimizer, lr_scheduler = prepare_optim_and_scheduler(model)
    ppo_trainer = PPOTrainer(ppo_config, model, ref_model, tokenizer, dataset=dataset, 
                             data_collator=collator, optimizer=optimizer, lr_scheduler=lr_scheduler)

    # load classifiers to last two cuda devices for efficient use of cuda memory
    cuda_devices = [max(0, torch.cuda.device_count()-1), max(0, torch.cuda.device_count()-2)]
    classifier = build_classifier(ppo_trainer, ppo_config.reward_model, f'cuda:{cuda_devices[0]}')
    hf_pipe = build_classifier(ppo_trainer, args.hf_model, f'cuda:{cuda_devices[1]}') if args.hf_model else None

    # arguments of `generate` function of the PPOTrainer, wrapper around `generate` function of model.
    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        # "pad_token_id": tokenizer.eos_token_id,
        "max_new_tokens": 512,
        "synced_gpus": args.synced_gpus
    }
    
    # classifier pipeline arguments.
    # `return_all_scores` True to get the score for each token.
    sent_kwargs = {
        "truncation": True, 
        "max_length": 512, # base model context length
        # "return_all_scores": True, # deprecated 
        "top_k": None,
        "function_to_apply": "none", 
        "batch_size": ppo_config.mini_batch_size
    }
    
    train(ppo_trainer, tokenizer, classifier, ppo_config.reward_model, generation_kwargs, sent_kwargs, 
          hf_pipe, args.hf_model_weight, args.normal_training, args.use_min)

if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, PPOConfig))
    args, ppo_config = parser.parse_args_into_dataclasses()

    main(args, ppo_config)