import os
import torch
import logging
import multiprocessing
from contextlib import nullcontext
from rich.console import Console
from rich.logging import RichHandler
from datasets import load_dataset
from dataclasses import dataclass, field
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, HfArgumentParser
from trl import DPOConfig, DPOTrainer, RichProgressCallback
from trl.commands.cli_utils import DPOScriptArguments, init_zero_verbose, TrlParser
from peft import LoraConfig
 

@dataclass
class ScriptArguments:
    trust_remote_code: bool = field(default=True, metadata={"help": "Enable `trust_remote_code`"})
    query_max_length: Optional[int] = field(default=125, metadata={"help": "allowed max length of queries in dataset"}) 
    synced_gpus: Optional[bool] = field(default=False, metadata={"help": "activate this flag when using Deepspeed ZeRO Stage 3"})
    # sanity_check: Optional[bool] = field(default=False,)
    save_model: Optional[bool] = field(default=False,)
    sys_prompt: Optional[bool] = field(default=True,)
    
    # LoraConfig
    use_peft: bool = field(default=False, metadata={"help": "whether to use peft"})
    lora_alpha: Optional[float] = field(default=128, metadata={"help": "the lora alpha parameter"})
    lora_r: Optional[int] = field(default=256, metadata={"help": "the lora r parameter"})

    # model config
    model_name: str = field(default="google/gemma-1.1-2b-it", metadata={"help": "the model name to train"})
    dataset_name: str = field(default="Intel/orca_dpo_pairs", metadata={"help": "the dataset to train on"})
    dataset_train_split: Optional[float] = field(default=0.8, metadata={"help": "the percentage of the dataset to use for training. The rest will be used for eval."})
    quantize: Optional[bool] = field(default=False, metadata={"help": "load model in 8 bits"}) # currently does not work due to cpu offloading
    flash_attn: Optional[bool] = field(default=True, metadata={"help": "load models with flash attention"})


def main(args, dpo_config):
    # assert not (args.quantize and args.flash_attn), "Quantization can not be used with flash attention 2!"
    
    init_zero_verbose()
    FORMAT = "%(message)s"
    logging.basicConfig(format=FORMAT, datefmt="[%X]", handlers=[RichHandler()], level=logging.INFO)

    # Force use our print callback
    dpo_config.disable_tqdm = True
    console = Console()

    ################
    # Model & Tokenizer
    ################
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
    ) if args.quantize else None
    model_kwargs = dict(
        trust_remote_code=args.trust_remote_code,
        attn_implementation="flash_attention_2" if args.flash_attn else None,
        torch_dtype=torch.bfloat16 if args.flash_attn else None,
        quantization_config=quantization_config,
    )
    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)
    # LoRA config based on QLoRA paper & Sebastian Raschka experiment
    peft_config = LoraConfig(
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        r=args.lora_r,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    ) if args.use_peft else None
    
    if not args.use_peft:
        model_ref = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)
    else:
        model_ref = None
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None:
        tokenizer.chat_template = "{% for message in messages %}{{message['role'] + ': ' + message['content'] + '\n\n'}}{% endfor %}{{ eos_token }}"
    # if args.ignore_bias_buffers:
    #     # torch distributed hack
    #     model._ddp_params_and_buffers_to_ignore = [
    #         name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
    #     ]

    ################
    # Optional rich context managers
    ###############
    init_context = console.status("[bold green]Initializing the DPOTrainer...")
    save_context = console.status(f"[bold green]Training completed! Saving the model to {dpo_config.output_dir}")

    ################
    # Dataset
    ################
    ds = load_dataset(args.dataset_name, split="train")
    # if args.sanity_check:
    #     for key in ds:
    #         ds[key] = ds[key].select(range(50))

    def process(row): # depends on the dataset!
        if "hassanjbara" in args.dataset_name:
            sys_prompt = "Provide naturally sounding answers."
            prompt = []
            if args.sys_prompt:
                prompt.append({"content": sys_prompt, "role": "system"})
            prompt.append({"content": row["query"], "role": "user"})
            
            row["prompt"] = tokenizer.apply_chat_template(prompt, tokenize=False)
            row["chosen"] = tokenizer.apply_chat_template([{"content": row["chosen"], "role": "assistant"}], tokenize=False)
            row["rejected"] = tokenizer.apply_chat_template([{"content": row["rejected"], "role": "assistant"}], tokenize=False)
        else:
            row["prompt"] = tokenizer.apply_chat_template(row["chosen"][:-1], tokenize=False)
            row["chosen"] = tokenizer.apply_chat_template([row["chosen"][-1]], tokenize=False)
            row["rejected"] = tokenizer.apply_chat_template([row["rejected"][-1]], tokenize=False)
    
        return row

    ds = ds.map(
        process,
        # num_proc=multiprocessing.cpu_count(),
        load_from_cache_file=False,
    )
    train_samples = int(args.dataset_train_split * len(ds))
    eval_samples = len(ds) - train_samples
    train_dataset = ds.shuffle().select(range(train_samples))
    eval_dataset = ds.shuffle().select(range(eval_samples))

    ################
    # Training
    ################
    with init_context:
        trainer = DPOTrainer(
            model,
            model_ref,
            args=dpo_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            peft_config=peft_config,
            callbacks=[RichProgressCallback],
        )

    trainer.train()

    if args.save_model:
        with save_context:
            trainer.save_model(dpo_config.output_dir)


if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, DPOConfig))
    args, dpo_config = parser.parse_args_into_dataclasses()
    
    main(args, dpo_config)