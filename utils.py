import torch
from tqdm import tqdm
from transformers import pipeline, QuantoConfig
from accelerate import Accelerator
from trl.import_utils import is_xpu_available
from trl import AutoModelForCausalLMWithValueHead
# from unsloth import FastLanguageModel
from accelerate.utils import DummyOptim, DummyScheduler
from peft import LoraConfig

def prepare_classifier_pipe(ppo_trainer, reward_model, device=None):
    # build classifier pipeline
    device = device if device else ppo_trainer.accelerator.device
    if ppo_trainer.accelerator.num_processes == 1:
        if is_xpu_available():
            device = "xpu:0"
        else:
            device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a `pipeline` bug
    ds_plugin = ppo_trainer.accelerator.state.deepspeed_plugin
    task, model_name = reward_model.split(":")
    if ds_plugin is not None and ds_plugin.is_zero3_init_enabled():
        with ds_plugin.zero3_init_context_manager(enable=False):
            classifier_pipe = pipeline(task, model=model_name, device=device)
    else:
        classifier_pipe = pipeline(task, model=model_name, device=device)
    
    # Some tokenizers like GPT-2's don't have a padding token by default, so we set one here.
    if classifier_pipe.tokenizer.pad_token_id is None:
        classifier_pipe.tokenizer.pad_token_id = tokenizer.pad_token_id
    
    if classifier_pipe.model.config.pad_token_id is None:
        classifier_pipe.model.config.pad_token_id = tokenizer.pad_token_id

    return classifier_pipe

def build_model(model_name, args):
    quantization_config = None
    if args.quantize:
        quantization_config = QuantoConfig(weights="int8")
        
    if args.use_peft:
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            bias="none",
            task_type="CAUSAL_LM",
        )
        ref_model = None
        # Copy the model to each device
        device_map = {"": Accelerator().local_process_index}
    else:
        ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
            model_name, 
            trust_remote_code=args.trust_remote_code, 
            quantization_config=quantization_config
        )
        device_map = None
        peft_config = None
    
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        model_name,
        trust_remote_code=args.trust_remote_code,
        device_map=device_map,
        peft_config=peft_config,
        quantization_config=quantization_config
    )

    return model, ref_model

def prepare_optim_and_scheduler(model):
    optimizer, lr_scheduler = None, None
    accelerator = Accelerator()
    
    if accelerator.state.deepspeed_plugin is None:
        return optimizer, lr_scheduler
    if "optimizer" in accelerator.state.deepspeed_plugin.deepspeed_config:
        optimizer = DummyOptim(model.parameters())
    if "scheduler" in accelerator.state.deepspeed_plugin.deepspeed_config:
        lr_scheduler = DummyScheduler(optimizer)

    return optimizer, lr_scheduler
    
# def build_model_unsloth(model_name, load_in_4bit, dtype=None):
#     max_seq_length = 1024 # Choose any! We auto support RoPE Scaling internally!
#     # dtype = torch.bfloat16 # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
#     # load_in_4bit = False # Use 4bit quantization to reduce memory usage. Can be False.
    
#     model, _ = FastLanguageModel.from_pretrained(
#         model_name = model_name,
#         max_seq_length = max_seq_length,
#         dtype = dtype,
#         load_in_4bit = load_in_4bit,
#     )

#     return model

def compute_human_scores(batch, classifier_pipe, sent_kwargs):
    classifier_output = classifier_pipe(batch["response"], **sent_kwargs)
    ref_classifier_output = classifier_pipe(batch["ref_response"], **sent_kwargs)
    human_scores = [torch.tensor(output[0]["score"]) for output in classifier_output]
    ref_human_scores = [torch.tensor(output[0]["score"]) for output in ref_classifier_output]

    return humans_scores, ref_human_scores

def compute_hf_scores(batch, hf_pipe, sent_kwargs):
    query_answer_pairs = [{"text": pair[0], "text_pair": pair[1]} for pair in list(zip(batch["query"], batch["response"]))]
    ref_query_answer_pairs = [{"text": pair[0], "text_pair": pair[1]} for pair in list(zip(batch["query"], batch["ref_response"]))]
    hf_outputs = hf_pipe(query_answer_pairs, **sent_kwargs)
    ref_hf_outputs = hf_pipe(ref_query_answer_pairs, **sent_kwargs)
    hf_scores = [torch.tensor(output[0]["score"]) for output in hf_outputs]
    ref_hf_scores = [torch.tensor(output[0]["score"]) for output in ref_hf_outputs]

    return hf_scores, ref_hf_scores
    
def compute_reward(batch, classifier_pipe, sent_kwargs, normal_training=False, hf_pipe=None, hf_model_weight=None):
    if normal_training:
        return compute_hf_scores(batch, classifier_pipe, sent_kwargs)

    else:
        human_scores, ref_human_scores = compute_human_scores(batch, classifier_pipe, sent_kwargs)

        if hf_pipe:
            hf_scores, ref_hf_scores = compute_hf_scores(batch, classifier_pipe, sent_kwargs)
            rewards, ref_rewards = [], []
    
            for i in range(len(batch["query"])):
                reward = hf_model_weight*hf_scores[i] + (1-hf_model_weight)*human_scores[i]
                ref_reward = hf_model_weight*ref_hf_scores[i] + (1-hf_model_weight)*ref_human_scores[i]
                
                rewards.append(reward)
                ref_rewards.append(ref_reward)
        
            return rewards, ref_rewards
        else:
            return human_scores, ref_human_scores

def train(ppo_trainer, tokenizer, classifier_pipe, generation_kwargs, sent_kwargs, hf_pipe=None, hf_model_weight=None, normal_training=False):
    tqdm.pandas()
    
    for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
        query_tensors = batch['input_ids']

        # Get response from model
        response_tensors, ref_response_tensors = ppo_trainer.generate(
            query_tensors, return_prompt=False, generate_ref_response=True, **generation_kwargs
        )

        batch["response"] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
        batch["ref_response"] = tokenizer.batch_decode(ref_response_tensors, skip_special_tokens=True)

        # Compute reward
        rewards, batch["ref_rewards"] = compute_reward(batch, classifier_pipe, sent_kwargs, normal_training, hf_pipe, hf_model_weight)

        # Run PPO step
        response_tensors_list = [rt for rt in response_tensors] # ppo_trainer.step expects a list
        stats = ppo_trainer.step(batch["input_ids"], response_tensors_list, rewards) # batch["input_ids"] for weights upate because you need a list here
        ppo_trainer.log_stats(stats, batch, rewards, columns_to_log=["query", "response", "ref_response", "ref_rewards"])
