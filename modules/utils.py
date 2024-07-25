import torch
from transformers import pipeline, QuantoConfig, AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator
from trl.import_utils import is_xpu_available
from trl import AutoModelForCausalLMWithValueHead
# from unsloth import FastLanguageModel
from accelerate.utils import DummyOptim, DummyScheduler
from peft import LoraConfig

# For loading binoculars
# accelerator = Accelerator()


def scale_rewards(reward, left_min, left_max, right_min, right_max) -> float:
    # Figure out how 'wide' each range is
    left_span = left_max - left_min
    right_span = right_max - right_min

    # Convert the left range into a 0-1 range (float)
    reward_scaled = float(reward - left_min) / float(left_span)

    # Convert the 0-1 range into a value in the right range.
    return right_min + (reward_scaled * right_span)

# @accelerator.on_local_main_process
# def prepare_binoculars():
#     print("Loading Binoculars...")
#     from binoculars import Binoculars # causes errors with deepspeed
#     return Binoculars()
#     # pass

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

def build_classifier(ppo_trainer, classifier_name: str, device=None):
    prepare_binoculars()
    if "binoculars" in classifier_name.lower():
        return prepare_binoculars()
    else:
        return prepare_classifier_pipe(ppo_trainer, classifier_name, device)

def build_model(model_name, args):
    assert not (args.quantize and args.flash_attn), "can not use quantization and flash-attn at the same time!"
    
    quantization_config = QuantoConfig(weights="int8") if args.quantize else None
    flash_attn_config = "flash_attention_2" if args.flash_attn else None
    torch_dtype= torch.bfloat16 if args.flash_attn else None
        
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
            quantization_config=quantization_config,
            attn_implementation=flash_attn_config,
            torch_dtype=torch_dtype, 
        )
        device_map = None
        peft_config = None
    
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        model_name,
        trust_remote_code=args.trust_remote_code,
        device_map=device_map,
        peft_config=peft_config,
        quantization_config=quantization_config,
        attn_implementation=flash_attn_config,
        torch_dtype=torch_dtype,
    )

    return model, ref_model

def build_model_for_benchmark(model_name: str, quantize: bool = False, flash_attn: bool = True, device="cuda:0"):
    assert not (quantize and flash_attn), "please use either quantization or flash_attn, not both!"
    
    quantization_config = BitsAndBytesConfig(load_in_8bit=True) if quantize else None
    dtype = torch.bfloat16 if flash_attn else None 
    attn = "flash_attention_2" if flash_attn else None
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                 quantization_config=quantization_config, # do not use with flash_attn2
                                                 torch_dtype=dtype,
                                                 attn_implementation=attn,
                                                 trust_remote_code=True
                                                ).to(device)

    return model, tokenizer

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

def word_count(text):
  return len(re.findall(r'\w+', text))

  return text_len - query_len

