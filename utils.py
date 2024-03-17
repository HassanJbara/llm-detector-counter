from transformers import pipeline
from trl.import_utils import is_xpu_available
from tqdm import tqdm
import torch

def prepare_classifier_pipe(ppo_trainer, reward_model):
    # build classifier pipeline
    device = ppo_trainer.accelerator.device
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

def train(ppo_trainer, classifier_pipe, tokenizer, generation_kwargs, sent_kwargs):
    tqdm.pandas() # may need to be in the beginning of the file, needs testing
    
    for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
        query_tensors = batch['input_ids']

        # Get response from model
        response_tensors, ref_response_tensors = ppo_trainer.generate(
            query_tensors, return_prompt=False, generate_ref_response=True, **generation_kwargs
        )

        batch["response"] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
        batch["ref_response"] = tokenizer.batch_decode(ref_response_tensors, skip_special_tokens=True)

        # Compute sentiment score
        pipe_outputs = classifier_pipe(batch["response"], **sent_kwargs)
        rewards = [torch.tensor(output[0]["score"]) for output in pipe_outputs]
        ref_pipe_outputs = classifier_pipe(batch["ref_response"], **sent_kwargs)
        ref_rewards = [torch.tensor(output[0]["score"]) for output in ref_pipe_outputs]
        batch["ref_rewards"] = ref_rewards

        # Run PPO step
        response_tensors_list = [rt for rt in response_tensors] # ppo_trainer.step expects a list
        stats = ppo_trainer.step(batch["input_ids"], response_tensors_list, rewards) # batch["input_ids"] for weights upate because you need a list here
        ppo_trainer.log_stats(stats, batch, rewards, columns_to_log=["query", "response", "ref_response", "ref_rewards"])
