import torch
from tqdm import tqdm

def compute_human_scores(batch, classifier_pipe, sent_kwargs):
    classifier_outputs_lists = classifier_pipe(batch["response"], **sent_kwargs) # list of lists
    ref_classifier_outputs_lists = classifier_pipe(batch["ref_response"], **sent_kwargs) #list of lists
    
    classifier_outputs = [output for outputs_list in classifier_outputs_lists for output in outputs_list] # flatten
    ref_classifier_outputs = [output for outputs_list in ref_classifier_outputs_lists for output in outputs_list] # flatten

    classifier_outputs = [output for output in classifier_outputs if output['label'].lower() == 'human'] # filter for human scores
    ref_classifier_outputs = [output for output in ref_classifier_outputs if output['label'].lower() == 'human'] # filter for human scores
    
    human_scores = [torch.tensor(output["score"]) for output in classifier_outputs] 
    ref_human_scores = [torch.tensor(output["score"]) for output in ref_classifier_outputs]

    return human_scores, ref_human_scores

def compute_hf_scores(batch, hf_pipe, sent_kwargs):
    query_answer_pairs = [{"text": pair[0], "text_pair": pair[1]} for pair in list(zip(batch["query"], batch["response"]))]
    ref_query_answer_pairs = [{"text": pair[0], "text_pair": pair[1]} for pair in list(zip(batch["query"], batch["ref_response"]))]
    hf_outputs = hf_pipe(query_answer_pairs, **sent_kwargs)
    ref_hf_outputs = hf_pipe(ref_query_answer_pairs, **sent_kwargs)
    hf_scores = [torch.tensor(output[0]["score"]) for output in hf_outputs]
    ref_hf_scores = [torch.tensor(output[0]["score"]) for output in ref_hf_outputs]

    return hf_scores, ref_hf_scores
    
def compute_reward(batch, classifier_pipe, sent_kwargs, normal_training=False, hf_pipe=None, hf_model_weight=None, use_min: bool = False):
    # if normal feedback reward model training 
    if normal_training:
        return compute_hf_scores(batch, classifier_pipe, sent_kwargs)

    # otherwise, with human score classifier
    human_scores, ref_human_scores = compute_human_scores(batch, classifier_pipe, sent_kwargs)

    if hf_pipe:
        hf_scores, ref_hf_scores = compute_hf_scores(batch, hf_pipe, sent_kwargs)
        rewards, ref_rewards = [], []

        for i in range(len(batch["query"])):
            if use_min:
                rewards.append(min(human_scores[i], hf_scores[i]))
                ref_rewards.append(min(ref_human_scores[i], ref_hf_scores[i]))
            else:
                normalized_human_score = (1/256) * human_scores[i] # 256 is half the max length of the response
                normalized_ref_human_score = (1/256) * ref_human_scores[i] # 256 is half the max length of the response
                reward = hf_model_weight*hf_scores[i] + (1-hf_model_weight)*normalized_human_score
                ref_reward = hf_model_weight*ref_hf_scores[i] + (1-hf_model_weight)*normalized_ref_human_score
                    
                rewards.append(reward)
                ref_rewards.append(ref_reward)
        
        return rewards, ref_rewards
    else:
        return human_scores, ref_human_scores

def train(ppo_trainer, tokenizer, classifier_pipe, generation_kwargs, sent_kwargs, hf_pipe=None, 
          hf_model_weight=None, normal_training: bool = False, use_min: bool = False):
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
        rewards, batch["ref_rewards"] = compute_reward(batch, classifier_pipe, sent_kwargs, normal_training, hf_pipe, hf_model_weight, use_min)

        # Run PPO step
        response_tensors_list = [rt for rt in response_tensors] # ppo_trainer.step expects a list
        stats = ppo_trainer.step(batch["input_ids"], response_tensors_list, rewards) # batch["input_ids"] for weights upate because you need a list here
        ppo_trainer.log_stats(stats, batch, rewards, columns_to_log=["query", "response", "ref_response", "ref_rewards"])
