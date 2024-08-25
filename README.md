## Introduction

[Blog Post](https://xela.blog/posts/human-like-llms/)

The goal of this project is implementing a method for fine-tuning LLMs to deceive any LLM detector using RL, a method that’s model- and detector-agnostic, meaning that it should theoretically work with any model and on any detector. The idea is to train an arbitrary LLM model to adapt its outputs in a way that deceives an arbitrary LLM detector using RL with the detector model as reward (punishment) model. Please take a look at the “Useful Links” and "Related Literature" sections for more on this topic.

### Project Questions

1. How good would it be, if it works on one detector, on another detector?  
2. How good are detectors really?  
3. Would this make the LLM output more natural, more human like?

### Scripts

The main training script is `train_dpo.py` and could be used as such:

```bash
python train_dpo.py \
        --dataset_name=hassanjbara/LONG-DPO \
        --model_name=mistralai/Mistral-Nemo-Instruct-2407 \
        --per_device_train_batch_size=1 \
        --learning_rate=1e-6 \
        --beta=0.6 \
        --gradient_accumulation_steps=8 \
        --warmup_steps=150 \
        --bf16 \
        --use_peft \
        --quantize \
        --num_train_epochs=1 \
        --dataset_train_split=1 \
```

The script also supports huggingface accelerate and could be used with the deepspeed configuration in the repository.

### Useful Links

* [Are LLMs the Beginning or End of NLP?](https://www.youtube.com/watch?v=KVDKWrsP3es\&t=2536s\&pp=ygUOaXMgbmxwIHRoZSBlbmQ%3D)  
* [Illustrating Reinforcement Learning from Human Feedback (RLHF)](https://huggingface.co/blog/rlhf)   
* [RLHF: Reinforcement Learning from Human Feedback | by Ms Aerin | Oct, 2023 | Towards Data Science](https://towardsdatascience.com/rlhf-reinforcement-learning-from-human-feedback-faa5ff4761d1)   
* [TRL \- Transformer Reinforcement Learning](https://huggingface.co/docs/trl/index) (how-to guides)  
* [Teach Llamas to Talk: Recent Progress in Instruction Tuning](https://gaotianyu.xyz/blog/2023/11/30/instruction-tuning/)   
* [huggingface/alignment-handbook: Robust recipes for to align language models with human and AI preferences](https://github.com/huggingface/alignment-handbook) 

### Runs

[W&B Project](https://wandb.ai/hasanjbara/LLM_Detector_Counter/overview)

## **Related Literature**

* [\[2304.04736\] On the Possibilities of AI-Generated Text Detection](https://arxiv.org/abs/2304.04736) \[[Twitter Thread](https://twitter.com/furongh/status/1645780628724502528)\]  
* [\[2203.02155\] Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)   
* [\[2305.15047\] Ghostbuster: Detecting Text Ghostwritten by Large Language Models](https://arxiv.org/abs/2305.15047) \[[Code](https://github.com/vivek3141/ghostbuster)\]
