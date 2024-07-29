import json, argparse

import wandb
from train_dpo import main
from trl import DPOConfig

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--count",
        type=int,
        default=10,
        help="Number of runs.",
    )
    parser.add_argument(
        "--project_name",
        type=str,
        default="trl",
    )
    parser.add_argument(
        "--sweep_id",
        type=str,
        default=None,
    )

    args = parser.parse_args()
    print("input args:\n", json.dumps(vars(args), indent=4, separators=(",", ":")))
    return args


def sweep():
    args_default = {
        # fixed
        "model_name": "microsoft/Phi-3-mini-4k-instruct",
        "dataset_name": "hassanjbara/LONG-DPO",
        "dataset_train_split": 1,
        "trust_remote_code": True,
        # "query_max_length": 125,
        "use_peft": False,
        # "lora_alpha":16,
        # "lora_r":16,
        "quantize": False,
        "flash_attn": True,
        "epochs": 1,
        "save_model": False,
        # sweep
        # "batch_size": 32,
        "mini_batch_size": 4,
        "gradient_accumulation_steps": 1,
        "learning_rate": 1e-05,
        "beta": 0.4,
        "warmup_steps": 150,
        "loss_type": "sigmoid",
        "label_smoothing": 0
    }
    wandb.init(config=args_default)
    config = wandb.config
    dpo_config = DPOConfig(
        output_dir="./sweep_models",
        per_device_train_batch_size=config.mini_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        beta=config.beta,
        num_train_epochs=config.epochs,
        loss_type=config.loss_type,
        label_smoothing=config.label_smoothing,
        logging_steps=10,
        logging_first_step=True,
        bf16=True,
        report_to="wandb",
        save_strategy="no",
    )
    
    main(config, dpo_config)

        
if __name__ == "__main__":
    cli_args = parse_arguments()
    wandb.login()

    sweep_configuration = {
        "method": "random",
        "program": "train_dpo.py",
        # "metric": {"goal": "minimize", "name": "train/rewards/margins"},
        "parameters": {
            'learning_rate': {
                # a flat distribution between 1e-5 and 1e-7
                'distribution': 'uniform',
                'min': 1e-7,
                'max': 1e-5
            },
            'beta': {
                # a flat distribution between 0.3 and 0.7
                'distribution': 'uniform',
                'min': 0.3,
                'max': 0.7
            },
            'label_smoothing': {
                # a flat distribution between 0 and 0.5
                'distribution': 'uniform',
                'min': 0,
                'max': 0.5
            },
            'warmup_steps': {
                'distribution': 'int_uniform',
                'min': 130,
                'max': 230
            },
            'mini_batch_size': {
                'distribution': 'int_uniform',
                'min': 1,
                'max': 4,
            },
            "gradient_accumulation_steps": {
                'distribution': 'int_uniform',
                'min': 1,
                'max': 4,
            },
            'loss_type': {
                'distribution': 'categorical',
                'values': ["sigmoid", "hinge", "ipo", "robust"]
            },
        },
    }

    sweep_id = f"{cli_args.project_name}/{cli_args.sweep_id}"
    if not cli_args.sweep_id:
        sweep_id = wandb.sweep(sweep=sweep_configuration, project=cli_args.project_name)
    wandb.agent(sweep_id, function=sweep, count=cli_args.count)