import os
import torch
import deepspeed
import wandb
import argparse
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.profiler import profile, record_function, tensorboard_trace_handler
import json
import torch.nn.functional as F

# from model import MambaMixerModel  
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel, MambaConfig
from data import get_clm_dataloader

def main():
    # Parse command line arguments.
    parser = argparse.ArgumentParser(description="Training script", allow_abbrev=False)
    parser.add_argument("--config", type=str, default="conf.yaml", help="Path to training config YAML")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank passed by deepspeed launcher")

    # This adds DeepSpeed's arguments (including --deepspeed_config)
    deepspeed.add_config_arguments(parser)

    args = parser.parse_args()

    # Load training configuration.
    config = OmegaConf.load(args.config)

    # Initialize WandB for logging.
    if args.local_rank == 0 or args.local_rank == -1:
        wandb.init(
            project=config.wandb.project,
            entity=config.wandb.entity,
            notes=config.wandb.notes,
            config=OmegaConf.to_container(config, resolve=True)
        )

    # Setup the data loader using configuration.
    train_loader, vocab_size, pad_token_id = get_clm_dataloader(
        tokenizer_name=config.dataloader.tokenizer_name,
        dataset_name=config.dataloader.dataset_name,
        dataset_config=config.dataloader.dataset_config,
        seq_len=config.dataloader.seq_len,
        batch_size=config.dataloader.batch_size,
    )

    # Create the MambaConfig
    mamba_config = MambaConfig(
                                d_model = config.model.n_embd,
                                n_layer = config.model.n_layer,
                                vocab_size = vocab_size,
                                ssm_cfg=config.model.ssm_cfg,  # Pass ssm_cfg from the configuration
                                attn_layer_idx=config.model.attn_layer_idx,  # Pass attn_layer_idx from the configuration
                                attn_cfg=config.model.attn_cfg,  # Pass attn_cfg from the configuration
                            )
    
    # Build the hybrid model using MambaLMHeadModel
    model = MambaLMHeadModel(mamba_config)

    # # Load DeepSpeed configuration.
    # with open(args.deepspeed_config, "r") as f:
    #     ds_config = json.load(f)  # Load the DeepSpeed config as a dictionary

    # Initialize DeepSpeed with your model, optimizer, and config.
    model_engine, optimizer, _, _ = deepspeed.initialize(
    args = args,
    model=model,
    model_parameters=model.parameters())

    if model_engine.global_rank == 0:
        print(mamba_config)
        print(model)

    global_step = 0
    model.train()

    # Set up the PyTorch profiler.
    profiler = profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
        on_trace_ready=tensorboard_trace_handler('./logs'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    )
    profiler.start()

    # Training loop.
    for epoch in range(config.training.epochs): 
        for batch in train_loader:
            # Assume batch contains input_ids and labels
            input_ids = batch["input_ids"].to(model_engine.device)
            labels = batch["labels"].to(model_engine.device)

            output = model_engine(input_ids=input_ids)  
            logits = output.logits  # shape: (B, T, V)

            loss = F.cross_entropy(
                logits[:, :-1, :].reshape(-1, logits.size(-1)),
                labels[:, 1:].reshape(-1),
                ignore_index=pad_token_id
            )
            print(loss.item())

            # Backward pass and optimization
            model_engine.backward(loss)  # Backward pass via DeepSpeed.
            model_engine.step()          # Update model parameters.

            global_step += 1

            # Log loss and step to WandB.
            if global_step % config.wandb.log_every_steps == 0:
                if model_engine.global_rank == 0:
                    wandb.log({"loss": loss.item(), "step": global_step})
                print(f"Step {global_step}: Loss {loss.item()}")

            # Step profiler (only from rank 0)
            if model_engine.global_rank == 0 and not model_engine.optimizer_overflow:
                profiler.step()

            if global_step >= config.training.train_steps:
                break
        if global_step >= config.training.train_steps:
            break

    if model_engine.global_rank == 0:
        profiler.stop()

    if model_engine.global_rank == 0:
        wandb.save("logs/*")

    # Save the final model checkpoint.
    os.makedirs(config.training.out_dir, exist_ok=True)
    model.save_checkpoint(config.training.out_dir)
    print(f"Model checkpoint saved to {config.training.out_dir}")

if __name__ == '__main__':
    main()