import os
import torch
import deepspeed
import wandb
import argparse
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.profiler import profile, record_function, tensorboard_trace_handler

from model import MambaMixerModel  
from data import get_clm_dataloader

def main():
    # Parse command line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="conf.yaml", help="Path to training config YAML")
    parser.add_argument("--deepspeed_config", type=str, default="ds_config.json", help="Path to DeepSpeed config JSON")
    args = parser.parse_args()

    # Load training configuration.
    config = OmegaConf.load(args.config)

    # Initialize WandB for logging.
    wandb.init(
        project=config.wandb.project,
        entity=config.wandb.entity,
        notes=config.wandb.notes,
        config=OmegaConf.to_container(config, resolve=True)
    )

    # Setup the data loader using configuration.
    train_loader, vocab_size = get_clm_dataloader(
        tokenizer_name=config.dataloader.tokenizer_name,
        dataset_name=config.dataloader.dataset_name,
        dataset_config=config.dataloader.dataset_config,
        seq_len=config.dataloader.seq_len,
        batch_size=config.dataloader.batch_size,
    )

    # Build the hybrid model using MambaMixerModel from the MambaFormer codebase.
    model = MambaMixerModel(
        n_dims=config.model.n_dims,
        n_embd=config.model.n_embd,
        n_layer=config.model.n_layer,
        interleave=config.model.interleave,
        vocab_size=vocab_size,
        mixed_attn=config.model.mixed_attn,
        n_positions=config.model.n_positions,
        ssm_cfg=config.model.ssm_cfg,  # Pass ssm_cfg from the configuration
        attn_cfg=config.model.attn_cfg,  # Pass attn_cfg from the configuration
    )
    print(model)
    # exit(0)

    # Create an optimizer.
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)

    # Add a learning rate scheduler.
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer, T_max=config.training.train_steps
    # )

    # Load DeepSpeed configuration.
    ds_config = OmegaConf.load(args.deepspeed_config)
    ds_config = OmegaConf.to_container(ds_config, resolve=True)

    # Initialize DeepSpeed with your model, optimizer, and config.
    model, optimizer, _, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        config=ds_config,
    )

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
            input_ids = batch["input_ids"].to(model.device)
            labels = batch["labels"].to(model.device)

            # Forward pass with input_ids and labels
            loss = model(input_ids=input_ids, labels=labels)

            # Backward pass and optimization
            model.backward(loss)  # Backward pass via DeepSpeed.
            model.step()          # Update model parameters.
            # scheduler.step()      # Update learning rate.

            global_step += 1

            # Log loss and step to WandB.
            if global_step % config.wandb.log_every_steps == 0:
                wandb.log({"loss": loss.item(), "step": global_step})
                print(f"Step {global_step}: Loss {loss.item()}")

            profiler.step()  # Update profiler state.

            if global_step >= config.training.train_steps:
                break
        if global_step >= config.training.train_steps:
            break

    profiler.stop()

    wandb.save("logs/*")

    # Save the final model checkpoint.
    os.makedirs(config.training.out_dir, exist_ok=True)
    model.save_checkpoint(config.training.out_dir)
    print(f"Model checkpoint saved to {config.training.out_dir}")

if __name__ == '__main__':
    main()