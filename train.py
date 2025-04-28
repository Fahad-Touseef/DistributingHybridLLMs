import os
import torch
import deepspeed
import wandb
import argparse
from omegaconf import OmegaConf
# from torch.utils.data import DataLoader
# from torch.profiler import profile, record_function, tensorboard_trace_handler
import torch.nn.functional as F

# from model import MambaMixerModel  
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel, MambaConfig
# from data import get_clm_dataloader
from torch.nn import BCEWithLogitsLoss
from data import get_imdb_dataset
from deepspeed import init_distributed
from deepspeed.pipe import PipelineModule
import torch.nn as nn

class ClassificationModel(nn.Module):
    def __init__(self, backbone, d_model, num_classes=1):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, inputs):
        # Unpack input_ids from the tuple
        if isinstance(inputs, tuple):
            input_ids = inputs[0]
        else:
            input_ids = inputs

        outputs = self.backbone(input_ids=input_ids)
        pooled_output = outputs[:, 0, :]  # Use the [CLS] token representation
        logits = self.classifier(pooled_output)
        return logits

def main():
    # Parse command line arguments.
    parser = argparse.ArgumentParser(description="Training script", allow_abbrev=False)
    parser.add_argument("--config", type=str, default="conf.yaml", help="Path to training config YAML")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank passed by deepspeed launcher")
    parser.add_argument("--backend", type=str, default="nccl", help="Distributed backend")

    # This adds DeepSpeed's arguments (including --deepspeed_config)
    deepspeed.add_config_arguments(parser)

    args = parser.parse_args()

    # Initialize the distributed backend
    init_distributed(dist_backend=args.backend)

    # Set the local rank and GPU device
    print(args.local_rank, int(os.environ["LOCAL_RANK"]))
    args.local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(args.local_rank)

    # Load training configuration.
    config = OmegaConf.load(args.config)

    # Initialize WandB for logging.
    if args.local_rank in (0, -1):
        wandb.init(
            project=config.wandb.project,
            entity=config.wandb.entity,
            notes=config.wandb.notes,
            config=OmegaConf.to_container(config, resolve=True)
        )

    # Setup the data loader using configuration.
    # train_loader, vocab_size, pad_token_id = get_clm_dataloader(
    #     tokenizer_name=config.dataloader.tokenizer_name,
    #     dataset_name=config.dataloader.dataset_name,
    #     dataset_config=config.dataloader.dataset_config,
    #     seq_len=config.dataloader.seq_len,
    #     batch_size=config.dataloader.batch_size,
    # )
    train_set, vocab_size, pad_token_id = get_imdb_dataset(
        seq_len=config.dataloader.seq_len,
    )

    # Create the MambaConfig
    mamba_config = MambaConfig(
        d_model=config.model.n_embd,
        n_layer=config.model.n_layer,
        vocab_size=vocab_size,
        ssm_cfg=config.model.ssm_cfg,
        attn_layer_idx=config.model.attn_layer_idx,
        attn_cfg=config.model.attn_cfg,
        rms_norm=True,
        fused_add_norm=False,        # <-- turn OFF the fused kernel
        residual_in_fp32=False,        # optional, but keep norm + residual in same dtype
        )
       
    # Build the hybrid model
    base_model = MambaLMHeadModel(mamba_config)
    model = ClassificationModel(base_model.backbone, d_model=config.model.n_embd)

    # Brought above after adding pipelining
    global_step = 0
    model.train()

    temp = [layer.mixer for layer in model.backbone.layers]
    print(len(temp), temp)

    # Define pipeline layers
    layers = [
        model.backbone.embedding,
        # lambda x: x[0] if isinstance(x, tuple) else x,  # Ensure inputs are unpacked before the first block
        *temp,
        # model.backbone.norm_f,
        lambda x: x[:, 0, :],  # Extract the [CLS] token representation
        model.classifier
    ]

    # Define custom loss function
    def loss_fn(outputs, labels):
        loss = BCEWithLogitsLoss()(outputs.squeeze(-1), labels.float())
        return loss

    pipeline_model = PipelineModule(
        layers=layers,
        loss_fn=loss_fn,
        num_stages=4,
        partition_method="parameters"
    )

    # Update the DeepSpeed initialization to include the configuration
    model_engine, optimizer, training_dataloader, _ = deepspeed.initialize(
        args=args,
        model=pipeline_model,
        model_parameters=pipeline_model.parameters(),
        training_data=train_set
        )

    if model_engine.global_rank == 0:
        print(mamba_config)
        print(pipeline_model)
        print(training_dataloader.loader.batch_size)

    # Training loop
    for step in range(config.training.train_steps):
        loss = model_engine.train_batch()
        if model_engine.global_rank == 0 and step % config.wandb.log_every_steps == 0:
            wandb.log({"loss": loss.item(), "step": step})
            print(f"Step {step}: Loss {loss.item()}")

    # # Set up the PyTorch profiler.
    # profiler = profile(
    #     schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
    #     on_trace_ready=tensorboard_trace_handler('./logs'),
    #     record_shapes=True,
    #     profile_memory=True,
    #     with_stack=True
    # )
    # profiler.start()

    # Training loop.
    # for epoch in range(config.training.epochs): 
    #     for batch in train_loader:
    #         input_ids = batch["input_ids"].to(model_engine.device)
    #         labels = batch["labels"].to(model_engine.device)

    #         output = model_engine(input_ids=input_ids)  
    #         logits = output.logits  # shape: (B, T, V)

    #         loss = F.cross_entropy(
    #             logits[:, :-1, :].reshape(-1, logits.size(-1)),
    #             labels[:, 1:].reshape(-1),
    #             ignore_index=pad_token_id
    #         )
    #         print(loss.item())

    #         # Backward pass and optimization
    #         model_engine.backward(loss)  # Backward pass via DeepSpeed.
    #         model_engine.step()          # Update model parameters.

    #         global_step += 1

    #         # Logging
    #         if model_engine.global_rank == 0:
    #             if global_step % config.wandb.log_every_steps == 0:
    #                 wandb.log({
    #                     "loss": loss.item(),
    #                     "step": global_step,
    #                     "lr": optimizer.param_groups[0]['lr'],
    #                 })
    #                 print(f"Step {global_step}: Loss {loss.item()}")

    #         # # Step profiler (only from rank 0)
    #         # if model_engine.global_rank == 0 and not model_engine.optimizer_overflow:
    #         #     profiler.step()

    #         if global_step >= config.training.train_steps:
    #             break
    #     if global_step >= config.training.train_steps:
    #         break

    # if model_engine.global_rank == 0:
    #     profiler.stop()

    # if model_engine.global_rank == 0:
    #     wandb.save("logs/*")

    # Save the final model checkpoint.
    if model_engine.global_rank == 0:
        wandb.finish()
        # model.save_pretrained(config.training.out_dir)
        # print(f"Model checkpoint saved to {config.training.out_dir}")

if __name__ == "__main__":
    main()