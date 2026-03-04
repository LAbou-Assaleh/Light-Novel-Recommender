import os
import sys
import torch
import torch.distributed as dist
from FlagEmbedding.finetune.embedder.encoder_only.m3.__main__ import main


if dist.is_available() and not dist.is_initialized():
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"

    dist.init_process_group(
        backend="gloo",
        rank=0,
        world_size=1
    )


def run_fine_tuning_programmatically():
    for k in ["RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"]:
        os.environ.pop(k, None)

    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    # Sanity check MPS
    print("MPS available:", torch.backends.mps.is_available())
    print("MPS built:", torch.backends.mps.is_built())

    sys.argv = [
        "prog",
        "--model_name_or_path", "BAAI/bge-m3",
        "--train_data", "bge_test.jsonl",
        "--output_dir", "./outputs/bge-m3-novel-ft",
        "--overwrite_output_dir",
        "--train_group_size", "8",
        "--query_max_len", "128",
        "--passage_max_len", "256",
        "--per_device_train_batch_size", "1",
        "--gradient_accumulation_steps", "8",

        "--num_train_epochs", "10",
        "--learning_rate", "2e-5",
        "--warmup_ratio", "0.1",

        "--knowledge_distillation", "True",
        "--kd_loss_type", "m3_kd_loss",
        "--unified_finetuning", "False",

        "--sentence_pooling_method", "cls",
        "--normalize_embeddings", "True",
        "--temperature", "0.05",

        "--logging_strategy", "steps",
        "--logging_steps", "20",

        "--save_strategy", "steps",
        "--save_steps", "200",

        "--report_to", "tensorboard",
        "--logging_dir", "./tb_logs",
    ]

    main()

if __name__ == "__main__":
    run_fine_tuning_programmatically()
