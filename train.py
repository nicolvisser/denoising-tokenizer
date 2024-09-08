from denoizr.tokenizer import (
    TokenizerModelArgs,
    TokenizerTrainingArgs,
    TransformerModelArgs,
    train,
)

model_args = TokenizerModelArgs(
    codebook_size=500,
    encoder=TransformerModelArgs(
        dim=1024,
        n_layers=6,
        head_dim=48,
        hidden_dim=2048,
        n_heads=16,
        n_kv_heads=16,
        norm_eps=1e-6,
        rope_theta=1_000_000,
    ),
    decoder=TransformerModelArgs(
        dim=1024,
        n_layers=6,
        head_dim=48,
        hidden_dim=2048,
        n_heads=16,
        n_kv_heads=16,
        norm_eps=1e-6,
        rope_theta=1_000_000,
    ),
)

train_args = TokenizerTrainingArgs(
    train_features_data_paths=[
        "/mnt/wsl/nvme/data/LibriSpeech/features/wavlm-large/layer-24/train-clean-100.h5",
    ],
    valid_features_data_paths=[
        "/mnt/wsl/nvme/data/LibriSpeech/features/wavlm-large/layer-24/dev-clean.h5",
    ],
    codebook_path="wavlm-large-layer-24-kmeans-500-centroids.npy",
    dpdp_lmbda=0.0,
    dpdp_num_neighbors=5,
    dedupe_tokens=True,
    betas=(0.9, 0.95),
    weight_decay=0.1,
    lr_max=5e-4,
    lr_final=5e-6,
    warmup_steps=10000,
    decay_steps=40000,
    batch_size=2,
    num_workers=63,
    precision="bf16-mixed",
    accumulate_grad_batches=1,
    gradient_clip_val=1.0,
    log_every_n_steps=1,
    val_check_interval=0.5,
    save_last=False,
    save_last_weights_only=False,
    save_best=True,
    save_best_top_k=1,
    save_best_weights_only=True,
    early_stopping=True,
    early_stopping_patience=5,
    max_epochs=None,
    max_steps=75000,
)

train(
    model_args=model_args,
    train_args=train_args,
    fast_dev_run=False,
)
