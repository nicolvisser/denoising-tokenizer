import math
from pathlib import Path

import lightning as L
import torch
import torch.optim as optim
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from denoizr.codename import codename

from .config import TokenizerModelArgs, TokenizerTrainingArgs
from .data import TokenizerTrainingInputBatch, TokenizerTrainingOutputBatch, collate_fn
from .dataset import TokenizerDataset
from .model import TokenizerModel


class TokenizerTrainingModule(L.LightningModule):
    def __init__(
        self,
        model_args: TokenizerModelArgs,
        train_args: TokenizerTrainingArgs,
    ):
        super().__init__()
        self.train_args = train_args
        self.model = TokenizerModel(model_args)

    def training_step(self, batch: TokenizerTrainingInputBatch, batch_idx):
        output: TokenizerTrainingOutputBatch = self.model.forward(batch)
        self.log(
            "train/loss",
            output.loss,
            batch_size=batch.batch_size,
        )
        self.log(
            "train/accuracy",
            output.accuracy,
            batch_size=batch.batch_size,
            prog_bar=True,
        )
        return output.loss

    def validation_step(self, batch: TokenizerTrainingInputBatch, batch_idx):
        output: TokenizerTrainingOutputBatch = self.model(batch)
        self.log(
            "val/loss",
            output.loss,
            batch_size=batch.batch_size,
        )
        self.log(
            "val/accuracy",
            output.accuracy,
            batch_size=batch.batch_size,
            prog_bar=True,
        )
        return output.loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.train_args.lr_max,
            betas=self.train_args.betas,
            weight_decay=self.train_args.weight_decay,
        )

        sched_config = {
            "scheduler": LinearRampCosineDecayScheduler(
                optimizer,
                n_linear_steps=self.train_args.warmup_steps,
                n_decay_steps=self.train_args.decay_steps,
                lr_init=0,
                lr_max=self.train_args.lr_max,
                lr_final=self.train_args.lr_final,
            ),
            "frequency": 1,
            "interval": "step",
        }

        return {"optimizer": optimizer, "lr_scheduler": sched_config}

    @classmethod
    def from_model_dir(cls, model_dir: str):
        model_dir = Path(model_dir)
        checkpoint_path: Path = model_dir / "best.ckpt"
        model_args_path: Path = model_dir / "model_config.json"
        train_args_path: Path = model_dir / "train_config.json"
        assert checkpoint_path.exists(), checkpoint_path
        assert model_args_path.exists(), model_args_path
        assert train_args_path.exists(), train_args_path
        return cls.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            model_args=TokenizerModelArgs.load(model_args_path),
            train_args=TokenizerTrainingArgs.load(train_args_path),
        )

    def save_vanilla_torch_checkpoint(self, output_dir: str):
        self.model.save_checkpoint(output_dir)


class LinearRampCosineDecayScheduler(optim.lr_scheduler._LRScheduler):
    """
    Custom learning rate scheduler that increases linearly for n_linear_steps,
    then decays cosine annealing for n_decay_steps,
    then stays at lr_final for the remaining steps.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        n_linear_steps (int): Number of steps for linear increase.
        n_decay_steps (int): Number of steps for cosine decay.
        lr_init (float, optional): Initial learning rate. Default is 0.
        lr_max (float, optional): Maximum learning rate. Default is 1e-5.
        lr_final (float, optional): Final learning rate. Default is 1e-6.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        n_linear_steps: int,
        n_decay_steps: int,
        lr_init: float,
        lr_max: float,
        lr_final: float,
    ):
        self.n_linear_steps = n_linear_steps
        self.n_decay_steps = n_decay_steps

        self.lr_init = lr_init
        self.lr_max = lr_max
        self.lr_final = lr_final

        super().__init__(optimizer, last_epoch=-1)

    def get_lr(self):
        current_step = self.last_epoch

        if current_step <= self.n_linear_steps:
            lr = self.lr_init + (self.lr_max - self.lr_init) * current_step / (
                self.n_linear_steps
            )
        elif current_step <= self.n_linear_steps + self.n_decay_steps:
            lr = (
                0.5
                * math.cos(
                    (current_step - self.n_linear_steps)
                    / (self.n_decay_steps)
                    * math.pi
                )
                + 0.5
            ) * (self.lr_max - self.lr_final) + self.lr_final
        else:
            lr = self.lr_final
        return [lr for _ in self.base_lrs]


def train(
    model_args: TokenizerModelArgs,
    train_args: TokenizerTrainingArgs,
    fast_dev_run=False,
):
    # create a unique readable name for this run
    run_id = codename()

    checkpoint_dir = Path("checkpoints") / "tokenizer" / run_id
    log_dir = Path("logs") / "tokenizer" / run_id

    if not fast_dev_run:
        # print the run name and directories:
        print()
        print("\033[32m\033[1m Training started\033[0m")
        print()
        print("\033[32m Run name: {} \033[0m".format(run_id))
        print("\033[32m Checkpoint dir: {}\033[0m".format(checkpoint_dir.absolute()))
        print("\033[32m Log dir: {}\033[0m".format(log_dir.absolute()))
        print()
        print("\033[32m Open tensorboard with: \033[0m")
        print(
            "\033[32m tensorboard --logdir={} \033[0m".format(log_dir.parent.absolute())
        )
        print()

        print("Model args:")
        print(model_args)
        print()

    print("Loading train dataset")
    train_dataset = TokenizerDataset(
        features_data_paths=train_args.train_features_data_paths,
        codebook_path=train_args.codebook_path,
        dpdp_lmbda=train_args.dpdp_lmbda,
        dpdp_num_neighbors=train_args.dpdp_num_neighbors,
        dedupe_tokens=train_args.dedupe_tokens,
    )

    print("Loading valid dataset")
    valid_dataset = TokenizerDataset(
        features_data_paths=train_args.valid_features_data_paths,
        codebook_path=train_args.codebook_path,
        dpdp_lmbda=train_args.dpdp_lmbda,
        dpdp_num_neighbors=train_args.dpdp_num_neighbors,
        dedupe_tokens=train_args.dedupe_tokens,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_args.batch_size,
        collate_fn=collate_fn,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
        num_workers=train_args.num_workers,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=train_args.batch_size,
        collate_fn=collate_fn,
        pin_memory=True,
        shuffle=False,
        num_workers=train_args.num_workers,
    )

    model = TokenizerTrainingModule(model_args, train_args)

    torch.set_float32_matmul_precision("medium")  # optimization for RTX 3090

    logger = TensorBoardLogger(
        save_dir=log_dir.parent.parent,
        name=log_dir.parent.name,
        version=log_dir.name,
    )
    if not fast_dev_run:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        model_args.save(checkpoint_dir / "model_config.json")
        train_args.save(checkpoint_dir / "train_config.json")

    callbacks = []

    lr_monitor_callback = LearningRateMonitor(logging_interval="step")
    callbacks.append(lr_monitor_callback)

    if train_args.save_best:
        best_checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="best",
            monitor="val/loss",
            mode="min",
            save_top_k=train_args.save_best_top_k,
            save_last=False,
            save_weights_only=train_args.save_best_weights_only,
            verbose=True,
        )
        callbacks.append(best_checkpoint_callback)

    if train_args.save_last:
        last_checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="last",
            save_last=False,
            save_weights_only=train_args.save_last_weights_only,
        )
        callbacks.append(last_checkpoint_callback)

    if train_args.early_stopping:
        assert train_args.early_stopping_patience is not None
        early_stopping_callback = EarlyStopping(
            monitor="val/loss",
            patience=train_args.early_stopping_patience,
            verbose=True,
            mode="min",
        )
        callbacks.append(early_stopping_callback)

    trainer = L.Trainer(
        accelerator="gpu",
        precision=train_args.precision,
        logger=logger,
        max_epochs=train_args.max_epochs,
        max_steps=train_args.max_steps,
        gradient_clip_val=train_args.gradient_clip_val,
        callbacks=callbacks,
        log_every_n_steps=train_args.log_every_n_steps,
        val_check_interval=train_args.val_check_interval,
        accumulate_grad_batches=train_args.accumulate_grad_batches,
        fast_dev_run=fast_dev_run,
    )
    trainer.fit(model, train_loader, valid_loader)
