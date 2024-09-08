from pathlib import Path

import torch
import torch.nn as nn
from xformers.ops.fmha.attn_bias import BlockDiagonalCausalMask

from denoizr.transformer import TransformerModel

from .config import TokenizerModelArgs
from .data import TokenizerTrainingInputBatch, TokenizerTrainingOutputBatch


class TokenizerModel(nn.Module):
    def __init__(self, config: TokenizerModelArgs):
        super().__init__()
        self.config = config

        self.encoder = TransformerModel(args=config.encoder)

        self.CLS_embedding = nn.Parameter(torch.FloatTensor(self.config.dim).uniform_())
        self.SEP_embedding = nn.Parameter(torch.FloatTensor(self.config.dim).uniform_())
        self.END_id = config.codebook_size
        self.embedding = nn.Embedding(
            num_embeddings=config.codebook_size,
            embedding_dim=config.encoder.dim,
        )

        self.decoder = TransformerModel(args=config.decoder)
        self.decoder.attn_mask_fn = BlockDiagonalCausalMask

        self.linear = nn.Linear(
            config.decoder.dim, config.codebook_size + 1
        )  # +1 for END token

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    def save_checkpoint(self, output_dir: str):
        output_dir = Path(output_dir)
        assert output_dir.exists(), output_dir
        config_path = output_dir / "model_config.json"
        checkpoint_path = output_dir / "best.pth"
        assert not config_path.exists(), config_path
        assert not checkpoint_path.exists(), checkpoint_path
        self.config.save(config_path)
        torch.save(self.state_dict(), checkpoint_path)

    @property
    def device(self):
        return next(self.parameters()).device

    @classmethod
    def from_model_dir(cls, model_dir: str):
        model_dir = Path(model_dir)
        config_path = model_dir / "model_config.json"
        checkpoint_path = model_dir / "best.pth"
        assert config_path.exists(), config_path
        assert checkpoint_path.exists(), checkpoint_path
        config = TokenizerModelArgs.load(config_path)
        model = cls(config=config)
        model.load_state_dict(torch.load(checkpoint_path))
        return model

    def forward(self, batch: TokenizerTrainingInputBatch):
        encoded_features = self.encoder.forward(
            embeddings=batch.features,
            q_seqlen=batch.encoder_q_seqlen,
        )
        embedded_tokens = self.embedding.forward(input=batch.tokens)

        decoder_input = torch.empty(
            size=(batch.decoder_input_length, self.config.dim),
            device=batch.device,
        )

        decoder_input[batch.decoder_input_CLS_token_mask] = self.CLS_embedding
        decoder_input[batch.decoder_input_features_mask] = encoded_features
        decoder_input[batch.decoder_input_SEP_token_mask] = self.SEP_embedding
        decoder_input[batch.decoder_input_token_mask] = embedded_tokens

        decoder_output = self.decoder.forward(
            embeddings=decoder_input,
            q_seqlen=batch.decoder_q_seqlen,
        )

        logits = self.linear.forward(decoder_output)

        targets = torch.full(
            size=(logits.shape[0],),
            fill_value=-100,
            dtype=torch.long,
            device=batch.device,
        )
        targets[batch.decoder_output_token_mask] = batch.tokens
        targets[batch.decoder_output_END_token_mask] = self.END_id

        loss = self.loss_fn(input=logits, target=targets)

        loss_logits = logits[batch.decoder_output_loss_mask]
        loss_targets = targets[batch.decoder_output_loss_mask]

        accuracy = (loss_logits.argmax(dim=-1) == loss_targets).float().mean()

        return TokenizerTrainingOutputBatch(
            loss=loss,
            accuracy=accuracy,
        )

    @torch.inference_mode()
    def encode(self, features: torch.Tensor):
        # always move everything to cuda
        self.cuda()
        features = features.cuda()

        T, D = features.shape
        STOP_FACTOR = 3  # predicted tokes is up to (STOP_FACTOR-1) times the length of the features

        encoded_features = self.encoder.forward(embeddings=features, q_seqlen=[T])

        decoder_input = torch.zeros(size=(STOP_FACTOR * T, D), device=features.device)
        decoder_input[: len(features)] = encoded_features
        decoder_input[len(features)] = self.CLS_embedding
        decoder_input_len = T + 1

        generated_tokens = torch.zeros(
            size=((STOP_FACTOR - 1) * T,), dtype=torch.long, device=features.device
        )
        num_generated_tokens = 0

        while True:
            decoder_output = self.decoder.forward(
                embeddings=decoder_input[:decoder_input_len],
                q_seqlen=[decoder_input_len],
            )  # (idx + 1, D)
            logits = self.linear.forward(decoder_output[-1:])  # (1, codebook_size + 1)

            next_token = logits.argmax(dim=-1)
            generated_tokens[num_generated_tokens] = next_token
            num_generated_tokens += 1

            if next_token.item() == self.END_id or decoder_input_len == STOP_FACTOR * T:
                break

            next_embedding = self.embedding.forward(input=next_token)  # (1, D)
            decoder_input[decoder_input_len] = next_embedding[0]
            decoder_input_len += 1

        return generated_tokens[:num_generated_tokens].cpu()
