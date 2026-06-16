import torch
import torch.nn as nn


def count_params(module):
    return sum(p.numel() for p in module.parameters())


class SelfAttnFusionModel(nn.Module):
    def __init__(
        self,
        seq_input_size: int,
        track_input_size: int,
        hidden_size: int = 64,
        fusion_size: int = 64,
        dropout: float = 0.3,
        **kwargs,
    ):
        super().__init__()
        assert hidden_size % 2 == 0, "hidden_size must be even"

        self.track_input_size = track_input_size
        self.hidden_size = hidden_size

        self.lstm_hidden = max(1, hidden_size // 2)
        self.seq_encoder = nn.LSTM(
            input_size=seq_input_size,
            hidden_size=self.lstm_hidden,
            batch_first=True,
            bidirectional=True,
        )
        self.seq_pool_score = nn.Linear(hidden_size, 1)

        self.track_encoder = nn.Sequential(
            nn.Linear(track_input_size, hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
        )
        self.tumor_encoder = nn.Sequential(
            nn.Linear(2, hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
        )
        self.stroma_encoder = nn.Sequential(
            nn.Linear(4, hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
        )
        self.immune_encoder = nn.Sequential(
            nn.Linear(2, hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
        )

        self.seq_norm = nn.LayerNorm(hidden_size)
        self.track_norm = nn.LayerNorm(hidden_size)
        self.tumor_norm = nn.LayerNorm(hidden_size)
        self.stroma_norm = nn.LayerNorm(hidden_size)
        self.immune_norm = nn.LayerNorm(hidden_size)

        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=1,
            batch_first=True,
            dropout=dropout,
        )
        self.attn_dropout = nn.Dropout(dropout)
        self.post_attn_norm = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.ffn_dropout = nn.Dropout(dropout)
        self.post_ffn_norm = nn.LayerNorm(hidden_size)
        self.modality_gate = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
        )

        fusion_in = 5 * hidden_size
        self.fusion_fc = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(fusion_in, fusion_size),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_size, 3),
        )

        seq_param_count = count_params(self.seq_encoder) + count_params(self.seq_pool_score) + count_params(self.seq_norm)
        print("Total Sequence parameters:", seq_param_count)
        print("Total Track parameters:", count_params(self.track_encoder) + count_params(self.track_norm))
        print("Total Tumor parameters:", count_params(self.tumor_encoder) + count_params(self.tumor_norm))
        print("Total Stroma parameters:", count_params(self.stroma_encoder) + count_params(self.stroma_norm))
        print("Total Immune parameters:", count_params(self.immune_encoder) + count_params(self.immune_norm))

    def _seq_token(self, x_seq: torch.Tensor) -> torch.Tensor:
        seq_outputs, _ = self.seq_encoder(x_seq)
        seq_logits = self.seq_pool_score(seq_outputs).squeeze(-1)
        seq_weights = torch.softmax(seq_logits, dim=1).unsqueeze(-1)
        seq_token = torch.sum(seq_outputs * seq_weights, dim=1)
        return self.seq_norm(seq_token)

    def _build_tokens(
        self,
        x_seq: torch.Tensor,
        x_track: torch.Tensor,
        x_pdosize: torch.Tensor | None = None,
        x_antigen: torch.Tensor | None = None,
        x_stroma: torch.Tensor | None = None,
        x_immune: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        seq_token = self._seq_token(x_seq)
        b = x_seq.size(0)
        tokens = [seq_token]
        track_token = self.track_encoder(x_track)
        tokens.append(self.track_norm(track_token))
        if x_pdosize is None:
            x_pdosize = torch.zeros(b, 1, dtype=seq_token.dtype, device=seq_token.device)
        if x_antigen is None:
            x_antigen = torch.zeros(b, 1, dtype=seq_token.dtype, device=seq_token.device)
        if x_stroma is None:
            x_stroma = torch.zeros(b, 4, dtype=seq_token.dtype, device=seq_token.device)
        if x_immune is None:
            x_immune = torch.zeros(b, 2, dtype=seq_token.dtype, device=seq_token.device)
        tumor_input = torch.cat([x_pdosize, x_antigen], dim=1)
        tokens.append(self.tumor_norm(self.tumor_encoder(tumor_input)))
        tokens.append(self.stroma_norm(self.stroma_encoder(x_stroma)))
        tokens.append(self.immune_norm(self.immune_encoder(x_immune)))
        return tokens

    def _fuse_tokens(self, tokens_stacked: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        attn_out, attn_weights = self.self_attn(
            query=tokens_stacked,
            key=tokens_stacked,
            value=tokens_stacked,
            need_weights=True,
        )
        tokens = self.post_attn_norm(tokens_stacked + self.attn_dropout(attn_out))
        ffn_out = self.ffn(tokens)
        tokens = self.post_ffn_norm(tokens + self.ffn_dropout(ffn_out))
        gates = self.modality_gate(tokens)
        gated_tokens = tokens * gates
        return gated_tokens, attn_weights

    def forward(
        self,
        x_seq: torch.Tensor,
        x_track: torch.Tensor,
        x_pdosize: torch.Tensor | None = None,
        x_antigen: torch.Tensor | None = None,
        x_stroma: torch.Tensor | None = None,
        x_immune: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        tokens = self._build_tokens(x_seq, x_track, x_pdosize, x_antigen, x_stroma, x_immune)
        tokens_stacked = torch.stack(tokens, dim=1)
        fused_tokens, _ = self._fuse_tokens(tokens_stacked)
        fused = fused_tokens.reshape(x_seq.size(0), -1)
        return self.fusion_fc(fused)

    def get_embedding(
        self,
        x_seq: torch.Tensor,
        x_track: torch.Tensor,
        x_pdosize: torch.Tensor | None = None,
        x_antigen: torch.Tensor | None = None,
        x_stroma: torch.Tensor | None = None,
        x_immune: torch.Tensor | None = None,
    ) -> torch.Tensor:
        tokens = self._build_tokens(x_seq, x_track, x_pdosize, x_antigen, x_stroma, x_immune)
        tokens_stacked = torch.stack(tokens, dim=1)
        fused_tokens, _ = self._fuse_tokens(tokens_stacked)
        return fused_tokens.reshape(x_seq.size(0), -1)

    def get_attn_weights(
        self,
        x_seq: torch.Tensor,
        x_track: torch.Tensor,
        x_pdosize: torch.Tensor | None = None,
        x_antigen: torch.Tensor | None = None,
        x_stroma: torch.Tensor | None = None,
        x_immune: torch.Tensor | None = None,
    ) -> torch.Tensor:
        tokens = self._build_tokens(x_seq, x_track, x_pdosize, x_antigen, x_stroma, x_immune)
        tokens_stacked = torch.stack(tokens, dim=1)
        _, attn_weights = self._fuse_tokens(tokens_stacked)
        return attn_weights
