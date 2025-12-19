import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------
#  Pointer Network 정의 (inference 전용)
# ------------------------------

class GlimpseAttention(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.W_ref = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, 1, bias=False)
        self.scale = 1.0 / math.sqrt(hidden_dim)

    def forward(self, query, ref, mask=None):
        """
        query: (B, H)
        ref  : (B, L, H)
        mask : (B, L)  True = masked
        """
        encoded_ref = self.W_ref(ref)                 # (B, L, H)
        encoded_q = self.W_q(query).unsqueeze(1)      # (B, 1, H)
        scores = self.v(torch.tanh(encoded_ref + encoded_q)).squeeze(-1)  # (B, L)
        scores = scores * self.scale

        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        attn = F.softmax(scores, dim=1)               # (B, L)
        glimpse = torch.bmm(attn.unsqueeze(1), ref).squeeze(1)  # (B, H)
        return glimpse, scores


def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name and param.dim() > 1:
            nn.init.xavier_uniform_(param)
        elif 'bias' in name:
            nn.init.constant_(param, 0.0)


class PointerNetwork(nn.Module):
    """
    학습에 사용한 구조를 inference 용으로 재현한 Pointer Network
    - Encoder: TransformerEncoder
    - Decoder: LSTMCell + Glimpse/Pointer Attention
    """
    def __init__(self, input_dim=2, hidden_dim=128, n_layers=2, n_heads=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Linear(input_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            batch_first=True,
            dim_feedforward=4 * hidden_dim,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.decoder_cell = nn.LSTMCell(hidden_dim, hidden_dim)

        self.glimpse = GlimpseAttention(hidden_dim)
        self.pointer = GlimpseAttention(hidden_dim)

        self.decoder_input0 = nn.Parameter(torch.zeros(hidden_dim))

        self.apply(init_weights)

    def encode(self, inputs, lengths):
        """
        inputs : (B, L, 2)
        lengths: (B,)
        """
        device = inputs.device
        B, L, _ = inputs.size()

        embedded = self.embedding(inputs)  # (B, L, H)

        range_tensor = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
        len_tensor = lengths.unsqueeze(1).expand(B, L)
        padding_mask = range_tensor >= len_tensor  # (B, L) True: pad

        enc_out = self.encoder(embedded, src_key_padding_mask=padding_mask)  # (B, L, H)

        return enc_out, embedded, padding_mask

    def decode_greedy(self, enc_out, embedded, padding_mask):
        """
        enc_out      : (1, L, H)
        embedded     : (1, L, H)
        padding_mask : (1, L)
        """
        device = enc_out.device
        B, L, H = enc_out.size()
        assert B == 1, "decode_greedy는 batch_size=1 기준으로 작성됨"

        dec_input = self.decoder_input0.unsqueeze(0).expand(B, -1)   # (1, H)
        h = torch.zeros(B, self.hidden_dim, device=device)
        c = torch.zeros(B, self.hidden_dim, device=device)

        mask = padding_mask.clone()  # (1, L)
        tour = []

        for _ in range(L):
            all_masked = mask.all(dim=1)    # (1,)
            if all_masked.any():
                mask[all_masked, 0] = False

            h, c = self.decoder_cell(dec_input, (h, c))

            attn_mask = mask.clone()
            context, _ = self.glimpse(h, enc_out, mask=attn_mask)
            _, scores = self.pointer(context, enc_out, mask=attn_mask)  # (1, L)
            log_scores = F.log_softmax(scores, dim=1)                   # (1, L)

            next_idx = torch.argmax(log_scores, dim=1)                  # (1,)
            idx = next_idx.item()
            tour.append(idx)

            # visited mask 업데이트
            sub_mask = mask
            sub_idx = next_idx.unsqueeze(1)                             # (1, 1)
            sub_mask.scatter_(1, sub_idx, True)
            mask = sub_mask

            # 다음 decoder 입력
            next_idx_expanded = next_idx.view(1, 1, 1).expand(1, 1, self.hidden_dim)
            dec_input = embedded.gather(1, next_idx_expanded).squeeze(1)  # (1, H)

        return tour


# ------------------------------
#  NeuralTSPSolver: pointer_network.pt 로드 + multi-start inference
# ------------------------------

class NeuralTSPSolver:
    """
    미리 학습된 Pointer Network (PyTorch, models/pointer_network.pt)를 사용하여
    TSP 경로를 추론하는 Solver.
    """
    def __init__(self, model_path: str = "models/pointer_network.pt", device: str = "cpu"):
        self.device = torch.device(device)

        if not os.path.exists(model_path):
            raise RuntimeError(f"Pointer Network 모델 파일을 찾을 수 없습니다: {model_path}")

        # 모델 구조는 학습 시 설정과 맞춰야 함 (여기서는 hidden_dim=128, n_layers=2, n_heads=4 가정)
        self.model = PointerNetwork(
            input_dim=2,
            hidden_dim=128,
            n_layers=2,
            n_heads=4,
        ).to(self.device)

        state = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()

    @staticmethod
    def _compute_tour_length(coords: torch.Tensor, tour: list) -> float:
        """
        coords: (N, 2) torch tensor on any device
        tour  : list of indices length N (ideally)
        """
        if len(tour) == 0:
            return float("inf")

        # 중복 제거/유효성 보정은 호출 측에서 어느 정도 해주지만, 여기서도 간단히 방어
        unique = []
        seen = set()
        for idx in tour:
            if idx not in seen:
                unique.append(idx)
                seen.add(idx)
        if len(unique) < coords.size(0):
            # 불완전 투어에는 큰 패널티
            return float("inf")

        tour_tensor = torch.tensor(unique, dtype=torch.long, device=coords.device)
        path_coords = coords[tour_tensor]                    # (N, 2)
        rolled = torch.roll(path_coords, shifts=-1, dims=0)  # (N, 2)
        dist = torch.norm(path_coords - rolled, dim=1).sum().item()
        return dist

    def _decode_single(self, coords: torch.Tensor) -> list:
        """
        coords: (N, 2) [0,1] 정규화된 좌표, device=self.device
        """
        self.model.eval()
        with torch.no_grad():
            n = coords.size(0)
            inputs = coords.unsqueeze(0)              # (1, N, 2)
            lengths = torch.tensor([n], device=self.device)

            enc_out, embedded, padding_mask = self.model.encode(inputs, lengths)
            tour = self.model.decode_greedy(enc_out, embedded, padding_mask)

        return tour

    def inference_multi_start(self, coords: torch.Tensor, num_starts: int = 8) -> tuple[list, float]:
        """
        coords: (N, 2) [0,1] torch tensor
        """
        self.model.eval()
        N = coords.size(0)
        best_tour = list(range(N))
        best_dist = self._compute_tour_length(coords, best_tour)

        with torch.no_grad():
            for _ in range(num_starts):
                # permute cities
                perm = torch.randperm(N, device=self.device)
                perm_coords = coords[perm]  # (N, 2)

                tour_perm = self._decode_single(perm_coords)  # indices in perm space
                if len(tour_perm) == 0:
                    continue

                tour_perm_tensor = torch.tensor(tour_perm, dtype=torch.long, device=self.device)
                # 원래 인덱스로 inverse
                tour_original = perm[tour_perm_tensor].detach().cpu().tolist()

                # 유효성 검사
                if len(set(tour_original)) != N:
                    continue

                dist = self._compute_tour_length(coords, tour_original)
                if dist < best_dist:
                    best_dist = dist
                    best_tour = tour_original

        return best_tour, best_dist

    def _preprocess_cities(self, cities_df) -> torch.Tensor:
        """
        cities_df: Streamlit에서 사용하는 DataFrame (컬럼: x, y, 범위 0~100)
        반환: (N, 2) torch tensor, [0,1] 정규화
        """
        coords = cities_df[['x', 'y']].values.astype(np.float32)  # (N, 2)
        normalized = coords / 100.0
        return torch.from_numpy(normalized).to(self.device)

    def solve(self, cities_df, num_starts: int = 16) -> list:
        """
        Streamlit용 공개 API.
        """
        coords = self._preprocess_cities(cities_df)  # (N, 2) [0,1]
        tour, _ = self.inference_multi_start(coords, num_starts=num_starts)
        return tour
