from abc import ABCMeta, abstractmethod
from collections import OrderedDict as odict

import torch
import torch.nn as nn
from torch import Tensor

from model.geometry import apply_frame, apply_frame_inverse, to_local


class FeedForward(nn.Module):
    def __init__(self, prev_dim, dims, act_fn="relu", dropout_p=None):
        super().__init__()
        self.dims = [prev_dim] + dims
        self.last_dim = dims[-1]
        self.layers = nn.ModuleList(
            [
                nn.Linear(self.dims[i], self.dims[i + 1])
                for i in range(len(self.dims) - 1)
            ]
        )
        activations = {"relu": nn.ReLU}
        self.activation = activations[act_fn]()
        self.use_dropout = dropout_p is not None
        if self.use_dropout:
            self.dropout = nn.Dropout(p=dropout_p)
        self.weight_init_()

    def weight_init_(self):
        for layer in self.layers:
            nn.init.zeros_(layer.bias)
            nn.init.kaiming_uniform_(layer.weight)

    def forward(self, x):
        assert x.shape[-1] == self.dims[0]
        for layer in self.layers:
            x = self.activation(layer(x))
            if self.use_dropout:
                x = self.dropout(x)
        return x

    def get_mask_hint(self):
        return None

    def get_input_annot(self):
        return "(b, .., m_in)"

    def get_output_annot(self):
        return "(b, .., m_out)"


class LayerNorm(nn.Module):
    def __init__(self, last_dim):
        super().__init__()
        self.last_dim = last_dim
        self._layernorm = nn.LayerNorm(self.last_dim)

    def forward(self, x):
        return self._layernorm(x)

    def get_mask_hint(self):
        return None

    def get_input_annot(self):
        return "(b, .., m_last)"

    def get_output_annot(self):
        return "(b, .., m_last)"


class MaskedMaxpool(nn.Module):
    def __init__(self, dim):
        """
        Args:
            dim ([type]): dimension to maxpool, expressed as a negative integer. It is assumed that every subsequent dimension is MDim(i.e. "model dimension")

        Raises:
            Exception: [description]
        """
        super().__init__()
        if dim >= 0:
            raise Exception("dim should be given as a negative integer")
        self.dim = dim

    def forward(self, x: Tensor, mask: Tensor):
        x = torch.where(
            mask, x, -torch.tensor(float("inf"), device=x.device, dtype=x.dtype)
        )
        x = torch.max(x, dim=self.dim).values
        return x

    def get_mask_hint(self):
        return "copy"

    def get_input_annot(self):
        # self.dim == -3, then (B, .., L_pool, M_1, M_2)
        s = "(b, .., l_pool"
        for i in range(1, -self.dim):
            s += f", m_{i}"
        s += ")"
        return s

    def get_output_annot(self):
        s = "(b, .."
        for i in range(1, -self.dim):
            s += f", m_{i}"
        s += ")"
        return s


class MultiModalRegressor(nn.Module):
    def __init__(self, last_dim, k):
        super().__init__()
        self.regressors = nn.ModuleList([nn.Linear(last_dim, 1) for _ in range(k)])
        self.weighters = nn.ModuleList([nn.Linear(last_dim, 1) for _ in range(k)])
        self.softmax = nn.Softmax(-1)
        self.weight_init_()

    def weight_init_(self):
        for regressor in self.regressors:
            nn.init.zeros_(regressor.bias)
            nn.init.xavier_uniform_(regressor.weight)
        for weighter in self.weighters:
            nn.init.zeros_(weighter.bias)
            nn.init.xavier_uniform_(weighter.weight)

    def forward(self, x):
        values = torch.stack([regressor(x) for regressor in self.regressors], dim=-1)
        weights = torch.stack([weighter(x) for weighter in self.weighters], dim=-1)
        weights = self.softmax(weights)
        return torch.sum(weights * values, dim=-1)


class MultiHeadRegressor(nn.Module):
    def __init__(
        self, last_dim: int, metrics: list[str], prefix=None, ks=None, zero_weight=[]
    ):
        super().__init__()
        self.last_dim = last_dim
        self.metrics = metrics
        self.prefix = prefix
        regressors = {}
        for metric in self.metrics:
            if ks is not None and metric in ks:
                k = ks[metric]
                regressors[metric] = MultiModalRegressor(self.last_dim, k)
            else:
                regressors[metric] = nn.Linear(self.last_dim, 1)
        self.regressors = nn.ModuleDict(regressors)
        self.zero_weight = zero_weight
        self.weight_init_()

    def weight_init_(self):
        for metric in self.metrics:
            if type(self.regressors[metric]) != nn.Linear:
                continue
            nn.init.zeros_(self.regressors[metric].bias)
            if metric in self.zero_weight:
                nn.init.zeros_(self.regressors[metric].weight)
            else:
                nn.init.xavier_uniform_(self.regressors[metric].weight)

    def get_key_name(self, metric):
        if self.prefix is None:
            return metric
        return f"{self.prefix}_{metric}"

    def forward(self, x):
        return {
            self.get_key_name(metric): self.regressors[metric](x).squeeze(-1)
            for metric in self.metrics
        }

    def get_mask_hint(self):
        return None

    def get_input_annot(self):
        return "(b, .., m_hidden)"

    def get_output_annot(self):
        return {metric: "(b, ..)" for metric in self.metrics}


class GeometricTransformerUnit(nn.Module, metaclass=ABCMeta):
    @property
    @abstractmethod
    def layer_cls(self):
        pass

    @property
    @abstractmethod
    def layer_abbrev(self):
        pass

    def __init__(self, hidden_dim=128, inter_dim=512, dropout_p=0.1, **kwargs):
        super().__init__()
        self.hidden_dim = hidden_dim
        setattr(
            self, self.layer_abbrev, self.layer_cls(hidden_dim=hidden_dim, **kwargs)
        )
        setattr(self, f"layernorm_after_{self.layer_abbrev}", nn.LayerNorm(hidden_dim))
        self.pointwise_ff = nn.Sequential(
            odict(
                [
                    ("linear1", nn.Linear(hidden_dim, inter_dim)),
                    ("activation", nn.ReLU()),
                    ("linear2", nn.Linear(inter_dim, hidden_dim)),
                ]
            )
        )
        self.layernorm_after_ff = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(p=dropout_p)
        self.weight_init_()

    def weight_init_(self):
        # zero bias
        nn.init.zeros_(self.pointwise_ff.linear1.bias)
        nn.init.zeros_(self.pointwise_ff.linear2.bias)

        # layers followed by relu: he initialization
        nn.init.kaiming_uniform_(self.pointwise_ff.linear1.weight)

        # layers right before residual connection
        nn.init.zeros_(self.pointwise_ff.linear2.weight)

    def forward(self, x, R, t, mask):
        assert x.shape[-1] == self.hidden_dim
        residual = x
        x = self.dropout(getattr(self, self.layer_abbrev)(x, R, t, mask))
        x = getattr(self, f"layernorm_after_{self.layer_abbrev}")(residual + x)
        residual = x
        x = self.dropout(self.pointwise_ff(x))
        x = self.layernorm_after_ff(residual + x)
        return x


class GeometricTransformer(nn.Module, metaclass=ABCMeta):
    @property
    @abstractmethod
    def unit_cls(self):
        pass

    def __init__(self, num_layers=1, **kwargs):
        super().__init__()
        self.units = nn.ModuleList([self.unit_cls(**kwargs) for _ in range(num_layers)])

    def forward(self, x, R, t, mask):
        for unit in self.units:
            x = unit(x, R, t, mask)
        return x


# RayAttention


class RayAttentionLayer(nn.Module):
    def __init__(
        self,
        hidden_dim=128,
        v_dim=12,
        num_heads=96,
        eps=1e-6,
        softplus=True,
        att_mode="linear",
    ):
        super().__init__()
        self.eps = eps
        self.v_dim = v_dim
        self.num_heads = num_heads
        self.to_v = nn.Linear(hidden_dim, self.v_dim * self.num_heads)
        self.to_a = nn.Linear(hidden_dim, 3 * self.num_heads)
        self.to_b = nn.Linear(hidden_dim, 3 * self.num_heads)
        self.final_proj = nn.Linear((self.v_dim + 5) * self.num_heads, hidden_dim)
        if softplus:
            init_alphabeta = -2 * torch.ones((1, 1, 1, num_heads))
        else:
            init_alphabeta = torch.zeros((1, 1, 1, num_heads))
        self.register_parameter("alpha", nn.Parameter(init_alphabeta))
        self.register_parameter("beta", nn.Parameter(init_alphabeta))
        if softplus:
            self.softplus = nn.Softplus()
        self.att_mode = att_mode
        self.att_softmax = nn.Softmax(-2)

        self.weight_init_()

    def weight_init_(self):
        # zero bias
        nn.init.zeros_(self.to_v.bias)
        nn.init.constant_(self.to_a.bias, self.eps)
        nn.init.constant_(self.to_b.bias, self.eps)
        nn.init.zeros_(self.final_proj.bias)

        # queries, keys, values
        nn.init.xavier_uniform_(self.to_v.weight)

        # weights that lead to geometry
        nn.init.zeros_(self.to_a.weight)
        nn.init.zeros_(self.to_b.weight)

        # layers right before residual additions
        nn.init.zeros_(self.final_proj.weight)

    def get_alpha(self):
        if hasattr(self, "softplus"):
            return self.softplus(self.alpha)
        return self.alpha

    def get_beta(self):
        if hasattr(self, "softplus"):
            return self.softplus(self.beta)
        return self.beta

    def get_unnormalized_att_weights(self, r_size, theta):
        if self.att_mode == "linear":
            return -self.get_alpha() * r_size - self.get_beta() * theta
        elif self.att_mode == "theta-weighted":
            return -self.get_beta() * theta * (1 + self.get_alpha() * r_size)
        raise Exception('No such "att_mode" for RA')

    def forward(self, x: Tensor, R: Tensor, t: Tensor, mask: Tensor):
        # two residue dims: original and surrounding
        # v : surrounding vector
        assert len(x.shape) == 3  # (B, L_res, M_hidden)
        N = x.shape[0]
        L = x.shape[1]
        assert R.shape == (N, L, 3, 3)
        assert t.shape == (N, L, 3)
        assert mask.shape == (N, L)
        a = self.to_a(x).reshape(N, L, 1, self.num_heads, 3)  # (B, L_res, 1, M_head, 3)
        b = self.to_b(x).reshape(N, L, 1, self.num_heads, 3)  # (B, L_res, 1, M_head, 3)
        local = to_local(R, t, t)  # (B, L_res, L_res, 3)
        r = local.unsqueeze(-2) - b  # (B, L_res, L_res, M_head, 3)

        r_size = torch.sqrt(torch.sum(r**2, dim=-1))  # (B, L_res, L_res, M_head)
        a_size = torch.sqrt(torch.sum(a**2, dim=-1))  # (B, L_res, L_res, M_head)
        r_dot_a = torch.sum(r * a, dim=-1)  # (B, L_res, L_res, M_head)
        theta = torch.acos(
            r_dot_a / ((r_size + self.eps) * (a_size + self.eps))
        )  # (B, L_res, L_res, M_head)

        # print(r_size.shape, self.alpha.shape)
        att_weights = self.get_unnormalized_att_weights(
            r_size, theta
        )  # (B, L_res, L_res, M_head)

        very_small_val = torch.tensor(-100.0, device=t.device, dtype=t.dtype)
        diag = torch.eye(L, device=t.device, dtype=bool)
        att_weights = torch.where(diag[None, :, :, None], very_small_val, att_weights)

        even_smaller_val = -torch.tensor(float("inf"), device=t.device, dtype=t.dtype)
        att_weights = torch.where(mask[:, None, :, None], att_weights, even_smaller_val)

        att_weights = self.att_softmax(att_weights)  # (B, L_res, L_res, M_head)

        v = self.to_v(x).reshape(
            N, 1, L, self.num_heads, self.v_dim
        )  # (B, 1, L_res, M_head, M_v)

        v_aggr = torch.sum(
            att_weights.unsqueeze(-1) * v, dim=2
        )  # (B, L_res, M_head, M_v)
        r_aggr = torch.sum(
            att_weights.unsqueeze(-1) * r, dim=2
        )  # (B, L_res, M_head, 3)
        r_aggr_size = torch.sqrt(torch.sum(r_aggr**2, dim=-1))  # (B, L_res, M_head)
        theta_aggr = torch.sum(att_weights * theta, dim=2)  # (B, L_res, M_head)

        concatenated = torch.cat(
            [
                v_aggr.reshape(N, L, -1),
                r_aggr.reshape(N, L, -1),
                r_aggr_size,
                theta_aggr,
            ],
            dim=-1,
        )
        out = self.final_proj(concatenated)  # (B, L_res, M_hidden)

        return out


class RayAttentionUnit(GeometricTransformerUnit):
    @property
    def layer_cls(self):
        return RayAttentionLayer

    @property
    def layer_abbrev(self):
        return "RA"


class RayAttention(GeometricTransformer):
    @property
    def unit_cls(self):
        return RayAttentionUnit


# QKRayAttention


class QKRayAttentionLayer(nn.Module):
    def __init__(
        self,
        hidden_dim=128,
        qk_dim=16,
        v_dim=16,
        num_heads=24,
        eps=1e-6,
        softplus=True,
        att_mode="linear",
    ):
        super().__init__()
        self.eps = eps
        self.qk_dim = qk_dim
        self.v_dim = v_dim
        self.num_heads = num_heads
        self.to_q = nn.Linear(hidden_dim, self.num_heads * self.qk_dim, bias=False)
        self.to_k = nn.Linear(hidden_dim, self.num_heads * self.qk_dim, bias=False)
        self.to_v = nn.Linear(hidden_dim, self.v_dim * self.num_heads)
        self.to_a = nn.Linear(hidden_dim, 3 * self.num_heads)
        self.to_b = nn.Linear(hidden_dim, 3 * self.num_heads)
        self.final_proj = nn.Linear((self.v_dim + 5) * self.num_heads, hidden_dim)
        if softplus:
            init_alphabeta = -2 * torch.ones((1, num_heads, 1, 1))
        else:
            init_alphabeta = torch.zeros((1, num_heads, 1, 1))
        self.register_parameter("alpha", nn.Parameter(init_alphabeta))
        self.register_parameter("beta", nn.Parameter(init_alphabeta))
        if softplus:
            self.softplus = nn.Softplus()
        self.att_mode = att_mode
        self.att_softmax = nn.Softmax(-1)

        self.weight_init_()

    def weight_init_(self):
        # zero bias
        nn.init.zeros_(self.to_v.bias)
        nn.init.constant_(self.to_a.bias, self.eps)
        nn.init.constant_(self.to_b.bias, self.eps)
        nn.init.zeros_(self.final_proj.bias)

        # queries, keys, values
        nn.init.xavier_uniform_(self.to_q.weight)
        nn.init.xavier_uniform_(self.to_k.weight)
        nn.init.xavier_uniform_(self.to_v.weight)

        # weights that lead to geometry
        nn.init.zeros_(self.to_a.weight)
        nn.init.zeros_(self.to_b.weight)

        # layers right before residual additions
        nn.init.zeros_(self.final_proj.weight)

    def get_alpha(self):
        if hasattr(self, "softplus"):
            return self.softplus(self.alpha)
        return self.alpha

    def get_beta(self):
        if hasattr(self, "softplus"):
            return self.softplus(self.beta)
        return self.beta

    def get_ray_att_weights(self, r_size, theta):
        if self.att_mode == "linear":
            return -self.get_alpha() * r_size - self.get_beta() * theta
        elif self.att_mode == "theta-weighted":
            return -self.get_beta() * theta * (1 + self.get_alpha() * r_size)
        raise Exception('No such "att_mode" for RA')

    def forward(self, x: Tensor, R: Tensor, t: Tensor, mask: Tensor):
        # two residue dims: original and surrounding
        # v : surrounding vector
        assert len(x.shape) == 3  # (B, L_res, M_hidden)
        N = x.shape[0]
        L = x.shape[1]
        assert R.shape == (N, L, 3, 3)
        assert t.shape == (N, L, 3)
        assert mask.shape == (N, L)

        a = (
            self.to_a(x).reshape(N, L, self.num_heads, 1, 3).transpose(1, 2)
        )  # (B, num_heads, L, 1, 3)
        b = (
            self.to_b(x).reshape(N, L, self.num_heads, 1, 3).transpose(1, 2)
        )  # (B, num_heads, L, 1, 3)
        local = to_local(R, t, t)  # (B, L, L, 3)
        r = local.unsqueeze(1) - b  # (B, num_heads, L, L, 3)

        r_size = torch.sqrt(torch.sum(r**2, dim=-1))  # (B, num_heads, L, L)
        a_size = torch.sqrt(torch.sum(a**2, dim=-1))  # (B, num_heads, L, 1)
        r_dot_a = torch.sum(r * a, dim=-1)  # (B, num_heads, L, L)
        theta = torch.acos(
            r_dot_a / ((r_size + self.eps) * (a_size + self.eps))
        )  # (B, num_heads, L, L)

        # print(r_size.shape, self.alpha.shape)
        ray_att_weights = self.get_ray_att_weights(
            r_size, theta
        )  # (B, num_heads, L, L )

        q = self.to_q(x).reshape(N, L, self.num_heads, self.qk_dim)
        k = self.to_k(x).reshape(N, L, self.num_heads, self.qk_dim)
        standard_att_weights = torch.matmul(
            q.permute(0, 2, 1, 3), k.permute(0, 2, 3, 1)
        ) * self.qk_dim ** (-0.5)  # (N, num_heads, L, L)

        att_weights = 2**-0.5 * (
            ray_att_weights + standard_att_weights
        )  # (N, num_heads, L, L)

        very_small_val = torch.tensor(-100.0, device=t.device, dtype=t.dtype)
        diag = torch.eye(L, device=t.device, dtype=bool)
        att_weights = torch.where(diag[None, None, :, :], very_small_val, att_weights)

        even_smaller_val = -torch.tensor(float("inf"), device=t.device, dtype=t.dtype)
        att_weights = torch.where(mask[:, None, None, :], att_weights, even_smaller_val)

        att_weights = self.att_softmax(att_weights)  # (B, num_heads, L, L)

        # Reshaping for the convenience of remaining calculations
        att_weights = att_weights.permute(0, 2, 3, 1)  # (B, L, L, num_heads)
        r = r.permute(0, 2, 3, 1, 4)  # (B, L, L, num_heads ,3)
        theta = theta.permute(0, 2, 3, 1)  # (B, L, L, num_heads)

        v = self.to_v(x).reshape(
            N, 1, L, self.num_heads, self.v_dim
        )  # (B, 1, L, num_heads, v_dim)

        v_aggr = torch.sum(
            att_weights.unsqueeze(-1) * v, dim=2
        )  # (B, L, num_heads, v_dim)
        r_aggr = torch.sum(att_weights.unsqueeze(-1) * r, dim=2)  # (B, L, num_heads, 3)
        r_aggr_size = torch.sqrt(torch.sum(r_aggr**2, dim=-1))  # (B, L, num_heads)
        theta_aggr = torch.sum(att_weights * theta, dim=2)  # (B, L, num_heads)

        concatenated = torch.cat(
            [
                v_aggr.reshape(N, L, -1),
                r_aggr.reshape(N, L, -1),
                r_aggr_size,
                theta_aggr,
            ],
            dim=-1,
        )
        out = self.final_proj(concatenated)  # (B, L, num_heads)

        return out


class QKRayAttentionUnit(GeometricTransformerUnit):
    @property
    def layer_cls(self):
        return QKRayAttentionLayer

    @property
    def layer_abbrev(self):
        return "RA"


class QKRayAttention(GeometricTransformer):
    @property
    def unit_cls(self):
        return QKRayAttentionUnit


# NewRayAttention


class NewRayAttentionLayer(nn.Module):
    def __init__(
        self,
        hidden_dim=128,
        num_heads=12,
        qk_dim=16,
        v_dim=16,
        num_qk_vec=4,
        num_v_vec=8,
        eps=1e-6,
        softplus=True,
    ):
        super().__init__()
        self.eps = eps
        self.num_heads = num_heads
        self.v_dim = v_dim
        self.qk_dim = qk_dim
        self.num_qk_vec = num_qk_vec
        self.num_v_vec = num_v_vec

        self.to_q = nn.Linear(
            hidden_dim, self.num_heads * self.qk_dim, bias=False
        )  # (128, 192)
        self.to_k = nn.Linear(
            hidden_dim, self.num_heads * self.qk_dim, bias=False
        )  # (128, 192)
        self.to_v = nn.Linear(
            hidden_dim, self.num_heads * self.v_dim, bias=False
        )  # (128, 192)

        self.to_a = nn.Linear(hidden_dim, self.num_heads * self.num_qk_vec * 3)
        self.to_b = nn.Linear(hidden_dim, self.num_heads * 3)

        # self.to_q_vec = nn.Linear(hidden_dim, self.num_heads* self.num_qk_vec * 3, bias=False) #(128, 144)
        self.to_k_vec = nn.Linear(
            hidden_dim, self.num_heads * self.num_qk_vec * 3, bias=False
        )  # (128, 144)
        self.to_v_vec = nn.Linear(
            hidden_dim, self.num_heads * self.num_v_vec * 3, bias=False
        )  # (128, 128)

        if softplus:
            init_alphabeta = -2 * torch.ones((1, num_heads, 1, 1))
        else:
            init_alphabeta = torch.zeros((1, num_heads, 1, 1))
        self.register_parameter("alpha", nn.Parameter(init_alphabeta))
        self.register_parameter("beta", nn.Parameter(init_alphabeta))

        # self.register_parameter('gamma', nn.Parameter(torch.zeros(self.num_heads)))
        self.softplus = nn.Softplus()
        self.att_softmax = nn.Softmax(-1)
        self.last_proj = nn.Linear(
            self.num_heads * (self.v_dim + self.num_v_vec * 4), hidden_dim
        )

        self.weight_init_()

    def weight_init_(self):
        # zero bias
        nn.init.zeros_(self.to_a.bias)
        nn.init.zeros_(self.to_b.bias)
        nn.init.zeros_(self.last_proj.bias)

        # queries, keys, values
        nn.init.xavier_uniform_(self.to_q.weight)
        nn.init.xavier_uniform_(self.to_k.weight)
        nn.init.xavier_uniform_(self.to_v.weight)

        # geometric
        nn.init.uniform_(self.to_k_vec.weight, a=-self.eps, b=self.eps)
        nn.init.uniform_(self.to_v_vec.weight, a=-self.eps, b=self.eps)
        nn.init.uniform_(self.to_a.weight, a=-self.eps, b=self.eps)
        nn.init.uniform_(self.to_b.weight, a=-self.eps, b=self.eps)

        # layers right before resudual additions
        nn.init.zeros_(self.last_proj.weight)

    def get_alpha(self):
        if hasattr(self, "softplus"):
            return self.softplus(self.alpha)
        return self.alpha

    def get_beta(self):
        if hasattr(self, "softplus"):
            return self.softplus(self.beta)
        return self.beta

    def get_ray_att_weights(self, r_size_sum, theta_sum):
        return -(
            self.get_alpha() * r_size_sum + self.get_beta() * theta_sum
        ) * self.num_qk_vec ** (-0.5)

    def forward(self, x: Tensor, R: Tensor, t: Tensor, mask: Tensor):
        # two residue dims: original and surrounding
        # v : surrounding vector
        assert len(x.shape) == 3  # (B, L_res, M_hidden)
        N = x.shape[0]
        L = x.shape[1]
        assert R.shape == (N, L, 3, 3)
        assert t.shape == (N, L, 3)
        assert mask.shape == (N, L)

        q = self.to_q(x).reshape(N, L, self.num_heads, self.qk_dim)
        k = self.to_k(x).reshape(N, L, self.num_heads, self.qk_dim)
        standard_att_weights = torch.matmul(
            q.permute(0, 2, 1, 3), k.permute(0, 2, 3, 1)
        ) * self.qk_dim ** (-0.5)  # (N, num_heads, L, L)

        b = self.to_b(x).reshape(N, L, self.num_heads, 1, 3)
        R_reshaped = R.reshape(N, L, 1, 1, 3, 3).broadcast_to(b.shape + (3,))
        t_reshaped = t.reshape(N, L, 1, 1, 3).broadcast_to(b.shape)
        b_global = apply_frame(R_reshaped, t_reshaped, b).unsqueeze(
            2
        )  # "query perspective"

        k_vec = self.to_k_vec(x).reshape(N, L, self.num_heads, self.num_qk_vec, 3)
        R_reshaped = R.reshape(N, L, 1, 1, 3, 3).broadcast_to(k_vec.shape + (3,))
        t_reshaped = t.reshape(N, L, 1, 1, 3).broadcast_to(k_vec.shape)
        k_vec_global = apply_frame(R_reshaped, t_reshaped, k_vec).unsqueeze(
            1
        )  # "key perspective"

        r_global = k_vec_global - b_global
        R_inverse = torch.inverse(R)
        R_inverse_reshaped = R_inverse.reshape(N, L, 1, 1, 1, 3, 3).broadcast_to(
            r_global.shape + (3,)
        )
        # R_reshaped = R.reshape(N, L, 1, 1, 1, 3, 3).broadcast_to(r_global.shape + (3,))
        # t_reshaped = t.reshape(N, L, 1, 1, 1, 3).broadcast_to(r_global.shape)
        r = torch.matmul(R_inverse_reshaped, r_global.unsqueeze(-1)).squeeze(-1)
        # r = apply_frame_inverse(R_reshaped, t_reshaped, r_global)
        assert r.shape == (N, L, L, self.num_heads, self.num_qk_vec, 3)

        a = (
            self.to_a(x)
            .reshape(N, L, 1, self.num_heads, self.num_qk_vec, 3)
            .broadcast_to(r.shape)
        )
        r_size = torch.sqrt(torch.sum(r**2, dim=-1))
        a_size = torch.sqrt(torch.sum(a**2, dim=-1))
        r_dot_a = torch.sum(r * a, dim=-1)
        theta = torch.acos(r_dot_a / ((r_size + self.eps) * (a_size + self.eps)))
        r_size_sum = torch.sum(r_size, dim=-1).permute(
            0, 3, 1, 2
        )  # (N, num_heads, L, L)
        theta_sum = torch.sum(theta, dim=-1).permute(0, 3, 1, 2)  # (N, num_heads, L, L)

        ray_att_weights = self.get_ray_att_weights(r_size_sum, theta_sum)
        att_weights = 2**-0.5 * (ray_att_weights + standard_att_weights)

        very_small_val = torch.tensor(-100.0, device=t.device, dtype=t.dtype)
        diag = torch.eye(L, device=t.device, dtype=bool)
        att_weights = torch.where(diag[None, None, :, :], very_small_val, att_weights)

        even_smaller_val = -torch.tensor(float("inf"), device=t.device, dtype=t.dtype)
        att_weights = torch.where(mask[:, None, None, :], att_weights, even_smaller_val)

        att_weights = self.att_softmax(att_weights)  # (N, num_heads, L, L)

        v = self.to_v(x)
        v = v.reshape(N, L, self.num_heads, self.v_dim).permute(
            0, 2, 1, 3
        )  # (N, num_heads, L, v_dim)
        o = torch.matmul(att_weights, v)  # (N, num_heads, L, v_dim)
        o = o.permute(0, 2, 1, 3).reshape(N, L, -1)  # (N, L, num_heads * v_dim)

        v_vec = self.to_v_vec(x)
        v_vec = v_vec.reshape(N, L, self.num_heads, self.num_v_vec, 3).permute(
            0, 2, 1, 3, 4
        )  # (N, num_heads, L, num_v_vec, 3)
        R_reshaped = R[:, None, :, None, :, :].broadcast_to(v_vec.shape + (3,))
        t_reshaped = t[:, None, :, None, :].broadcast_to(v_vec.shape)
        v_vec = apply_frame(
            R_reshaped, t_reshaped, v_vec
        )  # (N, num_heads, L, num_v_vec, 3)
        o_vec = torch.matmul(
            att_weights[None, :, :, :, :], v_vec.permute(4, 0, 1, 2, 3)
        ).permute(1, 3, 2, 4, 0)  # (N, L, num_heads, num_v_vec, 3)
        o_vec = o_vec.reshape(N, L, -1, 3)  # (N, L, num_heads * num_v_vec, 3)

        R_reshaped = R[:, :, None, :, :].broadcast_to(o_vec.shape + (3,))
        t_reshaped = t[:, :, None, :].broadcast_to(o_vec.shape)
        o_vec = apply_frame_inverse(
            R_reshaped, t_reshaped, o_vec
        )  # (N, L, num_heads * num_v_vec, 3)
        o_vec_norm = o_vec.norm(p=2, dim=-1)

        x = torch.cat([o, o_vec.reshape(N, L, -1), o_vec_norm], dim=-1)

        return self.last_proj(x)


class NewRayAttentionUnit(GeometricTransformerUnit):
    @property
    def layer_cls(self):
        return NewRayAttentionLayer

    @property
    def layer_abbrev(self):
        return "RA"


class NewRayAttention(GeometricTransformer):
    @property
    def unit_cls(self):
        return NewRayAttentionUnit


# IPA
class IPALayer(nn.Module):
    def __init__(
        self,
        hidden_dim=128,
        num_heads=12,
        qk_dim=16,
        v_dim=16,
        num_qk_vec=4,
        num_v_vec=8,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.v_dim = v_dim
        self.qk_dim = qk_dim
        self.num_qk_vec = num_qk_vec
        self.num_v_vec = num_v_vec

        self.to_q = nn.Linear(
            hidden_dim, self.num_heads * self.qk_dim, bias=False
        )  # (128, 192)
        self.to_k = nn.Linear(
            hidden_dim, self.num_heads * self.qk_dim, bias=False
        )  # (128, 192)
        self.to_v = nn.Linear(
            hidden_dim, self.num_heads * self.v_dim, bias=False
        )  # (128, 192)
        self.to_q_vec = nn.Linear(
            hidden_dim, self.num_heads * self.num_qk_vec * 3, bias=False
        )  # (128, 144)
        self.to_k_vec = nn.Linear(
            hidden_dim, self.num_heads * self.num_qk_vec * 3, bias=False
        )  # (128, 144)
        self.to_v_vec = nn.Linear(
            hidden_dim, self.num_heads * self.num_v_vec * 3, bias=False
        )  # (128, 128)
        self.register_parameter("gamma", nn.Parameter(torch.zeros(self.num_heads)))
        self.softplus = nn.Softplus()
        self.att_softmax = nn.Softmax(-1)
        self.last_proj = nn.Linear(
            self.num_heads * (self.v_dim + self.num_v_vec * 4), hidden_dim
        )

        self.weight_init_()

    def weight_init_(self):
        # zero bias
        nn.init.zeros_(self.last_proj.bias)

        # queries, keys, values
        nn.init.xavier_uniform_(self.to_q.weight)
        nn.init.xavier_uniform_(self.to_k.weight)
        nn.init.xavier_uniform_(self.to_v.weight)
        nn.init.xavier_uniform_(self.to_q_vec.weight)
        nn.init.xavier_uniform_(self.to_k_vec.weight)
        nn.init.xavier_uniform_(self.to_v_vec.weight)

        # layers right before resudual additions
        nn.init.zeros_(self.last_proj.weight)

    def forward(self, x: Tensor, R: Tensor, t: Tensor, mask: Tensor):
        # two residue dims: original and surrounding
        # v : surrounding vector
        assert len(x.shape) == 3  # (B, L_res, M_hidden)
        N = x.shape[0]
        L = x.shape[1]
        assert R.shape == (N, L, 3, 3)
        assert t.shape == (N, L, 3)
        assert mask.shape == (N, L)

        q = self.to_q(x).reshape(N, L, self.num_heads, self.qk_dim)
        k = self.to_k(x).reshape(N, L, self.num_heads, self.qk_dim)
        standard_att_weights = torch.matmul(
            q.permute(0, 2, 1, 3), k.permute(0, 2, 3, 1)
        ) * self.qk_dim ** (-0.5)  # (N, num_heads, L, L)

        q_vec = self.to_q_vec(x).reshape(N, L, self.num_heads, self.num_qk_vec, 3)
        k_vec = self.to_k_vec(x).reshape(N, L, self.num_heads, self.num_qk_vec, 3)
        shape = (N, L, L, self.num_heads, self.num_qk_vec, 3)
        R_reshaped = R[:, :, None, None, :, :].broadcast_to(q_vec.shape + (3,))
        t_reshaped = t[:, :, None, None, :].broadcast_to(q_vec.shape)
        vec_att_weight_term1 = (
            apply_frame(R_reshaped, t_reshaped, q_vec).unsqueeze(2).broadcast_to(shape)
        )
        vec_att_weight_term2 = (
            apply_frame(R_reshaped, t_reshaped, k_vec).unsqueeze(1).broadcast_to(shape)
        )
        vec_att_weights = ((vec_att_weight_term1 - vec_att_weight_term2) ** 2).sum(
            -1
        ).sum(-1).permute(0, 3, 1, 2) * self.num_qk_vec ** (
            -0.5
        )  # (N, self.num_heads, L, L)

        att_weights = 2**-0.5 * (
            standard_att_weights
            - self.softplus(self.gamma)[None, :, None, None] * vec_att_weights
        )  # (N, self.num_heads, L, L)

        very_small_val = torch.tensor(-100.0, device=t.device, dtype=t.dtype)
        diag = torch.eye(L, device=t.device, dtype=bool)
        att_weights = torch.where(diag[None, None, :, :], very_small_val, att_weights)

        even_smaller_val = -torch.tensor(float("inf"), device=t.device, dtype=t.dtype)
        att_weights = torch.where(mask[:, None, None, :], att_weights, even_smaller_val)

        att_weights = self.att_softmax(att_weights)  # (N, num_heads, L, L)

        v = self.to_v(x)
        v = v.reshape(N, L, self.num_heads, self.v_dim).permute(
            0, 2, 1, 3
        )  # (N, num_heads, L, v_dim)
        o = torch.matmul(att_weights, v)  # (N, num_heads, L, v_dim)
        o = o.permute(0, 2, 1, 3).reshape(N, L, -1)  # (N, L, num_heads * v_dim)

        v_vec = self.to_v_vec(x)
        v_vec = v_vec.reshape(N, L, self.num_heads, self.num_v_vec, 3).permute(
            0, 2, 1, 3, 4
        )  # (N, num_heads, L, num_v_vec, 3)
        R_reshaped = R[:, None, :, None, :, :].broadcast_to(v_vec.shape + (3,))
        t_reshaped = t[:, None, :, None, :].broadcast_to(v_vec.shape)
        v_vec = apply_frame(
            R_reshaped, t_reshaped, v_vec
        )  # (N, num_heads, L, num_v_vec, 3)
        o_vec = torch.matmul(
            att_weights[None, :, :, :, :], v_vec.permute(4, 0, 1, 2, 3)
        ).permute(1, 3, 2, 4, 0)  # (N, L, num_heads, num_v_vec, 3)
        o_vec = o_vec.reshape(N, L, -1, 3)  # (N, L, num_heads * num_v_vec, 3)

        R_reshaped = R[:, :, None, :, :].broadcast_to(o_vec.shape + (3,))
        t_reshaped = t[:, :, None, :].broadcast_to(o_vec.shape)
        o_vec = apply_frame_inverse(
            R_reshaped, t_reshaped, o_vec
        )  # (N, L, num_heads * num_v_vec, 3)
        o_vec_norm = o_vec.norm(p=2, dim=-1)

        x = torch.cat([o, o_vec.reshape(N, L, -1), o_vec_norm], dim=-1)

        return self.last_proj(x)


class IPAUnit(GeometricTransformerUnit):
    @property
    def layer_cls(self):
        return IPALayer

    @property
    def layer_abbrev(self):
        return "ipa"


class IPA(GeometricTransformer):
    @property
    def unit_cls(self):
        return IPAUnit


# BERT


class BERTLayer(nn.Module):
    def __init__(self, hidden_dim=128, num_heads=12, qk_dim=16, v_dim=16):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.qk_dim = qk_dim
        self.v_dim = v_dim
        self.dim_query = qk_dim

        self.to_query = nn.Linear(
            self.hidden_dim, self.num_heads * self.dim_query, bias=False
        )
        self.to_key = nn.Linear(
            self.hidden_dim, self.num_heads * self.qk_dim, bias=False
        )
        self.to_val = nn.Linear(
            self.hidden_dim, self.num_heads * self.v_dim, bias=False
        )
        self.softmax = nn.Softmax(dim=-1)
        self.proj = nn.Linear(self.num_heads * self.v_dim, self.hidden_dim, bias=False)

        self.weight_init_()

    def weight_init_(self):
        nn.init.xavier_normal_(self.to_query.weight)
        nn.init.xavier_normal_(self.to_key.weight)
        nn.init.xavier_normal_(self.to_val.weight)
        nn.init.xavier_normal_(self.proj.weight)

    def forward(self, x: Tensor, R: Tensor, t: Tensor, mask: Tensor):
        # (batch, seq, hidden)
        query = (
            self.to_query(x)
            .reshape(x.shape[0], x.shape[1], self.num_heads, self.dim_query)
            .transpose(1, 2)
        )  # (batch, head, next_seq, query)
        key = (
            self.to_key(x)
            .reshape(x.shape[0], x.shape[1], self.num_heads, self.qk_dim)
            .transpose(1, 2)
        )  # (batch, head, prev_seq, key)
        att_weights = torch.matmul(query, key.transpose(2, 3)) * (self.qk_dim) ** (
            -0.5
        )  # (batch, head, next_seq, prev_seq)

        small_val = torch.tensor(-100.0, device=t.device, dtype=t.dtype)
        att_weights = torch.where(mask[:, None, None, :], att_weights, small_val)

        att_weights = self.softmax(att_weights)  # (batch, head, next_seq, prev_seq)

        val = (
            self.to_val(x)
            .reshape(x.shape[0], x.shape[1], self.num_heads, self.v_dim)
            .transpose(1, 2)
        )  # (batch, head, prev_seq, val)
        x = torch.matmul(att_weights, val).transpose(1, 2)  # (batch, seq, head, val)

        x = x.reshape(
            x.shape[0], x.shape[1], x.shape[2] * x.shape[3]
        )  # (batch, seq, head * val)
        x = self.proj(x)  # (batch, seq, hidden)

        return x


class BERTUnit(GeometricTransformerUnit):
    @property
    def layer_cls(self):
        return BERTLayer

    @property
    def layer_abbrev(self):
        return "BERT"


class BERT(GeometricTransformer):
    @property
    def unit_cls(self):
        return BERTUnit


def test_SE3_invariance():
    from utils.random_rotations import get_random_rotation

    from model.geometry import compose_frames

    config = {
        "num_layers": 1,
        "hidden_dim": 128,
        "inter_dim": 512,
        "dropout_p": 0.1,
        "num_heads": 12,
        "qk_dim": 16,
        "v_dim": 16,
        "num_qk_vec": 4,
        "num_v_vec": 8,
    }
    model = NewRayAttention(**config)
    # model = IPA(**config)
    nn.init.xavier_uniform_(model.units[0].RA.last_proj.weight)
    # nn.init.xavier_uniform_(model.units[0].ipa.last_proj.weight)
    model.eval()
    x = torch.rand((10, 20, 128))
    R1 = torch.from_numpy(get_random_rotation(200).reshape((10, 20, 3, 3)))
    t1 = torch.rand((10, 20, 3))
    mask = torch.ones((10, 20), dtype=bool)
    out1 = model(x, R1, t1, mask)

    R0 = torch.from_numpy(get_random_rotation(1)[0])[None, None, :, :].broadcast_to(
        (10, 20, 3, 3)
    )
    t0 = torch.rand((1, 1, 3)).broadcast_to((10, 20, 3))
    R2, t2 = compose_frames(R0, t0, R1, t1)
    # R2, t2 = compose_frames(R1, t1, R0, t0)
    out2 = model(x, R2, t2, mask)
    print(out1 - out2)


if __name__ == "__main__":
    test_SE3_invariance()
