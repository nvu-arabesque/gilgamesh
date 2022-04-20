"""
@credit: 
- mchen@arabesque.ai
- https://github.com/jadore801120/attention-is-all-you-need-pytorch
- https://arxiv.org/pdf/1706.03762.pdf
"""
import torch
import torch.nn as nn

from gilgamesh.transformer.scaled_dot_product import ScaledDotProductAttention


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        n_head: int,
        d_model: int,
        d_k: int = None,
        d_v: int = None,
        dropout: float = 0.1,
        bias: bool = False,
        layer_norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.n_head = n_head
        self.d_model = d_model
        d_k_default = self.d_model // self.n_head
        self.d_k = d_k if d_k else d_k_default
        assert (
            self.d_k * n_head == d_model
        ), f"d_k * n_head should equal d_model, but got d_k: {d_k}, n_head: {n_head}, d_model: {d_model}"
        self.d_v = d_v if d_v else d_k_default
        self.wq = nn.Linear(
            in_features=d_model, out_features=n_head * self.d_k, bias=bias
        )
        self.wk = nn.Linear(
            in_features=d_model, out_features=n_head * self.d_k, bias=bias
        )
        self.wv = nn.Linear(
            in_features=d_model, out_features=n_head * self.d_v, bias=bias
        )
        self.w0 = nn.Linear(n_head * self.d_v, self.d_model, bias=bias)

        self.attention = ScaledDotProductAttention(temperature=self.d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask=None):
        """Forward step
        Args
        ---------
        q, k, v: torch.Tensor
            assumed to be of shape batch x length x d_model
        mask:

        Returns
        -----------
        """
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        # extract
        # q,k,v assumed to have shape [batch x seq length x d_model]
        batch_size = q.shape[0]
        len_q = q.shape[1]
        len_k = k.shape[1]
        len_v = v.shape[1]

        residual = q

        # Pass through the pre-attention projection
        # output to have shape [batch x seq length x (n*dv)]

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        # reshape and transpose for attention dot product: [batch x num heads x seq length x dv]
        q = q.view(batch_size, len_q, n_head, d_k).transpose(1, 2)
        k = k.view(batch_size, len_k, n_head, d_k).transpose(1, 2)
        v = v.view(batch_size, len_v, n_head, d_v).transpose(1, 2)
        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: [batch x seq length x n x dv]
        # Combine the last two dimensions to concatenate all the heads together: [b x lq x (n*dv)]
        q = q.transpose(1, 2).contiguous().view(batch_size, len_q, -1)
        q = self.dropout(self.w0(q))
        q += residual

        q = self.layer_norm(q)

        return q, attn
