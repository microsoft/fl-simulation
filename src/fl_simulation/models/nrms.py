"""The News Recommendation with Multi-Head Attention model."""
from typing import Optional, Tuple

import torch
import torch.nn as nn


class AdditiveAttention(nn.Module):
    """Additive attention module."""

    def __init__(self, in_dim: int, hidden_dim: int) -> None:
        """Create a new Additive Attention module.

        Args:
            in_dim (int): input dimension.
            hidden_dim (int): hiddent dimension.
        """
        super().__init__()

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.proj = nn.Linear(self.in_dim, self.hidden_dim)
        self.act = nn.Tanh()
        self.proj_v = nn.Linear(self.hidden_dim, 1, bias=False)

    def forward(self, context: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the Additive Attention on the give context.

        Args:
            context (tensor): [B, seq_len, in_dim]
            mask (tensor, optional): [B, seq_len]. True at positions to be ignored. Defaults to None.

        Returns:
            outputs, weights: [B, in_dim], [B, seq_len]
        """
        weights = self.proj_v(self.act(self.proj(context))).squeeze(-1)  # [B, seq_len]
        # mask the elements which should not contribute to the final result
        if mask is not None:
            weights.masked_fill_(mask, float("-inf"))
        weights = torch.softmax(weights, dim=-1)  # [B, seq_len]
        # [B, 1, seq_len] * [B, seq_len, in_dim] -> [B, 1, in_dim]
        return torch.bmm(weights.unsqueeze(1), context).squeeze(1), weights  # [B, in_dim], [B, seq_len]


class NewsEncoder(nn.Module):
    """The NewsEncoder module encodes the news articles into tensors of fixed size enc_size based on their titles."""

    def __init__(
            self,
            emb_size: int,
            num_heads: int,
            enc_size: int,
            hidden_dim: int,
    ) -> None:
        """Create a new `NewsEncoder` instance.

        Args:
            emb_size (int): the size of the embeddings used for the news articles.
            num_heads (int): number of heads in the internal multi-head attention.
            enc_size (int): the size of the finally produced encoding.
            hidden_dim (int): the hidden dimension of the internal additive attnetion module.
        """
        super(NewsEncoder, self).__init__()

        self.self_att = nn.MultiheadAttention(emb_size, num_heads=num_heads, batch_first=True)  # , dropout=0.1)
        self.proj = nn.Linear(emb_size, enc_size)
        self.additive_attn = AdditiveAttention(enc_size, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the representation for the news articles in `x`.

        Args:
            x (torch.Tensor): [num_articles, max_title_len, emb_size].

        Returns:
            torch.Tensor: [num_articles, enc_size]. Encoded news articles.
        """
        # [num_articles, max_title_len, emb_size] -> [num_articles, max_title_len, emb_size]
        output, _ = self.self_att(x, x, x)
        # [num_articles, max_title_len, emb_size] -> [num_articles, max_title_len, enc_size]
        output = self.proj(output)
        # [num_articles, max_title_len, enc_size] -> [num_articles, enc_size]
        output, _ = self.additive_attn(output)
        # [num_articles, enc_size]
        return output


class UserEncoder(nn.Module):
    """User Encoder.
    
    The `UserEncoder` class. Produces a user encoding based on user's history. User's history is a set of articles 
    (with maximum of `max_hist_len`), each of which will be encoded using the `NewsEncoder`.
    """

    def __init__(
            self,
            enc_size: int,
            num_heads: int,
            additive_att_hidden_dim: int,
            news_encoder: NewsEncoder,
            dropout: float = 0.0,
            batch_first: bool = True,
    ) -> None:
        """Create a new `UserEncoder` instance.

        Args:
            enc_size (int): size of the produces encoding.
            num_heads (int): number of heads in the underlying multi-head attention.
            additive_att_hidden_dim (int): hidden dimension in the additive attention module.
            news_encoder (NewsEncoder): the news encoder used for encoding the articles in the user's history.
            dropout (float, optional): the dropout rate. Defaults to 0.0.
            batch_first (bool, optional): whether the first dimension (dim=0) should be treated as the batch size, instead of the second one (dim=1). Defaults to True.
        """
        super().__init__()

        self.news_encoder = news_encoder
        self.self_att = nn.MultiheadAttention(enc_size, num_heads, batch_first=batch_first, dropout=dropout)
        self.proj = nn.Linear(enc_size, enc_size)
        self.additive_attn = AdditiveAttention(enc_size, additive_att_hidden_dim)

    def forward(self, history: torch.Tensor, hist_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward step.

        Args:
            history (torch.Tensor): [batch_size, max_hist_len, max_title_len, emb_size].
            hist_mask (Optional[torch.Tensor], optional): [batch_size, max_hist_len]. Indicates padded values in the history batch. If None, then all entries
                assumed to be unpadded. Defaults to None. 

        Returns:
            torch.Tensor: [batch_size, enc_size]. User encodings based on history.
        """
        batch_size = history.shape[0]
        max_hist_len = history.shape[1]
        max_title_len = history.shape[2]
        emb_size = history.shape[3]

        # [batch_size, max_hist_len, max_title_len, emb_size] -> [batch_size * max_hist_len, max_hist_len, emb_size]
        history = history.reshape(-1, max_title_len, emb_size)
        # [batch_size * max_hist_len, max_title_len, emb_size] -> [batch_size * max_hist_len, enc_size]
        hist_embed = self.news_encoder(history)
        # [batch_size * max_hist_len, enc_size] -> [batch_size, max_hist_len, enc_size]
        hist_embed = hist_embed.reshape(batch_size, max_hist_len, -1)
        # [batch_size, max_hist_len, enc_size] -> [batch_size, max_hist_len, enc_size]
        hist_output, _ = self.self_att(hist_embed, hist_embed, hist_embed, key_padding_mask=hist_mask)
        # [batch_size, max_hist_len, enc_size] -> [batch_size, max_hist_len, enc_size]
        projected_hist_output = self.proj(hist_output)
        # [batch_size, max_hist_len, enc_size] -> [batch_size, enc_size]
        user_repr, _ = self.additive_attn(projected_hist_output, mask=hist_mask)

        # [batch_size, enc_size]
        return user_repr


class NRMS(nn.Module):
    """News Recommendation with Multi-Head Self-Attention.
    
    The final model used for predicting users' clicks/non-clicks on the candidate articles based on the their histories.
    The forward pass takes `history` and `cands` tensors as an input. The `history` tensor is the users' histories,
    which is passed to the `UserEncoder`. The `cands` tensor is the candidate articles, which is passed to the
    `NewsEncoder`. The final prediction is made based on the dot product similarity of the produced encodings.
    """

    def __init__(
            self,
            emb_size: int,
            num_heads: int,
            enc_size: int,
            hidden_dim: int,
            dropout: float = 0.0,
            batch_first: bool = True,
    ) -> None:
        """Create the News Recommendation with Multi-Head Self-Attention model.

        Args:
            emb_size (int): the size of the embeddings used for the news articles.
            num_heads (int): number of heads in the underlying multi-head attention.
            enc_size (int): the encoding size for the user encoding and candidates encodings.
            hidden_dim (int): the hidden dimension for the underlying multi-head attention module. 
            dropout (float, optional): the dropout rate. Defaults to 0.0.
            batch_first (bool, optional): whether the first dimension (dim=0) should be treated as the batch size, instead of the second one (dim=1). Defaults to True.
        """
        super(NRMS, self).__init__()
        self.news_encoder = NewsEncoder(emb_size, num_heads, enc_size, hidden_dim)
        self.user_encoder = UserEncoder(
                enc_size, num_heads, hidden_dim, self.news_encoder, batch_first=batch_first, dropout=dropout
        )

    def forward(
            self,
            history: torch.Tensor,
            cands: torch.Tensor,
            hist_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Predict best article recommendations based on candidates and user history.

        Args:
            history (torch.Tensor): [batch_size, max_hist_len, max_title_len, emb_size]. User histories.
            cands (torch.Tensor): [batch_size, max_title_len, emb_size]. Candidate news.
            hist_mask (Optional[torch.Tensor], optional): [batch_size, max_hist_len]. Masks the ignored values in the history. If the corresponding values is True in the mask, the element is ignored.

        Return:
            torch.Tensor: predictions.
        """
        # [batch_size, max_hist_len, max_title_len, emb_size] -> [batch_size, enc_size]
        user_repr = self.user_encoder(history, hist_mask=hist_mask)
        # [batch_size, max_title_len, emb_size] -> [batch_size, enc_size]
        cand_repr = self.news_encoder(cands)

        # [batch_size, enc_size] x [batch_size, enc_size] -> [batch_size, enc_size] -> [batch_size]
        logits = torch.mul(user_repr, cand_repr).sum(dim=1)

        # [batch_size]
        return torch.sigmoid(logits)


class NRMSWithEmbeddings(nn.Module):
    """News Recommendation with Multi-Head Self-Attention with the Embeddings module.
    
    The final model used for predicting users' clicks/non-clicks on the candidate articles based on the their histories.
    The forward pass takes `history` and `cands` tensors as an input. The `history` tensor is the users' histories,
    which is passed to the `UserEncoder`. The `cands` tensor is the candidate articles, which is passed to the
    `NewsEncoder`. The final prediction is made based on the dot product similarity of the produced encodings.
    """

    def __init__(
            self,
            emb_size: int,
            num_heads: int,
            enc_size: int,
            hidden_dim: int,
            emb: nn.Embedding,
            batch_first: bool = True,
    ) -> None:
        """Create the News Recommendation with Multi-Head Self-Attention model.

        Args:
            emb_size (int): the size of the embeddings used for the news articles.
            num_heads (int): number of heads in the underlying multi-head attention.
            enc_size (int): the encoding size for the user encoding and candidates encodings.
            hidden_dim (int): the hidden dimension for the underlying multi-head attention module.
            emb (nn.Embeddings): the embedding module to be used. 
            batch_first (bool, optional): whether the first dimension (dim=0) should be treated as the batch size, instead of the second one (dim=1). Defaults to True.
        """
        super(NRMSWithEmbeddings, self).__init__()
        self.emb = emb
        self.nrms = NRMS(
                emb_size=emb_size,
                num_heads=num_heads,
                enc_size=enc_size,
                hidden_dim=hidden_dim,
                batch_first=batch_first
        )

    def forward(
            self,
            history: torch.Tensor,
            cands: torch.Tensor,
            hist_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Predict best article recommendations based on candidates and user history.

        Args:
            history (torch.Tensor): [batch_size, max_hist_len, max_title_len]. User histories.
            cands (torch.Tensor): [batch_size, max_title_len]. Candidate news.
            hist_mask (Optional[torch.Tensor], optional): [batch_size, max_hist_len]. Masks the ignored values in the history. If the corresponding values is True in the mask, the element is ignored.

        Return:
            torch.Tensor: predictions.
        """
        history = self.emb(history)
        cands = self.emb(cands)

        return self.nrms(history, cands, hist_mask)