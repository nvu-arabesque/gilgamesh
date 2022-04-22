"""
@credit:
- https://nlp.seas.harvard.edu/2018/04/03/attention.html#training-loop
"""
import torch


class Batch:
    "Object for holding a batch of data with mask during training."

    def __init__(self, src: torch.Tensor, tgt: torch.Tensor = None):
        """
        Args
        ------
        src: Tensor
            src.shape assumes to be batch x seq length x embed_dim
        """
        src_batch_size, src_seq_length, src_embed_dim = src.shape
        tgt_batch_size, tgt_seq_length, tgt_embed_dim = tgt.shape
        assert src_batch_size == tgt_batch_size
        self.batch_size = src_batch_size
        self.src = src
        self.tgt = None
        self.inp = None
        self.src_mask = None
        self.tgt_mask = None
        if tgt is not None:
            self.inp = tgt[:, :-1]
            self.tgt = tgt[:, 1:]
        self.tgt_mask = (
            None
            if self.tgt is None
            else self.get_subsequent_mask(self.tgt.shape[1], self.batch_size)
        )
        self.src_mask = self.get_subsequent_mask(self.src.shape[1], self.batch_size)

    @staticmethod
    def get_subsequent_mask(size, batch_size):
        """ Returns lower triangle, the upper fille with -inf for masking attention strictly up to time step t. """
        return torch.stack(
            [
                torch.triu(torch.full((size, size), float("-inf")), diagonal=1)
                for _ in range(batch_size)
            ]
        )
