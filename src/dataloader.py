from torch import Tensor


class DataLoaderLite:
    def __init__(self,
                 X: Tensor,
                 y: Tensor,
                 batch_size: int = 32):
        assert len(X) == len(y), "X and y must have the same length"
        self.X = X
        self.y = y
        self.N = len(X)
        self.bs = batch_size
        self.current_pos = 0
        self.epoch = 0

    def next_batch(self):
        # returns (x, y) of shape [B, ...]
        # always cycles; never raises StopIteration
        start = self.current_pos
        end = start + self.bs

        if end > self.N:
            # we've finished all samples, increment epoch and reset
            self.epoch += 1
            self.current_pos = 0
            start, end = 0, self.bs
        else:
            self.current_pos = end
        
        xb = self.X[start:end]
        yb = self.y[start:end]
        return xb, yb
    

class DataLoaderLiteDecoderOnly:
    """DataLoader for decoder-only models where input and target are the same sequence (shifted by 1)"""
    
    def __init__(self, sequences: Tensor, batch_size: int = 32):
        """
        Args:
            sequences: Tensor of shape [N, seq_len] containing combined sequences
                      Format: [EN_TOKENS] <sep> [HI_TOKENS] <eos> [PAD...]
            batch_size: Number of sequences per batch
        """
        self.sequences = sequences
        self.N = len(sequences)
        self.bs = batch_size
        self.current_pos = 0
        self.epoch = 0

    def next_batch(self):
        """
        Returns:
            input_ids: Tensor of shape [B, seq_len-1] - sequences without last token
            target_ids: Tensor of shape [B, seq_len-1] - sequences without first token (shifted)
        """
        start = self.current_pos
        end = start + self.bs

        if end > self.N:
            # Finished epoch, reset
            self.epoch += 1
            self.current_pos = 0
            start, end = 0, self.bs
        else:
            self.current_pos = end
        
        batch_sequences = self.sequences[start:end]
        
        # Create input (all tokens except last) and target (all tokens except first)
        input_ids = batch_sequences[:, :-1]   # [B, seq_len-1]
        target_ids = batch_sequences[:, 1:]   # [B, seq_len-1] (shifted by 1)
        
        return input_ids, target_ids

