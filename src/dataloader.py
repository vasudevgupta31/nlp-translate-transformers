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
