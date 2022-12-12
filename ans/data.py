import math
from typing import Iterator, Optional, Union

import torch


class BatchLoader:

    def __init__(
            self,
            x: torch.Tensor,
            y: Optional[torch.Tensor] = None,
            batch_size: Optional[int] = None,
            shuffle: bool = False
    ) -> None:
        """

        Args:
            x: The first dimension should correspond to batch index, i.e. x[i, ...] is the i-th sample
            y: In case of supervised problems (e.g. classification), e.g. y[i] should be the target for x[i]
            batch_size: How many samples in batch
            shuffle: If True, then the data should be randomly reordered on each __iter__
        """
        self.x = x
        self.y = y
        self.batch_size = batch_size or len(x)
        self.shuffle = shuffle

    def __iter__(self) -> Iterator[tuple[torch.Tensor, ...]]:
        """

        Returns:
            batch: If unsupervised (i.e. self.y is None), return single element tuple (x[batch_ids],). If supervised
                   (i.e. self.y is torch.Tensor), return the pair (x[batch_ids], y[batch_ids])
        """

        ########################################
        # TODO: implement

        # Recommended approach:
        # 1. Create tensor of indices into self.x. If self.shuffle, then the indices should be randomly reodered.
        # 2. Loop over the indices in groups (i.e. batches) of size self.batch_size
        #    2.1 `yield` rows of self.x (and self.y if it is not None) as indexed by the current batch of indices
        #    2.2 stop if there are no more batches



        # ENDTODO
        ########################################

    def __len__(self) -> int:
        return math.ceil(self.x.shape[0] / self.batch_size)

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}:\n" \
               f"    num_batches: {len(self)}\n" \
               f"    batch_shape: {(self.batch_size,) + tuple(self.x.shape[1:])}\n"


class DataLoader(torch.utils.data.DataLoader):
    """
    Hack to remove transferring of data to target device from training and inference logic
    """

    def __init__(self, *args, device: Union[str, torch.device] = 'cpu', **kwargs) -> None:
        self.device = device
        super().__init__(*args, **kwargs)

    def __iter__(self):
        for batch in super().__iter__():
            yield tuple(tensor.to(self.device) for tensor in batch)
