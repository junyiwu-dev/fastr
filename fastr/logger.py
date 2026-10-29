# SPDX-License-Identifier: MIT
# See LICENSE and NOTICE for copyright and source details.

"""Save training and evaluation values to TensorBoard."""

from datetime import datetime
from pathlib import Path
from uuid import uuid4


class Logger:
    def __init__(self, log_dir, name, active=True):
        run = f"{name}-{datetime.now():%Y%m%d-%H%M%S}-{uuid4().hex[:8]}"
        self.log_dir = Path(log_dir) / run
        self.writer = None
        if active:
            from torch.utils.tensorboard import SummaryWriter

            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(str(self.log_dir / "tb"))

    def log_scalars(self, prefix, metrics, step):
        if self.writer is not None:
            for key, value in metrics.items():
                self.writer.add_scalar(f"{prefix}/{key}" if prefix else key, value, step)

    def close(self):
        if self.writer is not None:
            self.writer.close()
