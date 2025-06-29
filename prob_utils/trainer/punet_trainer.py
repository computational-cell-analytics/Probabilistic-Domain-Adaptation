import time

import torch

from torch_em.trainer import DefaultTrainer


class PUNet_Trainer(DefaultTrainer):
    """Custom PUNet Trainer.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._kwargs = kwargs

    def _train_epoch_impl(self, progress, forward_context, backprop):
        self.model.train()

        n_iter = 0
        t_per_iter = time.time()

        for x, y in self.train_loader:
            x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            with forward_context():
                outputs, kld_loss = self.model(x, y)
                loss = self.loss(outputs, y, kld_loss, n_iter)

            backprop(loss)

            if self.logger is not None:
                lr = [pm["lr"] for pm in self.optimizer.param_groups][0]
                samples = self._sample() if self._iteration % self.log_image_interval == 0 else None
                self.logger.log_train(self._iteration, loss, lr, x, y, samples)

            self._iteration += 1
            n_iter += 1
            if self._iteration >= self.max_iteration:
                break
            progress.update(1)

        t_per_iter = (time.time() - t_per_iter) / n_iter
        return t_per_iter

    def _validate_impl(self, forward_context):
        self.model.eval()

        metric_val = 0.0
        loss_val = 0.0

        with torch.no_grad():
            for x, y in self.val_loader:
                x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)

                with forward_context():
                    outputs, kld_loss = self.model(x, y)
                    loss = self.loss(outputs, y, kld_loss)
                    metric = self.metric(outputs, y, kld_loss)

                loss_val += loss.item()
                metric_val += metric.item()

        metric_val /= len(self.val_loader)
        loss_val /= len(self.val_loader)

        if self.logger is not None:
            samples = self._sample()
            self.logger.log_validation(self._iteration, metric_val, loss_val, x, y, samples)

        return metric_val
