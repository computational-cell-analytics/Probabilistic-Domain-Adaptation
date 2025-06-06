import os
import time

import torch
import torch_em
from torchvision.utils import make_grid
from torch_em.trainer.logger_base import TorchEmLogger

from prob_utils.my_utils import dice_score
from prob_utils.my_models import l2_regularisation

class PUNetTrainer(torch_em.trainer.DefaultTrainer):

    def _sample(self, n_samples=16):
        samples = [self.model.sample() for _ in range(n_samples)]
        return samples

    def _train_epoch_impl(self, progress, forward_context, backprop):
        self.model.train()

        n_iter = 0
        t_per_iter = time.time()
        for x, y in self.train_loader:
            x, y = x.to(self.device), y.to(self.device)

            self.optimizer.zero_grad()

            with forward_context():
                self.model.forward(x, y, training=True)
                elbo = self.model.elbo(y)
                reg_loss = l2_regularisation(self.model.posterior) + l2_regularisation(self.model.prior) + l2_regularisation(self.model.fcomb.layers)
                loss = -elbo + 1e-5 * reg_loss

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
        dice_metric = 0.0

        with torch.no_grad():
            for x, y in self.val_loader:
                x, y = x.to(self.device), y.to(self.device)

                with forward_context():
                    self.model.forward(x, y, training=True)
                    elbo = self.model.elbo(y)
                    reg_loss = l2_regularisation(self.model.posterior) + l2_regularisation(self.model.prior) + l2_regularisation(self.model.fcomb.layers)
                    loss = -elbo + 1e-5 * reg_loss

                    n_samples = 8
                    samples_per_patch = []
                    for i in range(n_samples):
                        mysig = torch.nn.Sigmoid()
                        myval = self.model.sample(testing=False)
                        myval = mysig(myval)
                        samples_per_patch.append(myval)

                    mypred = torch.stack(samples_per_patch, dim=0).sum(dim=0)/len(samples_per_patch)
                    mypred = mypred.detach().cpu().numpy().squeeze()
                    mygt = y.detach().cpu().numpy().squeeze()
                    mymetric = dice_score(mypred, mygt)
                    _mymetric = 1. - mymetric

                dice_metric += mymetric
                loss_val += loss.item()
                metric_val += _mymetric

        metric_val /= len(self.val_loader)
        loss_val /= len(self.val_loader)
        dice_metric /= len(self.val_loader)
        print()
        print(f"The Average Dice Score for the Current Epoch is {dice_metric}")

        if self.logger is not None:
            samples = self._sample()
            self.logger.log_validation(self._iteration, metric_val, loss_val, x, y, samples)
        return metric_val

class PUNetLogger(TorchEmLogger):
    def __init__(self, trainer, save_root, **unused_kwargs):
        super().__init__(trainer, save_root)
        self.log_dir = f"./logs/{trainer.name}" if save_root is None else\
            os.path.join(save_root, "logs", trainer.name)
        os.makedirs(self.log_dir, exist_ok=True)

        self.tb = torch.utils.tensorboard.SummaryWriter(self.log_dir)
        self.log_image_interval = trainer.log_image_interval

    def add_image(self, x, y, samples, name, step):
        self.tb.add_image(tag=f"{name}/input", img_tensor=x[0], global_step=step)
        self.tb.add_image(tag=f"{name}/target", img_tensor=y[0], global_step=step)
        sample_grid = make_grid([sample[0] for sample in samples], nrow=4, padding=4)
        self.tb.add_image(tag=f"{name}/samples", img_tensor=sample_grid, global_step=step)

    def log_train(self, step, loss, lr, x, y, samples):
        self.tb.add_scalar(tag="train/loss", scalar_value=loss, global_step=step)
        self.tb.add_scalar(tag="train/learning_rate", scalar_value=lr, global_step=step)
        if step % self.log_image_interval == 0:
            self.add_image(x, y, samples, "train", step)

    def log_validation(self, step, metric, loss, x, y, samples):
        self.tb.add_scalar(tag="validation/loss", scalar_value=loss, global_step=step)
        self.tb.add_scalar(tag="validation/metric", scalar_value=metric, global_step=step)
        self.add_image(x, y, samples, "validation", step)