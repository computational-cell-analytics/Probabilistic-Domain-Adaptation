import os
import time

import torch
import torch_em
from torchvision.utils import make_grid
from torch_em.transform.raw import _normalize_torch
from torch_em.trainer.logger_base import TorchEmLogger

from prob_utils.my_utils import dice_score
from prob_utils.my_models import l2_regularisation


class FixMatchTrainer(torch_em.trainer.DefaultTrainer):
    """This trainer is meant to be used for FixMatch-based PUNet's training,
    where we also weight the ELBO based on consensus masks"""

    def __init__(self, ckpt_model=None, source_distribution=None, **kwargs):
        super().__init__(**kwargs)
        self._kwargs = kwargs
        self.sigmoid = torch.nn.Sigmoid()
        self.n_samples = 16
        self.ckpt_model = ckpt_model
        if source_distribution is None:
            self.source_distribution = None
        else:
            self.source_distribution = torch.FloatTensor(source_distribution).to(self.device)

        if self.ckpt_model is not None:
            weights = torch.load(self.ckpt_model, map_location=self.device)["model_state"]
            self.model.load_state_dict(weights)

    def sample_from_weak_model(self, weak_inputs, upper_thres=0.9, lower_thres=0.1):
        self.model.forward(weak_inputs, None, training=False)
        samples = [self.sigmoid(self.model.sample()) for _ in range(self.n_samples)]
        consensus = [
            torch.where((my_sample >= upper_thres) + (my_sample <= lower_thres),
                        torch.tensor(1.).to(self.device),
                        torch.tensor(0.).to(self.device))
            for my_sample in samples
        ]
        samples = torch.stack(samples, dim=0).sum(dim=0)/self.n_samples
        consensus = torch.stack(consensus, dim=0).sum(dim=0)/self.n_samples
        consensus = torch.where(consensus==1, 1, 0)
        return samples, consensus

    def sample_from_model(self):
        samples = [self.sigmoid(self.model.sample()) for _ in range(self.n_samples)]
        samples = torch.stack(samples, dim=0).sum(dim=0)/self.n_samples
        return samples

    def _train_epoch_impl(self, progress, forward_context, backprop):
        self.model.train()

        n_iter = 0
        t_per_iter = time.time()

        for x, x1, x2, gt_ in self.train_loader:
            x, x1, x2 = x.to(self.device), x1.to(self.device), x2.to(self.device)

            weak_aug, strong_aug = x1, x2

            with torch.no_grad():
                y, z = self.sample_from_weak_model(weak_aug)
            y, z = y.detach(), z.detach()

            if self.source_distribution is None:
                distribution_ratio = None
            else:
                y_binary = torch.where(y >= 0.5, 1, 0)
                _, target_distribution = torch.unique(y_binary, return_counts=True)
                target_distribution = target_distribution / target_distribution.sum()
                distribution_ratio = self.source_distribution / target_distribution
                y = torch.where(y < 0.5, y*distribution_ratio[0], y*distribution_ratio[1]).clip(0, 1)

            self.optimizer.zero_grad()
            with forward_context():
                self.model.forward(strong_aug, y, training=True)
                elbo = self.model.elbo(y, z)
                reg_loss = l2_regularisation(self.model.posterior) +\
                    l2_regularisation(self.model.prior) +\
                    l2_regularisation(self.model.fcomb.layers)
                loss = -elbo + 1e-5 * reg_loss

            backprop(loss)

            if self.logger is not None:
                lr = [pm["lr"] for pm in self.optimizer.param_groups][0]
                samples = self.sample_from_model() if self._iteration % self.log_image_interval == 0 else None
                self.logger.log_train(self._iteration, loss, lr, x, x1, x2, y, z, gt_, samples, distribution_ratio)

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
        gt_metric_val = 0.0

        with torch.no_grad():
            for x, x1, x2, gt_ in self.val_loader:
                x, x1, x2 = x.to(self.device), x1.to(self.device), x2.to(self.device)

                weak_aug, strong_aug = x1, x2

                y, z = self.sample_from_weak_model(weak_aug)

                with forward_context():
                    self.model.forward(strong_aug, y, training=True)
                    elbo = self.model.elbo(y, z)
                    reg_loss = l2_regularisation(self.model.posterior) +\
                        l2_regularisation(self.model.prior) +\
                        l2_regularisation(self.model.fcomb.layers)
                    loss = -elbo + 1e-5 * reg_loss

                    samples = self.sample_from_model()
                    mypred, mygt = samples.detach().cpu().numpy().squeeze(), y.detach().cpu().numpy().squeeze()
                    mymetric = dice_score(mypred, mygt)
                    _mymetric = 1. - mymetric

                    true_gt = gt_.detach().cpu().numpy().squeeze()
                    true_metric = dice_score(mypred, true_gt)
                    _true_metric = 1. - true_metric

                dice_metric += mymetric
                loss_val += loss.item()
                metric_val += _mymetric
                gt_metric_val += _true_metric

        metric_val /= len(self.val_loader)
        loss_val /= len(self.val_loader)
        dice_metric /= len(self.val_loader)
        gt_metric_val /= len(self.val_loader)
        print(f"The Average Dice Score for the Current Epoch is {dice_metric}")

        if self.logger is not None:
            samples = self.sample_from_model()
            self.logger.log_validation(
                self._iteration, metric_val, loss_val, x, x1, x2, y, z, gt_, samples, gt_metric=gt_metric_val
            )
        return metric_val


class FixMatchLogger(TorchEmLogger):
    def __init__(self, trainer, save_root, **unused_kwargs):
        super().__init__(trainer, save_root)
        self.my_root = save_root
        self.log_dir = f"./logs/{trainer.name}" if self.my_root is None else\
            os.path.join(self.my_root, "logs", trainer.name)
        os.makedirs(self.log_dir, exist_ok=True)

        self.tb = torch.utils.tensorboard.SummaryWriter(self.log_dir)
        self.log_image_interval = trainer.log_image_interval

    def add_image(self, x, x1, x2, y, z, gt, samples, name, step):
        # NOTE: we only show the first tensor per batch for all images

        weak_aug = _normalize_torch(x1[0])
        strong_aug = _normalize_torch(x2[0])
        pseudo_labels = y[0]
        prediction = samples[0]

        # self.tb.add_image(tag=f"{name}/input", img_tensor=_normalize_torch(x[0]), global_step=step)
        # self.tb.add_image(tag=f"{name}/weak_aug", img_tensor=weak_aug, global_step=step)
        # self.tb.add_image(tag=f"{name}/strong_aug", img_tensor=strong_aug, global_step=step)
        # self.tb.add_image(tag=f"{name}/pseudo-labels", img_tensor=pseudo_labels, global_step=step)
        # self.tb.add_image(tag=f"{name}/consensus-mask", img_tensor=z[0], global_step=step)
        # self.tb.add_image(tag=f"{name}/ground_truth", img_tensor=gt[0], global_step=step)
        # self.tb.add_image(tag=f"{name}/prediction", img_tensor=prediction, global_step=step)

        # I AM ONLY LOGGING THE GRID VIEW NOW, I FIND IT MORE EASY TO NAVIGATE
        grid = make_grid([weak_aug, strong_aug, pseudo_labels, prediction], nrow=2, padding=8)
        self.tb.add_image(tag=f"{name}/weak-strong-labels-pred", img_tensor=grid, global_step=step)

    def log_train(self, step, loss, lr, x, x1, x2, y, z, gt, samples, distribution_ratio=None):
        self.tb.add_scalar(tag="train/loss", scalar_value=loss, global_step=step)
        self.tb.add_scalar(tag="train/learning_rate", scalar_value=lr, global_step=step)
        # I ADDED LOGGING FOR THE DISTRIBUTION RATIOS, IT SHOWS NICELY WHEN THE MODELS DIVERGE
        if distribution_ratio is not None:
            self.tb.add_scalar(tag="train/distr_ratio_bg", scalar_value=distribution_ratio[0], global_step=step)
            self.tb.add_scalar(tag="train/distr_ratio_fg", scalar_value=distribution_ratio[1], global_step=step)
        if step % self.log_image_interval == 0:
            self.add_image(x, x1, x2, y, z, gt, samples, "train", step)

    def log_validation(self, step, metric, loss, x, x1, x2, y, z, gt, samples, gt_metric=None):
        self.tb.add_scalar(tag="validation/loss", scalar_value=loss, global_step=step)
        self.tb.add_scalar(tag="validation/metric", scalar_value=metric, global_step=step)
        if gt_metric is not None:
            self.tb.add_scalar(tag="validation/gt_metric", scalar_value=gt_metric, global_step=step)
        self.add_image(x, x1, x2, y, z, gt, samples, "validation", step)