import os
import time

import torch
import torch_em
from torch_em.transform.raw import _normalize_torch
from torch_em.trainer.logger_base import TorchEmLogger

from prob_utils.my_utils import dice_score
from prob_utils.my_models import l2_regularisation

class AdaMatchTrainer(torch_em.trainer.DefaultTrainer):
    """This trainer is meant to be used for AdaMatch-based PUNet's joint-training,
    where we also weight the ELBO based on consensus masks"""

    def __init__(self, source_train_loader, target_train_loader, do_consensus_masking=False, **kwargs):
        self.source_train_loader = source_train_loader
        self.target_train_loader = target_train_loader
        
        train_loader = source_train_loader if len(source_train_loader) < len(target_train_loader) else\
            target_train_loader

        super().__init__(train_loader=train_loader, **kwargs)
        self._kwargs = kwargs
        self.sigmoid = torch.nn.Sigmoid()
        self.n_samples = 16
        self.target_val_loader = None
        self.consensus_masking_is_activated = False
        self.do_consensus_masking = do_consensus_masking

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

        if self.do_consensus_masking:
            consensus = torch.where(consensus == 1, 1, 0)
        
        return samples, consensus

    def sample_from_model(self):
        samples = [self.sigmoid(self.model.sample()) for _ in range(self.n_samples)]
        samples = torch.stack(samples, dim=0).sum(dim=0)/self.n_samples
        return samples

    def _train_epoch_impl(self, progress, forward_context, backprop):
        self.model.train()

        n_iter = 0
        t_per_iter = time.time()

        for (xs, ys), (xt, xt1, xt2, yt) in zip(self.source_train_loader, self.target_train_loader):
            xs, ys = xs.to(self.device), ys.to(self.device)
            xt, xt1, xt2, yt = xt.to(self.device), xt1.to(self.device), xt2.to(self.device), yt.to(self.device)

            # source model supervised-training
            self.optimizer.zero_grad()
            with forward_context():
                self.model.forward(xs, ys, training=True)
                supervised_elbo = self.model.elbo(ys)
                supervised_reg_loss = l2_regularisation(self.model.posterior) +\
                    l2_regularisation(self.model.prior) +\
                    l2_regularisation(self.model.fcomb.layers)
                supervised_loss = -supervised_elbo + 1e-5 * supervised_reg_loss

            weak_aug, strong_aug = xt1, xt2

            # target dataset used to generate predictions from the joint-training model
            with torch.no_grad():
                y, z = self.sample_from_weak_model(weak_aug)

            y, z = y.detach(), z.detach()

            # target training based on pseudo labels generated on-the-fly from above
            with forward_context():
                self.model.forward(strong_aug, y, training=True)
                lr = [pm["lr"] for pm in self.optimizer.param_groups][0]
                if lr:
                    if not self.consensus_masking_is_activated:
                        print("Activating consensus masking the reconstruction loss in iteration :", self._iteration)
                        self.consensus_masking_is_activated = True
                    target_elbo = self.model.elbo(y, z)
                else:
                    target_elbo = self.model.elbo(y)
                target_reg_loss = l2_regularisation(self.model.posterior) +\
                    l2_regularisation(self.model.prior) +\
                    l2_regularisation(self.model.fcomb.layers)
                target_loss = -target_elbo + 1e-5 * target_reg_loss

            loss = (supervised_loss + target_loss) / 2

            backprop(loss)

            if self.logger is not None:
                lr = [pm["lr"] for pm in self.optimizer.param_groups][0]
                samples = self.sample_from_model() if self._iteration % self.log_image_interval == 0 else None
                self.logger.log_train(self._iteration, loss, lr, xt, xt1, xt2, y, z, yt, samples)

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
            self.target_val_loader = self.val_loader

            for xt, xt1, xt2, yt in self.target_val_loader:
                xt, xt1, xt2 = xt.to(self.device), xt1.to(self.device), xt2.to(self.device)

                weak_aug, strong_aug = xt1, xt2

                y, z = self.sample_from_weak_model(weak_aug)

                with forward_context():
                    self.model.forward(strong_aug, y, training=True)
                    elbo = self.model.elbo(y, z) if self.consensus_masking_is_activated else self.model.elbo(y)
                    reg_loss = l2_regularisation(self.model.posterior) +\
                        l2_regularisation(self.model.prior) +\
                        l2_regularisation(self.model.fcomb.layers)
                    loss = -elbo + 1e-5 * reg_loss

                    samples = self.sample_from_model()
                    mypred, mygt = samples.detach().cpu().numpy().squeeze(), y.detach().cpu().numpy().squeeze()
                    mymetric = dice_score(mypred, mygt)
                    _mymetric = 1. - mymetric

                    true_gt = yt.detach().cpu().numpy().squeeze()
                    true_metric = dice_score(mypred, true_gt)
                    _true_metric = 1. - true_metric

                dice_metric += mymetric
                loss_val += loss.item()
                metric_val += _mymetric
                gt_metric_val += _true_metric

        metric_val /= len(self.target_val_loader)
        loss_val /= len(self.target_val_loader)
        dice_metric /= len(self.target_val_loader)
        gt_metric_val /= len(self.target_val_loader)
        print(f"The Average Dice Score for the Current Epoch is {dice_metric}")

        if self.logger is not None:
            samples = self.sample_from_model()
            self.logger.log_validation(
                self._iteration, metric_val, loss_val, xt, xt1, xt2, y, z, yt, samples, gt_metric=gt_metric_val
            )
        return metric_val


class AdaMatchLogger(TorchEmLogger):
    def __init__(self, trainer, save_root, **unused_kwargs):
        super().__init__(trainer, save_root)
        self.my_root = save_root
        self.log_dir = f"./logs/{trainer.name}" if self.my_root is None else\
            os.path.join(self.my_root, "logs", trainer.name)
        os.makedirs(self.log_dir, exist_ok=True)

        self.tb = torch.utils.tensorboard.SummaryWriter(self.log_dir)
        self.log_image_interval = trainer.log_image_interval

    def add_image(self, xt, xt1, xt2, y, z, gt, samples, name, step):
        self.tb.add_image(tag=f"{name}/target_inputs", img_tensor=_normalize_torch(xt[0]), global_step=step)
        self.tb.add_image(tag=f"{name}/weak_aug", img_tensor=_normalize_torch(xt1[0]), global_step=step)
        self.tb.add_image(tag=f"{name}/strong_aug", img_tensor=_normalize_torch(xt2[0]), global_step=step)
        self.tb.add_image(tag=f"{name}/weak_model_predictions", img_tensor=y[0], global_step=step)
        self.tb.add_image(tag=f"{name}/weak_model_consensus", img_tensor=z[0], global_step=step)
        self.tb.add_image(tag=f"{name}/target_ground_truth", img_tensor=gt[0], global_step=step)
        self.tb.add_image(tag=f"{name}/model_samples", img_tensor=samples[0], global_step=step)

    def log_train(self, step, loss, lr, xt, xt1, xt2, y, z, gt, samples):
        self.tb.add_scalar(tag="train/loss", scalar_value=loss, global_step=step)
        self.tb.add_scalar(tag="train/learning_rate", scalar_value=lr, global_step=step)
        if step % self.log_image_interval == 0:
            self.add_image(xt, xt1, xt2, y, z, gt, samples, "train", step)

    def log_validation(self, step, metric, loss, xt, xt1, xt2, y, z, gt, samples, gt_metric=None):
        self.tb.add_scalar(tag="validation/loss", scalar_value=loss, global_step=step)
        self.tb.add_scalar(tag="validation/metric", scalar_value=metric, global_step=step)
        if gt_metric is not None:
            self.tb.add_scalar(tag="validation/gt_metric", scalar_value=gt_metric, global_step=step)
        self.add_image(xt, xt1, xt2, y, z, gt, samples, "validation", step)
