import os
import time
from copy import deepcopy

import torch

import torch_em
from torch_em.transform.raw import _normalize_torch
from torch_em.trainer.logger_base import TorchEmLogger

from prob_utils.my_utils import dice_score
from prob_utils.my_models import l2_regularisation


class MeanTeacherTrainer(torch_em.trainer.DefaultTrainer):
    """This trainer is meant to be used for MeanTeacher-based PUNet's training,
    where we also weight the ELBO based on consensus masks
    """
    def __init__(
        self,
        ckpt_model=None,
        ckpt_teacher=None,
        momentum=0.999,
        momentum_update_is_activated=False,
        do_consensus_masking=False,
        **kwargs
    ):
        super().__init__(**kwargs)

        self._kwargs = kwargs
        self.momentum = momentum
        self.ckpt_model = ckpt_model
        self.ckpt_teacher = ckpt_teacher
        self.momentum_update_is_activated = momentum_update_is_activated
        self.sigmoid = torch.nn.Sigmoid()
        self.n_samples = 16
        self.do_consensus_masking = do_consensus_masking

        with torch.no_grad():
            self.teacher = deepcopy(self.model)
            for param in self.teacher.parameters():
                param.requires_grad = False

        if self.ckpt_model is not None:
            weights = torch.load(self.ckpt_model, weights_only=False)["model_state"]
            self.model.load_state_dict(weights)

        if self.ckpt_teacher is not None:
            weights = torch.load(self.ckpt_teacher, weights_only=False)["model_state"]
            self.teacher.load_state_dict(weights)

    def _momentum_update(self):
        current_momentum = self.momentum
        for param, param_teacher in zip(self.model.parameters(), self.teacher.parameters()):
            param_teacher.data = param_teacher.data * current_momentum + param.data * (1. - current_momentum)

    def save_checkpoint(self, name, current_metric, best_metric, train_time=0.0):
        teacher_state = {"teacher_state": self.teacher.state_dict()}
        super().save_checkpoint(name, current_metric, best_metric, train_time=train_time, **teacher_state)

    def load_checkpoint(self, checkpoint="best"):
        save_dict = super().load_checkpoint(checkpoint)
        self.teacher.load_state_dict(save_dict["teacher_state"])
        self.teacher.to(self.device)
        return save_dict

    def _initialize(self, iterations, load_from_checkpoint, epochs=None):
        best_metric = super()._initialize(iterations, load_from_checkpoint, epochs=epochs)
        self.teacher.to(self.device)
        return best_metric

    def sample_from_teacher(self, teacher_inputs, upper_thres=0.9, lower_thres=0.1):
        self.teacher.forward(teacher_inputs, None, training=False)
        samples = [self.sigmoid(self.teacher.sample()) for _ in range(self.n_samples)]
        consensus = [
            torch.where(
                (my_sample >= upper_thres) + (my_sample <= lower_thres),
                torch.tensor(1.).to(self.device),
                torch.tensor(0.).to(self.device)
            ) for my_sample in samples
        ]
        samples = torch.stack(samples, dim=0).sum(dim=0) / self.n_samples
        consensus = torch.stack(consensus, dim=0).sum(dim=0) / self.n_samples

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

        for x, x1, x2, gt_ in self.train_loader:
            x, x1, x2 = x.to(self.device), x1.to(self.device), x2.to(self.device)

            teacher_inputs = x1

            with torch.no_grad():
                lr = [pm["lr"] for pm in self.optimizer.param_groups][0]
                y, z = self.sample_from_teacher(teacher_inputs)
                y, z = y.to(self.device), z.to(self.device)

            self.optimizer.zero_grad()
            with forward_context():
                self.model.forward(x2, y, training=True)
                elbo = self.model.elbo(y, z)
                reg_loss = l2_regularisation(self.model.posterior) + l2_regularisation(self.model.prior) + \
                    l2_regularisation(self.model.fcomb.layers)
                loss = -elbo + 1e-5 * reg_loss

            backprop(loss)

            if self.logger is not None:
                lr = [pm["lr"] for pm in self.optimizer.param_groups][0]
                samples = self.sample_from_model() if self._iteration % self.log_image_interval == 0 else None
                self.logger.log_train(self._iteration, loss, lr, x, x1, x2, y, z, gt_, samples)

            if lr:
                if not self.momentum_update_is_activated:
                    print("Activating momentum update in iteration : ", self._iteration)
                    self.momentum_update_is_activated = True
                with torch.no_grad():
                    self._momentum_update()

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

                teacher_inputs = x1

                y, z = self.sample_from_teacher(teacher_inputs)
                y, z = y.to(self.device), z.to(self.device)

                with forward_context():
                    self.model.forward(x2, y, training=True)
                    elbo = self.model.elbo(y, z)
                    reg_loss = l2_regularisation(self.model.posterior) + l2_regularisation(self.model.prior) + \
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


class MeanTeacherLogger(TorchEmLogger):
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

        from functools import partial
        norm = partial(_normalize_torch, minval=None, maxval=None, axis=None, eps=1e-7)

        self.tb.add_image(tag=f"{name}/input", img_tensor=norm(x[0]), global_step=step)
        self.tb.add_image(tag=f"{name}/aug_inputs_1", img_tensor=norm(x1[0]), global_step=step)
        self.tb.add_image(tag=f"{name}/aug_inputs_2", img_tensor=norm(x2[0]), global_step=step)
        self.tb.add_image(tag=f"{name}/teacher_predictions", img_tensor=y[0], global_step=step)
        self.tb.add_image(tag=f"{name}/teacher_consensus", img_tensor=z[0], global_step=step)
        self.tb.add_image(tag=f"{name}/ground_truth", img_tensor=gt[0], global_step=step)
        self.tb.add_image(tag=f"{name}/model_samples", img_tensor=samples[0], global_step=step)

    def log_train(self, step, loss, lr, x, x1, x2, y, z, gt, samples):
        self.tb.add_scalar(tag="train/loss", scalar_value=loss, global_step=step)
        self.tb.add_scalar(tag="train/learning_rate", scalar_value=lr, global_step=step)
        if step % self.log_image_interval == 0:
            self.add_image(x, x1, x2, y, z, gt, samples, "train", step)

    def log_validation(self, step, metric, loss, x, x1, x2, y, z, gt, samples, gt_metric=None):
        self.tb.add_scalar(tag="validation/loss", scalar_value=loss, global_step=step)
        self.tb.add_scalar(tag="validation/metric", scalar_value=metric, global_step=step)
        if gt_metric is not None:
            self.tb.add_scalar(tag="validation/gt_metric", scalar_value=gt_metric, global_step=step)
        self.add_image(x, x1, x2, y, z, gt, samples, "validation", step)
