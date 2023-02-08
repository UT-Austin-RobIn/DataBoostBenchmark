"""Torch implementation of Implicit Q-Learning (IQL)
https://github.com/ikostrikov/implicit_q_learning
"""
from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn
import databoost.models.iql.pytorch_utils as ptu


class LinearTransform(nn.Module):
    def __init__(self, m, b):
        super().__init__()
        self.m = m
        self.b = b

    def __call__(self, t):
        return self.m * t + self.b

class IQLModel(nn.Module):
    def __init__(
            self,
            # env,
            policy,
            qf1,
            qf2,
            vf,
            quantile=0.5,
            target_qf1=None,
            target_qf2=None,
            buffer_policy=None,
            z=None,

            discount=0.99,
            reward_scale=1.0,

            policy_lr=1e-3,
            qf_lr=1e-3,
            policy_weight_decay=0,
            q_weight_decay=0,
            optimizer_class=optim.Adam,

            policy_update_period=1,
            q_update_period=1,

            reward_transform_class=None,
            reward_transform_kwargs=None,
            terminal_transform_class=None,
            terminal_transform_kwargs=None,

            clip_score=None,
            soft_target_tau=1e-2,
            target_update_period=1,
            beta=1.0,

            device = None
    ):
        super(IQLModel, self).__init__()
        # self.env = env
        self.policy = policy
        self.qf1 = qf1
        self.qf2 = qf2
        self.target_qf1 = target_qf1
        self.target_qf2 = target_qf2
        self.soft_target_tau = soft_target_tau
        self.target_update_period = target_update_period
        self.vf = vf
        self.z = z
        self.buffer_policy = buffer_policy

        self.qf_criterion = nn.MSELoss()
        self.vf_criterion = nn.MSELoss()

        self.optimizers = {}

        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            weight_decay=policy_weight_decay,
            lr=policy_lr,
        )
        self.optimizers[self.policy] = self.policy_optimizer
        self.qf1_optimizer = optimizer_class(
            self.qf1.parameters(),
            weight_decay=q_weight_decay,
            lr=qf_lr,
        )
        self.qf2_optimizer = optimizer_class(
            self.qf2.parameters(),
            weight_decay=q_weight_decay,
            lr=qf_lr,
        )
        self.vf_optimizer = optimizer_class(
            self.vf.parameters(),
            weight_decay=q_weight_decay,
            lr=qf_lr,
        )

        if self.z:
            self.z_optimizer = optimizer_class(
                self.z.parameters(),
                weight_decay=q_weight_decay,
                lr=qf_lr,
            )

        self.discount = discount
        self.reward_scale = reward_scale
        self.eval_statistics = OrderedDict()
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True

        self.q_update_period = q_update_period
        self.policy_update_period = policy_update_period

        self.reward_transform_class = reward_transform_class or LinearTransform
        self.reward_transform_kwargs = reward_transform_kwargs or dict(m=1, b=0)
        self.terminal_transform_class = terminal_transform_class or LinearTransform
        self.terminal_transform_kwargs = terminal_transform_kwargs or dict(m=1, b=0)
        self.reward_transform = self.reward_transform_class(**self.reward_transform_kwargs)
        self.terminal_transform = self.terminal_transform_class(**self.terminal_transform_kwargs)

        self.clip_score = clip_score
        self.beta = beta
        self.quantile = quantile

        self.device = device

    def train_from_torch(self, batch, train=True, pretrain=False,):
        rewards = batch['rewards'][:, 0].float().to(self.device).reshape(-1, 1)
        terminals = batch['dones'][:, 0].float().to(self.device).reshape(-1, 1)
        obs = batch['observations'][:, 0].float().to(self.device)
        actions = batch['actions'][:, 0].float().to(self.device)
        next_obs = batch['observations'][:, 1].float().to(self.device)
        seed_bool = batch['seed'][:, 0].int().detach().numpy()
        if self.reward_transform:
            rewards = self.reward_transform(rewards)

        if self.terminal_transform:
            terminals = self.terminal_transform(terminals)
        """
        Policy and Alpha Loss
        """
        dist = self.policy(obs)

        """
        QF Loss
        """
        q1_pred = self.qf1(obs, actions)
        q2_pred = self.qf2(obs, actions)
        target_vf_pred = self.vf(next_obs).detach()

        q_target = self.reward_scale * rewards + (1. - terminals) * self.discount * target_vf_pred
        q_target = q_target.detach()
        qf1_loss = self.qf_criterion(q1_pred, q_target)
        qf2_loss = self.qf_criterion(q2_pred, q_target)

        """
        VF Loss
        """
        q_pred = torch.min(
            self.target_qf1(obs, actions),
            self.target_qf2(obs, actions),
        ).detach()
        vf_pred = self.vf(obs)
        vf_err = vf_pred - q_pred
        vf_sign = (vf_err > 0).float()
        vf_weight = (1 - vf_sign) * self.quantile + vf_sign * (1 - self.quantile)
        vf_loss = (vf_weight * (vf_err ** 2)).mean()

        """
        Policy Loss
        """
        policy_logpp = dist.log_prob(actions)

        adv = q_pred - vf_pred
        exp_adv = torch.exp(adv * self.beta)
        # exp_adv = torch.ones_like(q_pred)
        if self.clip_score is not None:
            exp_adv = torch.clamp(exp_adv, max=self.clip_score)


        weights = exp_adv[:, 0].detach()
        policy_loss = (-policy_logpp * weights).mean()

        """
        Update networks
        """
        if self._n_train_steps_total % self.q_update_period == 0:
            self.qf1_optimizer.zero_grad()
            qf1_loss.backward()
            self.qf1_optimizer.step()

            self.qf2_optimizer.zero_grad()
            qf2_loss.backward()
            self.qf2_optimizer.step()

            self.vf_optimizer.zero_grad()
            vf_loss.backward()
            self.vf_optimizer.step()

        if self._n_train_steps_total % self.policy_update_period == 0:
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

        """
        Soft Updates
        """
        if self._n_train_steps_total % self.target_update_period == 0:
            ptu.soft_update_from_to(
                self.qf1, self.target_qf1, self.soft_target_tau
            )
            ptu.soft_update_from_to(
                self.qf2, self.target_qf2, self.soft_target_tau
            )
            
        self._n_train_steps_total += 1

        seed_idx = np.where(seed_bool == 1)[0]
        prior_idx = np.where(seed_bool == 0)[0]
        
        logs = {
            'losses/qf1_loss': qf1_loss.detach().item(),
            'losses/qf2_loss': qf2_loss.detach().item(),
            'losses/vf_loss': vf_loss.detach().item(),
            'losses/policy_loss': policy_loss.detach().item(),
            'values/vf': target_vf_pred.mean().detach().item(),
            'values/q_target': q_target.mean().detach().item(),
            'values/q1_pred': q1_pred.mean().detach().item(),
            'values/q2_pred': q2_pred.mean().detach().item(),
            'values/adv_weight': exp_adv.mean().detach().item(),
            
            # seed specific logs
            'values_seed/vf': target_vf_pred[seed_idx].sum().detach().item(),
            'values_seed/q_target': q_target[seed_idx].sum().detach().item(),
            'values_seed/q1_pred': q1_pred[seed_idx].sum().detach().item(),
            'values_seed/q2_pred': q2_pred[seed_idx].sum().detach().item(),
            'values_seed/dones': terminals[seed_idx].sum().detach().item(),
            'values_seed/rewards': rewards[seed_idx].sum().detach().item(),
            'values_seed/num_samples': len(seed_idx),

            # prior specific logs
            'values_prior/vf': target_vf_pred[prior_idx].sum().detach().item(),
            'values_prior/q_target': q_target[prior_idx].sum().detach().item(),
            'values_prior/q1_pred': q1_pred[prior_idx].sum().detach().item(),
            'values_prior/q2_pred': q2_pred[prior_idx].sum().detach().item(),
            'values_prior/dones': terminals[prior_idx].sum().detach().item(),
            'values_prior/rewards': rewards[prior_idx].sum().detach().item(),
            'values_prior/num_samples': len(prior_idx)
        }

        return logs

    @property
    def networks(self):
        nets = [
            self.policy,
            self.qf1,
            self.qf2,
            self.target_qf1,
            self.target_qf2,
            self.vf,
        ]
        return nets

    def get_snapshot(self):
        return dict(
            policy=self.policy,
            qf1=self.qf1,
            qf2=self.qf2,
            target_qf1=self.target_qf1,
            target_qf2=self.target_qf2,
            vf=self.vf,
        )
    
    def get_all_state_dicts(self) -> dict:
        state_dicts = {
            "vf_optimizer": self.vf_optimizer.state_dict(),
            "qf1_optimizer": self.qf1_optimizer.state_dict(),
            "qf2_optimizer": self.qf2_optimizer.state_dict(),
            "policy_optimizer": self.policy_optimizer.state_dict(),
            "network_state_dicts": self.state_dict(),
            "n_train_steps_total": self._n_train_steps_total,
        }
        return state_dicts
    
    def load_from_checkpoint(self, state_dicts_dict: dict, load_optimizer: bool):
        self.load_state_dict(
            state_dicts_dict["network_state_dicts"], strict=False
        )  # strict false for if we disable the progress predictor
        if load_optimizer:
            self.vf_optimizer.load_state_dict(state_dicts_dict["vf_optimizer"])
            self.qf1_optimizer.load_state_dict(state_dicts_dict["qf1_optimizer"])
            self.qf2_optimizer.load_state_dict(state_dicts_dict["qf2_optimizer"])
            self.policy_optimizer.load_state_dict(state_dicts_dict["policy_optimizer"])
        self._n_train_steps_total = state_dicts_dict["n_train_steps_total"]
    
    def get_action(self, obs):
        act = self.policy(torch.tensor(obs).float().to('cuda')).mu.cpu().detach().numpy()
        return act
