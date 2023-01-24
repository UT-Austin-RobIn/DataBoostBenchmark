import copy
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.distributions import MultivariateNormal

import databoost
from databoost.utils.general import AttrDict
from databoost.models.mlp import mlp
from databoost.utils.model_utils import compute_batched, update_exponential_moving_average


DEFAULT_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EXP_ADV_MAX = 100.
LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


def asymmetric_l2_loss(u, tau):
    return torch.mean(torch.abs(tau - (u < 0).float()) * u**2)


class TwinQ(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, n_hidden=2):
        super().__init__()
        dims = [state_dim + action_dim, *([hidden_dim] * n_hidden), 1]
        self.q1 = mlp(dims, squeeze_output=True)
        self.q2 = mlp(dims, squeeze_output=True)

    def both(self, state, action):
        sa = torch.cat([state, action], 1)
        return self.q1(sa), self.q2(sa)

    def forward(self, state, action):
        return torch.min(*self.both(state, action))


class ValueFunction(nn.Module):
    def __init__(self, state_dim, hidden_dim=256, n_hidden=2):
        super().__init__()
        dims = [state_dim, *([hidden_dim] * n_hidden), 1]
        self.v = mlp(dims, squeeze_output=True)

    def forward(self, state):
        return self.v(state)


class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=256, n_hidden=2):
        super().__init__()
        self.net = mlp([obs_dim, *([hidden_dim] * n_hidden), act_dim])
        self.log_std = nn.Parameter(torch.zeros(act_dim, dtype=torch.float32))

    def forward(self, obs):
        mean = self.net(obs)
        std = torch.exp(self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX))
        scale_tril = torch.diag(std)
        return MultivariateNormal(mean, scale_tril=scale_tril)
        # if mean.ndim > 1:
        #     batch_size = len(obs)
        #     return MultivariateNormal(mean, scale_tril=scale_tril.repeat(batch_size, 1, 1))
        # else:
        #     return MultivariateNormal(mean, scale_tril=scale_tril)

    def get_action(self, ob, deterministic=False, enable_grad=False):
        with torch.set_grad_enabled(enable_grad):
            ob = torch.tensor(ob[None], dtype=torch.float).to(DEFAULT_DEVICE)
            dist = self(ob)
            return dist.mean.cpu().detach().numpy()[0] if deterministic else dist.sample().cpu().detach().numpy()[0]


class DeterministicPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=256, n_hidden=2):
        super().__init__()
        self.net = mlp([obs_dim, *([hidden_dim] * n_hidden), act_dim],
                       output_activation=nn.Tanh)

    def forward(self, obs):
        return self.net(obs)

    def get_action(self, ob, deterministic=False, enable_grad=False):
        with torch.set_grad_enabled(enable_grad):
            ob = torch.tensor(ob[None], dtype=torch.float).to(DEFAULT_DEVICE)
            return self(ob).cpu().detach().numpy()[0]


class ImplicitQLearning(nn.Module):
    def __init__(self, qf, vf, policy, optimizer_factory, max_steps,
                 tau, beta, discount=0.99, alpha=0.005):
        super().__init__()
        self.qf = qf.to(DEFAULT_DEVICE)
        self.q_target = copy.deepcopy(qf).requires_grad_(False).to(DEFAULT_DEVICE)
        self.vf = vf.to(DEFAULT_DEVICE)
        self.policy = policy.to(DEFAULT_DEVICE)
        self.v_optimizer = optimizer_factory(self.vf.parameters())
        self.q_optimizer = optimizer_factory(self.qf.parameters())
        self.policy_optimizer = optimizer_factory(self.policy.parameters())
        self.policy_lr_schedule = CosineAnnealingLR(self.policy_optimizer, max_steps)
        self.tau = tau
        self.beta = beta
        self.discount = discount
        self.alpha = alpha

    def update(self, observations, actions, next_observations, rewards, terminals):
        with torch.no_grad():
            target_q = self.q_target(observations, actions)
            next_v = self.vf(next_observations)

        # v, next_v = compute_batched(self.vf, [observations, next_observations])

        # Update value function
        v = self.vf(observations)
        adv = target_q - v
        v_loss = asymmetric_l2_loss(adv, self.tau)
        self.v_optimizer.zero_grad(set_to_none=True)
        v_loss.backward()
        self.v_optimizer.step()

        # Update Q function
        targets = rewards + (1. - terminals.float()) * self.discount * next_v.detach()
        qs = self.qf.both(observations, actions)
        q_loss = sum(F.mse_loss(q, targets) for q in qs) / len(qs)
        self.q_optimizer.zero_grad(set_to_none=True)
        q_loss.backward()
        self.q_optimizer.step()

        # Update target Q network
        update_exponential_moving_average(self.q_target, self.qf, self.alpha)

        # Update policy
        exp_adv = torch.exp(self.beta * adv.detach()).clamp(max=EXP_ADV_MAX)
        policy_out = self.policy(observations)
        if isinstance(policy_out, torch.distributions.Distribution):
            bc_losses = -policy_out.log_prob(actions)
        elif torch.is_tensor(policy_out):
            assert policy_out.shape == actions.shape
            bc_losses = torch.sum((policy_out - actions)**2, dim=1)
        else:
            raise NotImplementedError
        policy_loss = torch.mean(exp_adv * bc_losses)
        self.policy_optimizer.zero_grad(set_to_none=True)
        policy_loss.backward()
        self.policy_optimizer.step()
        self.policy_lr_schedule.step()


def main(benchmark, task_name, dataloader_configs, args):
    env = benchmark.get_env(task_name)
    dataloader = env._get_dataloader(**dataloader_configs)

    if args.deterministic_policy:
        policy = DeterministicPolicy(args.obs_dim, args.act_dim, hidden_dim=args.hidden_dim, n_hidden=args.n_hidden)
    else:
        policy = GaussianPolicy(args.obs_dim, args.act_dim, hidden_dim=args.hidden_dim, n_hidden=args.n_hidden)

    iql = ImplicitQLearning(
        qf=TwinQ(args.obs_dim, args.act_dim, hidden_dim=args.hidden_dim, n_hidden=args.n_hidden),
        vf=ValueFunction(args.obs_dim, hidden_dim=args.hidden_dim, n_hidden=args.n_hidden),
        policy=policy,
        optimizer_factory=lambda params: torch.optim.Adam(params, lr=args.learning_rate),
        max_steps=args.n_steps,
        tau=args.tau,
        beta=args.beta,
        alpha=args.alpha,
        discount=args.discount
    )

    for step in tqdm(range(args.n_steps)):
        for batch_num, traj_batch in tqdm(enumerate(dataloader)):
            observations = traj_batch["observations"][:, 0, :]
            actions = traj_batch["actions"][:, 0, :]
            next_observations = traj_batch["observations"][:, 1, :]
            rewards = traj_batch["rewards"][:, 0]
            terminals = traj_batch["dones"][:, 0]
            iql.update(observation, actions, next_observations, rewards, terminals)
        if (step+1) % args.eval_period == 0:
            success_rate, _ = benchmark.evaluate(
                task_name=task_name,
                policy=policy,
                n_episodes=args.n_eval_episodes,
                max_traj_len=args.max_episode_steps,
                goal_cond=False,
                render=False
            )
            print(f"step {step}: success_rate: {success_rate}")


if __name__=="__main__":

    benchmark_name = "metaworld"
    task_name = "pick-place-wall"
    boosting_method = "seed"
    # exp_name = f"{benchmark_name}-{task_name}-{boosting_method}-goal_cond-4"
    # dest_dir = f"/data/jullian-yapeter/DataBoostBenchmark/{benchmark_name}/models/{task_name}/{boosting_method}"
    # dest_dir = f"/data/jullian-yapeter/DataBoostBenchmark/{benchmark_name}/models/all"
    goal_condition = False

    iql_configs = AttrDict({
        "obs_dim": 39,
        "act_dim": 4,
        "discount": 0.99,
        "hidden_dim": 256,
        "n_hidden": 2,
        "n_steps": 500,
        "learning_rate": 3e-4,
        "alpha": 0.005,
        "tau": 0.7,
        "beta": 3.0,
        "deterministic_policy": True,
        "eval_period": 3,
        "n_eval_episodes": 20,
        "max_episode_steps": 500
    })

    dataloader_configs = {
        "dataset_dir": f"/data/jullian-yapeter/DataBoostBenchmark/{benchmark_name}/data/seed/{task_name}",
        # "dataset_dir": f"/data/jullian-yapeter/DataBoostBenchmark/{benchmark_name}/boosted_data/{task_name}/{boosting_method}",
        # "dataset_dir": f"/data/jullian-yapeter/DataBoostBenchmark/{benchmark_name}/data",
        "n_demos": None,
        "batch_size": 64,
        "seq_len": 2,
        "shuffle": True,
        "goal_condition": goal_condition
    }

    benchmark = databoost.get_benchmark(benchmark_name)
    main(benchmark, task_name, dataloader_configs, iql_configs)