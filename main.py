import random

import databoost


if __name__ == "__main__":
    # inspect list of available benchmarks (e.g. ["metaworld", "calvin", "vima"])
    benchmark_list = databoost.benchmark_list
    # choose DataBoostBenchmark
    chosen_benchmark = random.choice(benchmark_list)
    # get benchmark object
    benchmark = databoost.benchmark(chosen_benchmark)
    # get offline task-agnostic prior dataset
    '''
    # Notes: size, N, is number of demonstrations.
    #        seq_len is variable; can be different for each demonstration.
    #        Include failure demos? If so, need "terminals" and "timeouts", else
    #        just have a "done" key instead.

    prior_dataset = {
        observations: N x seq_len x obs_dim array of observations,
        actions: N x seq_len x action_dim array of actions,
        terminals: N x seq_len boolean array of episode termination flags
        timeouts: N x seq_len boolean array of episode timeout flags
        info: N x seq_len optional task-specific information
    }
    '''
    prior_dataset = benchmark.get_prior_dataset(size=1e3)
    # decide on a task that is compatible with the chosen benchmark
    # (e.g. ["open-drawer", "turn-on-lightbulb"])
    task_list = benchmark.task_list
    chosen_task = random.choice(task_list)
    # get task specific seed dataset
    '''
    # Notes: expert demos only?

    seed_dataset = {
        observations: N x seq_len x obs_dim array of observations,
        actions: N x seq_len x action_dim array of actions,
        rewards: N x seq_len float array of rewards,
        done: N x seq_len boolean array of termination flags,
        info: N x seq_len optional task-specific information
    }
    '''
    seed_dataset = benchmark.get_seed_dataset(task=chosen_task, size=10)
    # instantiate corresponding gym environment, sets specified task
    env = benchmark.get_env(task=chosen_task)
    # interact with environment
    obs = env.reset()
    a = env.action_space.sample()  # Sample an action
    obs, reward, done, info = env.step(a)
    # run evaluation
    metrics = benchmark.evaluate(
        task=chosen_task,
        policy=my_policy,
        num_runs=100
    )
