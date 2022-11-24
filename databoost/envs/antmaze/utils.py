import gym


def initialize_env(env: gym.Env) -> gym.Env:
    '''Initializes the env to be ready for task-specific rollouts; sets the
    target goal to the specified target cell (as a scaled position)

    Args:
        env [gym.Env]: AntMaze env
    Returns:
        env [gym.Env]: initialized AntMaze env
    '''
    env.set_target()
    return env