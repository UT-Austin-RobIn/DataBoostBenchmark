# TODO: Maybe only import the specific environment when required
from databoost.base import DataBoostBenchmarkBase
# from databoost.envs.antmaze import DataBoostBenchmarkAntMaze
# from databoost.envs.calvin import DataBoostBenchmarkCalvin
from databoost.envs.metaworld import DataBoostBenchmarkMetaworld


# dictionary of DataBoost's benchmark names and corresponding benchmark objects
benchmarks = {
    "metaworld": DataBoostBenchmarkMetaworld,
    # "antmaze": DataBoostBenchmarkAntMaze,
    # "calvin": DataBoostBenchmarkCalvin
}


# list of benchmark names
benchmarks_list = list(benchmarks.keys())


def get_benchmark(benchmark_name: str) -> DataBoostBenchmarkBase:
    '''Returns an initialized benchmark object corresponding to the specified
    benchmark name.

    Args:
        benchmark_name [str]: name of benchmark; must be valid (included in
                              benchmarks_list)
    Returns:
        benchmark [DataBoostBenchmarkBase]: initialized benchmark object
    '''
    assert benchmark_name in benchmarks_list
    benchmark = benchmarks[benchmark_name]()
    return benchmark

__all__ = [get_benchmark, benchmarks_list]
