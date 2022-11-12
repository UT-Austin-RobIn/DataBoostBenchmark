from databoost.envs.antmaze import DataBoostBenchmarkAntMaze
from databoost.envs.metaworld import DataBoostBenchmarkMetaworld



benchmarks = {
    "metaworld": DataBoostBenchmarkMetaworld,
    "antmaze": DataBoostBenchmarkAntMaze
}

benchmarks_list = list(benchmarks.keys())

def get_benchmark(benchmark_name: str):
    assert benchmark_name in benchmarks_list
    return benchmarks[benchmark_name]()

__all__ = [get_benchmark, benchmarks_list]
