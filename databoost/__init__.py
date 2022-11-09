from databoost.envs.metaworld import DataBoostBenchmarkMetaworld


benchmark = {
    "metaworld": DataBoostBenchmarkMetaworld,
}

benchmarks_list = list(benchmark.keys())


__all__ = [benchmark, benchmarks_list]
