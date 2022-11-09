from databoost.envs import metaworld


benchmark = {
    "metaworld": metaworld,
}

benchmarks_list = list(benchmark.keys())


__all__ = [benchmark, benchmarks_list]
