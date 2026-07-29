from llmpidtuner.parallel_api import partition_groups


def test_partition_groups_splits_100_cases_across_10_workers() -> None:
    shards = partition_groups(range(1, 101), workers=10)

    assert len(shards) == 10
    assert shards[0] == tuple(range(1, 11))
    assert shards[-1] == tuple(range(91, 101))
    assert sorted(group for shard in shards for group in shard) == list(range(1, 101))


def test_partition_groups_does_not_create_empty_workers() -> None:
    assert partition_groups([1, 2, 3], workers=10) == [(1,), (2,), (3,)]
