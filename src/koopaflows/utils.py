from typing import Callable

from prefect.futures import PrefectFuture


def wait_for_task_runs(
    results: list,
    buffer: list[PrefectFuture],
    max_buffer_length: int = 6,
    result_insert_fn: Callable = lambda r: r.result(),
):
    while len(buffer) >= max(1, max_buffer_length):
        results.append(result_insert_fn(buffer.pop(0)))
