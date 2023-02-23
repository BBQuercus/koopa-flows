from typing import Callable

from prefect.futures import PrefectFuture


def wait_for_task_runs(
    results: list,
    buffer: list[PrefectFuture],
    max_buffer_length: int = 6,
    result_insert_fn: Callable = lambda r: r.result(),
):
    while len(buffer) >= max(1, max_buffer_length):
        i = 0
        while i < len(buffer):
            result = buffer[i].wait(1)
            if result is not None:
                results.append(
                    result_insert_fn(result)
                )
                buffer.pop(i)
                i = len(buffer)
            else:
                i += 1
