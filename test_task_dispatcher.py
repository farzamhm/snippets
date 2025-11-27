import asyncio
import time
import random
from collections import deque
from typing import Any, Dict, List, Tuple


class RateLimitedExecutor:
    def __init__(
        self,
        max_concurrent: int,
        max_new_per_n_seconds: int,
        n_seconds: float,
    ):
        """
        max_concurrent:
            Maximum number of in-flight tasks at any time.

        max_new_per_n_seconds:
            Maximum number of *new* tasks that can be started per `n_seconds`.

        n_seconds:
            The length of the rate-limit window in seconds.
            - If n_seconds = 1.0 → you get "per second" behaviour.
            - If n_seconds = 2.0 → "per 2 seconds", etc.
        """
        self.max_concurrent = max_concurrent
        self.max_new_per_n_seconds = max_new_per_n_seconds
        self.n_seconds = n_seconds

    async def __make_request_with_semaphore(
        self,
        semaphore: asyncio.Semaphore,
        building_id: str,
    ) -> Tuple[str, float]:
        """
        Dummy request function that simulates an HTTP call using sleep.
        Returns (response_text, response_time).
        """
        async with semaphore:
            start = time.time()
            print(f"{start:.3f} - START   building={building_id}")
            # Simulate variable latency
            delay = random.uniform(0.3, 0.4)
            await asyncio.sleep(delay)
            end = time.time()
            duration = end - start
            print(f"{end:.3f} - FINISH  building={building_id}, duration={duration:.3f}s")
            return f"response-{building_id}", duration

    async def execute_tasks(
        self,
        batches: List[List[Dict[str, Any]]],
    ) -> Tuple[List[Any], float, int, List[Dict[str, Any]]]:
        """
        Execute tasks with:
          - a global concurrency limit
          - a generalized rate limit: at most `max_new_per_n_seconds` new tasks
            per `n_seconds` window.

        Args:
            batches: list of batches; each batch is a list of building dicts
                     with at least {"building_id": "..."}.

        Returns:
            results: list of responses or exceptions
            avg_task_time: average duration of successful tasks
            failed_tasks: number of failed tasks
            failed_buildings: list of building dicts that failed
        """
        if not batches:
            return [], 0.0, 0, []

        max_concurrent = self.max_concurrent
        max_new_per_n_seconds = self.max_new_per_n_seconds
        n_seconds = self.n_seconds

        semaphore = asyncio.Semaphore(max_concurrent)

        # Flatten batches into a single queue
        queue: deque[Dict[str, Any]] = deque()
        for batch in batches:
            for building in batch:
                queue.append(building)

        in_flight: dict[asyncio.Task, Dict[str, Any]] = {}
        results: List[Any] = []
        failed_buildings: List[Dict[str, Any]] = []
        failed_tasks = 0

        total_task_time = 0.0
        total_successful_tasks = 0

        # Rate limiting state
        window_start = time.time()
        started_in_window = 0

        overall_start = time.time()
        print(
            f"Starting execution: "
            f"{len(batches)} batches, "
            f"{sum(len(b) for b in batches)} total tasks, "
            f"max_concurrent={max_concurrent}, "
            f"max_new_per_{n_seconds}_seconds={max_new_per_n_seconds}"
        )

        while queue or in_flight:
            loop_start = time.time()

            # Reset the window if needed
            now = time.time()
            if now - window_start >= n_seconds:
                window_start = now
                started_in_window = 0

            # How many tasks can we start right now?
            available_concurrency = max_concurrent - len(in_flight)
            available_quota = max_new_per_n_seconds - started_in_window
            can_start = min(available_concurrency, available_quota, len(queue))

            # Start up to can_start new tasks
            for _ in range(can_start):
                building = queue.popleft()
                building_id = building["building_id"]
                task = asyncio.create_task(
                    self.__make_request_with_semaphore(semaphore, building_id)
                )
                in_flight[task] = building
                started_in_window += 1

            # If nothing in flight and queue is empty -> done
            if not in_flight and not queue:
                break

            # If nothing in flight but queue has items, just loop again to start more
            if not in_flight and queue:
                continue

            # Wait for at least one task to finish, or until the current
            # window is over so we can start more.
            time_passed_in_window = time.time() - window_start
            time_left_in_window = max(0.0, n_seconds - time_passed_in_window)
            timeout = time_left_in_window if time_left_in_window > 0 else None
            if timeout is not None:
                print(f"Waiting up to {timeout:.3f}s for the rest of the window to elapse, all completed tasks time: time_passed_in_window={time_passed_in_window:.3f}s")
            done, _ = await asyncio.wait(
                set(in_flight.keys()),
                timeout=timeout,
                return_when=asyncio.FIRST_COMPLETED,
            )

            # Process finished tasks
            for task in done:
                building = in_flight.pop(task)
                try:
                    resp_text, resp_time = await task
                    results.append(resp_text)
                    total_task_time += resp_time
                    total_successful_tasks += 1
                except Exception as exc:
                    failed_tasks += 1
                    failed_buildings.append(building)
                    results.append(exc)

            loop_end = time.time()
            loop_elapsed = loop_end - loop_start
            # You could log loop_elapsed if you want per-iteration timing.

        overall_end = time.time()
        overall_elapsed = overall_end - overall_start

        avg_task_time = (
            total_task_time / total_successful_tasks if total_successful_tasks else 0.0
        )

        print("\n=== Summary ===")
        print(f"Overall elapsed time: {overall_elapsed:.3f}s")
        print(f"Successful tasks:     {total_successful_tasks}")
        print(f"Failed tasks:         {failed_tasks}")
        print(f"Average task time:    {avg_task_time:.3f}s")

        return results, avg_task_time, failed_tasks, failed_buildings


async def main():
    random.seed(0)

    # Example: 3 batches of 5 tasks
    batches: List[List[Dict[str, Any]]] = []
    for b in range(3):
        batch = []
        for i in range(10):
            batch.append({"building_id": f"batch{b+1}-task{i+1}"})
        batches.append(batch)

    # Example:
    #   - max_concurrent = 5
    #   - at most 5 new tasks per 2 seconds
    executor = RateLimitedExecutor(
        max_concurrent=len(batches[0]),
        max_new_per_n_seconds=len(batches[0]),
        n_seconds=5,
    )

    results, avg_time, failed_tasks, failed_buildings = await executor.execute_tasks(
        batches
    )

    print("\nResults count:", len(results))
    print("Average task time (returned):", avg_time)
    print("Failed tasks:", failed_tasks)
    print("Failed buildings:", failed_buildings)


if __name__ == "__main__":
    asyncio.run(main())

