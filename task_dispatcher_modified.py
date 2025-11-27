import asyncio
import time
import random
from collections import deque
from typing import Any, Dict, List, Tuple
import statistics


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

    @staticmethod
    def generate_long_tail_delay() -> float:
        """
        Generate one response time with a long-tail distribution.
        90% fast, 10% slow outliers (up to 60s).
        """
        if random.random() < 0.90:  # 90% fast responses
            # Exponential distribution for fast responses
            delay = random.expovariate(1 / 0.02)
        else:  # 10% slow outliers
            # Pareto-like distribution for outliers
            delay = random.paretovariate(1.5)
            delay = min(delay, 60.0)  # Cap at 60s

        return delay

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
            # Approximate current concurrency from semaphore
            current_concurrency = self.max_concurrent - semaphore._value
            print(
                f"{start:.3f} - START   building={building_id}, "
                f"concurrent_tasks={current_concurrency}"
            )

            # Simulate variable latency
            delay = self.generate_long_tail_delay()
            await asyncio.sleep(delay)

            end = time.time()
            duration = end - start
            current_concurrency = self.max_concurrent - semaphore._value
            print(
                f"{end:.3f} - FINISH  building={building_id}, "
                f"duration={duration:.3f}s, "
                f"concurrent_tasks={current_concurrency}"
            )
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
        successful_durations_list: List[float] = []  # <-- store all durations

        # Rate limiting state
        window_start = time.time()
        started_in_window = 0

        # Per-window stats for completed tasks
        window_completed = 0
        window_total_duration = 0.0
        window_index = 0

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
                # Emit stats for the previous window, if any
                if window_completed > 0:
                    window_avg = window_total_duration / window_completed
                    current_concurrency = len(in_flight)
                    print(
                        f"[WINDOW {window_index}] "
                        f"completed={window_completed}, "
                        f"avg_duration={window_avg:.3f}s, "
                        f"concurrent_tasks={current_concurrency}"
                    )

                window_index += 1
                window_start = now
                started_in_window = 0
                window_completed = 0
                window_total_duration = 0.0

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

            # ✅ Only do "waiting up to ..." when there are still tasks in the queue.
            # If the queue is empty, we just wait for in-flight tasks with no timeout
            # (no extra sleep / no rate-limit wait log).
            if queue and time_left_in_window > 0:
                timeout = time_left_in_window
                current_concurrency = len(in_flight)
                print(
                    f"Waiting up to {timeout:.3f}s for the rest of the window "
                    f"to elapse; time_passed_in_window={time_passed_in_window:.3f}s; "
                    f"concurrent_tasks={current_concurrency}"
                )
            else:
                timeout = None  # wait until something finishes, no artificial sleep/log

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

                    # Global stats
                    total_task_time += resp_time
                    total_successful_tasks += 1
                    successful_durations_list.append(resp_time)
                    running_avg = total_task_time / total_successful_tasks

                    # Per-window stats
                    window_completed += 1
                    window_total_duration += resp_time

                    current_concurrency = len(in_flight)
                    print(
                        f"[STATS] completed={total_successful_tasks}, "
                        f"last_duration={resp_time:.3f}s, "
                        f"running_avg={running_avg:.3f}s, "
                        f"concurrent_tasks={current_concurrency}"
                    )

                except Exception as exc:
                    failed_tasks += 1
                    failed_buildings.append(building)
                    results.append(exc)

            loop_end = time.time()
            loop_elapsed = loop_end - loop_start
            # print(f"Loop iteration elapsed: {loop_elapsed:.3f}s")

        # Flush final window stats if we still have unreported completions
        if window_completed > 0:
            window_avg = window_total_duration / window_completed
            current_concurrency = len(in_flight)  # likely 0 here
            print(
                f"[WINDOW {window_index}] "
                f"completed={window_completed}, "
                f"avg_duration={window_avg:.3f}s, "
                f"concurrent_tasks={current_concurrency}"
            )

        overall_end = time.time()
        overall_elapsed = overall_end - overall_start

        avg_task_time = (
            total_task_time / total_successful_tasks if total_successful_tasks else 0.0
        )

        # ============================
        # Distribution report
        # ============================
        if successful_durations_list:
            durations = successful_durations_list
            durations_sorted = sorted(durations)

            min_v = min(durations)
            max_v = max(durations)
            mean_v = statistics.mean(durations)
            median_v = statistics.median(durations)
            stdev_v = statistics.pstdev(durations) if len(durations) > 1 else 0.0

            def pct(p: float) -> float:
                idx = int(len(durations_sorted) * p)
                idx = min(idx, len(durations_sorted) - 1)
                return durations_sorted[idx]

            p50 = pct(0.50)
            p75 = pct(0.75)
            p90 = pct(0.90)
            p95 = pct(0.95)
            p99 = pct(0.99)

            print("\n=== Duration Distribution ===")
            print(f"Min:        {min_v:.3f} s")
            print(f"Max:        {max_v:.3f} s")
            print(f"Mean:       {mean_v:.3f} s")
            print(f"Median:     {median_v:.3f} s")
            print(f"Std Dev:    {stdev_v:.3f} s")
            print("")
            print(f"P50:        {p50:.3f} s")
            print(f"P75:        {p75:.3f} s")
            print(f"P90:        {p90:.3f} s")
            print(f"P95:        {p95:.3f} s")
            print(f"P99:        {p99:.3f} s")

            # Histogram (10 buckets)
            print("\n--- Histogram (10 buckets) ---")
            bucket_count = 10
            if max_v == min_v:
                bucket_size = 1.0
            else:
                bucket_size = (max_v - min_v) / bucket_count

            buckets = [0] * bucket_count
            for d in durations:
                if max_v == min_v:
                    buckets[0] += 1
                else:
                    idx = int((d - min_v) / bucket_size)
                    idx = min(idx, bucket_count - 1)
                    buckets[idx] += 1

            for i in range(bucket_count):
                bucket_min = min_v + i * bucket_size
                bucket_max = bucket_min + bucket_size
                marks = "#" * buckets[i]
                print(f"{bucket_min:5.2f} - {bucket_max:5.2f} s | {marks}")

            # Tasks per duration-second bucket (rough feel of where durations land)
            print("\n--- Durations Per Whole-Second Bucket ---")
            timing_buckets: Dict[int, int] = {}
            for d in durations:
                second = int(d)
                timing_buckets.setdefault(second, 0)
                timing_buckets[second] += 1

            for sec in sorted(timing_buckets.keys()):
                print(f"{sec:2d}s : {timing_buckets[sec]} tasks")
        else:
            print("\nNo successful durations to produce a distribution report.")

        print("\n=== Summary ===")
        print(f"Overall elapsed time: {overall_elapsed:.3f}s")
        print(f"Successful tasks:     {total_successful_tasks}")
        print(f"Failed tasks:         {failed_tasks}")
        print(f"Average task time:    {avg_task_time:.3f}s")

        return results, avg_task_time, failed_tasks, failed_buildings


async def main():
    random.seed(0)

    # Example: 90 batches of 70 tasks
    batches: List[List[Dict[str, Any]]] = []
    for b in range(90):
        batch = []
        for i in range(70):
            batch.append({"building_id": f"batch{b+1}-task{i+1}"})
        batches.append(batch)

    executor = RateLimitedExecutor(
        max_concurrent=len(batches[0]),
        max_new_per_n_seconds=len(batches[0]),
        n_seconds=1.0,
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
