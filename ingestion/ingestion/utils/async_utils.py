import asyncio
from typing import List

async def producer(tasks: List[int], q: asyncio.Queue, k: int) -> None:
    print(f"Producer started working!")
    for task in tasks:
        await q.put(task)  # put tasks to Queue

    # poison pill technique
    for _ in range(k):
        await q.put(None)  # put poison pill to all worker/consumers

    print("Producer finished working!")


async def consumer(
        consumer_name: str,
        q: asyncio.Queue,
        semaphore: asyncio.Semaphore,
) -> None:
    print(f"{consumer_name} started working!")
    while True:
        task = await q.get()

        if task is None:  # stop if poison pill was received
            break

        async with semaphore:
            await task

    print(f"{consumer_name} finished working!")


async def concurrent_calls(tasks: List, k: int) -> None:
    q = asyncio.Queue(maxsize=k)
    s = asyncio.Semaphore(value=k)
    consumers = [
        consumer(
            consumer_name=f"Consumer {i + 1}",
            q=q,
            semaphore=s,
        ) for i in range(k)
    ]
    await asyncio.gather(producer(tasks=tasks, q=q, k=k), *consumers)