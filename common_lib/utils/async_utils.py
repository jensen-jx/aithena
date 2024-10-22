import asyncio
from typing import List, Any

 
async def producer(tasks_to_do: List[int], q: asyncio.Queue, concurrent_req: int) -> None:
    print(f"Producer started working!")
    for i, task in enumerate(tasks_to_do):
        await q.put((i,task))  # put tasks to Queue

    for _ in range(concurrent_req):
        await q.put((None,None))  # put poison pill to all worker/consumers

    print("Producer finished working!")


async def consumer(
        q: asyncio.Queue,
        s: asyncio.Semaphore,
        results: List[Any]
) -> None:
    while True:
        i, task = await q.get()

        if task is None:  # stop if poison pill was received
            break
 
        async with s:
            print(f"Start task {i}!")
            task_result = await task
            results[i] = task_result
            print(f"Task {i} has completed!")


async def async_run(tasks: List[Any], concurrent_req: int) -> List[Any]:
    q = asyncio.Queue()
    s = asyncio.Semaphore(concurrent_req)
    results = [None] * len(tasks)
    consumers = [
        consumer( q=q, s=s, results=results) 
        for i in range(concurrent_req)
    ]
    await asyncio.gather(producer(tasks_to_do=tasks, q=q, concurrent_req=concurrent_req), *consumers)

    return results