"""
GPU 推理队列管理

单卡使用 asyncio.Queue 串行化所有推理请求，
避免多个 forward 同时抢显存导致 OOM 或延迟抖动。
"""

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Tuple

logger = logging.getLogger(__name__)

_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gpu-worker")


class InferenceQueue:
    """
    异步推理队列。

    所有推理请求通过 ``submit`` 入队，后台 worker 逐个消费。
    底层推理函数在独立线程池中运行，不阻塞事件循环。
    """

    def __init__(self, maxsize: int = 32):
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
        self._worker_task: asyncio.Task | None = None

    async def start(self):
        self._worker_task = asyncio.create_task(self._worker_loop())
        logger.info("GPU 推理队列 worker 已启动")

    async def stop(self):
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            logger.info("GPU 推理队列 worker 已停止")

    @property
    def qsize(self) -> int:
        return self._queue.qsize()

    async def submit(
        self,
        fn: Callable[..., Any],
        args: Tuple = (),
        timeout: float = 60.0,
    ) -> Any:
        """
        提交推理任务并等待结果。

        Raises:
            asyncio.TimeoutError: 排队 + 推理总耗时超过 timeout
            Exception: 推理函数抛出的原始异常
        """
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        item = (fn, args, future)

        try:
            await asyncio.wait_for(self._queue.put(item), timeout=timeout)
        except asyncio.TimeoutError:
            raise asyncio.TimeoutError("推理队列已满，请稍后重试")

        return await asyncio.wait_for(future, timeout=timeout)

    async def _worker_loop(self):
        loop = asyncio.get_running_loop()
        while True:
            fn, args, future = await self._queue.get()
            if future.cancelled():
                self._queue.task_done()
                continue
            try:
                result = await loop.run_in_executor(_executor, fn, *args)
                if not future.cancelled():
                    future.set_result(result)
            except Exception as exc:
                if not future.cancelled():
                    future.set_exception(exc)
            finally:
                self._queue.task_done()
