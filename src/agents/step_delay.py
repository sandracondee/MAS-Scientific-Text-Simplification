"""Small helper utilities to slow down graph steps for readability."""

import asyncio
import time


STEP_DELAY_SECONDS = 0.5


def pause_step_sync(seconds: float = STEP_DELAY_SECONDS) -> None:
    """Block briefly after a synchronous graph node finishes."""
    time.sleep(seconds)


async def pause_step_async(seconds: float = STEP_DELAY_SECONDS) -> None:
    """Await briefly after an asynchronous graph node finishes."""
    await asyncio.sleep(seconds)