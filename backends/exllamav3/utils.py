import os
import platform
from typing import Optional

import torch
from common.logger import xlogger


def exllama_supports_nccl():
    if platform.system() == "Windows":
        unsupported_message = "The NCCL tensor parallel backend is not supported on Windows."
        xlogger.warning(unsupported_message)
        return False

    return torch.cuda.is_available() and torch.distributed.is_nccl_available()


def cpu_moe_host_thread_cap() -> Optional[int]:
    """
    Torch intra-op thread count that fits the cores exllamav3 leaves the
    server process when CPU MoE offload is active, or None when it leaves the
    process unconfined.

    The engine pins this process to EXL3_MOE_HOST_CORES physical cores
    (default 1) once the expert worker starts, keeping the rest for the
    worker's threads.
    """

    try:
        host_cores = int(os.environ.get("EXL3_MOE_HOST_CORES", "1"))
    except ValueError:
        host_cores = 1

    return host_cores if host_cores > 0 else None


def cap_host_torch_threads() -> Optional[int]:
    """
    Shrink torch's intra-op thread pool to the cores CPU MoE offload leaves
    this process. Returns the previous thread count when it was lowered (for
    restoring on unload), otherwise None.

    exllamav3 confines the process without resizing the pool, so any CPU-side
    tensor op big enough to open a parallel region runs a full-size pool on a
    core or two. That stalls for tens of milliseconds; the MTP drafter's input
    layer pays it on every draft, which makes drafting slower than not drafting.
    The expert worker is a separate process with its own pool and is unaffected.
    """

    cap = cpu_moe_host_thread_cap()
    if cap is None:
        return None

    previous = torch.get_num_threads()
    if previous <= cap:
        return None

    torch.set_num_threads(cap)
    xlogger.info(
        f"CPU MoE offload: torch threads in the server process capped at {cap} (was {previous})."
    )
    return previous
