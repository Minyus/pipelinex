import logging
from functools import wraps
from typing import Callable

from importlib.util import find_spec

if find_spec("pynvml"):
    from pynvml import (
        nvmlInit,
        nvmlSystemGetDriverVersion,
        nvmlSystemGetNVMLVersion,
        nvmlDeviceGetCount,
        nvmlDeviceGetHandleByIndex,
        nvmlDeviceGetName,
        nvmlDeviceGetMemoryInfo,
        nvmlShutdown,
    )
elif find_spec("py3nvml"):
    from py3nvml.py3nvml import (
        nvmlInit,
        nvmlSystemGetDriverVersion,
        nvmlSystemGetNVMLVersion,
        nvmlDeviceGetCount,
        nvmlDeviceGetHandleByIndex,
        nvmlDeviceGetName,
        nvmlDeviceGetMemoryInfo,
        nvmlShutdown,
    )
else:
    raise ImportError("Install pynvml or py3nvml.")


def _func_full_name(func: Callable):
    return getattr(func, "__qualname__", repr(func))


def get_nv_info():
    nv_info = dict()
    try:
        nvmlInit()

        nv_info["_Driver_Version"] = nvmlSystemGetDriverVersion()
        nv_info["_NVML_Version"] = nvmlSystemGetNVMLVersion()

        device_count = nvmlDeviceGetCount()
        nv_info["Device_Count"] = device_count

        devices = []

        for i in range(device_count):
            dev_info = dict()

            handle = nvmlDeviceGetHandleByIndex(i)
            dev_info["_Name"] = nvmlDeviceGetName(handle)

            memory_info = nvmlDeviceGetMemoryInfo(handle)
            dev_info["Total_Memory"] = memory_info.total
            dev_info["Free_Memory"] = memory_info.free
            dev_info["Used_Memory"] = memory_info.used

            devices.append(dev_info)

        nv_info["Devices"] = devices

        nvmlShutdown()

    except Exception as e:
        nv_info["Exception"] = str(e)

    return nv_info


def nvml_profile(func: Callable) -> Callable:
    @wraps(func)
    def _nvml_profile(*args, **kwargs):
        log = logging.getLogger(__name__)

        init_nv_info = get_nv_info()
        init_devices = init_nv_info.get("Devices", [])
        result = func(*args, **kwargs)
        nv_info = get_nv_info()
        devices = nv_info.get("Devices", [])
        device_count = nv_info.get("Device_Count", 0)

        used_memory_diffs = []
        for i in range(device_count):
            init_used_memory = init_devices[i].get("Used_Memory", 0)
            used_memory = devices[i].get("Used_Memory", 0)
            try:
                used_memory_diff = used_memory - init_used_memory
            except:
                used_memory_diff = None
            used_memory_diffs.append(used_memory_diff)

        log.info(
            "Ran: '{}', NVML returned: {}, Used memory diff: {}".format(
                _func_full_name(func), nv_info, used_memory_diffs,
            )
        )
        return result

    return _nvml_profile
