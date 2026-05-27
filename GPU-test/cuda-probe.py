import os
import json
import socket
import sys
import torch

def check_cuda():
    try:
        torch.cuda.current_device()
        names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        return True, names
    except Exception as e:
        return False, str(e)

ok, result = check_cuda()

print(json.dumps({
    "host": socket.gethostname(),
    "local_rank": os.environ.get("LOCAL_RANK"),
    "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "cuda_available": torch.cuda.is_available(),
    "device_count": torch.cuda.device_count(),
    "cuda_init_ok": ok,
    "device_names_or_error": result,
}, indent=2))

if not ok:
    sys.exit(1)