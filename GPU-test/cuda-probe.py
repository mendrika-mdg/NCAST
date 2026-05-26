import os, json, socket

def get_device_names():
    try:
        import torch
        torch.cuda.current_device()  # force CUDA init
        return [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except Exception as e:
        return f"ERROR: {e}"

if __name__ == "__main__":
    import torch

    print("CUDA available:", torch.cuda.is_available())
    print("CUDA device count:", torch.cuda.device_count())

    print(json.dumps({
        "host": socket.gethostname(),
        "local_rank": os.environ.get("LOCAL_RANK"),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device_count": torch.cuda.device_count(),
        "device_names": get_device_names(),
    }, indent=2))
