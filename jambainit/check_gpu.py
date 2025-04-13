import pynvml

def get_gpu_with_max_free_memory():
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()

    max_free_mem = 0
    selected_gpu = 0

    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        free_mem = mem_info.free
        print(f"GPU {i}: Free memory = {free_mem / 1024**2:.2f} MB")

        if free_mem > max_free_mem:
            max_free_mem = free_mem
            selected_gpu = i

    pynvml.nvmlShutdown()
    return selected_gpu

best_gpu = get_gpu_with_max_free_memory()
print(f"\nSelected GPU with most free memory: GPU {best_gpu}")