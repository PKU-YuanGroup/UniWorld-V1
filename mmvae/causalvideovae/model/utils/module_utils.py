import importlib
import torch
import gc

Module = str
MODULES_BASE = "causalvideovae.model.modules."
def resolve_str_to_obj(str_val, append=True):
    if append:
        str_val = MODULES_BASE + str_val
    module_name, class_name = str_val.rsplit('.', 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)

def create_instance(module_class_str: str, **kwargs):
    module_name, class_name = module_class_str.rsplit('.', 1)
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    return class_(**kwargs)

def gpu_memory_test(name, func, *args, **kwargs):
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    result = func(*args, **kwargs)
    print(name, f"{torch.cuda.max_memory_reserved() / 1024 ** 2:.2f} MB")
    return result

def tensor_memory_test(name, tensor):
    memory_usage_bytes = tensor.element_size() * tensor.numel()
    memory_usage_mb = memory_usage_bytes / (1024 ** 2)
    print(name, f"{memory_usage_mb:.2f} MB")