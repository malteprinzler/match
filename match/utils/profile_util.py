import torch

def peak_gpu_memory_usage() -> tuple[float, float]:
  """Returns the currently allocated and the total reserved GPU memory."""
  if not torch.cuda.is_available():
    return 0.0, 0.0

  pytorch_allocated = torch.cuda.memory_allocated() // (1024**3)
  pytorch_reserved = torch.cuda.memory_reserved() // (1024**3)

  allocated_memory = pytorch_allocated
  reserved_memory = pytorch_reserved
  return allocated_memory, reserved_memory