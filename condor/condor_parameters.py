import gin
import dataclasses


@gin.configurable
@dataclasses.dataclass(kw_only=True)
class CondorParameters:
  name: str 
  batch_name: str
  cmd: str
  base_dir: str
  request_gpus: int = 1
  request_cpus: int = 16
  request_memory: int = 64_000
  request_disk: str = "50G"
  requirements: str = '(TARGET.CUDAGlobalMemoryMb >=40000)'
  njobs: int = 1
  environment: dict = dataclasses.field(default_factory=lambda: {})
  backup_root: str | None = None
  backup_excludepatterns: list[str] = dataclasses.field(default_factory=lambda: [])
  retries: int = 0
  
