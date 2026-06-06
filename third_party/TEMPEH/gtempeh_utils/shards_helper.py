# GOOGLE:
# from google3.file.base.python import shards

from src.utils import file_helper


def GenerateShardedFilenames(pattern: str) -> list[str]:
  # GOOGLE: return shards.GenerateShardedFilenames(str(p))

  pattern_path = file_helper.Path(pattern.replace("@*", "-*"))
  pattern_parent = pattern_path.parent
  pattern_name = pattern_path.name
  pattern_files = sorted(pattern_parent.glob(pattern_name))
  pattern_files = [str(x) for x in pattern_files]

  return pattern_files
