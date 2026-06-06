from collections import defaultdict
from src.utils import file_helper


def read_proto(path: str) -> dict:
  out = defaultdict(list)
  with file_helper.open_file(file_helper.Path(path), "rt") as f:
    for line in f.readlines():
      line = line.strip()
      key, value = line.split(": ")
      try:
        value = float(value)
      except Exception:
        pass
      out[key].append(value)

  return out
