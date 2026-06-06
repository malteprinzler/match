"""File path utility functions."""

from collections.abc import Mapping
import io
import multiprocessing
import os
from pathlib import Path
import shutil
import tempfile
import accelerate
import numpy as np


class RemoteFileWrapper(object):
  """Makes a local temp copy of a file in CNS so it can be changed locally.

  OpenCV and PIL don't have wrappers to handle GFile, so a lot of boiler plate
  copying locally, manipulating the image and writing it back to its location.
  This also cleans up the temp file when done.

  NOTE: Requires you call Close manually to clean up the object or use within
  a context manager.
  """

  def __init__(
      self, filename, should_writeback=True, temp_interface=tempfile.mkstemp
  ):
    self._remote_filename = filename

    _, file_extension = os.path.splitext(os.path.basename(filename))
    unused_file_handle, self._local_tmp_filename = temp_interface(
        file_extension
    )

    # Close the handle so OS can reclaim the disk space when Close() finishes.
    os.closerange(unused_file_handle, unused_file_handle + 1)

    # Determines if we should write back changes operated on the local file
    # to the remote file.
    self._should_writeback = should_writeback

    # This flag will determine if the file has been lazily copied on access.
    # It's possible this object may be constructed, but the filename never
    # accessed and this saves significant I/O time.
    self._copied_locally = False

  def GetFilename(self):
    # Copy only if we intend to use the file.
    if not self._copied_locally and exists(self._remote_filename):
      copy(self._remote_filename, self._local_tmp_filename, True)
      self._copied_locally = True
    return self._local_tmp_filename

  def Flush(self):
    """Writes results back to remote file."""
    if self._should_writeback:
      copy(self._local_tmp_filename, self._remote_filename, True)

  def Close(self):
    """Flushes and deletes temporary file."""
    if self._copied_locally or self._should_writeback:
      self.Flush()
      remove(self._local_tmp_filename)

  def __enter__(self):
    return self

  def __exit__(self, unused_type, unused_value, unused_traceback):
    """We need to flush and delete temporary file on context exit."""
    self.Close()


class AccelerateRemoteDirWrapper(RemoteFileWrapper):
  """Makes a local temp copy of a directory in CNS so it can be changed locally."""

  def __init__(
      self,
      filename: str,
      tmp_filename: str,
      accelerator: accelerate.Accelerator|None,
      should_writeback=True,
  ):
    self._remote_filename = filename
    self._local_tmp_filename = tmp_filename

    # Determines if we should write back changes operated on the local file
    # to the remote file.
    self._should_writeback = should_writeback

    # This flag will determine if the file has been lazily copied on access.
    # It's possible this object may be constructed, but the filename never
    # accessed and this saves significant I/O time.
    self._copied_locally = False

    self._accelerator = accelerator

  def wait_for_everyone(self):
    if self._accelerator is not None:
      self._accelerator.wait_for_everyone()

  @property
  def is_main_process(self):
    if self._accelerator is not None:
      return self._accelerator.is_main_process
    else:
      return True

  def Close(self):
    """Flushes and deletes temporary directory."""
    self.wait_for_everyone()
    if self.is_main_process:
      if self._copied_locally:
        self.Flush()
        delete_recursively(self._local_tmp_filename)
    self.wait_for_everyone()

  def Flush(self):
    """Writes results back to remote file."""
    if self._should_writeback:
      copy(self._local_tmp_filename, self._remote_filename, True)

  def GetFilename(self):
    self.wait_for_everyone()
    # Copy only if we intend to use the file.
    if not self._copied_locally and exists(self._remote_filename):
      # Downloading data to local directory in main process only, others wait
      if self.is_main_process:
        # cleaning up local directory if it exists before downloading
        if exists(self._local_tmp_filename):
          delete_recursively(self._local_tmp_filename)
        makedirs(os.path.dirname(self._local_tmp_filename), exist_ok=True)
        copy(self._remote_filename, self._local_tmp_filename, True)
      self._copied_locally = True
    self.wait_for_everyone()
    return self._local_tmp_filename


def open_file(file_path: Path | str, *args, **kwargs):
  """Opens a file."""
  file_path = Path(file_path)
  return file_path.open(*args, **kwargs)


def get_extension(file_path: Path) -> str:
  """Get file extension from the path."""
  return file_path.suffix


def get_filename(file_path: Path) -> str:
  """Get the filename from the path."""
  return file_path.stem


def get_sub_directories(file_path: Path) -> list[Path]:
  """Get list of paths of all sub directories.

  Args:
    file_path: path of a directory.

  Returns:
    List of paths of all subdirectories.
  """

  sub_dir = [x for x in file_path.iterdir() if x.is_dir()]
  return sub_dir


def list_dir(file_path: Path | str) -> list[str]:
  """Get list of paths of all files in a directory."""
  return os.listdir(str(file_path))


def makedirs(file_path: Path | str, exist_ok: bool = False) -> None:
  """Make directories."""
  os.makedirs(str(file_path), exist_ok=exist_ok)


def is_directory(file_path: Path|str):
  return Path(file_path).is_dir()


def delete_recursively(file_path: Path):
  """Delete a directory and all its contents."""
  shutil.rmtree(str(file_path))


def remove(file_path: Path):
  """Remove a file."""
  os.remove(str(file_path))


def copy_file(
    file_path: Path, target_path: Path, overwrite: bool = False
) -> None:
  """Copy a file, preserving metadata.

  Args:
    file_path: Path of the file to copy.
    target_path: Path of the target file.
    overwrite: Whether to overwrite the target file if it exists.
  """
  if overwrite:
    if target_path.exists():
      target_path.unlink()
  shutil.copy2(str(file_path), str(target_path))


def copy(
    file_path: Path | str, target_path: Path | str, overwrite: bool = False
) -> None:
  """Copy a file or directory."""
  file_path = Path(file_path)
  target_path = Path(target_path)
  if file_path.is_dir():
    if target_path.exists() and overwrite:
      shutil.rmtree(str(target_path))
    shutil.copytree(str(file_path), str(target_path), dirs_exist_ok=True)
  else:
    copy_file(file_path, target_path, overwrite)


def rename(file_path: Path, target_path: Path, overwrite: bool = False) -> None:
  if target_path.exists():
    if not overwrite:
      raise FileExistsError(f"Target {target_path} already exists.")
    if file_path.is_dir():
      delete_recursively(target_path)
    else:
      target_path.unlink()
  os.rename(str(file_path), str(target_path))


def exists(file_path: Path | str) -> bool:
  """Check if a file or directory exists."""
  file_path = Path(file_path)
  return file_path.exists()


def get_resource_filename(resource_path: str) -> str:
  """Get the filename of a resource."""
  return resource_path.strip("google3/")


def glob(file_path: Path | str, pattern: str) -> list[Path]:
  """Glob files.

  Returns:
    List of file paths matching the pattern.
  """
  file_path = Path(file_path)
  return list(file_path.glob(pattern))


def get_sub_folders(
    file_path: Path | str,
    num_parallel_workers: int = 64,
) -> list[str]:
  """Get list of names of all subfolders.

  Args:
    file_path: Path of a directory.
    num_parallel_workers: Number of parallel workers to check if a file is a
      directory.

  Returns:
    List of names of all subfolders.
  """
  file_path = Path(file_path)

  if not is_directory(file_path):
    return []
  list_sub_dir = list_dir(file_path)

  frame_paths = [Path(file_path) / sub_dir for sub_dir in list_sub_dir]

  with multiprocessing.Pool(processes=num_parallel_workers) as pool:
    is_directory_ = pool.map(is_directory, frame_paths)

  return [
      sub_dir for sub_dir, is_dir in zip(list_sub_dir, is_directory_) if is_dir
  ]


# def get_file_paths(
#     file_path: _PathType, file_ext: str = '.ply'
# ) -> list[_PathType]:
#   """Return list of pathnames of files with the file specified extension.

#   Args:
#     file_path: path of the directory.
#     file_ext: specified file extension pattern.

#   Returns:
#     List of file pathnames.
#   """

#   file_paths = list(
#       file_path.glob('*' + file_ext, mode=gpath.GlobMode.NON_RECURSIVE)
#   )
#   return file_paths


def save_npz(output_path: Path | str, data: Mapping[str, np.ndarray]) -> None:
  """Saves the data dictionary to a NumPy ZIP File.

  Saving to `.npz` does not work on CNS, so we save the data in a temporary
  in-memory buffer and then write it to a CNS file.

  Args:
    output_path: The path where the data will be saved.
    data: A dictionary of NumPy arrays to be saved.
  """
  output_path = Path(output_path)
  io_buffer = io.BytesIO()
  np.savez_compressed(io_buffer, **data)
  with open_file(output_path, "wb") as f:
    f.write(io_buffer.getvalue())
