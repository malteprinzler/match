from gin.config import configurable


@configurable
def join_strings(strings: list[str], separator: str = ''):
  """Simple gin configurable function to join two strings.

  This function is used within gin files to join strings such as:
  file_pattern = @string_join()
  string_join.left = "/cns/.../data_dir"
  string_join.right = "*.sstable"
  string_join.separator = "/"

  Args:
    left: The string left of the separator.
    right: The string left of the separator.
    separator: The string separator between the two strings.

  Returns:
    The joined string.
  """
  return separator.join(map(str, strings))

