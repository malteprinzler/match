import logging


def getLogger(name: str) -> logging.Logger:
  return logging.getLogger(name)


def basicConfig(**kwargs):
  logging.basicConfig(**kwargs)



