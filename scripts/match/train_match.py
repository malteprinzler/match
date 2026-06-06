from match.runner import MatchRunner
import argparse
import gin
from match.options import Options
from match.utils import gin_util

def main(argv: list[str]|None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--gin_configs", action="append", default=[])
    parser.add_argument("--gin_bindings", action="append", default=[])
    FLAGS = parser.parse_args(argv)

    # explicitly initialize the warm pool as the first line of main!
    gin.parse_config_files_and_bindings(
        config_files=FLAGS.gin_configs, bindings=None, skip_unknown=True
    )
    opt = Options()
    runner = MatchRunner(opt)
    runner.prepare_training(save_config_files=FLAGS.gin_configs)
    runner.run_training()
    runner.graceful_exit(0)



if __name__ == "__main__":
  main()
