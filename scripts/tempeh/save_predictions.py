
import os
import sys

sys.path.append('third_party/TEMPEH')
from trainer.global_trainer import save_uvmaps
from option_handler.train_options_global import TrainOptions

def get_date_string():
    from datetime import datetime
    mydate = datetime.now()
    return '%s%02d__%02d-%02d-%02d' % (mydate.strftime("%B"), mydate.day, mydate.hour, mydate.minute, mydate.second)

def execute_locally(config_fname, outpath):
    process_idx = int(os.environ.get('CONDOR_Process', '0'))
    world_size = int(os.environ.get('CONDOR_WORLD_SIZE', '1'))
    save_uvmaps(config_fname=config_fname, outpath=outpath, process_idx=process_idx, world_size=world_size)

def main():
    parser = TrainOptions()
    config_args = parser.parse()
    config_fname = config_args.config_filename
    outpath = config_args.outpath
    execute_locally(config_fname, outpath)

if __name__ == '__main__':
    main()
    print('Done')



