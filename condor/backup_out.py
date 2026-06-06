#!/bin/python3

import os
from datetime import datetime
import shutil


CONDOR_OUTFILENAME_VARNAME = 'CONDOR_OUTFILE'


def timestamp_str():
    return datetime.now().strftime("%m_%d-%H_%M")

if CONDOR_OUTFILENAME_VARNAME in os.environ:
    outfile_path = os.environ[CONDOR_OUTFILENAME_VARNAME]
    if os.path.exists(outfile_path):
        backup_path = outfile_path + '-backup-' + timestamp_str() + '.txt'
        shutil.copy(outfile_path, backup_path)
        print(f'Copied from {outfile_path} to {backup_path}')
    else:
        print(f'Couldnt find {outfile_path}')
else:
    print(f'Environment variable {CONDOR_OUTFILENAME_VARNAME} not defined.')