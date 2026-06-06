#!/bin/python3

import os
import sys

CONDOR_EXITFILENAME_VARNAME = 'CONDOR_EXITFILE'

if CONDOR_EXITFILENAME_VARNAME in os.environ:
    exit_code_path = os.environ[CONDOR_EXITFILENAME_VARNAME]
    if not os.path.exists(exit_code_path):
        raise FileNotFoundError(f'Couldnt find exit code file {exit_code_path}.')
    else:
        with open(exit_code_path, 'r') as f:
            exit_code = int(f.readline().strip())
    sys.exit(exit_code)
else:
    raise EnvironmentError(f'Environment variable {CONDOR_EXITFILENAME_VARNAME} not defined!')