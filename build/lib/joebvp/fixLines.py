######################################################
#
# Script to take parameter file output from VPmeasure,
# set parameter ties to 'fixed', and write new output.
#
######################################################

from __future__ import print_function, absolute_import, division, unicode_literals
from astropy.io import ascii

def fixinput(fname,outfile='fixedVP.dat',fixpars=['n','b','v']):
    inp=ascii.read(fname)
    if isinstance(fixpars,list):
        for col in fixpars:
            if col not in ['n','b','v','N']:
                raise ValueError('Unknown line parameter.  Please use any of \'n\', \'N\', \'b\', \'v\'.')
            else:
                if col=='N': col='n'
                colname=col+'flag'
                inp[colname]=1
    else:
        if fixpars not in ['n', 'b', 'v', 'N']:
            raise ValueError('Unknown line parameter.  Please use any of \'n\', \'N\', \'b\', \'v\'.')
        else:
            if fixpars == 'N': fixpars = 'n'
            colname = fixpars + 'flag'
            inp[colname] = 1
    ascii.write(inp,outfile,delimiter='|')



