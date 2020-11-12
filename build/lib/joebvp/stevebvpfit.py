#!/usr/bin/env python
# coding: utf-8

#######################################################################
###
###  params format (each entry a list): [restwave,coldens,bval,z,vel]
###  restwave (of an actual line), column density, bval - width of Gaussian, z (bulk), velocity (relative to z)
###
#######################################################################

from __future__ import print_function, absolute_import, division, unicode_literals

from PyQt5 import QtGui, QtCore

from joebvp import joebgoodies as jbg
import numpy as np
from scipy.signal import convolve
from scipy.special import wofz
from scipy.optimize import least_squares
import scipy.stats
from joebvp.atomicdata import atomicdata
from astropy import constants as const
import sys, os
try:
    import joebvp_cfg as cfg
except:
    print(
        "joebvp.makevoigt: No local joebvp_cfg.py found, using default cfg.py file from joebvp."
    )
    from joebvp import cfg
from sklearn.cluster import MeanShift, estimate_bandwidth
import astropy.units as u
from astropy.io import ascii
from astropy.table import Table
from linetools import utils as ltu
from linetools.spectra.lsf import LSF
import pandas as pd
from glob import glob
from joebvp import VPmeasure
from joebvp import makevoigt
from joebvp import nmpfit

ln2 = np.log(2)
c = const.c.to('km/s').value

from linetools.spectra.xspectrum1d import XSpectrum1D

from scipy.optimize import leastsq

# directly lifted from joebvpfit:


def foldpars(pars, numpars=5):
    rows = len(pars) / numpars
    fpars = []
    for i in np.arange(numpars, dtype='int'):
        fparrow = []
        for j in np.arange(rows, dtype='int'):
            fparrow.append(pars[i + numpars * j])
        fpars.append(fparrow)
    return fpars


def unfoldpars(pars, numpars=5):
    rows = len(pars[0])
    ufpars = []
    for j in range(rows):
        for i in np.arange(numpars, dtype='int'):
            ufpars.append(pars[i][j])
    return ufpars


def voigtfunc(vwave, vpars):
    ### Check to see if cfg variables are set
    if isinstance(cfg.fitidx, int) | isinstance(cfg.wave, int):
        cfg.fitidx = fitpix(vwave, vpars)
        cfg.wave = vwave
    if len(cfg.lsfs) == 0:
        makevoigt.get_lsfs()

    vflux = np.zeros(len(vwave)) + 1.
    ### redo voigt function w/ numpy?
    factor = makevoigt.voigt(vwave, vpars[0], vpars[1], vpars[2], vpars[3],
                             vpars[4])
    convfactor = makevoigt.convolvecos(vwave, factor, vpars[0], vpars[3])
    vflux *= convfactor

    return vflux


def readpars(filename, wave1=None, wave2=None):
    '''

    Parameters
    ----------
    filename : str
        Name of parameter input (or joebvp output) file
        File should at least have following columns:
            specfile|restwave|zsys|col|sigcol|bval|sigbval|vel|sigvel|nflag|bflag|vflag|vlim1|vlim2
    wave1 : float, optional
        Beginning of wavelength range over which to load lines (must be set with wave2)
    wave2 : float, optional
        End of wavelength range over which to load lines (must be set with wave1)

    Returns
    -------
    fitpars : list of lists
        Parameters for fit ready for fitter!
    fiterrors : array of numpy vectors
        Error array for the fitting initialized to '0' for each param
    parinfo : array of arrays
        Flags to be used in fit
    linecmts: list of lists
        Reliability and comment flags, e.g., from igmguesses
    '''
    linelist = ascii.read(filename)
    linerestwave = linelist['restwave'].data
    linez = linelist['zsys'].data
    if (wave1 == None) & (wave2 == None):
        lineshere = np.arange(len(linelist))
    elif ((wave1 == None) | (wave2 == None)) | (wave1 >= wave2):
        lineshere = np.arange(len(linelist))
        warnings.warn(
            'Note that both \'wave1\' and \'wave2\' must be declared or neither must be. \n Loading all lines in list.'
        )
    else:
        lineobswave = linerestwave * (1. + linez)
        lineshere = np.where((lineobswave > wave1) & (lineobswave < wave2))[0]
    linelist = linelist[lineshere]
    linelist['ions'] = atomicdata.lam2ion(linelist['restwave'])

    linelist.sort(['ions', 'zsys', 'vel', 'restwave'])

    linerestwave = linelist['restwave']
    zs = linelist['zsys']
    linecol = linelist['col']
    lineb = linelist['bval']
    linevel = linelist['vel']
    linevlim1 = linelist['vlim1']
    linevlim2 = linelist['vlim2']
    colflag = linelist['nflag']
    bflag = linelist['bflag']
    velflag = linelist['vflag']
    restwaves = linerestwave
    if (('rely' in linelist.colnames) & ('comment' in linelist.colnames)):
        pass
    elif ('rely' in linelist.colnames):
        linelist['comment'] = ['none'] * len(linelist)
    else:
        linelist['rely'] = ['-'] * len(linelist)
        linelist['comment'] = ['none'] * len(linelist)

    reliability = linelist['rely']
    comment = linelist['comment']
    initinfo = [colflag, bflag, velflag]
    initpars = [restwaves, linecol, lineb, zs, linevel, linevlim1, linevlim2]
    fitpars, parinfo = initlinepars(zs, restwaves, initpars, initinfo=initinfo)
    fiterrors = np.zeros([5, len(fitpars[0])])  # Initialize errors to zero
    #fiterrors[1] = colsig
    #fiterrors[2] = bsig
    #fiterrors[4] = velsig
    linecmts = [reliability, comment]

    return fitpars, fiterrors, parinfo, linecmts


def initlinepars(zs, restwaves, initvals=[], initinfo=[]):
    '''

    Parameters
    ----------
    zs : numpy vector of floats
        Redshifts of lines (this parameter will be fixed during fitting)
    restwaves : numpy vector of floats
        Rest frame wavelengths of lines to be fitted
    initvals : list of numpy vectors, optional
        Contains the following (in order): [restwaves,linecol,lineb,zs,linevel,linevlim1,linevlim2]
        Will default to values set in cfg.py if not set
    initinfo : list of numpy vectors, optional
        Contains the flags for fitting (in order): [colflag,bflag,velflag]
        Parameters with flags = 0 and 1 will freely value and fixed, respectively
        If 2 or more lines have the same flag value for the same parameter, the parameters will
            be tied to one another.

    Returns
    -------
    initpars : list of lists
        Parameters for fit ready for fitter!
    parinfo : array of arrays
        Flags to be used in fit
    '''

    ### Set atomic data for each line
    lam, fosc, gam = atomicdata.setatomicdata(restwaves)
    cfg.lams = lam
    cfg.fosc = fosc
    cfg.gam = gam

    initpars = [[], [], [], [], [], [], []]
    defaultcol = cfg.defaultcol
    defaultb = cfg.defaultb
    if initvals == []:
        for i in range(len(restwaves)):
            initpars[0].extend([restwaves[i]])
            initpars[1].extend([defaultcol])
            initpars[2].extend([defaultb])
            initpars[3].extend([zs[i]])
            initpars[4].extend([0.])
            initpars[5].extend([-cfg.defaultvlim])
            initpars[6].extend([cfg.defaultvlim])
    else:
        if len(initvals) == 5:
            for i in range(len(restwaves)):
                initpars[0].extend([initvals[0][i]])
                initpars[1].extend([initvals[1][i]])
                initpars[2].extend([initvals[2][i]])
                initpars[3].extend([initvals[3][i]])
                initpars[4].extend([initvals[4][i]])
                initpars[5].extend([-cfg.defaultvlim])
                initpars[6].extend([cfg.defaultvlim])
        else:
            initpars = [[], [], [], [], [], [], []]
            for i in range(len(restwaves)):
                initpars[0].extend([initvals[0][i]])
                initpars[1].extend([initvals[1][i]])
                initpars[2].extend([initvals[2][i]])
                initpars[3].extend([initvals[3][i]])
                initpars[4].extend([initvals[4][i]])
                initpars[5].extend([initvals[5][i]])
                initpars[6].extend([initvals[6][i]])

    ### If hard limits on Doppler b-value are smaller or greater than cfg.lowblim or cfg.upperblim,
    ### modify those limits
    maxb = np.max(initpars[2][:])
    minb = np.min(initpars[2][:])
    if maxb > cfg.upperblim:
        cfg.upperblim = maxb + 10.
    if minb < cfg.lowblim: cfg.lowblim = minb - 2.

    parinfo = np.zeros([5, len(restwaves)], dtype=int)
    parinfo[0] = parinfo[0] + 1
    parinfo[3] = parinfo[3] + 1
    if ((initinfo == []) & (initvals == [])):

        ### Look for multiplet membership of each line
        seriesassoc = np.zeros(len(restwaves)) - 99
        for i in range(len(restwaves)):
            for j in range(len(cfg.multiplets)):
                currmult = np.array(cfg.multiplets[j])
                if (abs(restwaves[i] -
                        currmult[jbg.closest(currmult, restwaves[i])]) < 0.01):
                    seriesassoc[i] = j

        uqions = np.unique(seriesassoc).tolist()

        if -99 in uqions: uqions.remove(-99)
        flagctr = 2
        for uqion in uqions:
            ionmatch = np.where(seriesassoc == uqion)[0]
            uqzs = np.unique(zs[ionmatch])
            for uz in uqzs:
                matchcrit = (zs == uz) & (seriesassoc == uqion)
                rellines = np.where(matchcrit)[0]
                uqlams = np.unique(restwaves[rellines])
                if len(uqlams) > 1:
                    complist = []
                    numcomps = []
                    for ul in uqlams:
                        matchcrit2 = matchcrit & (restwaves == ul)
                        matches = np.where(matchcrit2)[0]
                        complist.append(matches)
                        numcomps.append(len(matches))
                    numcomps = np.array(numcomps)
                    complist = np.array(complist)
                    compidxsort = sorted(range(len(numcomps)),
                                         key=lambda x: numcomps[x],
                                         reverse=True)
                    numcompsort = numcomps[compidxsort]
                    complistsort = complist[compidxsort]
                    maxcomps = numcompsort[0]
                    for compidx in range(maxcomps):
                        for li in range(len(complistsort)):
                            if compidx < numcompsort[li]:
                                parinfo[1][complistsort[li][compidx]] = flagctr
                                parinfo[2][complistsort[li][compidx]] = flagctr
                                parinfo[4][complistsort[li][compidx]] = flagctr
                            else:
                                continue
                        flagctr += 1

    elif initinfo != []:
        parinfo[1] = initinfo[0]
        parinfo[2] = initinfo[1]
        parinfo[4] = initinfo[2]
    elif ((initinfo == []) & (initvals != [])):

        ### Look for multiplet membership of each line
        seriesassoc = np.zeros(len(restwaves)) - 99
        for i in range(len(restwaves)):
            for j in range(len(cfg.multiplets)):
                currmult = np.array(cfg.multiplets[j])
                if (abs(restwaves[i] -
                        currmult[jbg.closest(currmult, restwaves[i])]) < 0.01):
                    seriesassoc[i] = j

        ### Fix measurements that are imported
        for i in range(len(restwaves)):
            if ((initpars[1][i] != defaultcol) & (initpars[2][i] != defaultb)):
                parinfo[1][i] = 1
                parinfo[2][i] = 1
                parinfo[4][i] = 1
        uqions = np.unique(seriesassoc).tolist()
        if -99 in uqions: uqions.remove(-99)
        flagctr = 2
        for uqion in uqions:
            ionmatch = np.where(seriesassoc == uqion)[0]
            uqzs = np.unique(zs[ionmatch])
            for uz in uqzs:
                matchcrit = (zs == uz) & (seriesassoc == uqion)
                rellines = np.where(matchcrit)[0]
                uqlams = np.unique(restwaves[rellines])
                if len(uqlams) > 1:
                    complist = []
                    numcomps = []
                    for ul in uqlams:
                        matchcrit2 = matchcrit & (restwaves == ul)
                        matches = np.where(matchcrit2)[0]
                        complist.append(matches.tolist())
                        numcomps.append(len(matches))
                    numcomps = np.array(numcomps)
                    complist = np.array(complist)
                    compidxsort = sorted(range(len(numcomps)),
                                         key=lambda x: numcomps[x])
                    complistsort = complist[compidxsort]
                    complistsort = complistsort.tolist()
                    for i in range(len(complistsort) - 1):
                        for idx in complistsort[i]:
                            parinfo[1][idx] = flagctr
                            parinfo[2][idx] = flagctr
                            parinfo[4][idx] = flagctr
                            for j in range(i + 1, len(complistsort)):
                                idxidx = jbg.closest(
                                    initvals[4][complistsort[j]],
                                    initvals[4][idx])
                                clidx = complistsort[j][idxidx]
                                parinfo[1][clidx] = flagctr
                                parinfo[2][clidx] = flagctr
                                parinfo[4][clidx] = flagctr
                                complistsort[j].remove(clidx)
                            flagctr += 1

    return initpars, parinfo


def prepparinfo(linepars, parflags):
    parinfo = []
    parflags = np.array(parflags)
    numpars = 5
    for i in range(len(parflags[0])):
        parinfo.extend([{
            'fixed': 1
        }, {
            'fixed': 0
        }, {
            'fixed': 0
        }, {
            'fixed': 1
        }, {
            'fixed': 0
        }])
        for j in range(1, len(parflags)):
            if parflags[j][i] == 1: parinfo[i * numpars + j]['fixed'] = 1
            elif parflags[j][i] <= 0: parinfo[i * numpars + j]['fixed'] = 0
            else:
                matches = np.where(np.array(parflags[j]) == parflags[j][i])[0]
                if matches[0] != i:
                    tiedpar = int(matches[0] * numpars + j)
                    parinfo[i * numpars +
                            j]['tied'] = 'p[' + str(tiedpar) + ']'
        col = round(linepars[1][i], 2)
        vel = round(linepars[4][i], 2)
        bpar = round(linepars[2][i], 2)
        parinfo[numpars * i + 1]['limited'] = [1, 1]
        parinfo[numpars * i +
                1]['limits'] = [round(col - 5., 2),
                                round(col + 5., 2)]
        parinfo[numpars * i + 2]['limited'] = [1, 1]
        ### adjust b-value limits to allow for broadened HI features
        lydiff = abs(linepars[0][i] - cfg.lyseries)
        lymatch = np.where(abs(lydiff) <= 0.05)[0]
        if lymatch:
            parinfo[numpars * i + 2]['limits'] = [
                max([cfg.lowblim, bpar - 10.]),
                min([bpar + 10, cfg.upperblim_HI])
            ]
        else:
            parinfo[numpars * i + 2]['limits'] = [
                max([cfg.lowblim, bpar - 10.]),
                min([bpar + 10, cfg.upperblim])
            ]
        parinfo[numpars * i + 2]['step'] = 0.5
        parinfo[numpars * i + 2]['mpside'] = 2
        #parinfo[numpars*i+2]['relstep']=0.0001
        parinfo[numpars * i + 4]['limited'] = [1, 1]
        ### Allow velocity to flop around
        if parflags[4][i] < 0:
            flopamt = abs(parflags[4][i])
            parinfo[numpars * i + 4]['limits'] = [
                round(vel - flopamt, 2),
                round(vel + flopamt, 2)
            ]
        elif len(linepars) > 5:
            v1 = round(linepars[5][i], 2)
            v2 = round(linepars[6][i], 2)
            parinfo[numpars * i + 4]['limits'] = [v1, v2]
        else:
            parinfo[numpars * i +
                    4]['limits'] = [round(vel - 50., 2),
                                    round(vel + 50., 2)]
        parinfo[numpars * i + 4]['step'] = 1.
        #parinfo[numpars*i+4]['relstep']=0.01
    return parinfo


def fitpix(wave, pararr, find_bad_pixels=True):

    if find_bad_pixels:
        # define bad pixels
        cfg.bad_pixels = update_bad_pixels(
        )  # this variable stores the indices of bad pixels
    else:
        cfg.bad_pixels = []

    ll = pararr[0]
    lz = pararr[3]
    lv1 = pararr[5]
    lv2 = pararr[6]
    relpix = []
    for i in range(len(ll)):
        vels = jbg.veltrans(lz[i], wave, ll[i])
        p1 = jbg.closest(vels, lv1[i])
        p2 = jbg.closest(vels, lv2[i])

        if ((p1 >= 10) & (p2 <= (len(wave) - 1 - 10))):
            relpix.extend(range(p1 - 10, p2 + 10))
        elif (p1 < 10):
            relpix.extend(range(0, p2 + 10))
        else:
            relpix.extend(range(p1 - 10, len(wave) - 1))
    rp = np.unique(np.array(relpix))
    clean_rp = np.array([i for i in rp if i not in cfg.bad_pixels])
    return clean_rp


def stevevoigterrfunc(x, xall0, notfixed, indices, wavelength, flux, sig):
    xall = xall0.copy()

    xall[notfixed] = x[indices]

    folded_xall = foldpars(xall)
    model = voigtfunc(wavelength, folded_xall)
    status = 0

    residuals = (flux[cfg.fitidx] - model[cfg.fitidx]) / sig[cfg.fitidx]

    try:
        residuals = residuals.value
    except AttributeError:
        pass
    return (residuals)


def update_bad_pixels():
    # define bad pixels
    cond_badpix = (cfg.spectrum.wavelength <= cfg.spectrum.wvmin) | \
                  (cfg.spectrum.wavelength >= cfg.spectrum.wvmax) | \
                  (cfg.spectrum.sig <= 0)
    #(cfg.spectrum.flux / cfg.spectrum.sig < cfg.min_sn)  # bad S/N
    # spectral gaps
    for gap in cfg.spectral_gaps:
        cond_gap = (cfg.spectrum.wavelength >=
                    gap[0] * u.AA) & (cfg.spectrum.wavelength <= gap[1] * u.AA)
        cond_badpix = cond_badpix | cond_gap
    bad_pixels = np.where(cond_badpix)[0]
    return bad_pixels


def various_indices(parinfo):
    """
    Parameters:
    -----------

    parinfo : list of dictionaries

    Returns:
    --------

    pfixed : list of integers in numpy array
        fixed/unfixedness of parameters.
    ptied : list of strings in numpy array
        tiedness or lack thereof of parameters.
    ifree : list of integers in numpy array
        indices of free parameters.
    indices : list of floats in numpy array
        index indicating which other tied parameter a given parameter is tied to.

    """

    npar = len(parinfo)

    indices = np.zeros([npar])

    ## FIXED parameters ?
    pfixed = parameterInformationFunction(parinfo, 'fixed', default=0, n=npar)

    ptied = parameterInformationFunction(parinfo, 'tied', default='', n=npar)

    qanytied = 0
    for i in range(npar):
        ptied[i] = ptied[i].strip()
        if (ptied[i] != ''): qanytied = 1

    kk = 0
    for ii in range(len(indices)):
        if pfixed[ii] == 1:
            continue
        else:
            if ptied[ii] == '':

                indices[ii] = kk

                kk = kk + 1
            else:
                ll = int(ptied[ii][2:-1])
                indices[ii] = indices[ll]

    ## Finish up the free parameters
    ifree = np.where(np.array(pfixed) == 0)
    nfree = len(ifree)
    if nfree == 0:
        raise ValueError('no free parameters.')

    inotfixed = np.where(np.array(pfixed) == 0)

    indices = indices[inotfixed]

    for i in range(npar):
        pfixed[i] = pfixed[i] or (
            ptied[i] != '')  ## Tied parameters are also effectively fixed
    return (pfixed, ptied, ifree, indices)


def parameterInformationFunction(parinfo=None, key='a', default=None, n=0):
    #if (self.debug): print('Entering parinfo...')

    # makes sure n is the length of parinfo:
    if (n == 0) and (parinfo is not None): n = len(parinfo)

    # returns nothing if parinfo is empty:
    if (n == 0):
        values = default
        return (values)

    # makes a list of the values of the specified key/item for each parameter
    values = []
    for i in range(n):
        if ((parinfo is not None) and (key in parinfo[i])):
            values.append(parinfo[i][key])
        else:
            values.append(default)

    # Convert to numeric arrays if possible
    test = default
    if (type(default) == list): test = default[0]
    if (type(test) == int):
        values = np.asarray(values, dtype=np.int)
    elif (type(test) == float):
        values = np.asarray(values, dtype=np.float)

    return (values)


# here are the functions needed for calculating fit errors in stevebvpfit:


def calc_covar(rr, ipvt=None, tol=1.e-14):

    if np.rank(rr) != 2:
        print('ERROR: r must be a two-dimensional matrix')
        return (-1)
    s = np.shape(rr)
    n = s[0]
    if s[0] != s[1]:
        print('ERROR: r must be a square matrix')
        return (-1)

    if (ipvt is None): ipvt = np.arange(n)
    r = rr.copy()
    r.shape = [n, n]

    ## For the inverse of r in the full upper triangle of r
    l = -1
    tolr = tol * abs(r[0, 0])
    for k in range(n):
        if (abs(r[k, k]) <= tolr): break
        r[k, k] = 1. / r[k, k]
        for j in range(k):
            temp = r[k, k] * r[j, k]
            r[j, k] = 0.
            r[0:j + 1, k] = r[0:j + 1, k] - temp * r[0:j + 1, j]
        l = k

    ## Form the full upper triangle of the inverse of (r transpose)*r
    ## in the full upper triangle of r
    if l >= 0:
        for k in range(l + 1):
            for j in range(k):
                temp = r[j, k]
                r[0:j + 1, j] = r[0:j + 1, j] + temp * r[0:j + 1, k]
            temp = r[k, k]
            r[0:k + 1, k] = temp * r[0:k + 1, k]

    ## For the full lower triangle of the covariance matrix
    ## in the strict lower triangle or and in wa
    wa = np.repeat([r[0, 0]], n)
    for j in range(n):
        jj = ipvt[j]
        sing = j > l
        for i in range(j + 1):
            if sing: r[i, j] = 0.
            ii = ipvt[i]
            if ii > jj: r[ii, jj] = r[i, j]
            if ii < jj: r[jj, ii] = r[i, j]
        wa[jj] = r[j, j]

    ## Symmetrize the covariance matrix in r
    for j in range(n):
        r[0:j + 1, j] = r[j, 0:j + 1]
        r[j, j] = wa[j]

    return (r)


def calc_perrors(jac, ifree, numpars=5):
    n = len(ifree)
    try:
        rr = np.linalg.inv(jac.T @ jac)
    except np.linalg.LinAlgError:
        # idk should I do this?
        rr = np.linalg.pinv(jac.T @ jac)
        print('using pseudo inverse for Jacobian')
    covar = np.zeros([numpars, numpars], np.float)

    for i in range(n):
        idx = ifree + ifree[i] * numpars
        np.put(covar, idx, rr[:, i])

    d = np.sqrt(np.diagonal(covar))

    return (d)


# The main event:
def stevebvpfit(wave, flux, sig, flags, linepars=None, xall=None):

    if (linepars is None):
        raise ValueError('Must pass parameters in linepars.')
        return

    # Only feed to the fitter the parameters that go into the model
    partofit = linepars[:5]

    parinfo = prepparinfo(partofit, flags)

    npar = len(parinfo)

    ## Be sure that PARINFO is of the right type
    if (parinfo is not None):
        if (type(parinfo) != list):
            raise ValueError('PARINFO must be a list of dictionaries.')
            return

        else:
            if (type(parinfo[0]) != dict):
                raise ValueError('PARINFO must be a list of dictionaries.')
                return

    ## If the parameters were not specified at the command line, then
    ## extract them from PARINFO

    if (xall is None):

        xall = parameterInformationFunction(parinfo, 'value')
        if (xall is None):
            raise ValueError(
                'either xall or PARINFO(*)["value"] must be supplied.')
    xall = np.ravel(np.array(xall).T)

    # function that deals with tied and fixed parameters and creates
    # indices list to go from non-redundant to redundant array:

    indices_and_friends = various_indices(parinfo)

    pfixed, ptied, ifree, indices = indices_and_friends

    indices = indices.astype(np.int)
    ifree = np.squeeze(ifree)
    ## Compose only VARYING parameters
    x = np.squeeze(xall)[np.squeeze(
        pfixed == 0)]  ## x is the set of free parameters

    ## LIMITED parameters ?
    limited = parameterInformationFunction(parinfo,
                                           'limited',
                                           default=[0, 0],
                                           n=npar)
    limits = parameterInformationFunction(parinfo,
                                          'limits',
                                          default=[0., 0.],
                                          n=npar)

    if (limited is not None) and (limits is not None):

        ## Error checking on limits in parinfo
        wh = np.nonzero((limited[:, 0] & (xall < limits[:, 0]))
                        | (limited[:, 1] & (xall > limits[:, 1])))

        if (len(wh[0]) > 0):
            raise ValueError('parameters are not within PARINFO limits')
            return

        wh = np.nonzero((limited[:, 0] & limited[:, 1])
                        & (limits[:, 0] >= limits[:, 1]) & (pfixed == 0))
        if (len(wh[0]) > 0):
            raise ValueError('PARINFO parameter limits are not consistent')
            return
        freeanduntied = np.where(np.squeeze(pfixed == 0))[0]
        ## Transfer structure values to local variables
        qulim = np.take(limited[:, 1], freeanduntied)
        ulim = np.take(limits[:, 1], freeanduntied)
        qllim = np.take(limited[:, 0], freeanduntied)
        llim = np.take(limits[:, 0], freeanduntied)

    # Save the velocity windows to add back to the parameter array
    vlim1 = linepars[5]
    vlim2 = linepars[6]
    # Get atomic data
    lam, fosc, gam = atomicdata.setatomicdata(linepars[0])
    cfg.lams = lam
    cfg.fosc = fosc
    cfg.gam = gam
    # Set fit regions
    cfg.fitidx = fitpix(wave, linepars)

    # Prep parameters for fitter

    partofit = unfoldpars(partofit)

    modelvars = {'l': wave, 'y': flux, 'err': sig}

    bnds = (np.squeeze(np.where(qllim == 1, llim, -np.inf)),
            np.squeeze(np.where(qulim == 1, ulim, np.inf)))

    arg = [
        xall, ifree, indices, modelvars['l'], modelvars['y'], modelvars['err']
    ]

    # here is where fitting happens:

    m = least_squares(stevevoigterrfunc,
                      x,
                      bounds=bnds,
                      args=arg,
                      kwargs={},
                      verbose=True)

    if m.status < 0: print('Fitting error:', m.message)

    xall_fitted = xall.copy()

    xall_fitted[ifree] = m['x'][indices]

    fitpars = foldpars(xall_fitted)

    # This is a super-hacky fix to the issue of parameters having Jacobians of 0.
    # It could obscure other problems and so it would make sense to replace it
    # at some point.
    badvarmask = np.all(np.isclose(m.jac, 1e-10), axis=0)
    if np.any(badvarmask):
        try:
            bad_var_ind, = np.where(badvarmask)
            # Replacing bad columns with value near 0.
            m.jac[:, bad_var_ind] = 1e-5
            # Giving rest wavelength and redshift of bad lines. Also really janky in and
            # of itself so that's fun.
            bad_var_indices_mask = np.in1d(indices, bad_var_ind)
            bad_ifree = ifree[bad_var_indices_mask]
            bad_comp_inds = np.unique(bad_ifree // 5)
            for comp in bad_comp_inds:
                bad_wav = fitpars[0][comp]
                bad_red = fitpars[3][comp]
                print('Bad component of {0:.5f} at {1:.5f}.'.format(
                    np.float(bad_wav), np.float(bad_red)))
        except IndexError:
            pass

    perr = calc_perrors(m.jac, freeanduntied, numpars=xall.size)

    fitperr = foldpars(perr)

    for par in perr:
        par = np.ravel(np.array(par))

    # Adding in reduced chi-squared bit:
    rchi2 = 2 * m.cost / (len(cfg.fitidx) - len(freeanduntied))
    # Add velocity windows back to parameter array
    fitpars.append(vlim1)
    fitpars.append(vlim2)

    # prints results in terminal/notebook/whatever thing you have python running from
    print('\nFit results: \n')
    for i in range(len(fitpars[0])):
        print(
            jbg.tabdelimrow([
                round(fitpars[0][i], 2),
                jbg.decimalplaces(fitpars[3][i], 5),
                jbg.roundto(fitpars[1][i], 5),
                jbg.roundto(fitpars[2][i], 5),
                jbg.roundto(fitpars[4][i], 5)
            ])[:-2])
        print(
            jbg.tabdelimrow([
                ' ', ' ', ' ',
                round(fitperr[1][i], 3),
                round(fitperr[2][i], 3),
                round(fitperr[4][i], 3)
            ]))
    print('\nReduced chi-squared: {0:f}'.format(rchi2))
    return fitpars, fitperr, rchi2


# ... the other main event:
def fit_to_convergence(wave,
                       flux,
                       sig,
                       flags,
                       linepars,
                       xall,
                       maxiter=50,
                       itertol=0.0001):
    '''

    Parameters
    ----------
    wave
    flux
    sig
    flags
    linepars
    xall

    maxiter : int
        Maximum number of times to run the fit while striving for convergence

    itertol : float
        Maximum difference in any parameter from one fitting iteration to the next.  Routine will fit again if
        any difference in the measurements exceeds itertol.


    Returns
    -------

    '''
    fitpars = linepars
    oldfitpars = np.zeros([7, len(fitpars[0])]) - 99
    ctr = 0
    okay = 1
    while ((np.max(np.abs(fitpars - oldfitpars)) > itertol) & (ctr < maxiter)):
        ctr += 1

        try:
            # so the fitpars from the last iteration doesn't get erased:
            oldfitpars = fitpars
            fitpars, fiterrors, rchi2 = stevebvpfit(wave,
                                                    flux,
                                                    sig,
                                                    flags,
                                                    linepars=linepars,
                                                    xall=xall)
            fitpars = np.array(fitpars)
            print('Iteration', ctr, '-')

        except:
            print('Fitting error!')
            print("Unexpected error:", sys.exc_info()[0])
            okay = 0
            raise

    if okay != 0:
        print('Fit converged after', ctr, 'iterations.')
        return fitpars, fiterrors, rchi2
    else:
        return linepars, fiterrors, rchi2


def writerchi2(rchi2, outfilename):
    '''
    Writes reduced input group's chi-squared to file.
    '''
    rchi2file = open(outfilename, 'wt')
    rchi2file.write(str(rchi2))
    rchi2file.close()


def writelinepars(fitpars,
                  fiterrors,
                  parinfo,
                  specfile,
                  outfilename,
                  linecmts=None):
    '''
    Write fit parameters out to file.

    Parameters
    ----------
    fitpars : list of lists
        Parameters for fit ready for fitter!
    fiterrors : array of numpy vectors
        Error array for the fitting initialized to '0' for each param
    parinfo : array of arrays
        Flags to be used in fit
    specfile : str
        Name of the input file containing the spectrum
    outfilename : str
        Parameter output filename
    linecmts : list of lists, optional
        Reliability flags and comments, e.g., from igmguesses

    '''
    import os
    ### Set outputs and open files
    bigfiletowrite = cfg.largeVPparfile
    filetowrite = outfilename
    if os.path.isfile(filetowrite):
        VPparfile = open(filetowrite, 'wb')
        bigparfile = open(bigfiletowrite, 'ab')  # Append to the running list
    else:
        VPparfile = open(filetowrite, 'wb')
        bigparfile = open(bigfiletowrite, 'wb')

    ### Prep header of line parameter file
    if linecmts is not None:
        header = b'specfile|restwave|zsys|col|sigcol|bval|sigbval|vel|sigvel|nflag|bflag|vflag|vlim1|vlim2|wobs1|wobs2|pix1|pix2|z_comp|trans|rely|comment \n'
    else:
        header = b'specfile|restwave|zsys|col|sigcol|bval|sigbval|vel|sigvel|nflag|bflag|vflag|vlim1|vlim2|wobs1|wobs2|pix1|pix2|z_comp|trans \n'
    VPparfile.write(header)
    bigparfile.write(header)

    ### Grab parameters/info for each line
    for i in range(len(fitpars[0])):
        zline = fitpars[3][i]
        vlim1 = fitpars[5][i]
        vlim2 = fitpars[6][i]
        restwave = fitpars[0][i]
        wobs1 = restwave * (1 + zline + vlim1 / c)
        wobs2 = restwave * (1 + zline + vlim2 / c)
        pix1 = jbg.closest(cfg.wave, wobs1)
        pix2 = jbg.closest(cfg.wave, wobs2)
        trans = atomicdata.lam2ion(fitpars[0][i])
        z_comp = ltu.z_from_dv(fitpars[4][i] * u.km / u.s, zline)
        if linecmts is not None:
            towrite = jbg.pipedelimrow([
                specfile, restwave,
                round(zline, 5),
                round(fitpars[1][i], 3),
                round(fiterrors[1][i], 3),
                round(fitpars[2][i], 3),
                round(fiterrors[2][i], 3),
                round(fitpars[4][i], 3),
                round(fiterrors[4][i], 3), parinfo[1][i], parinfo[2][i],
                parinfo[4][i], vlim1, vlim2, wobs1, wobs2, pix1, pix2,
                round(z_comp, 5), trans, linecmts[0][i], linecmts[1][i]
            ])
        else:
            towrite = jbg.pipedelimrow([
                specfile, restwave,
                round(zline, 5),
                round(fitpars[1][i], 3),
                round(fiterrors[1][i], 3),
                round(fitpars[2][i], 3),
                round(fiterrors[2][i], 3),
                round(fitpars[4][i], 3),
                round(fiterrors[4][i], 3), parinfo[1][i], parinfo[2][i],
                parinfo[4][i], vlim1, vlim2, wobs1, wobs2, pix1, pix2,
                round(z_comp, 5), trans
            ])
        VPparfile.write(towrite.encode())
        bigparfile.write(towrite.encode())
    VPparfile.close()
    bigparfile.close()
    print('Line parameters written to:')
    print(filetowrite)


def writeVPmodel(outfile, wave, fitpars, normflux, normsig):

    from astropy.table import Table

    model = voigtfunc(wave, fitpars)
    modeltab = Table([wave, model, normflux, normsig],
                     names=['wavelength', 'model', 'normflux', 'normsig'])

    dummycont = np.ones(len(wave))

    spec = XSpectrum1D.from_tuple((modeltab['wavelength'], modeltab['model'],
                                   modeltab['normsig'], dummycont))
    spec.write_to_fits(outfile)
