#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STIMGEN

All the functions for generating stimuli (almost) like in Wyart et al. 2012

Created on Tue Sep  1 11:04:21 2020

@author: ebalestr

"""

import numpy as np
import copy
from scipy.ndimage import gaussian_filter


#%% STIM DESIGN

# grating

def do_grating(va, pixXva, sf, rador, phase, contrast):
    """
    function for creating sine wave
    following  Lu & Dosher Visual Psychophysics textbook (2014)
    
    Parameters
    ----------
    va : scalar
        radius (visual angles).
    pixXva : scalar
        pixels x visual angle.
    sf : scalar
        spatial frequency. The present defines [-va + va].
    rador : scalar
        sine wave orientation (rad).
    phase : scalar
        sine wave phase angle (rad).
    contrast : scalar
        sine wave amplitude (1 max).

    Returns
    -------
    grid : numpy array
        patch subtending 2 va.
    Xg : numpy array
        va grid on x axis.
    Yg : numpy array
        va grid on y axis.

    """
        
    # define number of pixels contained in the patch
    npixs = 2*va*pixXva
    
    # define meshgrid for size definition
    axs = np.linspace(-va, va, npixs)
    Xg, Yg = np.meshgrid(axs, axs)
    grid = contrast * np.sin(2 * np.pi * sf *
                             (Yg*np.sin(rador) + Xg*np.cos(rador)))
    
    # define grid size
    lgcl_mask = (Xg**2 + Yg**2) > va**2
    
    # null everything not corresponding to the size
    grid[lgcl_mask] = 0
    
    return grid, Xg, Yg


# clip the center & extrema

def do_donut(fixva, Xg, Yg, grid):
    """
    clip the stimulus provided to create a fixation donut

    Parameters
    ----------
    fixva : scalar
        radius (visual angles).
    Xg : numpy array
        va grid on x axis.
    Yg : numpy array
        va grid on y axis.
    grid : numpy array
        patch subtending 2 va.

    Returns
    -------
    None.

    """
    
    # define fixation area
    lgcl_mask_fix = (Xg**2 + Yg**2) <= fixva**2
    
    # define surrounding area
    lgcl_surr_area = (Xg**2 + Yg**2) >= np.max(Xg)**2 

    # clip
    donut = copy.deepcopy(grid)
    donut[lgcl_mask_fix] = 0
    donut[lgcl_surr_area] = 0 
    
    return donut



# noise  patch

def do_noisepatch(va, fixva, pixXva, sigmanoise, sigmafilter, signal, addsignal=True):
    
    # define number of pixels contained in the patch
    npixs = 2*va*pixXva

    # 42 is the answer. Moreover, is the value suggested in Lu & Dosher (2014),
    # but this is less relevant. (SD=10% used by Wyart, but I don't know how to interpret this yet)
    # noisepatch =  (42/128) * np.random.randn(npixs, npixs)
    noisepatch =  sigmanoise * np.random.randn(npixs, npixs)
    
    # define meshgrid for size definition
    axs = np.linspace(-va, va, npixs)
    Xg, Yg = np.meshgrid(axs, axs)
    
    # smooth the thing
    noisepatch = gaussian_filter(noisepatch, sigmafilter)

    # add signal
    if addsignal:
        
        noiseANDsignal = noisepatch + signal

    donutsignal = do_donut(fixva, Xg, Yg, signal)
    noisepatch = do_donut(fixva, Xg, Yg, noisepatch)
    noiseANDsignal = do_donut(fixva, Xg, Yg, noiseANDsignal)
    
    # CC_signal = np.correlate(noiseANDsignal.flatten(order='C'),
    #                          donutsignal.flatten(order='C'))
    
    # CC_noise = np.correlate(noiseANDsignal.flatten(order='C'),
    #                         noisepatch.flatten(order='C'))
    
    power_signal = (donutsignal**2).sum()
    power_den = ((donutsignal - noiseANDsignal)**2).sum()
    
    # compute SNR and energy of the signal
    SNR = 10 * np.log10(power_signal / power_den)


    return noiseANDsignal, donutsignal, SNR

    






#%% generate noise patch with some degree of correlation
    
def do_corrnoise(grid, R, sigmanoise):

    # vectorize stims    
    flattengrid = grid.flatten()
    flattengrid = flattengrid[np.newaxis]
    noise = np.random.normal(loc=0, scale=sigmanoise, size=(1, flattengrid.size))

    # choleski decomposition from covariance matrix
    r = [[1, R],
         [R, 1]]
    L = np.linalg.cholesky(r)

    # concatenate together signal & noise 
    uncorrelated =np.concatenate((flattengrid, noise))

    # generate correlated noise
    correlated = np.dot(L, uncorrelated)

    # reshape into grid
    corrnoise = correlated[1, :]
    noisegrid = corrnoise.reshape(grid.shape)
    
    # get an enery estimate
    enest = (corrnoise[np.newaxis] @ flattengrid.T) / np.sum(flattengrid!=0)
    
    return noisegrid, enest