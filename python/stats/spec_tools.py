#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Some functions allowing statistical tests (ttests and ANOVAs) on 2d

They include cluster permutation tests


Created on Tue Jan  5 15:59:56 2021

@author: elio
"""

import numpy as np
from scipy.stats import f, t
from skimage import measure
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF






def clustperm_repANOVA(mat, alphathresh=.05, alphacluster=.05, nperms=1000, plot=True):
    """
    function for cluster permutation (2D) on repeated measures ANOVA
    
    requires a 4D matrix as input, with conditions on 3rd dimension and
    subjects on 4th dimension
    
    returns masked F values and cluster significance table
    
    written by Elio Balestrieri @Buschlab (2021)
    elio.balestrieri@gmail.com
    """
    
    # run repeated measures ANOVA
    Fvals_mat, df_num, df_den = mat_repANOVA(mat)

    # define critical F value for thresholding given the alpha threshold defined, and
    # the degrees of freedom
    crit_F = f.ppf(1-alphathresh, df_num, df_den)
    
    # compute clustermasses
    clustermasses = repANOVA_clustermasses(Fvals_mat, crit_F)

    # start permutations
    permuted_clustmasses = []
    
    for iperm in range(nperms):
        
        swap_ = np.copy(mat)
        swap_ = scramble(swap_)
        
        Fvals_perm, df_num, df_den = mat_repANOVA(swap_)

        temp_ = repANOVA_clustermasses(Fvals_perm, crit_F)

        permuted_clustmasses.append(temp_.max())


    # compute the ecdf
    ecdf_shffld = ECDF(permuted_clustmasses)
    # extract pvalues
    Pvals_clusts = 1-ecdf_shffld(clustermasses) 

    dict_out = {'clustermasses' : clustermasses, 'p' : Pvals_clusts}

    return dict_out




def scramble(array, axis_shuffle=2, axis_ind=3):
    
    
    swapped_array = array.swapaxes(0, axis_shuffle)
    swapped_array = swapped_array.swapaxes(-1, axis_ind)        
    
    dims = swapped_array.shape
   
    for idim4 in range(dims[-1]):
        
        idx = np.random.choice(dims[0], dims[0], replace=False)
        swapped_array[:, :, :, idim4] = swapped_array[idx, :, :, idim4]         

    swapped_array = swapped_array.swapaxes(-1, axis_ind)        
    swapped_array = swapped_array.swapaxes(0, axis_shuffle)

                
    return swapped_array



def repANOVA_clustermasses(Fvals_mat, crit_F):
    
    # find all the values in the maps exceeding the critical F
    above_thresh = (Fvals_mat>crit_F)*1

    # label continuous clusters
    clusters_labeled = measure.label(above_thresh, background=0)

    # find cluster masses
    clustermasses = []
        
    clustvals = np.unique(clusters_labeled)
    
    # get rid of background
    pruned_clustvals = clustvals[clustvals>0]

    if pruned_clustvals.size==0:
        
        clustermasses.append(0)

    else:
        
        for iclust in pruned_clustvals:
            
            temp_mask = clusters_labeled == iclust
            clust_Fs = Fvals_mat[temp_mask]

            clustermasses.append(clust_Fs.sum())


    # convert clustermass list into np array and return
    clustermasses = np.array(clustermasses)
    
    return clustermasses    




def mat_repANOVA(mat):
    
    nsubjs = mat.shape[3]
    nconds = mat.shape[2]

    cnd_mean = mat.mean(axis=3)
    grand_mean = cnd_mean.mean(axis=2)
    subj_mean = mat.mean(axis=2)
    
    # preallocate 0 matrix
    ss_cond = np.zeros((mat.shape[0], mat.shape[1]))

    # within group variation
    ss_w = np.zeros((mat.shape[0], mat.shape[1]))

    
    for icond in range(nconds):
        
        # SS condition
        this_cond = cnd_mean[:, :, icond]
        temp_ = nsubjs * (this_cond-grand_mean)**2
        ss_cond += temp_
    
        # SS within
        for isubj in range(nsubjs):
            
            this_subj = mat[:, :, icond, isubj]
            temp2_ = (this_subj - this_cond)**2
            ss_w += temp2_

    # compute ss subjects
    ss_subjs = np.zeros((mat.shape[0], mat.shape[1]))
            
    for isubj in range(nsubjs):
                                
        ss_subjs += nconds * (subj_mean[:, :, isubj] - grand_mean)**2


    # compute ss error
    ss_error = ss_w - ss_subjs
    
    # mean square for our experimental condition
    ms_cond = ss_cond / (nconds-1)

    # mean square error
    ms_error = ss_error / ((nconds-1)*(nsubjs-1))
    
    # finally compute F vals
    F_vals_mat = ms_cond / ms_error
    
    # store df of numerator and denominator to compute critical F
    df_num = (nconds-1)
    df_den = ((nconds-1)*(nsubjs-1))
    
    return F_vals_mat, df_num, df_den




def rep_ttest(mat1, mat2):
    
    df = mat1.shape[2]
    
    if df != mat2.shape[2]:
        raise ValueError('df of mat 1 is different from mat2')
        
    diffs_ = mat1 - mat2
    tvals_mat = np.sqrt(df) * diffs_.mean(axis=2) / diffs_.std(axis=2)    

    return tvals_mat


def clustermass_t(tvals_mat, crit_t):
    
    # find all the values in the maps exceeding the critical t
    above_thresh = (tvals_mat>np.abs(crit_t))*1

    # label contiguous clusters
    posclusters_labeled = measure.label(above_thresh, background=0)

    # find cluster masses
    clusters = {'masses' : [], 'masks' : []}
        
    posclustvals = np.unique(posclusters_labeled)
    
    # get rid of background
    pruned_posclustvals = posclustvals[posclustvals>0]

    if pruned_posclustvals.size==0:
        
        clusters['masses'].append(0)
        temp_mask = posclusters_labeled == 99999999
        clusters['masks'].append(temp_mask)


    else:
        
        for iclust in pruned_posclustvals:
            
            temp_mask = posclusters_labeled == iclust
            clust_pos_Ts = tvals_mat[temp_mask]

            clusters['masses'].append(clust_pos_Ts.sum())
            clusters['masks'].append(temp_mask)


    # repeat procedure for negative clusters
    below_thresh = (tvals_mat<-np.abs(crit_t))*1

    # label contiguous clusters
    negclusters_labeled = measure.label(below_thresh, background=0)
            
    negclustvals = np.unique(negclusters_labeled)
    
    # get rid of background
    pruned_negclustvals = negclustvals[negclustvals>0]

    if pruned_negclustvals.size==0:
        
        clusters['masses'].append(0)

    else:
        
        for iclust in pruned_negclustvals:
            
            temp_mask = negclusters_labeled == iclust
            clust_neg_Ts = tvals_mat[temp_mask]

            clusters['masses'].append(clust_neg_Ts.sum())
            clusters['masks'].append(temp_mask)


    return clusters   


def clustperm_ttest(mat1, mat2, alphathresh=.05, alphacluster=.05, nperms = 1000):
    
    df = mat1.shape[2]
    crit_t = t.ppf(1-alphathresh, df-1)

    tvals_mat = rep_ttest(mat1, mat2)

    clusters_data = clustermass_t(tvals_mat, crit_t)
    
    aggr_mat = np.concatenate((mat1[:, :, :, np.newaxis], 
                               mat2[:, :, :, np.newaxis]), axis=3)
    
    
    clust_shffld_pop = []
     
    for iperm in range(nperms):
        
        swap_idxs1 = np.random.randint(0, high=2, size=df) 
        swap_idxs2 = np.abs(swap_idxs1-1)
    
        swap_mat1 = np.zeros(mat1.shape)
        swap_mat2 = np.zeros(mat1.shape)
        
        for isubj in range(df):
               
            swap_mat1[:, :, isubj] = aggr_mat[:, :, isubj, swap_idxs1[isubj]]
            swap_mat2[:, :, isubj] = aggr_mat[:, :, isubj, swap_idxs2[isubj]]
    
        
        tvals_mat_shffld = rep_ttest(swap_mat1, swap_mat2)
        clustermasses_shffld = clustermass_t(tvals_mat_shffld, crit_t)
                
        clust_shffld_pop.append(max(clustermasses_shffld['masses'], key=abs))


    # compute the ecdf
    ecdf_shffld = ECDF(clust_shffld_pop)
    
    Pvals_clusts = ecdf_shffld(clusters_data['masses']) 
    Pvals_clusts[Pvals_clusts>.5] = 1-Pvals_clusts[Pvals_clusts>.5]

    clusters_data.update({'p': Pvals_clusts, 'tvals_map' : tvals_mat})
    
    # add masked tval map
    sign_clusts = Pvals_clusts < alphacluster

    strt_mask = np.zeros(tvals_mat.shape) == 1

    acc_clust = 0
    for iclust_is_sign in sign_clusts:
        
        if iclust_is_sign:
            
            strt_mask = (strt_mask) | (clusters_data['masks'][acc_clust])


        acc_clust += 1

    copy_t = np.copy(tvals_mat)
    copy_t[strt_mask==False] = np.nan
    
    clusters_data.update({'sign_masked_clusts' : copy_t})
    

    return clusters_data



















































































