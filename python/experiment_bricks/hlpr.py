#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The script collects all the helper function and classes needed to run a psypy exp
on BUSCHLAB.

The usage of these classes and function is for one specific experiment on a specific computer.  

The most generalizable class is StartObserver, the core of the custom bayesian adaptive procedure for maintaining subjects' performance constant. 

Created on Sat Sep 19 12:54:57 2020
@author: ebalestr
"""


import numpy as np
import pandas as pd
from itertools import product
import copy
from scipy.ndimage import gaussian_filter
from psychopy import visual, event, logging, core
import h5py
from scipy.stats import norm
import telegram
import time
from EyelinkWrapper import *
from tkinter.filedialog import askopenfilename



#%% CFG class definition
"""
This first class defines the basic features of the experiment
"""

class PrepareConfig:

    # it follows a nested schema: 
    # it breaks the class in dictionaries, based on theme


    #%% dictionaries for stable, invariant features
    
    def __init__(self, isQUEST=False, isINSTRUCTIONS=False,
                 isET=False, isEEG=False, language=1, isVPSTUNDEN=False):
        
        # define telegrambot coordinates to receive notifications
        global bot, receiver
        receiver = NumeriCodeForYourReceiver
        bot = telegram.Bot(token='SusbstituteWithYourToken')        
        
        # attach flags for devices currently operating (EEG and/or eyetracking)
        self.eyetrackingON = isET
        self.eegON = isEEG
        self.language=language
        self.VPStunden = isVPSTUNDEN
        
        self.va = {
            
            "in_crcl" : 4,
            "out_crcl" : 8,
            "outout_crcl" : 9,
            "patch_width" : 512/55,
            "int_fix" : .05,
            "out_fix" : .15,
            "unity" : 1,
            "starter_cue" : 2
            
            } 
                
        self.time = {
            
            "int_dur" : 2,
            "pre_stim" : 1.7,
            "post_stim" : 1.5,
            "int_break" : 1,
            "cue_on" : 2,
            "stim_on" : 6/120,
            "jitter_range" : .5                         # define the upper jitter range lower default 1 frame defined in trialexecution
            
            }
                        
        self.colors = {
            
            "azure" : [0, 90, 179],                         # azure; HSV = [210 100 75]
            "ochre" : [179, 90, 0],                       # OCRA; HSV = [30, 100, 75]
            "grey" : [128, 128, 128],      
            "darkgrey" : [64, 64, 64],
            "yellowrand" : [93, 99, 0],
            "greencorrect" : [0, 102, 0],
            "redwrong" : [153, 0, 0],
            
            }
        
        
            # "azure" : [35, 192, 192],                       # azure; HSV = [180 82 75]
            # "magenta" : [192, 35, 189],                     # magenta; HSV = [301, 82, 75]
        
        
        self.triggers = {
            
            "noise" : 1,                                   # these 3 triggers give info on which stimulus has been displayed
            "signal" : 2,
            "threshold": 3,
            "reported_noise" : 10,                         # these 2 triggers give info on which response was given by the participant
            "reported_signal" : 20,
            "low_conf" : 100,                               # these 2 triggers code for confidence
            "high_conf" : 200,
            "pause_EEG_save" : 255,                         # these 3 triggers deal with the saving policies of EEG, and will be used to identify start end block or start end calibration
            "start_EEG_save" : 254,
            "savebox_EEG" : 253,
            "start_trial" : 66,                             # starting trials and blocks
            "start_block" : 77
            
            }
        
        self.generic = {
            
            "subj2screen" : 860,                            # distance to screen, mm
            "trigdur" : .002,                               # trigger duration, ms
            "pixXmm" : 1/.2715,                             # pixel X mm
            "screenSize" : [1920, 1080],
            "debugScreen" : [800, 800],
            "edgesCircle" : 128
            
            }
        
        # reward contingencies
        self.reward_cunt = {
            
            "H"              : 1,
            "CR"             : 1,
            "wrong"          : 0,
            "highconfright"  : 1,
            "highconfwrong"  : 0,
            "lowconf"        : .5
            
            }
        
        
        # reward boundaries for EUR conversion
        self.boundaries_EUR = {'const_highacc' : .75, 
                               'const_lowacc' : .5,
                               'const_high_conf' : .75, 
                               'const_low_conf' : .5,
                               'const_EUR' : 30, 
                               'max_EUR' : 40
                               }
        
        # initialize saccade counter for the present block
        self.saccadecounter = 0 
                
        
        # self called methods for features requiring computation
        
        # conversion from va to pixels
        self.__to_pixs()            
            
        # preallocate experimental conditions
        # before that, check whether by mistake tehre is overlapping between flags, that should NEVER happen
        if isQUEST and isINSTRUCTIONS:
            raise ValueError('impossible to run QUEST AND instructions')
        
        self.__preallocate(isQUEST, isINSTRUCTIONS)
        
        # change RGB space
        self.__to_psychopyRGB()
        
        # call grating definition at last, since require pix X VA
        self.gratings = {
            
            "sf" : 2,                                       # spatial frequency, cpd
            "rador" : 0,                                    # gating orientation, rad
            "sigmanoise" : .1,                              # Wyart suggests 10 (.1)
            "sigmafilter" : self.pxs['unity']*.083          # as in Wyart et al. 2012
                       
            }    
                            
  
    
    # conversion from visual angles into pixels
    def __to_pixs(self):
    
        self.pxs = self.va.copy()
        
        for k, v in self.pxs.items():
            
            inrad = np.radians(v)
            
            self.pxs[k] = int(round(2 * self.generic["subj2screen"] * 
                                np.tan(inrad / 2) *
                                self.generic["pixXmm"]))
            
            
    # preallocate conditions within a block
    def __preallocate(self, isQUEST, isINSTRUCTIONS):
        
        # define all experimental conditions in order to obtain perfect balancing
        # NOTE !!!: the unprobed stimulus' magnitude is randomly assigned
              
        neutcue = ['neutral']
        neutprob = [0, 1]
        matchthresh = [0, 2]
        neutfeed = ['fair']
        
        colnames = ['cue_type',
                    'cue_color',
                    'cue_valence',
                    'target_present', 
                    'feedback']
        
        # obtain all combinations of the aforementioned condition
        # this process is different for each trial type
        
        pre_neutral = pd.DataFrame(list(product(neutcue, ['darkgrey'], ['neutral'], neutprob, neutfeed)), 
                                   columns = colnames)      
        pre_thresh = pd.DataFrame(list(product(neutcue, ['darkgrey'], ['neutral'], matchthresh, neutfeed)),
                                  columns = colnames)
       
        # keep the module standard for the QUEST/training part
        if isQUEST:

            # create a mini module of trials to get the subject aquainted with the task
            
            NEUT_start = pd.concat([pre_neutral]*3) 
            NEUT_start = NEUT_start.sample(frac=1).reset_index(drop=True)
            
            NEUT_mod = pd.concat([pre_neutral]*25 + [pre_thresh]*5) # 25x, 5x
            NEUT_mod = NEUT_mod.sample(frac=1).reset_index(drop=True)

                                 
            mini_module = pd.concat([NEUT_start, NEUT_mod])

            self.preallocate = mini_module                            
        
        
        
        elif isINSTRUCTIONS:
            
            NEUT_mod = pre_neutral                                 
            instruction_module = pd.concat([NEUT_mod])

            self.preallocate = instruction_module 

        
        else:
            
            # create mini module (36 trials) with all the experimental repetitions                      
            mini_module = pd.concat([pre_neutral]*14 + [pre_thresh]*4)

            # the following loop allows true randomization by maintaining perfect condition balance across quartiles of the experiment            
            list_modules = []
            for iModule in range(4):
                
                shuffled_module = mini_module.sample(frac=1).reset_index(drop=True)
                list_modules.append(shuffled_module)
            
            self.preallocate = pd.concat(list_modules)
            
            


        # preallocate a 1024 X 1024 X n trials empty array to store the noisedpatches
        self.signals_arrays = np.empty([1024, 1024, self.preallocate.shape[0]], dtype='uint8')
        
        # additionally, preallocate a duplicate of subj_info.data in numerical form only
        # in order to add robustness in case of bug and save the duplicate info
        # with the image arrays.
        # dimord: rows -> trials
        # columns: 0 -> cue type (0 = neutral, 1 = probability, 2 = relevance)
        #          1 -> cue valence (0 = neutral, 1 = low, 2 = high)
        #          2 -> signal (0 = "absent", 1 = present)
        #          3 -> signal contrast (float)
        #          4 -> SNR (float)
        #          5 -> confidence (float, [-1, 1])     
        self.data_duplicate = np.empty([self.preallocate.shape[0], 6], dtype='float32')

        # save structure of experiment to allow flexible looping
        self.expstruct = {
            
            "trlXblock"     : self.preallocate.shape[0],
            "nblocks"       : 7
            
            }
        
    
 
    # change from a canonical rgb255 to a -1 1 rgb space
    def __to_psychopyRGB(self):
                
        for k, v in self.colors.items():
            
            self.colors[k] = (np.array(v) - 128) /128




    #%% conversion from seconds to frames
    def to_frames(self, fps):
        
        self.frames = self.time.copy()
        
        for k, v in self.frames.items():
            
            self.frames[k] = int(v * np.round(fps))
            
    #%% convert scores to EUR
    def from_score_to_EUR(self, score, ntrials, tot_trials):
    
        score = np.array(score)
        
        # extract stuff from the dictionary        
        const_highacc = self.boundaries_EUR['const_highacc']
        const_lowacc = self.boundaries_EUR['const_lowacc']
        const_EUR = self.boundaries_EUR['const_EUR']
        max_EUR = self.boundaries_EUR['max_EUR']
        const_high_conf = self.boundaries_EUR['const_high_conf']
        const_low_conf = self.boundaries_EUR['const_low_conf']
        
        max_score = (const_highacc * ntrials) + const_high_conf * ntrials
        min_score = (const_lowacc * ntrials) + const_low_conf * ntrials    
    
        score[score < min_score] = min_score
        score[score > max_score] = max_score
        
        if not self.VPStunden:
            
            reward = ((max_EUR - const_EUR)*(score - min_score)/(max_score-min_score) 
                + const_EUR) * (ntrials / tot_trials)
        
        else:

            reward = ((max_EUR - const_EUR)*(score - min_score)/(max_score-min_score) 
                ) * (ntrials / tot_trials)
        
            
        return reward

            
            
    #%% update the current array patch, and save when required 
    def imagecontainer(self, this_trial=None, 
                       itrl=None, save=False, blocknumber=None, subjcode='',
                       path='', resp=None):
        
        if not (this_trial == None):
            
            if not (itrl == None):
                
                # compress image to uint 8
                compressed = np.uint8(this_trial.signalarray * 128 + 128)
                
                # append to array
                self.signals_arrays[:, :, itrl] = compressed
                
                # convert info in numerical values the meta level information of the trial
                # this gives us a second save of the data in case the main csv file would be lost
                # or lack of correspondence would be observed
                
                # 0. cue type
                if this_trial.intval_def['cue_type'] == 'neutral':                  
                    self.data_duplicate[itrl, 0] = 0
                    
                elif this_trial.intval_def['cue_type'] == 'probability':                  
                    self.data_duplicate[itrl, 0] = 1
                    
                elif this_trial.intval_def['cue_type'] == 'relevance':                  
                    self.data_duplicate[itrl, 0] = 2
                    
                # 1. cue valence     
                if this_trial.intval_def['cue_valence'] == 'neutral':                  
                    self.data_duplicate[itrl, 1] = 0

                elif this_trial.intval_def['cue_valence'] == 'low':                  
                    self.data_duplicate[itrl, 1] = 1

                elif this_trial.intval_def['cue_valence'] == 'high':                  
                    self.data_duplicate[itrl, 1] = 2

                # 2. signal presence               
                self.data_duplicate[itrl, 2] = this_trial.intval_def['target_present']
                
                # 3. signal contrast
                self.data_duplicate[itrl, 3] = this_trial.intval_def['trgt_cnt']
                
                # 4. SNR
                self.data_duplicate[itrl, 4] = this_trial.intval_def['SNR']
                              
                # 5. confidence
                self.data_duplicate[itrl, 5] = resp

            
            else:
                
                raise ValueError('Trial number cannot be NoneType, failed image array updating')

        else:
            
            if save:
                
                fname = path + subjcode + '.hdf5'
                
                if blocknumber == 0:
                    
                    # create hdf5 file for the first time, and relative subgroup 
                    # for the present block
                    hd_file = h5py.File(fname, 'w')
                    
                else:
                    
                    # load file in append mode
                    hd_file = h5py.File(fname, 'a')                    
                    
                blockname = 'block' + str(blocknumber+1)                 
                grp = hd_file.create_group(blockname)
                grp.create_dataset('sigpatches', data=self.signals_arrays)
                grp.create_dataset('labels', data=self.data_duplicate)
                
                hd_file.close()
                
    #%% correct the current trial order after fixation break occurred            
    def correct_trialorder(self, itrl):
        
        # create a copy of the preallocate dataframe
        swap_ = self.preallocate.copy()
        
        # extract the current and the last line from preallocate
        thisline, endline = self.preallocate.iloc[itrl], self.preallocate.iloc[-1]
        
        # swap the lines
        swap_.iloc[itrl] = endline
        swap_.iloc[-1] = thisline
        
        # reassign back the swap dataframe to the original
        self.preallocate = swap_


#%% Trial
"""
What needs to be done in order to run a trial
"""

class TrialExecution:
     
    def __init__(self, cfg_info, subj_info, i_trl):

        # fetch the current trial
        self.trialdef = cfg_info.preallocate.iloc[i_trl]

        # save in dictionary the properties necessary to draw the current trial    
        # (interval colors in RGB, order of probed trials...)
        self.intval_def = {
            
            'cue_rgb' : cfg_info.colors[self.trialdef['cue_color']],
            'target_present' : self.trialdef['target_present'],
            'cue_valence' : self.trialdef['cue_valence'],
            'cue_type' : self.trialdef['cue_type'],
            'jitter_frames' : np.random.randint(1, high=cfg_info.frames['jitter_range'])
            
            }        


        # define dictionary for triggers in this trial
        expcondname = self.intval_def['cue_valence'] + '_' + self.intval_def['cue_type']
        trialonset = cfg_info.triggers['start_trial']
        self.intval_trigs = {
            
            'cue_off' : trialonset,
            'patch_on' : self.intval_def['target_present']
            
            }
        

        # initialize a list for time intervals
        self.time_intervals = []
                    
            
        # finalize the dictionary definition by converting the theoretical magnitude
        # into the contrast level obtained, and adding redundancy for easy coding
        # edit: add randomized phase angle as well
        self.intval_def.update({
            
            'trgt_cnt' : subj_info.cntvals[self.intval_def['target_present']],
            'phase' : np.pi * np.random.rand()
            
            })
        
        # store separately nframes for each experiment portion, from cfg
        self.frames = cfg_info.frames
    

    #%% some methods for the gabor design    
    @staticmethod
    def do_grating(patchva, pixXva, sf, rador, phase, contrast):
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
        npixs = int(np.round(2*patchva*pixXva))
        
        # define meshgrid for size definition
        axs = np.linspace(-patchva, patchva, npixs)
        Xg, Yg = np.meshgrid(axs, axs)
        grid = contrast * np.sin(2 * np.pi * sf *
                                 (Yg*np.sin(rador) + Xg*np.cos(rador))+
                                 phase)
        
        grid_sin = np.sin(2 * np.pi * 2 * (Yg*np.sin(0) + Xg*np.cos(0)))
        grid_cos = np.cos(2 * np.pi * 2 * (Yg*np.sin(0) + Xg*np.cos(0)))
        
        # # define grid size
        # lgcl_mask = (Xg**2 + Yg**2) > va**2
        
        # # null everything not corresponding to the size
        # grid[lgcl_mask] = 0
        
        return grid, grid_sin, grid_cos
 
    
 
    @staticmethod
    def do_donut(fixva, extva, Xg, Yg, grid):
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
        lgcl_surr_area = (Xg**2 + Yg**2) > extva**2 
    
        # clip
        donut = copy.deepcopy(grid)
        donut[lgcl_mask_fix] = 0
        donut[lgcl_surr_area] = 0 
        
        return donut, lgcl_surr_area


    def do_noisepatch(self, patchva, circleva, fixva, pixXva, sigmanoise, 
                      sigmafilter, signal, grid_sin, grid_cos):
    
        # define number of pixels contained in the patch
        npixs = int(np.round(2*patchva*pixXva))
        # copy, for debug
        self.npixs = npixs
    
        # get current random state before defining the noise patch, so that we're gonna be able to reproduce that even without saving them.
        RNGstate = np.random.get_state()
    
        # 42 is the answer. Moreover, is the value suggested in Lu & Dosher (2014),
        # but this is less relevant. (SD=10% used by Wyart)
        # noisepatch =  (42/128) * np.random.randn(npixs, npixs)
        noisepatch =  sigmanoise * np.random.randn(npixs, npixs)
        
        # define meshgrid for size definition
        axs = np.linspace(-patchva, patchva, npixs)
        Xg, Yg = np.meshgrid(axs, axs)
        
        # smooth the thing
        noisepatch = gaussian_filter(noisepatch, sigmafilter)
        
        # reconvert the gaussian filtered noisepatch to the original std
        noisepatch = noisepatch * (sigmanoise/noisepatch.std())
    
        # add signal
        noiseANDsignal = noisepatch + signal    
    
        donutsignal, mask = self.do_donut(fixva, circleva, Xg, Yg, signal)
        noisepatch, mask = self.do_donut(fixva, circleva, Xg, Yg, noisepatch)
        noiseANDsignal, mask = self.do_donut(fixva, circleva, Xg, Yg, noiseANDsignal)
        
        # CC_signal = np.correlate(noiseANDsignal.flatten(order='C'),
        #                          donutsignal.flatten(order='C'))
        
        # CC_noise = np.correlate(noiseANDsignal.flatten(order='C'),
        #                         noisepatch.flatten(order='C'))
        
        
        # convert mask in a format understood in psychopy
        out_mask = np.abs(mask.astype(float)-1)*2 -1
                
        
        # compute energy of the signal (but keep SNR for variable name compatibility)
        SNR = np.sqrt((noiseANDsignal.reshape((1, noiseANDsignal.size)) @ grid_sin.reshape((grid_sin.size, 1))) ** 2 + 
                 (noiseANDsignal.reshape((1, noiseANDsignal.size)) @ grid_cos.reshape((grid_cos.size, 1))) ** 2) / noiseANDsignal.size 
        
        
        # return donutsignal, SNR # debug purpose
        return noiseANDsignal, SNR, out_mask, RNGstate

    
    #%% callable methods
    
        
    # preallocate stimuli array
    def preallocate_arrays(self, cfg, win):
        
        # append measures of the inner circle and unity for ET online control
        self.incircle = cfg.va['in_crcl']
        self.pxXva = cfg.pxs['unity']
        
        p = self.intval_def # shortcut handle
        
        signal, grid_sin, grid_cos = self.do_grating(cfg.va['patch_width'], 
                                                     cfg.pxs['unity'],
                                                     cfg.gratings['sf'],
                                                     cfg.gratings['rador'],
                                                     p['phase'],
                                                     np.exp(p['trgt_cnt'])
                                                     )    
        
        self.signalarray, SNR, mask, RNGstate = self.do_noisepatch(cfg.va['patch_width'],
                                                                   cfg.va['out_crcl'],
                                                                   cfg.va['in_crcl'], 
                                                                   cfg.pxs['unity'], 
                                                                   cfg.gratings['sigmanoise'], 
                                                                   cfg.gratings['sigmafilter'], 
                                                                   signal,
                                                                   grid_sin,
                                                                   grid_cos)
        
        self.RNGstate = RNGstate

        self.intval_def.update({
            
            'SNR' : SNR,
            
            })
        
        
        self.signal = visual.GratingStim(win, 
                                         mask=mask,
                                         tex=self.signalarray,
                                         size=self.signalarray.shape,
                                         autoLog=False)
        
        
        self.extcircle = visual.Circle(win, 
                                       radius=cfg.pxs['outout_crcl'], 
                                       fillColor=list(cfg.colors['darkgrey']), 
                                       lineColor=list(cfg.colors['darkgrey']),
                                       edges=cfg.generic['edgesCircle'],
                                       autoLog=False
                                       )
        
        
        self.greycircle = visual.Circle(win,
                                        radius=cfg.pxs['out_crcl'],
                                        fillColor=list(cfg.colors['grey']), 
                                        lineColor=list(cfg.colors['grey']),
                                        edges=cfg.generic['edgesCircle']  ,
                                        autoLog=False
                                        )
        
        self.fix_dict = {
            
            'internal' : visual.Circle(win,
                                       radius=cfg.pxs['int_fix'],
                                       fillColor=list(cfg.colors['grey']), 
                                       lineColor=list(cfg.colors['grey']),
                                       autoLog=False
                                       ),
            
            'external' : visual.Circle(win,
                                       radius=cfg.pxs['out_fix'],
                                       fillColor=list(cfg.colors['darkgrey']), 
                                       lineColor=list(cfg.colors['darkgrey']),
                                       autoLog=False
                                       ),
                    
            }
        
        
        self.cuecircles_dict = {
            
            'internal' : visual.Circle(win,
                                       radius=cfg.pxs['unity'],
                                       fillColor=list(cfg.colors['grey']), 
                                       lineColor=list(cfg.colors['grey']),
                                       edges=cfg.generic['edgesCircle']  ,
                                       autoLog=False
                                       ),

            'external' : visual.Circle(win,
                                       radius=cfg.pxs['starter_cue'],
                                       fillColor=list(self.intval_def['cue_rgb']), 
                                       lineColor=list(self.intval_def['cue_rgb']),
                                       edges=cfg.generic['edgesCircle'],
                                       autoLog=False
                                       ),
                        
            'nsteps' : np.array(cfg.pxs['outout_crcl'] - cfg.pxs['starter_cue']) 
            
            
            }
        
        
        
    #--------------------------------------------------------------------------
    ###################### ACTUAL EXPERIMENT EXECUTION !!!!!! #################
    #--------------------------------------------------------------------------
        
    def run_interval(self, win, ET=None, EEGport=None, eyetrackon=False, isEEG=False):
                
        # initialize flag to determine whether a saccade has occurred, hence whether the trial has to be repeated
        self.saccadeflag = False
        
        # prestimulus interval
        strt_ = time.time()
        for iflip in range(self.frames['pre_stim']):
            
            self.extcircle.draw()
            self.greycircle.draw()
            self.fix_dict['external'].draw()
            self.fix_dict['internal'].draw()
            win.flip()
            
            if iflip==0:
                
                send_trigger(self.intval_trigs['cue_off'], ET=ET, 
                             EEGport=EEGport,
                             isET=eyetrackon, 
                             isEEG=isEEG)
           
            if eyetrackon:
                # fetch info from eyetracking
                thispos = EyelinkGetGaze((0,0), self.incircle, (1920, 1080), 
                                         el=ET, PixPerDeg=self.pxXva)
                self.saccadeflag = thispos['hsmvd']
       
                # if a saccade was done, return current function 
                if self.saccadeflag:            
                                   
                    return
        
        self.time_intervals.append(time.time()-strt_)


        # include jitter
        strt_ = time.time()        
        for iflip in range(self.intval_def['jitter_frames']):
            
            self.extcircle.draw()
            self.greycircle.draw()
            self.fix_dict['external'].draw()
            self.fix_dict['internal'].draw()
            win.flip()
             
            if eyetrackon:
                # fetch info from eyetracking
                thispos = EyelinkGetGaze((0,0), self.incircle, (1920, 1080), 
                                         el=ET, PixPerDeg=self.pxXva)
                self.saccadeflag = thispos['hsmvd']
            
                # if a saccade was done, return current function 
                if self.saccadeflag:            
                                       
                    return
         
        self.time_intervals.append(time.time()-strt_)
           
           
        # flip stimulus on screen for the required number of frames
        strt_ = time.time()        
        for iflip in range(self.frames['stim_on']):
           
            self.extcircle.draw()
            self.signal.draw()
            self.fix_dict['external'].draw()
            self.fix_dict['internal'].draw()
            win.logOnFlip(level=logging.EXP, msg='stim flip ' + str(iflip))
            win.flip()
            
            if iflip==0:
                
                send_trigger(self.intval_trigs['patch_on']+1, ET=ET, 
                             EEGport=EEGport,
                             isET=eyetrackon,
                             isEEG=isEEG)

            
            if eyetrackon:
                # fetch info from eyetracking
                thispos = EyelinkGetGaze((0,0), self.incircle, (1920, 1080), 
                                         el=ET, PixPerDeg=self.pxXva)
                self.saccadeflag = thispos['hsmvd']
            
                # if a saccade was done, return current function 
                if self.saccadeflag:            
                                       
                    return

        self.time_intervals.append(time.time()-strt_)
        

        # post stimulus interval
        strt_ = time.time()        
        for iflip in range(self.frames['post_stim']):
            
            self.extcircle.draw()
            self.greycircle.draw()
            self.fix_dict['external'].draw()
            self.fix_dict['internal'].draw()
            win.flip()
            
            # not control fixation here to let people avoid "cheating" (as improbable as it can be)

        self.time_intervals.append(time.time()-strt_)

    
   
    def run_break(self, win):
        
        for iflip in range(self.frames['int_break']):
            
            self.extcircle.draw()
            self.greycircle.draw()
            self.fix_dict['external'].draw()
            self.fix_dict['internal'].draw()
            win.flip()
                             
                        
    # wait for response
    def get_rating(self, win, cfg, joy, ET=None, EEGport=None, 
                   isET=False, isEEG=False):
               
        
        # choose language
        if cfg.language==1:
            textNOISE = 'RAUSCHEN'
            textSIGNAL = 'SIGNAL'
            textSURE = 'Sicher'
            textUNSURE = 'Unsicher'
        else:
            textNOISE = 'NOISE'
            textSIGNAL = 'SIGNAL'
            textSURE = 'Sure'
            textUNSURE = 'Unsure'


        anchor_notseen = visual.TextStim(win,
                                         text= textNOISE,
                                         height = 20,
                                         color = 'black',
                                         pos = (-0.1*1920/2, 0),
                                         autoLog=False
                                         )
        
        anchor_seen = visual.TextStim(win,
                                      text= textSIGNAL,
                                      height = 20,
                                      color = 'black',
                                      pos = (0.1*1920/2, 0),
                                      autoLog=False
                                      )

        anchor_sure = visual.TextStim(win,
                                      text= textSURE,
                                      height = 20,
                                      color = 'black',
                                      pos = (0, 0.11*1080/2),
                                      autoLog=False
                                      )

        anchor_unsure = visual.TextStim(win,
                                      text= textUNSURE,
                                      height = 20,
                                      color = 'black',
                                      pos = (0, -0.11*1080/2),
                                      autoLog=False
                                      )


        keepwaiting = True
        strt_ = time.time()
        while keepwaiting:

            anchor_notseen.draw()
            anchor_seen.draw()

            win.flip()             

            if joy.get_a(): #joy.get_x():
                
                response = 0
                keepwaiting = False
                
            elif joy.get_x(): #joy.get_b():
                
                response = 1
                keepwaiting = False
                        
    
        RT = time.time()-strt_
        trigcode = (response+1)*10
        send_trigger(trigcode, ET=ET,
                     EEGport=EEGport,
                     isET=isET, isEEG=isEEG)




        keepwaiting = True
        strt_ = time.time()
        while keepwaiting:

            anchor_sure.draw()
            anchor_unsure.draw()

            win.flip()             

            if joy.get_b(): # joy.get_a():
                
                rating = 0
                keepwaiting = False
                
            elif joy.get_y():
                
                rating = 1
                keepwaiting = False


        
        # send trigger
        trigcode = (rating+1)*100 
        send_trigger(trigcode, ET=ET, 
                     EEGport=EEGport,
                     isET=isET, isEEG=isEEG)
        
        # 3. get actual rating on (-1) - 1 scale (change this however you want!)
        # print(rating) # (not absolutely necessary, included this for checking)
        
        # 4. return (rounded value) from function (round just for easy checking purposes)
        return response, rating, RT

        
    @staticmethod    
    def give_them_a_break(txt, win, printmessage=None, language=1):
        
        if language == 1:            
            go_on_text = 'Drücke eine beliebige Taste auf der Tastatur'
        
        else:
            go_on_text = 'Please press a button on the keyboard'
            
            
        
        
        textpatch = visual.TextStim(win,
                                    text = txt,
                                    height = 30,
                                    color= 'black',
                                    pos = (0, 100),
                                    autoLog=False
                                    )

        textpatch_st = visual.TextStim(win,
                                       text = go_on_text,
                                       height = 30,
                                       color= 'black',
                                       pos = (0, -100),
                                       autoLog=False
                                       )

        
        while not event.getKeys():
    
            textpatch.draw()
            textpatch_st.draw()
            win.flip()
            if not(printmessage is None):
                
                print(printmessage)


        
    
#%% Observer
"""
Definition of Bayes observer -and data storage
"""
class StartObserver:
    
    def __init__(self, nblocks=7, strtblock=1):

        # define expected HR and FA rates ( and d' and criterion, as a consequence)
        self.expctd_HR = .667
        self.expctd_FA = .25
        self.brake_low = .51
        self.brake_up = .7
        
        # start an accumulator for trial number
        self.trialnum = 0 
        
        # store the total number of blocks
        self.nblocks = nblocks
        
        # expected SDT metrics
        self.expctd_dP = norm.ppf(self.expctd_HR) - norm.ppf(self.expctd_FA)
        self.expctd_crit = -.5 * (norm.ppf(self.expctd_HR) + 
                                  norm.ppf(self.expctd_FA))
              
        self.list_SDT = ['CR', 'FA', 'M', 'H']
        
        # aimed presps
        self.aimed_p_resp = 999
        self.p_thresh = 999        

        # start money counter
        self.dasKapital = str(0)
        
        # store the random states used for patch generatiions
        self.randomStates = []
        
        # define boundaries for prior reshaping      
        self.bound_dP = {
            'lower': self.expctd_dP - .4,
            'upper': self.expctd_dP + .4,
            'lowerlower' : self.expctd_dP - .7,
            'upperupper' : self.expctd_dP + .7,
            }

        self.bound_crit = {
            'lower': self.expctd_crit - .3,
            'upper': self.expctd_crit + .3,
            'lowerlower': self.expctd_crit - .5,
            'upperupper': self.expctd_crit + .5
            }

        
        # initialize objects needed for bayesian estimates
        self.depth = 61
        self.cntdepth = 151 
        self.ranges = {
            
            'alpha' : np.linspace(np.log(.0078), np.log(.2), num=self.depth),
            'beta' : 6.3644, # np.linspace(.5, 15, num=self.depth),
            'gamma' : np.linspace(0, .5, num=self.depth),
            'contrast' : np.linspace(np.log(.001), np.log(.3), num=self.cntdepth)
                            
            }
        
        self.prior = np.ones([self.depth,
                              self.depth])/(self.depth ** 2)
        
        a_, g_ = np.meshgrid(self.ranges['alpha'],
                             self.ranges['gamma'],
                             sparse=True,
                             indexing='ij'
                             )
        
        self.space = {
            
            'alpha' : a_,
            'beta' : self.ranges['beta'],
            'gamma' : g_
            
            }
    
        self.__pmf_model()
        
        # initialize DataFrame for storage
        self.colnames_data = ['cue_type',
                              'cue_valence',
                              'signal',
                              'color_cue',
                              'cnt_probed',
                              'energy_probed',
                              'phase_angle_probed',
                              'bin_resp',
                              'confidence',
                              'RT',
                              'feed_type',
                              'score_detection', 
                              'score_confidence', 
                              'score_trl',
                              'cumulative_score', 
                              'cumulative_EUR',
                              'jitter', 
                              'accurate_detection', 
                              'SDT_type',
                              'Presp_aimed', 
                              'alpha_est', 
                              'gamma_est']

        self.data = pd.DataFrame(columns=self.colnames_data)

        
        # initialize DataFrame for timing storage
        self.timestamps_colnames = ['prestim',
                                    'jitter',
                                    'stimon',
                                    'postim']
        self.timestamps = pd.DataFrame(columns=self.timestamps_colnames)
        
        # initialize contrast vals. 
        self.cntvals = np.log([.001, .2, .03])   # third value for p = .5
        
        # initialize subject score
        self.expscore = 0
        
        # initialize subject P in players cdf
        self.SD = 1
        
        # add a QUEST counter
        self.QUESTcounter = 0
        
        # add an "out of range" counter
        self.outofrange_counter = 0
        
        # add a smoothing prior counter
        self.counter_smoothingprior = 0
        
        # ask to repeat QUEST anyway
        self.repeatQUESTanyway = False
        
        
        if strtblock > 1:
            
            try:
                
                print('####### EXTRA-ATTENTION NEEDED!!!!')
                print('\nYou are starting the experiment from block ' + str(strtblock))
                print('We are trying to load a couple of files that will be needed to continue the experiment exactly from where we left it')
                print('If this will work out, we will authomatically avoid the QUEST.')
                print('If not, the QUEST will be authomatically initialized')

                
                subjcode = input('\n\nSubject code?\n(Case sensitive, no spaces)\n>')
                
                initialdir = '../Logfiles/' + subjcode 
                
                prior_filename = askopenfilename(title='Select the PRIOR file for the current participant', initialdir=initialdir)
                data_filename = askopenfilename(title='Select the Block file for the current participant', initialdir=initialdir)
                
                data_previous = pd.read_csv(data_filename)    
                prior_previous = np.load(prior_filename)
                
                self.data = data_previous
                self.prior = prior_previous
                self.QUESTcounter = 1
                self.expscore = data_previous['cumulative_score'].iloc[-1]
                self.trialnum = data_previous.shape[0] + 66
                
                
                print('\n\nOK, it looks like all the files have been loaded correctly, we are ready to start!')
                print('##############################################################')
        
            except:
                
                self.repeatQUESTanyway = True
                print('\nSomething went wrong in the loading process\n\nWe are hence going to repeat the QUEST.')
    
        # update coefficients (even if uniform prior, just to initialize values)       
        self.coeffs = {
            
            'alpha' : self.ranges['alpha'] @ self.prior.sum(axis=1),
            'beta' : self.ranges['beta'],
            'gamma' : self.ranges['gamma'] @ self.prior.sum(axis=0)
            
            }

    
    
    
    def __pmf_model(self):
        
        alpha = self.space['alpha']
        beta = self.space['beta']
        gamma = self.space['gamma']
        
        cnt = self.ranges['contrast']
        
        presp = np.zeros([self.depth, self.depth, self.cntdepth])
        
        accpos = 0
        for x in cnt:
            
            presp[:, :, accpos] = gamma + (1 - gamma)/(1 + np.exp(-beta*(x-alpha)))   # 1/(1 + np.exp(-beta*(x-alpha)))

            accpos +=1
        
        self.presp = presp
        
    
    
    def __invert(self, p):
        
        x = -(1/self.coeffs['beta']) * np.log((1-p)/(p - self.coeffs['gamma'])) + self.coeffs['alpha']
        
        return(x)
        
    
    def update(self, this_resp, conf_resp, this_trial, win, cfg, RT=0):
        
        # before updating the bayes:
        # i) convert score from [-1 1] to 0 or 1
        # ii) log trial features into data
        # iii) provide feedback
        
        
        resp = this_resp   
                                   
        # current contrast
        x = this_trial.intval_def['trgt_cnt']
                
        # performance-based both for trials at threshold (alpha) and expected HR
        if this_trial.intval_def['target_present'] == 2:            
            istrialpresent = 1
        else:
            istrialpresent = this_trial.intval_def['target_present']
        
        
        iscorrectresponse = istrialpresent == resp
        
                                             
        if iscorrectresponse:
            
            colorscore = cfg.colors['greencorrect']
            
            if resp == 1:
                
                this_score = cfg.reward_cunt['H']
                
            else:
                
                this_score = cfg.reward_cunt['CR']
                
                
            if conf_resp == 1:
            
                this_confscore = cfg.reward_cunt['highconfright']
                colorscore_conf = cfg.colors['greencorrect']
                
            else:
                
                this_confscore = cfg.reward_cunt['lowconf']
                colorscore_conf = cfg.colors['yellowrand']

                                
        else:
                
            colorscore = cfg.colors["redwrong"]
            this_score = cfg.reward_cunt['wrong']
            
            if conf_resp == 1:
            
                this_confscore = cfg.reward_cunt['highconfwrong']
                colorscore_conf = cfg.colors['redwrong']

            else:
                
                this_confscore = cfg.reward_cunt['lowconf']
                colorscore_conf = cfg.colors['yellowrand']
                
        
        # compute new score based on the previous and current
        oldscore = self.expscore
        newscore = oldscore + this_score + this_confscore   
        
        # update trial number
        self.trialnum += 1
        
        # this new compensation
        tot_trials = 66 + 60*(self.QUESTcounter-1) + self.nblocks * 144
        dasKapital = cfg.from_score_to_EUR(newscore, self.trialnum, tot_trials)
        self.dasKapital = str(np.round(dasKapital, 2))
        
        # assign SDT label 
        signal_tag = this_trial.trialdef['target_present']
        signal_tag = (signal_tag>0)*2
        sdt_code = self.list_SDT[resp + signal_tag]
        
        acc_det = (resp == (signal_tag/2))*1
        
        # which presp assigned
        presps_list = [0, self.aimed_p_resp, self.p_thresh]
        this_presp = presps_list[signal_tag]
        
        # log data in the required format, based on the current trial occurrences     
        dat = [[
            this_trial.intval_def['cue_type'],          # 'cue_type'
            this_trial.intval_def['cue_valence'],       # 'cue_valence'
            this_trial.trialdef['target_present'],      # 'signal'
            this_trial.trialdef['cue_color'],           # 'color_cue'
            this_trial.intval_def['trgt_cnt'],          # 'cnt_probed'
            this_trial.intval_def['SNR'],               # 'energy_probed'
            this_trial.intval_def['phase'],             # 'phase_angle_probed'
            resp,                                       # 'bin_resp'
            conf_resp,                                  # 'confidence'
            RT,                                         # 'RT'
            this_trial.trialdef['feedback'],            # 'feed_type'
            this_score,                                 # 'score_detection'
            this_confscore,                             # 'score_confidence'
            this_score + this_confscore,                # 'score_trl'
            newscore,                                   # 'cumulative_score'
            dasKapital,                                 # 'cumulative_EUR'
            this_trial.intval_def['jitter_frames'],     # 'jitter
            acc_det,                                    # 'accurate_detection'
            sdt_code,                                   # 'SDT_type'
            this_presp,                                 # 'Presp aimed'
            self.coeffs['alpha'],                       # 'alpha_est'
            self.coeffs['gamma'],                       # 'gamma_est'
            ]]

        # convert to dataframe...
        thisdata = pd.DataFrame(data=dat, columns=self.colnames_data)                

        # ... and append to the original dataset
        self.data = self.data.append(thisdata)

        # append timestamps for the current trial
        temp_ts = pd.DataFrame(data=[this_trial.time_intervals], 
                               columns=self.timestamps_colnames)

        self.timestamps = self.timestamps.append(temp_ts)
        
        
        # append current randomstate tuple
        self.randomStates.append(this_trial.RNGstate)
  
        # dynamic animation for score
        animated_score = np.round(np.linspace(oldscore, newscore, this_trial.frames['int_break']), 2)

        myscore = visual.TextStim(win,
                                  text = "this score (detection): " + str(this_score),
                                  height = 30,
                                  color= list(colorscore),
                                  pos = (0, 160),
                                  autoLog=False
                                  )

        myscore_confidence = visual.TextStim(win,
                                             text = "this score (confidence): " + str(this_confscore),
                                             height = 30,
                                             color= list(colorscore_conf),
                                             pos = (0, 100),
                                             autoLog=False
                                             )

        dynamicscore = visual.TextStim(win,
                                  text = "overall score: " + str(oldscore),
                                  height = 30,
                                  color= list(cfg.colors['darkgrey']),
                                  pos = (0, -100),
                                  autoLog=False
                                  )

        for iframe in range(this_trial.frames['int_break']):
            
            myscore.draw()
            myscore_confidence.draw()
            win.flip()
            
        
        for iframe in range(this_trial.frames['int_break']):
            
            flippedscore = animated_score[iframe]
            dynamicscore.text = "overall score: " + str(flippedscore)
            
            myscore.draw()
            myscore_confidence.draw()
            dynamicscore.draw()
            win.flip()
            
        
        self.expscore = newscore
        
        if x == -np.inf:
            
            idx_x = 0

        else:
        
            idx_x = np.abs(self.ranges['contrast'] - x).argmin() # find the contrast value in the grid closer to the contrast acually presented

                
        # update prior                
        if resp == 1:            
            p = self.presp[:, :, idx_x]
            
        elif resp == 0:            
            p = 1 - self.presp[:, :, idx_x]
                    
        self.prior = (p * self.prior) / (p * self.prior).sum() # Bayes theorem
    
        # update coefficients        
        self.coeffs = {
            
            'alpha' : self.ranges['alpha'] @ self.prior.sum(axis=1),
            'beta' : self.ranges['beta'],
            'gamma' : self.ranges['gamma'] @ self.prior.sum(axis=0)
            
            }
    
        # estimated function
        self.estimated = self.coeffs['gamma'] + (
            1- self.coeffs['gamma'])/(1 + np.exp(-self.coeffs['beta']*(
            self.ranges['contrast']-self.coeffs['alpha'])))
        
        # debug purpose
        # print(x)
        # print(idx_x)
        # print(self.coeffs)
        # time.sleep(.5)
        
        # get current magnitude levels
        # Mag1 = standard .001 (-inf). p is assumed to be ~gamma
        # Mag2 = func_inv(.2867*2 - gamma)
        # Mag3 = func_inv(.5)
        # Mag4 = func_inv(.83)   
        
        
        # determine FAR to get the next better estimate of expected HR
        FA_total = self.data['bin_resp'].loc[self.data['signal']==0].mean()

        if (FA_total is None) or (FA_total == 0):
            
            FA_total = .0000001   
        
        
        
        aimed_p_resp = norm.cdf(self.expctd_dP + norm.ppf(FA_total))
        p_thresh = (1 + self.coeffs['gamma'])/2
        
        # enforce safety breaks for the range not to go too much up (or down)
        if aimed_p_resp < self.brake_low:
            
            aimed_p_resp = self.brake_low
            
        elif aimed_p_resp > self.brake_up:
            
            aimed_p_resp = self.brake_up
            
            
        if p_thresh < self.brake_low:
            
            p_thresh = self.brake_low
            
        elif p_thresh > self.brake_up:
            
            p_thresh = self.brake_up
            
            
        self.aimed_p_resp = np.copy(aimed_p_resp)
        self.p_thresh = np.copy(p_thresh)        
              
        Mnoise = -np.inf # stable noise at contrast 0.                         before:  self.__invert(self.expctd_FA)
        Msignal = self.__invert(aimed_p_resp)
        Mtreshold = self.__invert(p_thresh)
            
        # get the current contrast definition for noise and signal ( and threshold, obtained in a subset of neutral trials)
        self.cntvals = np.array([Mnoise, Msignal, Mtreshold])


        print('#################### TRIAL SUMMARY #########################\n\n')
        print('signal type = ' + str(this_trial.trialdef['target_present']))
        print('participant response = ' + sdt_code)
        print('expected p = ' + str(this_presp))
        print('participant score = ' + str(newscore))
        print('participant reward = ' + str(self.dasKapital) + ' EUR')
        print('trial : ' + str(self.trialnum) + '/' + str(tot_trials) + '\n\n')

        print('################ technical, nerdy stuff ####################\n\n')     

        out_cnt = [round(entry, 3) for entry in self.cntvals]
        out_coeffs = {key : round(self.coeffs[key], 3) for key in self.coeffs}
        print('aimed p thresh = ' + str(p_thresh))
        print('aimed p resp = ' + str(aimed_p_resp) + '\n')

        print(out_cnt)
        print(out_coeffs)

        print('\n\n############################################################')
        
        
    def fakeupdate(self, this_resp, conf_resp, this_trial, win, cfg):
        
        # before updating the bayes:
        # i) convert score from [-1 1] to 0 or 1
        # ii) log trial features into data
        # iii) provide feedback
           
        resp = this_resp   
                                   
        # current contrast
        x = this_trial.intval_def['trgt_cnt']
                
        # performance-based both for trials at threshold (alpha) and expected HR
        if this_trial.intval_def['target_present'] == 2:            
            istrialpresent = 1
        else:
            istrialpresent = this_trial.intval_def['target_present']
                
        iscorrectresponse = istrialpresent == resp
                     
        if iscorrectresponse:
            
            colorscore = cfg.colors['greencorrect']
            
            if resp == 1:
                
                this_score = cfg.reward_cunt['H']
                
            else:
                
                this_score = cfg.reward_cunt['CR']
                
                
            if conf_resp == 1:
            
                this_confscore = cfg.reward_cunt['highconfright']
                colorscore_conf = cfg.colors['greencorrect']
                
            else:
                
                this_confscore = cfg.reward_cunt['lowconf']
                colorscore_conf = cfg.colors['yellowrand']

                                
        else:
                
            colorscore = cfg.colors["redwrong"]
            this_score = cfg.reward_cunt['wrong']
            
            if conf_resp == 1:
            
                this_confscore = cfg.reward_cunt['highconfwrong']
                colorscore_conf = cfg.colors['redwrong']

            else:
                
                this_confscore = cfg.reward_cunt['lowconf']
                colorscore_conf = cfg.colors['yellowrand']
                        
        
        # compute new score based on the previous and current
        oldscore = self.expscore
        newscore = oldscore + this_score + this_confscore   
        
        # update trial number
        self.trialnum += 1
        
        # this new compensation
        tot_trials = 66 + 60*self.QUESTcounter + self.nblocks * 144
        dasKapital = cfg.from_score_to_EUR(newscore, self.trialnum, tot_trials)
        self.dasKapital = str(np.round(dasKapital, 2))

        # assign SDT label 
        signal_tag = this_trial.trialdef['target_present']
        signal_tag = (signal_tag>0)*2
        sdt_code = self.list_SDT[resp + signal_tag]
        
        acc_det = (resp == (signal_tag/2))*1
        
        # which presp assigned
        presps_list = [0, self.aimed_p_resp, self.p_thresh]
        this_presp = presps_list[signal_tag]
        
        # log data in the required format, based on the current trial occurrences     
        dat = [[
            this_trial.intval_def['cue_type'],          # 'cue_type'
            this_trial.intval_def['cue_valence'],       # 'cue_valence'
            this_trial.trialdef['target_present'],      # 'signal'
            this_trial.trialdef['cue_color'],           # 'color_cue'
            this_trial.intval_def['trgt_cnt'],          # 'cnt_probed'
            this_trial.intval_def['SNR'],               # 'energy_probed'
            this_trial.intval_def['phase'],             # 'phase_angle_probed'
            resp,                                       # 'bin_resp'
            conf_resp,                                  # 'confidence'
            0,                                          # 'RT'
            this_trial.trialdef['feedback'],            # 'feed_type'
            this_score,                                 # 'score_detection'
            this_confscore,                             # 'score_confidence'
            this_score + this_confscore,                # 'score_trl'
            newscore,                                   # 'cumulative_score'
            dasKapital,                                 # 'cumulative_EUR'
            this_trial.intval_def['jitter_frames'],     # 'jitter
            acc_det,                                    # 'accurate_detection'
            sdt_code,                                   # 'SDT_type'
            this_presp,                                 # 'Presp aimed'
            self.coeffs['alpha'],                       # 'alpha_est'
            self.coeffs['gamma'],                       # 'gamma_est'
            ]]

        # convert to dataframe...
        thisdata = pd.DataFrame(data=dat, columns=self.colnames_data)                

        # append data
        self.data = self.data.append(thisdata)

          
        
        oldscore = self.expscore
        newscore = oldscore + this_score        
        animated_score = np.round(np.linspace(oldscore, newscore, this_trial.frames['int_break']), 2)


        myscore_confidence = visual.TextStim(win,
                                             text = "this score (confidence): " + str(this_confscore),
                                             height = 30,
                                             color= list(colorscore_conf),
                                             pos = (0, 100),
                                             autoLog=False
                                             )


        myscore = visual.TextStim(win,
                                  text = "this score: " + str(this_score),
                                  height = 30,
                                  color= list(colorscore),
                                  pos = (0, 160),
                                  autoLog=False
                                  )

        dynamicscore = visual.TextStim(win,
                                  text = "overall score: " + str(oldscore),
                                  height = 30,
                                  color= list(cfg.colors['darkgrey']),
                                  pos = (0, -100),
                                  autoLog=False
                                  )

        for iframe in range(this_trial.frames['int_break']):
            
            myscore.draw()
            myscore_confidence.draw()
            win.flip()
            
        
        for iframe in range(this_trial.frames['int_break']):
            
            flippedscore = animated_score[iframe]
            dynamicscore.text = "overall score: " + str(flippedscore)
            
            myscore.draw()
            myscore_confidence.draw()
            dynamicscore.draw()
            win.flip()
            
        
        self.expscore = newscore
        # debug purpose
        # print(x)
        # print(idx_x)
        # print(self.coeffs)
        # time.sleep(.5)
        
        # get current magnitude levels
        # Mag1 = standard .001 (-inf). p is assumed to be ~gamma
        # Mag2 = func_inv(.2867*2 - gamma)
        # Mag3 = func_inv(.5)
        # Mag4 = func_inv(.83)   
        
        print(self.cntvals)
        
        print('################# TRIAL SUMMARY (QUEST) ####################\n\n')
        print('signal type = ' + str(this_trial.trialdef['target_present']))
        print('participant response = ' + sdt_code)
        print('expected p = ' + str(this_presp))
        print('participant score = ' + str(newscore))
        print('participant reward = ' + str(self.dasKapital) + ' EUR')
        print('trial : ' + str(self.trialnum) + '/' + str(tot_trials) + '\n\n')
        print('################ technical, nerdy stuff ####################\n\n')

        out_cnt = [round(entry, 3) for entry in self.cntvals]
        out_coeffs = {key : round(self.coeffs[key], 3) for key in self.coeffs}
        print(out_cnt)
        print(out_coeffs)

        print('\n\n############################################################')

        
        
    # null things that should not stay after quest, but by leaving the prior
    def end_quest(self):
          
        self.QUESTcounter += 1
        
        # assume here that the performance is OK, and check afterward
        repeatQUEST = False
        
        # determine d' and criterion in the passed block
        HR_block = self.data['bin_resp'].loc[self.data['signal']==1].mean()
        FA_block = self.data['bin_resp'].loc[self.data['signal']==0].mean()
 
        # correct for infinite values in SDT
        if HR_block == 1:           
            HR_block = .9999999           
        elif FA_block == 0:           
            FA_block = .0000001           
        elif HR_block == 0:           
            HR_block = .0000001           
        elif FA_block == 1:           
            FA_block = .9999999

        
        dprime_block = norm.ppf(HR_block) - norm.ppf(FA_block)
        crit_block = -.5 * (norm.ppf(HR_block) + norm.ppf(FA_block))
        
        print('QUEST dprime: ' + str(round(dprime_block, 2)))
        print('QUEST criterion: ' + str(round(crit_block, 2)))
        
        # check whether d' and criterion are out of boundaries                     
        isoutdprime = ((dprime_block<self.bound_dP['lower']) or
                       (dprime_block>self.bound_dP['upper']))
                     
        isoutoutdprime = ((dprime_block<self.bound_dP['lowerlower']) or
                          (dprime_block>self.bound_dP['upperupper']))

        badintervals = isoutdprime
        badbadintervals = isoutoutdprime

        # if d' or criterion are out of boundaries...
        if badintervals:
            
            if badbadintervals:
            
                # check how many times we've already run QUEST
                if self.QUESTcounter < 2:
                    
                    # if less than 2 times repeat it
                    repeatQUEST = True                   
                    print('WARNING: QUEST is gonna be repeated, values out of range')

                else:
                    print('WARNING: QUEST is NOT gonna be repeated, (already repeated once), but values are still out of range')
                    print('Hence please keep an eye on performance')
                    
                    
                # in either cases, (quest > 1 or not), heavily smooth the prior
                self.smooth_prior(kernellength=5)
                    
            else:
                
                # just apply a mild prior smoothing
                self.smooth_prior(kernellength=3)
                
                
        # null the present data (it has already been saved on disk)        
        self.data = pd.DataFrame(columns=self.colnames_data)

        # 0 subject score
        if self.expscore < 0:
            self.expscore = 0
                
        # return indication of repetition to be observed in main
        return repeatQUEST
        
        
            
    # smooth prior in case of poor estimation of HR or FA
    def smooth_prior(self, kernellength=3):
        
        self.prior = gaussian_filter(self.prior, kernellength)
        
        # re-normalize prior to sum 1, in case the gaussian kernel
        # did not achieve a perfect result in this sense
        self.prior = self.prior/self.prior.sum()
        


    # get a summary to pipe into telegrambot        
    def get_summary(self, ntrials):
        
        # determine d' and criterion for the whole experiment so far
        HR_total = self.data['bin_resp'].loc[self.data['signal']==1].mean()
        FA_total = self.data['bin_resp'].loc[self.data['signal']==0].mean()
 
        # correct for infinite values in SDT
        if HR_total == 1:           
            HR_total = .9999999           
        elif FA_total == 0:           
            FA_total = .0000001           
        elif HR_total == 0:           
            HR_total = .0000001           
        elif FA_total == 1:           
            FA_total = .9999999
        
        dprime_total = norm.ppf(HR_total) - norm.ppf(FA_total)
        crit_total = -.5 * (norm.ppf(HR_total) + norm.ppf(FA_total))
            
        # determine d' and criterion in the last N trials
        red_data = self.data.tail(ntrials)
        
        HR_ntrials = red_data['bin_resp'].loc[red_data['signal']==1].mean()
        FA_ntrials = red_data['bin_resp'].loc[red_data['signal']==0].mean()
 
        # correct for infinite values in SDT
        if HR_ntrials == 1:           
            HR_ntrials = .9999999           
        elif FA_ntrials == 0:           
            FA_ntrials = .0000001           
        elif HR_ntrials == 0:           
            HR_ntrials = .0000001           
        elif FA_ntrials == 1:           
            FA_ntrials = .9999999
        
        dprime_ntrials = norm.ppf(HR_ntrials) - norm.ppf(FA_ntrials)
        crit_ntrials = -.5 * (norm.ppf(HR_ntrials) + norm.ppf(FA_ntrials))

        # if there's a critical discrepancy from the expected values in the last N trials
        # apply a mild prior smoothing                     
        isoutoutdprime = ((dprime_ntrials<self.bound_dP['lowerlower']) or
                          (dprime_ntrials>self.bound_dP['upperupper']))
                  
        badbadintervals = isoutoutdprime

        if badbadintervals:
            
            self.smooth_prior(kernellength=3)     
            
            # add the present smoothing to the counter
            self.counter_smoothingprior += 1
            print('prior smoothing procedure applied')


        # fetch current contrast definition for signal and noise
        noise_cnt = self.cntvals[0]
        signal_cnt = self.cntvals[1]

        # create summary dictionary
        Nstr = str(ntrials)
        
        summary_dict = {
            
            'HR_total' : round(HR_total, 2),
            'FA_total' : round(FA_total, 2),
            'dprime_total' : round(dprime_total, 2),
            'crit_total' : round(crit_total, 2),
            'HR_' + Nstr : round(HR_ntrials, 2),
            'FA_' + Nstr : round(FA_ntrials, 2),
            'dprime_' + Nstr : round(dprime_ntrials, 2),
            'crit_' + Nstr : round(crit_ntrials, 2),
            'noise_cnt' : round(np.exp(noise_cnt), 2),
            'signal_cnt' : round(np.exp(signal_cnt), 2),
            'alpha' : round(np.exp(self.coeffs['alpha']), 3),
            'beta' : round(self.coeffs['beta'], 3),
            'OUT_OF_BOUNDARIES': badbadintervals

            }

        # convert dictionary into nicely formatted string for sending 
        msg_send = ''
        for k,v in summary_dict.items():
            
            temp_ = k + '=' + str(v) + '\n'
            msg_send += temp_

        # send everything to the current receiver via telegram
        bot.sendMessage(chat_id=receiver, text=msg_send)


            
            
                
                
            
#%% Create a function (NOT A CLASS THIS TIME) to send triggers to EEG and Eyetracking togetehr

def send_trigger(msg, ET=None, EEGport=None, isEEG=True, isET=True, trigdur=.002):
    
    if isET:
            
        notify(message=str(msg), el=ET)
    
    if isEEG:
        
        EEGport.setData(msg)
        core.wait(trigdur)
        EEGport.setData(0)
    
    return


    
    
    
    
    
