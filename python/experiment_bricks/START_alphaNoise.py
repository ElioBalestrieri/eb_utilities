#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 15:39:22 2020

@author: ebalestr
"""

from exp_bricks import DialogBox, ExpBuilder
from hlpr import StartObserver
from psychopy import parallel
from EyelinkWrapper import EyelinkStart, EyelinkStop, EyelinkCalibrate


# get the main settings of the current experiment
howruns = DialogBox()

# initialize experiment structure for the main experiment
exp_handle = ExpBuilder(howruns)

#%% give subject interactive instructions
giveINSTRUCTIONS = howruns.inputsubject['instructions']
if giveINSTRUCTIONS:
    
    # initialize subject object    
    foo = StartObserver()
    exp_handle.run_instructions(foo)


#%% QUEST the subject until reasonable performance is achieved (or 2 QUESTs run)
repeatQUEST = howruns.inputsubject['QUEST']

# initialize subject object
subj_info = StartObserver(nblocks=howruns.inputsubject['end_block'],
                          strtblock=howruns.inputsubject['start_block'])

repeatQUEST = repeatQUEST or subj_info.repeatQUESTanyway


while repeatQUEST:    

    # actually run QUEST
    subj_info, repeatQUEST = exp_handle.run_QUEST(subj_info)



#%% for loop through blocks

# initialize eyetracking if required by the experimenter
if howruns.inputsubject['eyetracking']:   
    ET = EyelinkStart((1920, 1080), 'AC_' + howruns.inputsubject['subjcode'], 
                      exp_handle.win)
else:
    ET = None
    
# open trigger port if EEG is being recorded
if howruns.inputsubject['EEG']:
    EEGport = parallel.ParallelPort('/dev/parport0')
else:    
    EEGport = None

# loop through the experiment blocks
for iblock in exp_handle.blockrange:
    
    # jump into calibration if necessary
    if (((iblock + 1) > howruns.inputsubject['start_block']) and 
         howruns.inputsubject['eyetracking']):
        
        ET =  EyelinkCalibrate(el=ET)
           
    subj_info = exp_handle.run_block(subj_info, iblock, ET=ET, EEGport=EEGport)

# stop eyetracking under the same conditions
if howruns.inputsubject['eyetracking']:   
    EyelinkStop('AC_' + howruns.inputsubject['subjcode'], el=ET)



#%% QUESTIONNAIRE

# exp_handle.run_questionnaire()



