#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 15:39:39 2020

@author: ebalestr
"""


import tkinter as tk
import numpy as np
import os
from psychopy import visual, core, event, logging
from psychopy.hardware import joystick
from hlpr import PrepareConfig, TrialExecution, send_trigger
import pandas as pd
from datetime import datetime
import random
from scipy.stats import norm
import pickle

from EyelinkWrapper import *




#%% =============================================================================
# DIALOG BOX
# =============================================================================

class DialogBox:
    
    def __init__(self):
        
        self.window = tk.Tk()

        mainframe = tk.Frame(self.window)
        mainframe.pack()
        
        # subject code
        tk.Label(mainframe, text='Subj code').grid(
            row=0, column=0, sticky='W')
        SUBJCODE = tk.StringVar(self.window, value='00AA')
        subj = tk.Entry(mainframe, textvariable=SUBJCODE)
        subj.grid(row=0, column=1, sticky='W')
                        
        # starting block number
        tk.Label(mainframe, text='start block number').grid(
            row=1, column=0, sticky='W')
        STARTBLOCK = tk.IntVar(self.window, value=1)
        strtblck = tk.Entry(mainframe, textvariable=STARTBLOCK)
        strtblck.grid(row=1, column=1, sticky='W')
        
        # ending block number
        tk.Label(mainframe, text='end block number').grid(
            row=2, column=0, sticky='W')
        ENDBLOCK = tk.IntVar(self.window, value=7)
        endblck = tk.Entry(mainframe, textvariable=ENDBLOCK)
        endblck.grid(row=2, column=1, sticky='W')
        
        # instructions?
        tk.Label(mainframe, text='Instructions?').grid(
            row=3, column=0, sticky='W')
        INSTRUCTIONS = tk.BooleanVar(self.window, value=True)
        instr = tk.Entry(mainframe, textvariable=INSTRUCTIONS)
        instr.grid(row=3, column=1, sticky='W')
        
        # QUEST?
        tk.Label(mainframe, text='QUEST?').grid(
            row=4, column=0, sticky='W')
        QUEST = tk.BooleanVar(self.window, value=True)
        qst = tk.Entry(mainframe, textvariable=QUEST)
        qst.grid(row=4, column=1, sticky='W')

        # nplayers?
        tk.Label(mainframe, text='how many players?').grid(
            row=5, column=0, sticky='W')
        PLAYERS = tk.IntVar(self.window, value=0)
        nplayers = tk.Entry(mainframe, textvariable=PLAYERS)
        nplayers.grid(row=5, column=1, sticky='W')

        # language?
        tk.Label(mainframe, text='Language? (1=DE, 2=ENG)').grid(
            row=6, column=0, sticky='W')
        LANGUAGE = tk.IntVar(self.window, value=1)
        whichlang = tk.Entry(mainframe, textvariable=LANGUAGE)
        whichlang.grid(row=6, column=1, sticky='W')

        # EYETRACKING?
        tk.Label(mainframe, text='EYETRACKING?').grid(
            row=7, column=0, sticky='W')
        EYETRACKING = tk.BooleanVar(self.window, value=True)
        eytrack = tk.Entry(mainframe, textvariable=EYETRACKING)
        eytrack.grid(row=7, column=1, sticky='W')

        # EEG?
        tk.Label(mainframe, text='EEG?').grid(
            row=8, column=0, sticky='W')
        EEG = tk.BooleanVar(self.window, value=True)
        eeg = tk.Entry(mainframe, textvariable=EEG)
        eeg.grid(row=8, column=1, sticky='W')

        # VPStunden?
        tk.Label(mainframe, text='VP Stunden?').grid(
            row=9, column=0, sticky='W')
        VPStunden = tk.BooleanVar(self.window, value=False)
        vps = tk.Entry(mainframe, textvariable=VPStunden)
        vps.grid(row=9, column=1, sticky='W')


        
        # button to close window                
        button = tk.Button(self.window, text='Enter', command=self.__close_window)
        button.pack()
        
        
        self.window.mainloop()

        # collect all items into a dict
        self.inputsubject = {
            
            'subjcode' : SUBJCODE.get(),
            'start_block' : STARTBLOCK.get(),
            'end_block' : ENDBLOCK.get(),
            'instructions' : INSTRUCTIONS.get(),
            'QUEST' : QUEST.get(),
            'language' : LANGUAGE.get(),
            'eyetracking' : EYETRACKING.get(),
            'EEG' : EEG.get(),
            'VPStunden' : VPStunden.get()
            
            }
        
        # determine whether path for logfiles exist. if not, create that
        self.logfolder = '../Logfiles/' + self.inputsubject['subjcode'] + '/'
        
        if not os.path.exists(self.logfolder):
            
            os.makedirs(self.logfolder)
            
        # determine blocks range given the interval defined
        self.blockrange = np.arange(self.inputsubject['start_block']-1, 
                                    self.inputsubject['end_block']) 
        
        # determine (here in the code, manually) how many players will take part in the competition
        # (apart from the current observer)
        self.nplayers = PLAYERS.get()
        
        
        

    def __close_window(self):
        
        self.window.destroy()



#%% =============================================================================
# BLOCK BUILDER
# =============================================================================

class ExpBuilder:
    
    def __init__(self, howruns):
        
        # assign some input from dialog box important for structuring experiment execution
        self.logfolder = howruns.logfolder
        self.subjcode = howruns.inputsubject['subjcode']
        self.blockrange = howruns.blockrange
        self.nplayers = howruns.nplayers
        self.language = howruns.inputsubject['language']
        self.eyetrackingON = howruns.inputsubject['eyetracking']
        self.eegON = howruns.inputsubject['EEG']
        self.VPStunden = howruns.inputsubject['VPStunden']
        
        # generate a player dictionary
        # this call creates as well a pd dataframe containing the scores of the subject
        # and of all the other players blockwise, and the d' of the subject at each single block
        # + the decision of the algorithm to broaden or not the prior
        self.__generate_players_dict()

        # open psychopy win
        self.win = visual.Window(units ='pix',
                                 size = [1920, 1080],
                                 fullscr = True,
                                 screen = 0,
                                 winType='pyglet')
        
        self.fps = self.win.getActualFrameRate(nIdentical=60, 
                                               nMaxFrames=120, 
                                               nWarmUpFrames=120)

        print('\n\n############################################################\n\n')
        print('detected fps: ' + str(round(self.fps)))
        print('\n\n############################################################\n\n')
        

        # genrate questions for the end of experiemnt
        self.__generate_questions()
        
        # attach joystick to the experiment class
        joystick.backend='pyglet'  # must match the Window
        nJoys = joystick.getNumJoysticks()  # to check if we have any
        self.joy = joystick.XboxController(nJoys-1)  # id must be <= nJoys - 1


        
    def run_block(self, subj_info, iblock, ET=None, EEGport=None):
        
        cfg_info = PrepareConfig(isET=self.eyetrackingON, isEEG=self.eegON,
                                 language=self.language,
                                 isVPSTUNDEN=self.VPStunden)
        
        cfg_info.to_frames(self.fps)
        
        itrl = 0
        consecutivefixbreak = 0
        
        
        send_trigger(cfg_info.triggers['savebox_EEG'], ET=ET, 
                             EEGport=EEGport,
                             isET=self.eyetrackingON,
                             isEEG=self.eegON)
        
        while itrl < cfg_info.expstruct['trlXblock']: 
            
            this_trial = TrialExecution(cfg_info, subj_info, itrl)
        
            if (itrl == 0) and (consecutivefixbreak == 0):
                
                if self.language==1:
                    this_trial.give_them_a_break('Block ' + str(iblock + 1) + ' startet', 
                                                 self.win, language=self.language) 
                   
                else:
                    this_trial.give_them_a_break('Block ' + str(iblock + 1) + ' is starting', 
                                                 self.win, language=self.language) 
                
                send_trigger(cfg_info.triggers['start_EEG_save'], ET=ET, 
                             EEGport=EEGport,
                             isET=self.eyetrackingON,
                             isEEG=self.eegON)
    
            this_trial.preallocate_arrays(cfg_info, self.win)  
            this_trial.run_interval(self.win, ET=ET, EEGport=EEGport, 
                                    eyetrackon=self.eyetrackingON,
                                    isEEG=self.eegON)
            
            if this_trial.saccadeflag:
                
                cfg_info.correct_trialorder(itrl)
                consecutivefixbreak += 1
                
                if self.language==1:
                    this_trial.give_them_a_break('Sorry, du hast das Zentrum nicht fixiert', 
                                                 self.win, language=self.language)
                else:
                    this_trial.give_them_a_break('Sorry, you did not fixate the center', 
                                                 self.win, language=self.language)
                    

                
                if consecutivefixbreak > 5:
                    
                    # jump back into recalibration
                    if self.language==1:

                        this_trial.give_them_a_break('Es tut mir leid, wir müssen die Eyetracking neu kalibrieren...\n' + 
                                                     'wende dich bitte an den/die Versuchleiter*in', 
                                                     self.win, printmessage='RECALIBRATTION NEEDED!!', 
                                                     language=self.language)
                    else:
                        this_trial.give_them_a_break('Sorry, we have to recalibrate...\n' + 
                                                     'Please contact the experimenter',
                                                     self.win, printmessage='RECALIBRATTION NEEDED!!', 
                                                     language=self.language)
                        
                    
                    
                    self.win.flip()
                    
                    send_trigger(cfg_info.triggers['pause_EEG_save'], ET=ET, 
                             isET=self.eyetrackingON,
                             EEGport=EEGport,
                             isEEG=self.eegON)                    
                    
                    EyelinkCalibrate(el=ET)
                    
                    send_trigger(cfg_info.triggers['start_EEG_save'], ET=ET, 
                             isET=self.eyetrackingON,
                             EEGport=EEGport,
                             isEEG=self.eegON)

                
                
                continue                          # this prevents the while loop to go on, and continues from the previous iteration                

                
            else:
                
               consecutivefixbreak = 0         
                
            
            
            this_resp, this_rating, RT = this_trial.get_rating(self.win, cfg_info, 
                                                               self.joy, ET=ET,
                                                               EEGport=EEGport,
                                                               isET=self.eyetrackingON,
                                                               isEEG=self.eegON)
            
            cfg_info.imagecontainer(this_trial=this_trial, itrl = itrl, 
                                    resp=this_rating)
            
            subj_info.update(this_resp, this_rating, this_trial, self.win, cfg_info, RT=RT)
    
            if np.mod(itrl+1, 36)==0:

                subj_info.get_summary(36)
                
                if itrl < 36*4:
                    
                    if self.language==1:
                        this_trial.give_them_a_break('Eine kurze Pause machen!\n\n'
                                                     'Sie haben ' + subj_info.dasKapital + ' EUR verdient\n' 
                                                     'Sie haben ' + str(subj_info.expscore) + ' Punkte erzielt\n\n',
                                                     self.win)
                    else:
                        this_trial.give_them_a_break('Take a short break!\n\n' 
                                                     'You earned ' + subj_info.dasKapital + ' EUR!\n'
                                                     'You scored ' + str(subj_info.expscore) + ' points!\n\n',
                                                     self.win)

            # update trial number        
            itrl += 1


        out_data = subj_info.data
        out_timestamps = subj_info.timestamps
        
        # get timestamp
        dateObj = datetime.now()
        tstamp = dateObj.strftime("%Y%b%d%H%M%S")
        
        # generate filenames
        dataname = self.logfolder + tstamp + '_' + self.subjcode + '_block' + str(iblock+1) + '.csv'
        priorname = self.logfolder + tstamp + '_' + self.subjcode + '_prior'
        timestampsname = self.logfolder + tstamp + '_' + self.subjcode + '_TIMESTAMPS_b' + str(iblock+1) + '.csv'
        RNGstatename = self.logfolder + self.subjcode + '_RNGstates_b' + str(iblock+1) + '.pickle'
        
        # save behavioural data
        out_data.to_csv(path_or_buf=dataname)
        out_timestamps.to_csv(path_or_buf=timestampsname)
        
        # save prior
        out_prior = subj_info.prior
        np.save(priorname, out_prior)
        
        # save the randm states that genrated the images
        with open(RNGstatename, 'wb') as f:
            pickle.dump(subj_info.randomStates, f)
    
        # save images
        cfg_info.imagecontainer(save=True, blocknumber = iblock, 
                                subjcode = self.subjcode,
                                path=self.logfolder)
        
        # measure current performance
        # if necessary, broaden prior
        # if required, start the hunger games
        subj_info = self.__close_block(out_data, subj_info, iblock, 
                                       hungergames=False)
        
        
        dfplayersname = self.logfolder + self.subjcode + '_PLAYERS' + '_block' + str(iblock+1) + '.csv'
        self.summaryDF.to_csv(path_or_buf=dfplayersname)
    
    
    
        if self.language==1:
            this_trial.give_them_a_break('Der Block Nummer ' + str(iblock+1) + ' ist vorbei!\n\n'
                                         'Sie haben ' + subj_info.dasKapital + ' EUR verdient\n'
                                         'Sie haben ' + str(subj_info.expscore) + ' Punkte erzielt\n\n',
                                         self.win)
        else:
            this_trial.give_them_a_break('The block number ' + str(iblock+1) + ' is over!\n\n'
                                         'You earned ' + subj_info.dasKapital + ' EUR!\n'
                                         'You scored ' + str(subj_info.expscore) + ' points!\n\n',
                                         self.win)

    
        # announce the jump back into recalibration
        if self.language==1:
            this_trial.give_them_a_break('Wir müssen die Eyetracking neu kalibrieren...\n' + 
                                         'wende dich bitte an den/die Versuchleiter*in', 
                                         self.win, printmessage='RECALIBRATTION NEEDED!!', 
                                         language=self.language)
        else:
            this_trial.give_them_a_break('We have to calibrate again...\n' + 
                                         'Please contact the experimenter',
                                         self.win, printmessage='RECALIBRATTION NEEDED!!', 
                                         language=self.language)
            

        self.win.flip()



        # pause EEG recording during calibration / inter block break        
        send_trigger(cfg_info.triggers['pause_EEG_save'], ET=ET, 
                     EEGport=EEGport,
                     isET=self.eyetrackingON,
                     isEEG=self.eegON)                    



        # # if this is the last block, close window
        # if iblock == self.blockrange[-1]:          
        #     self.win.close()
            
        # return the subject info with associated prior    
        return subj_info



    def run_QUEST(self, subj_info):
        
        cfg_info = PrepareConfig(isQUEST=True,
                                 language=self.language,
                                 isVPSTUNDEN=self.VPStunden)
        cfg_info.to_frames(self.fps)

        if subj_info.QUESTcounter == 0:
            
            itrl = 0
            
        else:
            
            itrl = 6
            
            

        while itrl < cfg_info.expstruct['trlXblock']: 
            
            this_trial = TrialExecution(cfg_info, subj_info , itrl)
            
            if itrl == 0:
                
                if self.language == 1:
                    this_trial.give_them_a_break('QUEST startet', self.win, 
                                                 language=self.language)                  
                    
                else:
                    this_trial.give_them_a_break('QUEST is starting', self.win, 
                                                 language=self.language)  
            
            this_trial.preallocate_arrays(cfg_info, self.win)  
            this_trial.run_interval(self.win)
            this_resp, this_rating, RT = this_trial.get_rating(self.win, cfg_info, self.joy)
            
            if itrl > 5:
            
                subj_info.update(this_resp, this_rating, this_trial, self.win, cfg_info)
                cfg_info.imagecontainer(this_trial=this_trial, itrl = itrl, 
                                        resp=this_rating)
                                
            else:
                
                subj_info.fakeupdate(this_resp, this_rating, this_trial, self.win, cfg_info)
        
            
            itrl += 1
        
        cfg_info.imagecontainer(save=True, blocknumber = 0, subjcode = self.subjcode + 'QUEST',
                                path=self.logfolder)
        
        
        QUESTdata = subj_info.data
        QUESTprior = subj_info.prior
        
        # save values
        QUESTdata.to_csv(path_or_buf= self.logfolder + self.subjcode + 'QUESTdata.csv')
        np.save(self.logfolder + self.subjcode + 'QUESTprior', QUESTprior)
        
        
        subj_info.get_summary(30)
        repeatQUEST = subj_info.end_quest()
        
        if (not repeatQUEST) or (subj_info.QUESTcounter>1):

            if self.language == 1:

                this_trial.give_them_a_break('QUEST ist fertig!\n\n'
                                             'Sie haben ' + subj_info.dasKapital + ' EUR verdient!\n'
                                             'Sie haben ' + str(subj_info.expscore) + ' Punkte erzielt!\n\n',
                                             self.win)
 
            else:

                this_trial.give_them_a_break('QUEST is over!\n\n'
                                             'You earned ' + subj_info.dasKapital + ' EUR!\n'
                                             'You scored ' + str(subj_info.expscore) + ' points!\n\n',
                                             self.win)
       
        self.win.flip()

        return subj_info, repeatQUEST
        

    def run_instructions(self, subj_info):

        cfg_info = PrepareConfig(isINSTRUCTIONS=True,
                                 language=self.language)
        cfg_info.to_frames(self.fps)

        list_steps = ['start', 'continue', 'signal_example', 'noise_example',
                      'scoring1', 'scoring2', 'neutral', 'close_practice1', 
                      'close_practice2']

        istep = 0
        itrl = 0
        
        while istep < len(list_steps):

            thi_label = list_steps[istep]
            
            if istep < 2:
            
                istep, moveforward = self.__give_instructions(thi_label, istep)
                
            elif istep == 2:

                istep, moveforward = self.__give_instructions(thi_label, istep)

                if moveforward:
                    
                    this_trial = TrialExecution(cfg_info, subj_info , 1)        
                    this_trial.preallocate_arrays(cfg_info, self.win)  
                    this_trial.run_interval(self.win)

            elif istep == 3:

                istep, moveforward = self.__give_instructions(thi_label, istep)

                if moveforward:
                    
                    this_trial = TrialExecution(cfg_info, subj_info , 0)        
                    this_trial.preallocate_arrays(cfg_info, self.win)  
                    this_trial.run_interval(self.win)

            elif (istep>3) and (istep<6):
            
                istep, moveforward = self.__give_instructions(thi_label, istep)
               
            elif istep==6:
                
                itrl = 1
                this_trial = TrialExecution(cfg_info, subj_info , itrl)        
                this_trial.preallocate_arrays(cfg_info, self.win)  
        
                istep, moveforward = self.__give_instructions(thi_label, istep)
                
                if moveforward:
                
                    this_trial.run_interval(self.win)
                    this_resp, this_rating, RT = this_trial.get_rating(self.win, cfg_info, self.joy)            
                    subj_info.fakeupdate(this_resp, this_rating, this_trial, self.win, cfg_info)
                    
            elif istep > 6:
                
                istep, moveforward = self.__give_instructions(thi_label, istep)

    

    def run_questionnaire(self):
        
        cfg_info = PrepareConfig(isINSTRUCTIONS=True)
        cfg_info.to_frames(self.fps)
       
        questionnaire_answers = {}
        
        nkeys = len(self.dict_questionnaire.keys())
        
        for iQ in range(nkeys):
            
            code = 'Q' + str(iQ+1)

            rating = self.__get_questionnaire_answer(code, iQ, cfg_info)

            questionnaire_answers.update({code : rating})

    
        # transform into dataframe
        questionnaire_df = pd.DataFrame.from_dict([questionnaire_answers])

        # save
        questionnaire_df.to_csv(path_or_buf= self.logfolder + self.subjcode + '_FINALquestionnaire.csv')


        # now it's finally time to close the window
        self.win.close()


    # =============================================================================
    # internal methods    
    # =============================================================================


    def __generate_players_dict(self):
        
        self.playerdict = {}
        self.DFcol_list = []

        pl_code = np.arange(self.nplayers)+1
        np.random.shuffle(pl_code)

        for iPlayer in range(self.nplayers):
            
            thiname = 'subj' + str(pl_code[iPlayer])
            
            if iPlayer == 0:
 
               # add always a super player
                thirange = [1.35, 1.4]            
                
            elif iPlayer == 1:
                
                # add always a shitty player
                thirange = [.3, .4]
                
            elif iPlayer == 2:
                
                # add always a pretty good player
                thirange = [1.25, 1.39]

            elif iPlayer == 3:
                
                # add always a player approximately at participant's level
                thirange = [.9, 1.1]

            else:
                
                #all the other players
                thirange = [.5, .9]
                
            
            self.playerdict.update(
                {thiname : thirange}
                )
            
            self.DFcol_list.append(thiname)

        
        # generate pandas dataframe to contain block by block info on other players
        # and on the current one
        self.DFcol_list.append('subj_score')
        self.DFcol_list.append('subj_d')
        self.DFcol_list.append('subj_crit')
        self.DFcol_list.append('broaden_prior')
        self.DFcol_list.append('outofrange_counter')
        
        self.summaryDF = pd.DataFrame(columns=self.DFcol_list)


    def __close_block(self, out_data, subj_info, iblock, hungergames=False):

        # get only the last 120 trials to get an estimate of the last block
        red_data = out_data.tail(120)
        
        # determine d' and criterion in the passed block
        HR_block = red_data['bin_resp'].loc[red_data['signal']==1].mean()
        FA_block = red_data['bin_resp'].loc[red_data['signal']==0].mean()
 
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

        # do not smooth prior here, but keep the count of how many times this happened      
        smoothprior = subj_info.counter_smoothingprior
        
        # keep the count of how many out of range were detected
        outofrange = subj_info.outofrange_counter
        
        
        # determine where the sd of the participant based on his current score            
        thisblockscore = red_data['cumulative_score'].sum()    
        if iblock > 0:
            
            try:
                
                previousblockscore = self.summaryDF['subj_score'].iloc[iblock-1]

            except:
                
                previousblockscore = thisblockscore
            
            if previousblockscore > thisblockscore:
                
                # worse
                subj_info.SD -= .025
                
            else:
                
                # better
                subj_info.SD += .1
            
        # define scaling factor 
        gaus_scale = subj_info.expscore/(subj_info.SD)
                        
        # determine the score of the other "participants"
        dict_players_scores = {}
        block_DF_updater = []

        for iPlayer in range(self.nplayers):
        
            thiname = self.DFcol_list[iPlayer]           
            thirange = self.playerdict[thiname]           
            thiSD = random.uniform(thirange[0], thirange[1])
              
            if hungergames:
                thiscore = np.abs(gaus_scale * thiSD)
            else:
                thiscore = 0
            
            dict_players_scores.update(
                {thiname : [thiscore]}
                )
        
            block_DF_updater.append(thiscore)
        
        
        # append participant score and name to dictionary
        dict_players_scores.update(
            {'your_score' : [subj_info.expscore]}
            )
        
        print(dict_players_scores)
        
        # append the remaining data to create a df for this block
        block_DF_updater.append(thisblockscore)
        block_DF_updater.append(dprime_block)
        block_DF_updater.append(crit_block)
        block_DF_updater.append(smoothprior)
        block_DF_updater.append(outofrange)

        
        temp_ = pd.DataFrame(data=[block_DF_updater], columns=self.DFcol_list)
        
        self.summaryDF = self.summaryDF.append(temp_)
        
        if hungergames:
            
            self.__show_parts_scores(dict_players_scores)   
        
        return subj_info
    
    
    def __show_parts_scores(self, dict_players_scores):
        
        # sort dictionary with magical line found on the internet that I haven't understood at all
        sorted_dict = {k: v for k, v in sorted(dict_players_scores.items(), 
                                               key=lambda item: item[1],
                                               reverse=True)}
    
        darkdarkgrey = (np.array([32, 32, 32])-128) / 128
        myrectang = visual.Rect(self.win, width=400, height=600, 
                                fillColor=darkdarkgrey)
        
        
        while not event.getKeys():
      
            myrectang.draw()
            
            ypos = 200
            for key,value in sorted_dict.items():
                
                if key == 'your_score':
                    
                    drawcolor = (np.array([0, 204, 204])-128) / 128
                
                else:
                    
                    drawcolor = (np.array([160, 160, 160])-128) / 128
                
                
                txt = key + ' : ' + str(round(value[0], 2))
                
                textpatch = visual.TextStim(self.win,
                                        text = txt,
                                        height = 30,
                                        color= drawcolor,
                                        pos = (0, ypos),
                                        autoLog=False
                                        )
                
                textpatch.draw()
                
                ypos -= 50
            
            
            
            self.win.flip()            
    
    
    
    def __give_instructions(self, instructions_step, istep):
        
        if instructions_step=='start':
            
            if self.language == 1:
                
                incipit = 'Vielen Dank für deine Teilnahme an diesem Experiment!\n\n\n'
                goon = 'Heute wirst du eine Detektionsaufgabe durchführen, in der du zwischen einem visuellen SIGNAL und einem visuellen RAUSCHEN differenzieren musst'
                
            else:
                                
                incipit = 'Thank you for participating to this experiment!\n\n\n'
                goon = 'Today you\'re going to perform a detection task\n where you will have to differentiate SIGNAL from NOISE'
    
        elif instructions_step=='continue':
            
            if self.language == 1:
                
                incipit = 'Jeder Versuchsdurchgang wird mit einem Fixationskreuz starten\n\n'
                goon = 'Dann wird entweder ein SIGNAL oder ein RAUSCHEN \nkurz auf dem Bildschirm erscheinen.\n\n'
                
            else:
                    
                incipit = 'Each trial will start with a fixation point\n\n'
                goon = 'Then SIGNAL or NOISE will shortly appear on screen.\n\n '

        elif instructions_step=='signal_example':
            
            if self.language == 1:
                
                incipit = 'Hier ist ein Beispiel für ein SIGNAL…'
                goon = ''
                
            else:            

                incipit = 'This is an example of SIGNAL...'
                goon = ''

        elif instructions_step=='noise_example':
            
            if self.language == 1:
                
                incipit = 'Hier ist ein Beispiel für ein RAUSCHEN…'
                goon = ''
                
            else:            
            
                incipit = 'This is an example of NOISE...'
                goon = ''

        elif instructions_step=='scoring1':
            
            if self.language == 1:
                
                incipit = 'Nach der Stimuluspräsentation wirst du gefragt werden, wie überzeugt du bist das SIGNAL oder das RAUSCHEN gesehen zu haben.\n\n Dann werden Sie gefragt, wie sicher Sie sich bei Ihrer Antwort sind..\n\n'
                goon = 'Alle Antworten werden über das Gamepad gegeben. \n X (left) -> Rauschen \n B (right) -> Signal \n\n\n Y (up) -> sicher \n A (down) -> unsicher'
                
            else:            
          
                incipit = 'After the stimulus presentation, you will be asked whether you have seen the SIGNAL or the NOISE \n\n Then, you will be asked how much sure you are of your reply\n\n'
                goon = 'All responses are given on the gamepad:\n X (left) -> noise \n B (right) -> signal \n\n\n Y (up) -> sure \n A (down) -> unsure'

        elif instructions_step=='scoring2':

            if self.language == 1:
                
                incipit = 'Nach jedem Versuchsdurchgang erhältst du 1 Punkt für eine korrekte Antwort, oder 0 Punkte für eine falsche Antwort. \n\n'
                goon = 'Für das Vertrauen erhalten Sie 1 Punkt für SICHER-RICHTIG, 0 Punkte für SICHER-FALSCH \n\n Wenn Sie "unsicher" wählen, erhalten Sie 0,5 Punkte unabhängig vom Ergebnis (richtig oder falsch)'
                
            else:            
          
                incipit = 'At every trial you will receive 1 point for a correct response, or 0 points for a wrong response. \n\n'
                goon = 'For the confidence, you get 1 point for SURE-CORRECT, 0 points for SURE-WRONG \n\n If you choose "unsure" you get 0.5 points regardless of the outcome (right or wrong)'

        elif instructions_step=='neutral':

            if self.language == 1:
                
                incipit = 'Nun wird sie einem Beispiel folgen.\n\n'
                goon = ''
                
            else:            
            
                incipit = 'Now it will follow a trial example.\n\n'
                goon = ''

        elif instructions_step=='close_practice1':

            if self.language == 1:
                
                incipit = 'Ihre Punktzahl wird in jedem Miniblock (36 trials) in einen Geldbetrag umgerechnet.\n\n '
                goon = 'Sie erhalten daher eine Rückmeldung, wie viel Geld Sie bisher verdient haben.'
                
            else:            

                incipit = 'Your score will be converted, in any mini-block (36 trials), to a monetary compensation.\n\n'
                goon = 'You will receive hence a feedback on how much money you earned so far.'

        elif instructions_step=='close_practice2':

            if self.language == 1:
                
                incipit = 'An dieser Stelle sollte alles klar sein.\n Sollte dies nicht der Fall sein, wende dich bitte an den/die Versuchleiter*in\n\n'
                goon = 'Viel Glück!'
                
            else:            

                incipit = 'At this point everything should be clear.\n If not, please communicate that to the experimenter\n or go back to the points that are not clear\n\n'
                goon = 'Good luck!'


        fintext = incipit + goon
        textpatch_inst = visual.TextStim(self.win,
                                        text = fintext,
                                        height = 35,
                                        color= 'black',
                                        pos = (0, 100),
                                        autoLog=False,
                                        wrapWidth = 1000
                                        )
        if self.language == 1:            
            text_forward = 'WEITER>'            
        else:            
            text_forward = 'FORWARD>'
        
        if instructions_step=='start':
            text_back = ''
        else:
            if self.language == 1:            
                text_back = '<ZURÜCK'
            else:                
                text_back = '<BACK'
                        
    
        textpatch_back = visual.TextStim(self.win,
                                        text = text_back,
                                        height = 35,
                                        color= 'black',
                                        pos = (-600, -300),
                                        autoLog=False
                                        )

        textpatch_forward = visual.TextStim(self.win,
                                        text = text_forward,
                                        height = 35,
                                        color= 'black',
                                        pos = (600, -300),
                                        autoLog=False
                                        )

        # force to wait
        textpatch_inst.draw()
        textpatch_back.draw()
        textpatch_forward.draw()
    
        self.win.flip()

        core.wait(.5)

        # get response                   
        keepwaiting=True
        
        while keepwaiting:
            
            textpatch_inst.draw()
            textpatch_back.draw()
            textpatch_forward.draw()
        
            self.win.flip()
            
            if self.joy.get_a(): #joy.get_x():
                
                istep -= 1
                keepwaiting = False
                moveforward = False
                
            elif self.joy.get_x(): #joy.get_b():
                
                istep += 1
                keepwaiting = False
                moveforward = True
                           
            else:
            
                moveforward = False
            
        if istep < 0:
            istep = 0 # correct for negative cases
            
        return istep, moveforward
    
    

    
    def __generate_questions(self):
    
        # generate a dictionary with all the questions to be asked the participant
        if self.language == 1:
            
            self.dict_questionnaire = {
            
                'Q1' : 'Wie schwer fandest du das Experiment?',
                'Q2' : 'Wie stark hast du dich mit den vorherigen Versuchspersonen im Wettkampf gefühlt?',
                'Q3' : 'Wie sehr dachtest du, dass du eine bessere Punktzahl hättest erreichen können?',
                'Q4' : 'Wie sehr hast du die Hinweisreize deine Entscheidungen beeinflussen lassen?',
                'Q5' : 'Wie häufig warst du sicher ein SIGNAL in der Bedingung "HIGH PROBABILITY" gesehen zu haben, und stattdessen war es ein RAUSCHEN?',
                'Q6' : 'Wie häufig warst du sicher, dass der Stimulus in "LOW PROBABILITY" ein RAUSCHEN war, aber stattdessen war es ein SIGNAL?',
                'Q7' : 'Wie sehr hat dich der Wettkampf mit den anderen Versuchspersonen motiviert?',
                'Q8' : 'Wie sehr hat dich der Wettkampf mit den anderen Versuchspersonen gestresst?',
                'Q9' : 'Wie wahrscheinlich ist es, dass du mindestens einen deiner "Gegner" persönlich kennst?',
                'Q10' : 'Wie viel VORSICHTIG warst du mit deiner Antwort in der Bedingung "HIGH RELEVANCE"?',
                'Q11' : 'Wie viel VORSICHTIGER warst du mit deiner Antwort in der Bedingung "LOW RELEVANCE"?',
                'Q12' : 'Wie motiviert warst du Punkte zu ERHALTEN?',
                'Q13' : 'Wie viel Angst hattest du Punkte zu VERLIEREN?',
                'Q14' : 'Wie wahrscheinlich ist es, dass du mindestens eine der anderen Versuchspersonen, mit denen du konkurriert hast, NACH dem Experiment treffen wirst?',            
                'Q15' : 'Wie viel GENAUER (weniger Fehler) glaubst du warst du in der Bedingung "HIGH RELEVANCE" im Vergleich zu den anderen Bedingungen?',
                'Q16' : 'Wie gerne würdest du die anderen Versuchspersonen fragen, wie diese ihre Punktzahlen erreicht haben?',
                'Q17' : 'Wie sehr hast du die Existenz der anderen Versuchspersonen ANGEZWEIFELT?'
         
            }

        else:
        
            self.dict_questionnaire = {
                
                'Q1' : 'How much did you find the experiment difficult?',
                'Q2' : 'How much did you feel in competition with the previous participants?',
                'Q3' : 'How much did you think you could achieve a better score?',
                'Q4' : 'How much did you use the cues to guide your decision?',
                'Q5' : 'How often were you sure to have seen a SIGNAL in the HIGH PROBABILITY condition, and instead it was not there?',
                'Q6' : 'How often were you sure that the stimulus showed in the LOW PROBABILITY was NOISE, and it was SIGNAL instead?',
                'Q7' : 'How much did the competition with the other subjects motivate you?',
                'Q8' : 'How much did the competition with the other participants stress you?',
                'Q9' : 'How probable is it that you knew in person at least one of your opponents?',
                'Q10' : 'How much were you CAREFUL in giving your answer in the HIGH RELEVANCE condition?',
                'Q11' : 'How much were you CAREFUL in giving your answer in the LOW RELEVANCE condition?',
                'Q12' : 'How much were you motivated in ACHIEVING points?',
                'Q13' : 'How much were you scared of LOSING points?',
                'Q14' : 'How probable is it that you meet at least one of the other participants you competed against AFTER the experiment?',            
                'Q15' : 'How much do you think you were more PRECISE (less mistakes) in the HIGH RELEVANCE condition compared to the others?',
                'Q16' : 'How much would you like to ask in person to the other participants how did they achieve their scores?',
                'Q17' : 'How much do you DOUBT of the existence of the other participants?'
             
                }
    
    
    
    
    def __get_questionnaire_answer(self, code, iQ, cfg):
        
        
        this_question = self.dict_questionnaire[code]
        
        if (iQ == 'Q6') or (iQ == 'Q7'):
            
            if self.language==1:

                leftend = 'NIEMALS'
                rightend = 'IMMER'                
                
            else:

                leftend = 'NEVER'
                rightend = 'ALWAYS'
            
        else:

            if self.language==1:

                leftend = 'GAR NICHT'
                rightend = 'SEHR VIEL'                
                
            else:
            
                leftend = 'NOT AT ALL'
                rightend = 'A LOT'
    
        # bring mouse position to the center
        cfg.mousehandle.setPos()
               
        # rating part
        cross_vert = [(0, 0), (.03*1920, .03*1080), 
                      (0, 0), (-.03*1920,-.03*1080), 
                      (0, 0), (.03*1920, -.03*1080), 
                      (0, 0), (-.03*1920, .03*1080)]
        
        cross = visual.ShapeStim(self.win, 
                                 vertices = cross_vert, 
                                 closeShape = False, 
                                 lineWidth = 10, 
                                 lineColor = list(cfg.colors['darkgrey']), 
                                 pos=(0, -0.5*1080/2),
                                 autoLog=False
                                 )        
            
        line = visual.Line(self.win,
                           start=(-0.5*1920/2, -0.5*1080/2), 
                           end=(0.5*1920/2, -0.5*1080/2), 
                           lineColor= list(cfg.colors['darkgrey']), 
                           lineWidth = 8,
                           autoLog=False
                           )
        
        placeholder = visual.Line(self.win,
                                  start=(0, -0.5*1080/2 -.05*1080),
                                  end=(0, -0.5*1080/2 + .05*1080),
                                  lineWidth=8,
                                  lineColor=list(cfg.colors['darkgrey']),
                                  autoLog=False
                                  )
        

        
        
        mouse = visual.CustomMouse(self.win, 
                                   leftLimit=-0.5*1920/2, 
                                   topLimit=-0.5*1080/2, 
                                   rightLimit=0.5*1920/2, 
                                   bottomLimit=-0.5*1080/2, 
                                   showLimitBox = False, 
                                   clickOnUp = True, 
                                   newPos =(0, -0.5*1920/2), 
                                   pointer = cross,
                                   autoLog=False
                                   )
        
        anchor_notseen = visual.TextStim(self.win,
                                         text= leftend,
                                         height = 30,
                                         color = list(cfg.colors['darkgrey']),
                                         pos = (-0.8*1920/2, -0.5*1080/2),
                                         autoLog=False
                                         )

        anchor_seen = visual.TextStim(self.win,
                                      text= rightend,
                                      height = 30,
                                      color = list(cfg.colors['darkgrey']),
                                      pos = (+0.8*1920/2, -0.5*1080/2),
                                      autoLog=False
                                      )
        
        instructions = visual.TextStim(self.win,
                                       text = this_question,
                                       height = 30,
                                       color= list(cfg.colors['darkgrey']),
                                       pos = (0, 0),
                                       autoLog=False
                                       )

        
        
        while not mouse.getClicks():

            anchor_notseen.draw()
            anchor_seen.draw()
            instructions.draw()
            line.draw()
            placeholder.draw()
            
            mouse.draw()   

            self.win.flip()             
                
            if mouse.getClicks():
                
                x = mouse.getPos()[0] # !! getPos returns float !!
                self.win.flip()
    
        # 3. get actual rating on (-1) - 1 scale (change this however you want!)
        rating = round(4 *x / 1920, 2) # round(x, 2) * 2
        
        # 4. return (rounded value) from function (round just for easy checking purposes)
        return (rating + 1)/2

    
    
    
    
    
    
    
    
    
    
    
    