#!/usr/bin/python
# -*- coding: utf-8 -*-

import ML_OPE

from common import utilities

class runMLOPE:
    
    def __init__(self, mult, settings, model_folder, tops):
        self.mult = mult
        self.settings = settings
        self.model_folder = model_folder
        self.tops = tops
        
    def run(self, write2file=False):
        # Initialize the algorithm
        # print'initialize the algorithm ...'
        ml_ope = ML_OPE.MLOPE(self.settings['num_terms'], self.settings['num_topics'], self.settings['alpha'], 
                              self.settings['tau0'], self.settings['kappa'], self.settings['iter_infer'])
        # Start
        # print'start!!!'
        i = 0
        while i < self.settings['iter_train']:
            i += 1
            # print'\n***iter_train:%d***\n'%(i)
            j = 0
            idx = 0
            while True:
                j += 1
                idx, (wordids, wordcts) = utilities.read_minibatch_list_frequencies(self.mult,
                                                                                    self.settings['batch_size'],
                                                                                    begin_idx=idx)
                # Stop condition
                if len(wordids) == 0:
                    break
                # 
                # print'---num_minibatch:%d---'%(j)
                (time_e, time_m, theta) = ml_ope.static_online(wordids, wordcts)
                # Compute sparsity
                ##sparsity = utilities.compute_sparsity(theta, theta.shape[0], theta.shape[1], 't')
                # sparsity = 0.
                # # Compute perplexities
                # ##LD2 = utilities.compute_perplexities_vb(ml_ope.beta, self.settings['alpha'], self.settings['eta'],
                #                                         # self.settings['iter_infer'], self.test_data)
                # # LD2 = []
                # # # Search top words of each topics
                # # list_tops = utilities.list_top(ml_ope.beta, self.tops)
                # # # Write files
                # # utilities.write_file(i, j, ml_ope.beta, time_e, time_m, theta, sparsity, LD2, list_tops, self.tops,
                # #                      self.model_folder)

        # # Write settings
        # print'write setting ...'
        # file_name = '%s/setting.txt'%(self.model_folder)
        # utilities.write_setting(self.settings, file_name)

        # Write final model to file
        if write2file:
            print'write final model ...'
            file_name = '%s/beta_final.dat'%(self.model_folder)
            utilities.write_topics(ml_ope.beta, file_name)

        # Finish
        # print'done!!!'

        return theta, ml_ope.beta
