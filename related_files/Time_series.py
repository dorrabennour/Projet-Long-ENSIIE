#!/usr/bin/env python
# coding: utf-8

# In[1]:

# Blocking time series split
class BlockingTimeSeriesSplit():
  
  def __init__(self, n_splits):

    self.n_splits = n_splits
  
  def get_n_splits(self, X, y, groups):
      return self.n_splits
  
  def split(self, X, y=None, groups=None):
      import numpy as np
      n_samples = len(X)
      k_fold_size = n_samples // self.n_splits
      indices = np.arange(n_samples)

      margin = 0
      for i in range(self.n_splits):
          start = i * k_fold_size
          stop = start + k_fold_size
          mid = int(0.5 * (stop - start)) + start
          yield indices[start: mid], indices[mid + margin: stop]

import pandas as pd
import datetime
from datetime import datetime as dt
from dateutil.relativedelta import *


# Time Based cross validation
class TimeBasedCV(object):
    '''
    Parameters 
    ----------
    train_period: int
        number of time units to include in each train set
        default is 30
    test_period: int
        number of time units to include in each test set
        default is 7
    freq: string
        frequency of input parameters. possible values are: days, months, years, weeks, hours, minutes, seconds
        possible values designed to be used by dateutil.relativedelta class
        deafault is days
    '''
    
    
    def __init__(self, train_period=30, test_period=7, freq='days'):
        self.train_period = train_period
        self.test_period = test_period
        self.freq = freq

        
        
    def split(self, data, validation_split_date=None, date_column='record_date', gap=0):
        '''
        Generate indices to split data into training and test set
        
        Parameters 
        ----------
        data: pandas DataFrame
            your data, contain one column for the record date 
        validation_split_date: datetime.date()
            first date to perform the splitting on.
            if not provided will set to be the minimum date in the data after the first training set
        date_column: string, deafult='record_date'
            date of each record
        gap: int, default=0
            for cases the test set does not come right after the train set,
            *gap* days are left between train and test sets
        
        Returns 
        -------
        train_index ,test_index: 
            list of tuples (train index, test index) similar to sklearn model selection
        '''
        
        # check that date_column exist in the data:
        try:
            data[date_column]
        except:
            raise KeyError(date_column)
                    
        train_indices_list = []
        test_indices_list = []

        if validation_split_date==None:
            validation_split_date = data[date_column].min().date() + eval('relativedelta('+self.freq+'=self.train_period)')
        
        start_train = validation_split_date - eval('relativedelta('+self.freq+'=self.train_period)')
        end_train = start_train + eval('relativedelta('+self.freq+'=self.train_period)')
        start_test = end_train + eval('relativedelta('+self.freq+'=gap)')
        end_test = start_test + eval('relativedelta('+self.freq+'=self.test_period)')

        while end_test < data[date_column].max().date():
            # train indices:
            cur_train_indices = list(data[(data[date_column].dt.date>=start_train) & 
                                     (data[date_column].dt.date<end_train)].index)

            # test indices:
            cur_test_indices = list(data[(data[date_column].dt.date>=start_test) &
                                    (data[date_column].dt.date<end_test)].index)
            
            print("Train period:",start_train,"-" , end_train, ", Test period", start_test, "-", end_test,
                  "# train records", len(cur_train_indices), ", # test records", len(cur_test_indices))

            train_indices_list.append(cur_train_indices)
            test_indices_list.append(cur_test_indices)

            # update dates:
            start_train = start_train + eval('relativedelta('+self.freq+'=self.test_period)')
            end_train = start_train + eval('relativedelta('+self.freq+'=self.train_period)')
            start_test = end_train + eval('relativedelta('+self.freq+'=gap)')
            end_test = start_test + eval('relativedelta('+self.freq+'=self.test_period)')

        # mimic sklearn output  
        index_output = [(train,test) for train,test in zip(train_indices_list,test_indices_list)]

        self.n_splits = len(index_output)
        
        return index_output
    
    
    def get_n_splits(self):
        """Returns the number of splitting iterations in the cross-validator
        Returns
        -------
        n_splits : int
            Returns the number of splitting iterations in the cross-validator.
        """
        return self.n_splits 


# Expanding window cross validation
#%%
'''
Expanding window cross validation
Similar to sklearn format
@ germayne  
'''
import numpy as np

class expanding_window(object):
    '''	
    Parameters 
    ----------
    
    Note that if you define a horizon that is too far, then subsequently the split will ignore horizon length 
    such that there is validation data left. This similar to Prof Rob hyndman's TsCv 
    
    
    initial: int
        initial train length 
    horizon: int 
        forecast horizon (forecast length). Default = 1
    period: int 
        length of train data to add each iteration 
    '''
    

    def __init__(self,initial= 1,horizon = 1,period = 1):
        self.initial = initial
        self.horizon = horizon 
        self.period = period 


    def split(self,data):
        '''
        Parameters 
        ----------
        
        Data: Training data 
        
        Returns 
        -------
        train_index ,test_index: 
            index for train and valid set similar to sklearn model selection
        '''
        self.data = data
        self.counter = 0 # for us to iterate and track later 


        data_length = data.shape[0] # rows 
        data_index = list(np.arange(data_length))
         
        output_train = []
        output_test = []
        # append initial 
        output_train.append(list(np.arange(self.initial)))
        progress = [x for x in data_index if x not in list(np.arange(self.initial)) ] # indexes left to append to train 
        output_test.append([x for x in data_index if x not in output_train[self.counter]][:self.horizon] )
        # clip initial indexes from progress since that is what we are left 
         
        while len(progress) != 0:
            temp = progress[:self.period]
            to_add = output_train[self.counter] + temp
            # update the train index 
            output_train.append(to_add)
            # increment counter 
            self.counter +=1 
            # then we update the test index 
            
            to_add_test = [x for x in data_index if x not in output_train[self.counter] ][:self.horizon]
            output_test.append(to_add_test)

            # update progress 
            progress = [x for x in data_index if x not in output_train[self.counter]]	
            
        # clip the last element of output_train and output_test
        output_train = output_train[:-1]
        output_test = output_test[:-1]
        
        # mimic sklearn output 
        index_output = [(train,test) for train,test in zip(output_train,output_test)]
        
        return index_output

import numpy as np


class timefold(object):
    """
    Cross-validation methods for timeseries data.
    Available methods
        * nested
            Generates train-test pair indices with a growing training window.
            Example (folds=3):
            TRAIN: [0 1 2] TEST: [3 4 5]
            TRAIN: [0 1 2 3 4 5] TEST: [6 7 8]
            TRAIN: [0 1 2s 3 4 5 6 7] TEST: [8 9]
        * window
            Generates train-test pair indices with a moving window.
            Example (folds=3):
            TRAIN: [0 1 2] TEST: [3 4 5]
            TRAIN: [3 4 5] TEST: [6 7 8]
            TRAIN: [6 7] TEST: [8 9]
        * step
            Generates one step ahead train-test pair indices with specified testing size.
            Fold argument is ignored. The maximum possible number of folds is generated based on
            the number of samples and specified testing window size.
            Example (test_size=1):
            TRAIN: [0] TEST: [1]
            TRAIN: [0 1] TEST: [2]
            TRAIN: [0 1 2] TEST: [3]
            TRAIN: [0 1 2 3] TEST: [4]
            TRAIN: [0 1 2 3 4] TEST: [5]
            TRAIN: [0 1 2 3 4 5] TEST: [6]
            TRAIN: [0 1 2 3 4 5 6] TEST: [7]
            TRAIN: [0 1 2 3 4 5 6 7] TEST: [8]
            TRAIN: [0 1 2 3 4 5 6 7 8] TEST: [9]
        * shrink
            Generates train-test pair indices with a shrinking training window and constant testing window.
            Example (folds=3):
            TRAIN: [0 1 2 3 4 5 6 7] TEST: [8 9]
            TRAIN: [3 4 5 6 7] TEST: [8 9]
            TRAIN: [6 7] TEST: [8 9]
        * stratified
            Generates stratified train-test pair indices where a ratio is preserved per fold.
            To be implemented
    """

    def __init__(self, folds=10, method='nested', min_train_size=1, min_test_size=1, step_size=1):
        self.folds = folds
        self.method = method
        self.min_train_size = min_train_size
        self.min_test_size = min_test_size
        self.step_size = step_size

    def split(self, X):
        """
        Split data into train-test pairs based on specified cross-validation method.
        """
        folds = self.folds
        method = self.method
        min_train_size = self.min_train_size
        min_test_size = self.min_test_size
        step_size = self.step_size

        X_obs = X.shape[0]
        indices = np.arange(X_obs)

        if folds >= X_obs:
            raise ValueError(
                ("The number of folds {0} must be smaller than the number of observations {1}".format(folds, X_obs)))

        folds += 1
        fold_indices = np.array_split(indices, folds, axis=0)
        fold_sizes = [len(fold) for fold in fold_indices][:-1]
        train_starts = [fold[0] for fold in fold_indices][:-1]
        train_ends = [fold[0] for fold in fold_indices][1:]

        if method == 'nested':
            for end, size in zip(train_ends, fold_sizes):
                yield(indices[:end], indices[end:end + size])

        elif method == 'window':
            for start, end, size in zip(train_starts, train_ends, fold_sizes):
                yield(indices[start:end], indices[end:end + size])

        elif method == 'step':
            steps = np.arange(min_train_size, indices[-1], step_size)
            for step in steps:
                yield(indices[:step], indices[step:step + min_test_size])


        elif method == 'shrink':
            for start, size in zip(train_starts, fold_sizes):
                yield(indices[start:train_ends[-1]], indices[-fold_sizes[-1]:])

        elif method == 'stratified':
            pass

        else:
            raise ValueError("Unknown method supplied '{0}'. Method must be one of: 'nested', 'window', 'step', "
                             "'stratified'".format(method))