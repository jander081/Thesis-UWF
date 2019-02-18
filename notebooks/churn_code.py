# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict
from sklearn.base import BaseEstimator, TransformerMixin


#------------------------------------------------------------


class TypeSelector(BaseEstimator, TransformerMixin):
    
    '''np.object, np.number, np.bool_'''
    
    def __init__(self, dtype1, dtype2=None, dtype3=None):
        self.dtype1 = dtype1
        self.dtype2 = dtype2
        self.dtype3 = dtype3

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        
        assert isinstance(X, pd.DataFrame), "Gotta be Pandas"
        
        if self.dtype3 != None:
            
            output = (X.select_dtypes(include=[self.dtype1]),
                   X.select_dtypes(include=[self.dtype2]),
                   X.select_dtypes(include=[self.dtype3]))
            
        elif self.dtype2 != None:
            output = (X.select_dtypes(include=[self.dtype1]),
                   X.select_dtypes(include=[self.dtype2]))
            
        else:
            
            output = (X.select_dtypes(include=[self.dtype1]))
            
        return output
        

#------------------------------------------------------------

from sklearn.preprocessing import StandardScaler 

class StandardScalerDf(StandardScaler):
    
    """
    DataFrame Wrapper around StandardScaler; Recursive override
    """

    def __init__(self, copy=True, with_mean=True, with_std=True):
        super(StandardScalerDf, self).__init__(copy=copy,
                                               with_mean=with_mean,
                                               with_std=with_std)

    def transform(self, X, y=None):
        z = super(StandardScalerDf, self).transform(X.values)
        return pd.DataFrame(z, index=X.index, columns=X.columns)


#------------------------------------------------------------

from fancyimpute import SoftImpute

class SoftImputeDf(SoftImpute):
    
    """
    DataFrame Wrapper around SoftImpute
    """

    def __init__(self, shrinkage_value=None, convergence_threshold=0.001,
                 max_iters=100,max_rank=None,n_power_iterations=1,init_fill_method="zero",
                 min_value=None,max_value=None,normalizer=None,verbose=True):
        
        super(SoftImputeDf, self).__init__(shrinkage_value=shrinkage_value, 
                                           convergence_threshold=convergence_threshold,
                                           max_iters=max_iters,max_rank=max_rank,
                                           n_power_iterations=n_power_iterations,
                                           init_fill_method=init_fill_method,
                                           min_value=min_value,max_value=max_value,
                                           normalizer=normalizer,verbose=False)

    

    def fit_transform(self, X, y=None):
        
        assert isinstance(X, pd.DataFrame), "Must be pandas dframe"
        
        for col in X.columns:
            if X[col].isnull().sum() < 10:
                X[col].fillna(0.0, inplace=True)
        
        z = super(SoftImputeDf, self).fit_transform(X.values)
        return pd.DataFrame(z, index=X.index, columns=X.columns)


#------------------------------------------------------------



class SelectFeatures(BaseEstimator, TransformerMixin):
    
    """
    Used with Kbins to select features with sufficient cardinality. Could
    probably just join this with kbins
    """
    
    def __init__(self, val_count=50, categorical=False):
        self.val_count = val_count
        self.categorical = categorical
        
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        feat = pd.DataFrame()
        
        if self.categorical==False:           
            for col in X.columns:
                if len(X[col].value_counts()) > self.val_count:              
                    X[col + '_bin'] = X[col]
                    feat = pd.concat([feat, X[col + '_bin']], axis=1)
        else:
            for col in X.columns:
                if len(X[col].value_counts()) > self.val_count: 
                    feat = pd.concat([feat, X[col]], axis=1)                    
        return feat

    
#------------------------------------------------------------
from sklearn.preprocessing import KBinsDiscretizer

class KBins(KBinsDiscretizer):
    
    """DataFrame Wrapper around KBinsDiscretizer. Sometimes this will throw 
    the monotonically increase/decrease error. You can either reduce bins 
    or modify the selected features by value counts (increase)"""

    def __init__(self, n_bins=5, encode='onehot', strategy='quantile'):
        super(KBins, self).__init__(n_bins=n_bins,
                                    encode='ordinal',
                                    strategy=strategy)                               
        
       
    def transform(self, X, y=None):
        
        assert isinstance(X, pd.DataFrame), "Must be pandas dframe"
        
        
        z = super(KBins, self).transform(X)
        binned = pd.DataFrame(z, index=X.index, columns=X.columns)
        binned = binned.applymap(lambda x: 'category_' + str(x))
#         final = pd.concat([X, binned], axis=1)        
        return binned


    
#------------------------------------------------------------

import re

class RegImpute(BaseEstimator, TransformerMixin):
    
    '''consider adding methods to check for special characters or return
    indices for nans, since nans can be different types. If bool, shut off
    regex'''
    
    def __init__(self, regex=True):
        self.regex = regex
        
    def find_nulls(self, X, y=None):
        '''this returns all dframe indices with nans. Useful to determine
        type of null'''
        return pd.isnull(X).any(1).nonzero()[0]
    
    def null_cols(self, X, y=None):
        '''Prints list of null cols with number of nulls'''
        null_columns=X.columns[X.isnull().any()]
        print(X[null_columns].isnull().sum())
              
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        
        for col in X.columns:
            X[col].fillna(X[col].mode().iloc[0], inplace=True)
            
        if self.regex == True:
            X = X.applymap(lambda x: re.sub(r'\W+', '', x)) 
            
        return X

#------------------------------------------------------------
 

class FreqFeatures(BaseEstimator, TransformerMixin):
    
    """
    returns a dict for freqs. This can then be mapped to 
    any col to create freq feature. Must be run prior to freq_group
    """
       
    def __init__(self, val_count=50):
        self.val_count = val_count
        self.drops = []
        
    def make_dict(self, col):
        
        df = pd.DataFrame(self.data[col].value_counts())
        df.reset_index(level=0, inplace=True)
        df.rename(columns={'index': 'key', col: 'value'}, inplace=True)
        df_dict = defaultdict(list)
        for k, v in zip(df.key, df.value):
            df_dict[k] = (int(v))
        return df_dict
    
    @staticmethod
    def reduce(x,y):
        if x <= 10:
            return 'rare'
        elif x <= 50:
            return 'infrequent'
        elif x <= 100:
            return 'less common'
        else:
            return y

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        self.data = X
        
        assert isinstance(self.data, pd.DataFrame), 'pls enter dframe'
        
        for col in self.data.columns:
            dict_ = self.make_dict(col)        
            freq_vec = self.data[col].map(dict_)
            
            if len(set(self.data[col])) > 2000:
                self.drops.append(col)
                self.data.drop([col], axis=1, inplace=True)
                self.data[col + '_freq'] = freq_vec
                
            elif len(set(self.data[col])) > 100:
                y = self.data[col]
                vectfunc = np.vectorize(self.reduce,cache=False)
                vec = np.array(vectfunc(freq_vec,y))
                
                self.data[col + '_freq'] = freq_vec
                self.data[col + '_reduce'] = vec
                self.data.drop([col], axis=1, inplace=True)
                
        return self.data
                


'''Notes: built-in super() function, which is a function for delegating method calls to some class in the instance’s ancestor tree. For our purposes, think of super() as a generic instance of our parent class.

http://flennerhag.com/2017-01-08-Recursive-Override/



'''


#------------------------------------------------------------
            
class MetricVec:
    
    '''Run this before features'''
    
    def __init__(self, X, y=None):
        self.data = X
        self.label = y
        
        
    def join(self, features=['churn', 'metric_renew', 'metric']):
        
        '''run this first'''
        
        assert isinstance(features, list), "as a list pls"
        
        features = [self.data[feat].apply(str) for feat in features]
        
        return pd.concat(features, axis=1).apply(lambda x: ' '.join(x), axis=1)   
      
    
    def metric_merge(self, string):
        
        '''If the account is New, the metric is taken from the metric column - which
        has more values for New. Otherwise, it comes from metric at Renewal - which has
        more values for existing accounts'''

        churn = string.split()[0]
        m1 =string.split()[1]
        m2 = string.split()[2]
        if churn == 'New':
            metric = m2
        else:
            metric = m1
        if metric == 'nan':
            metric = metric
        else:
            metric = int(float(metric))

        return metric
    
    def modify(self):
        
        '''Only need to run this method. Everything else is linked. It works,
        but probably not set up correctly'''
        
        merge = self.join()
        
        self.data['metRic'] = np.float64(merge.apply(self.metric_merge))
        self.data.drop(['metric_renew', 'metric'], axis=1, inplace=True)

#------------------------------------------------------------
        
import calendar 

class TimeFeatures:
    
    def __init__(self, X, y=None):
        self.data = X
        self.label = y

        
    def index(self, col='eff_dt', output=False):
        
        self.data.index = self.data[col]
        self.data.drop([col], axis=1, inplace=True)
        self.data.index.rename(col, inplace=True)
        # args
        if output:
            return self.data
        
        
    def time_feats(self, col = 'Effective YearMonth', output=False):    
        
        self.data['eff_dt']= self.data[col].apply(lambda x: datetime.strptime(str(x), '%Y%m'))
        self.data.drop([col], axis=1, inplace=True)
        
        series = self.data['eff_dt']
        # financial quarter
        self.data['quarter'] = series.apply(lambda x: int(x.month // 3.25 + 1))
        # extract time features      
        self.data['month_nom'] = series.apply(lambda x: calendar.month_abbr[x.month]) 
        self.data['month_prog'] = series.apply(lambda x: np.float(x.month)) 
        self.data['year'] = series.apply(lambda x: str(x.year) + '_')
        
        # create term for absolute quarter
        term = self.data['year'] + self.data['quarter'].astype(str)

        abs_quarter = { '2016_1': 1, '2016_2': 2, '2016_3': 3 ,'2016_4': 4,
                       '2017_1': 5, '2017_2': 6, '2017_3': 7,'2017_4': 8,'2018_1': 9,
                       '2018_2': 10}

        self.data['abs_q'] = term.map(abs_quarter)
        
        self.index()
        
        if output:
            return self.data
          
        
        
#------------------------------------------------------------
#------------------------------------------------------------
#------------------------------------------------------------
#------------------------------------------------------------
#------------------------------------------------------------
#------------------------------------------------------------











