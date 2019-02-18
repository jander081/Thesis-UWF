"""
Author: Jake Anderson
Original code and method by:
License: 
    
"""

'''TECH dictionary for mapping'''

import numpy as np
import pandas as pd

tech_index = {'AK': 44.86, 'AZ': 54.88, 'CO': 80.40, 'ID': 46.30, 'MT': 43.73,
          'NM': 55.19,'NV': 32.76, 'OR': 62.33, 'UT': 69.14, 'WA': 71.84, 
          'WY': 43.02, 
          
          'IL': 59.51, 'IN': 49.23, 'IA': 43.52, 'KY': 30.53, 
          'MI': 58.75, 'MN': 69.58, 'NE': 53.53, 'ND': 49.73, 'OH': 52.32, 
          'SD': 41.55, 'WI': 55.06,
          
          'AL': 42.67, 'FL': 38.82, 'GA': 53.53, 
          'MS': 29.84, 'NC': 62.64, 'SC': 35.84, 'TN': 40.22,
          
          'CT': 71.05, 'MA': 83.67,  'ME': 38.39,  'NH': 65.32,  
          'NY': 57.55,  'RI': 59.84,  'VT': 52.58, 
         
          'DC': 72.00, 'DE': 65.38, 'MD': 80.31, 'NJ': 59.40, 
          'PA': 61.54, 'VA': 65.88, 'WV': 25.84, 'CA':75.94, 
         
          'AR': 27.95, 'KS': 48.44, 'LA': 31.40, 'MO': 50.60, 
          'OK': 34.62, 'TX': 58.66}

#---------------------------------------------------------------------  

#NEW FEATURES
'''Education dictionary for mapping'''

ed_index = {'AK': 27.9, 'AZ': 25.9, 'CO': 36.4, 'ID': 24.4, 'MT': 28.8,
          'NM': 25.0,'NV': 21.7, 'OR': 28.8, 'UT': 29.3, 'WA': 31.1, 
          'WY': 24.1, 
          
          'IL': 30.8, 'IN': 22.7, 'IA': 24.9, 'KY': 20.5, 
          'MI': 25.2, 'MN': 31.8, 'NE': 28.6, 'ND': 27.6, 'OH': 24.6, 
          'SD': 26.3, 'WI': 26.3,
          
          'AL': 21.90, 'FL': 25.8, 'GA': 27.3, 
          'MS': 19.5, 'NC': 26.5, 'SC': 24.5, 'TN': 23.1,
          
          'CT': 35.5, 'MA': 39.0,  'ME': 26.8,  'NH': 32.8,  
          'NY': 32.5,  'RI': 30.2,  'VT': 33.6, 'CA': 30.1,
         
          'DC': 50.1, 'DE': 27.8, 'MD': 36.1, 'NJ': 35.4, 
          'PA': 27.1, 'VA': 34.2, 'WV': 17.5,
         
          'AR': 19.5, 'KS': 29.8, 'LA': 21.4, 'MO': 25.6, 
          'OK': 22.9, 'TX': 25.9}

#------------------------------------------------------------
def csf(state):
    
    '''maps for competitive state fund'''
    
    csf = ['AZ', 'CA', 'CO', 
       'HI', 'ID', 'KY', 'LA', 'ME',
       'MD', 'MN', 'MO', 'MT', 'NM', 
       'NY', 'OK', 'OR', 'PA', 'RI', 'TX', 'UT'] 

    if state in csf:
        state = '1'
    else:
        state = '0'
    return(state)    

#---------------------------------------------------------------------  
def market_sh(state):
    
    '''market share for states with competitve state funds'''
    
    high = ['CO', 'ID', 'MT', 'RI', 'ME', 'OR', 'UT'] #  > 50%
    med = ['HI', 'KY', 'NM', 'NY', 'OK', 'TX']  # > 25%
    low = ['CA', 'MN', 'PA', 'AZ', 'LA', 'MD', 'MO']  # > 0
    
    if state in high:
        state = 'high'
    elif state in med:
        state = 'med'
    elif state in low:
        state = 'low'
    else:
        state = 'none'
    return state 

#---------------------------------------------------------------------   


def div(state):
    
    '''Div buckets for states with competitive state funds'''
    
    high = ['LA', 'MT', 'OR', 'TX'] #  > 18%
    med = ['CO', 'ME', 'NY', 'UT']  # > 5%
    low = ['AZ', 'CA', 'HI', 'ID', 'MD', 'MN', 'MO', 'RI']  # > 0
    
    if state in high:
        state = 'high'
    elif state in med:
        state = 'med'
    elif state in low:
        state = 'low'
    else:
        state = 'none'
    return state

#--------------------------------------------------------------------

class FeatureMaps:
    
    def __init__(self, X, y=None, model=None):
        self.data = X
        self.target = y
        self.model = model
        
    
    
        
    def diff_indicator(self, col_1, col_2, col_2_drop=False, output=False):
        '''takes in 2 colnames and creates an indicator'''
        # create the new colname
        col_name = 'diff_' + col_1[:3] + '_' + col_2
        # compare feature values
        self.data[col_name] = pd.np.where(self.data[col_1] != self.data[col_2] , 1, 0).astype('bool')
    
        # args
        if col_2_drop:
            self.data.drop([col_2], axis=1, inplace=True)             
        if output:
            return self.data
        
    def csf_indices(self, output=False):
        '''dicts: market_sh, csf, div, tech, and ed are imported'''
        state = 'risk_state'
        # Indicator for csf
        self.data['csf_ind'] = self.data[state].map(csf).astype('bool')
        # Market share - need to modify the dict, this is a bandaid
        self.data['csf_market_sh'] = self.data[state].map(market_sh)
        self.data['csf_market_sh'] = np.float64(self.data['csf_market_sh'].map({'none':0, 'low':1, 'med':2, 'high':3}))
        # Dividend percentage - need to modify
        self.data['csf_div'] = self.data[state].map(div)
        self.data['csf_div'] = np.float64(self.data['csf_div'].map({'none':0, 'low':1, 'med':2, 'high':3}))
        # state index scores
        self.data['tech_score'] = self.data[state].map(tech_index)
        self.data['ed_score'] = self.data[state].map(ed_index)
        # args
        if output:
            return self.data
        
    def process_num(self, output=False):
    
        process = self.data['processing']
        # convert to ordinal 1 - 3 amount of personal interaction
        self.data['processing_num'] = np.float64(process.map({'least_personal':1, 
                                                              'personal':2, 
                                                              'most_personal':3}))
        # args
        if output:
            return self.data
        
    
    def to_cat(self, list_):
        assert isinstance(list_, list), 'pls enter list'
        for feat in list_:
            self.data[feat + '_cat'] = self.data[feat].apply(lambda x: 'cat_' + str(x))
            self.data.drop([feat], axis=1, inplace=True)
          
        
    def run_all(self):
        
        self.diff_indicator('risk_state', 'agnt_state', col_2_drop=True)
        self.diff_indicator('agnt', 'mstr_agnt', col_2_drop=True)
        self.csf_indices()
        self.process_num()
        self.to_cat(['zip', 'terr', 'quarter', 'abs_q'])
        
        
        
        
        
        
        
