"""
Author: Jake Anderson
Original code and method by:
License: 
    
"""
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin



class LagGenerator(BaseEstimator, TransformerMixin):
    
    """
    Lag Generator for Trend thesis. There were two main concerns with this particular
    dataset when trying to construct features.
    
    1. the policies are through independent agents
    2. individual policy status are displayed monthly for agents; the policy status
    if updated annually. 
    
    For a more detailed analysis, see thesis document 
    
    The inital code for the lag features was very messy and ended up pieced together
    due to time constraints. This is an attempt to further modularize and generalize the 
    code. Unfortunately, this is not a common dataset concern, but feel free to use and 
    improve. 
    
    
    Summary: 
        
        
    -Premium annual lags: aggregate features were able to be generated for the independent agents and 
    applied at the policy level for predictive purposes. There was limited policy level
    information, however, independent agent information was available and features that
    may be able to capitalize on collective behavior were generated. One of the more 
    informative features in commercial insurance is policy premium. Therefore, aggregate
    agent quarterly premium lag features were engineered. The challenge was the yearly
    individual policy status update for a given independent agent holding numerous policies
    (with a particular carrier). This class makes certain assumptions about a base (beginning) 
    agent's annual policies and then updates the annual policies quarterly in order to 
    obtain aggregate features. 
    
    -Quarterly lags: lags for types of policies per agent could only be lagged quarterly
    due to data constraints. Annual aggregate data could be captured, but an entire year
    on data would need to be sacrificed. For the premium features, the base year could
    be assumed by the presence of policies a year later - regardless of whether they were renews
    or not. 
    
    Parameters
    -------------
    value: string, default = 'prem'
        This could also be any of the features related to premium.
     
    entity: string, default = 'agnt'
        The aggregate value. These are the independent agents that have client list 
        ownership
    
    aggfunc: string, default='sum'  
        see pivot tables 
        
    entity2: string, default=None  
        This is a potentially useful hyperparameter. It allows for the aggregated 
        feature to be broken down further by another feature. For exampe, this was 
        used to further breakdown agent premium by insurance line. So if a given agency
        was seeing a quarterly decline in auto insurance premium, then this may prove a 
        better indicator for certain policies (i.e. smaller business, policies for transportation
        companies). While I was excited about the feature synthesis possibilites, preliminary
        testing showed poor results with models treating the further refined aggregate features
        as groups of correlated features. This introduced a host of issues discussed towards th 
        end of the thesis. I ended up scraping entity2. But it could easily be reintroduced
        for further experimenting. 
    
    colname: string, default='prem'
        Column name for lagged features
    
    annual: bool, default=True
    
        Setting this to False will bypass the base updater and generate aggregated quarterly
        features.  Use this for policy type counts.
        
    policy: string, default='Non-Renew'
         Used for policy type counts. Choices: 'New', 'Non-Renew', 'Renew'  
     
    References
    ---------------
    Trend thesis
    
    """
    
    
    def __init__(self, value='prem', entity='agnt', aggfunc='sum',
                 entity2=None, colname='prem', annual=True, policy='Non-Renew'):
        self.value = value
        self.entity = entity
        self.aggfunc = aggfunc
        self.colname = colname
        self.separater = '_'
        self.annual = annual
        self.policy = policy
    
 
    
    
    def relative_q(self, absolute_q):
        # this was used to convert absolute quarter into local quarters - see base
        # updater
        
        if absolute_q < 5:
            return absolute_q
        elif ((absolute_q > 4) & (absolute_q < 9)):
            return absolute_q - 4
        else:
            return absolute_q - 8




    def dict_gen(self, base_update):
        # returns dict. Modin.pandas may cause problems with pivot tables
        
        pv = pd.pivot_table(base_update, values=self.value, index=self.entity,
                            aggfunc=self.aggfunc).fillna(0)
        
        return pd.Series(pv.loc[:, self.value], index=pv.index).to_dict() 




    def quarter_updater(self, quarter, lags=2):
        """
        Used if annual parameter = False. Intended for policy count lags. Given
        the limited data, lags are kept to 2 for all features. However, with a large
        enough dataset, this could be a lot larger (and possible more predictive)
        """
        
        lag_dicts=[]
        for lag in range(1, lags+1):
            sub = self.data[ (self.data['abs_q'] == quarter-lag) & (self.data['churn'] == self.policy)]
            lag_dict = self.dict_gen(sub)
            lag_dicts.append(lag_dict)
            
        return tuple(lag_dicts)
       



    def base_updater(self, base, quarter):
        """
        This must be initiated with the original base (see tranform()). It will
        then update the base and take aggregate feature information. Be very careful
        if you try to further generalize this. 
        """
        # notice the abs quarter to select the update quarter made up with new  and 
        # renew policies. 
        sub = self.data[ (self.data['abs_q'] == quarter-2) & (self.data['churn'] != 'Non-Renew') ]
        # the relative quarter in the base is then discarded
        base = base[ -( base['quarter'] == self.relative_q(quarter-2) ) ]  
        # and updated with the abs quarter of newer information
        base_update1 = base.append(sub)

        lag_2_dict = self.dict_gen(base_update1)

        sub = self.data[ (self.data['abs_q'] == quarter-1) & (self.data['churn'] != 'Non-Renew') ]
        base = base_update1[ -( base_update1['quarter'] == self.relative_q(quarter-1) ) ]  
        base_update2 = base.append(sub)

        lag_1_dict = self.dict_gen(base_update2)

        return lag_1_dict, lag_2_dict, base_update2

  
    
    
    def stacker(self, quarter, final, lag1, lag2):
        # takes in the two dicts and final dframe. The final dframe should be those
        # quarters that will later be discarded (i.e. abs Q1 and Q2). Leaving 
        # these quarters off with throw of other feature generators
        
        df_ = self.data[self.data.abs_q == quarter]
        # Colname from parameter
        df_['lag_1_' + self.colname] = df_[self.entity].map(lag1)
        df_['lag_2_' + self.colname] = df_[self.entity].map(lag2)
        # row-wise join. 
        return pd.concat([final, df_], axis=0)
 
    
    
    
    def fit(self, X, y=None):
        # can now use in sklearn pipeline
        return self


    
    def transform(self, X, y=None):
        self.data = X
        self.target = y
        # Create empty colvecs in order to keep abs_q 1 and 2
        final_df = self.data[self.data.abs_q.isin([1, 2])]
        for lag in range(1,3):
            name = self.separater.join(['lag', str(lag), self.colname])
            final_df[name] = 0
            
        # set the starting base. This is the active policies as of Q4 2015
        base_origin =  self.data[(self.data['churn'] != 'New') & (self.data['year'] == '2016_')]               
        base = base_origin 

        for i in range(3, 11): 
            if self.annual:
                lag1_d, lag2_d, base = self.base_updater(base, quarter=i)
                final_df = self.stacker(i, final_df, lag1_d, lag2_d)
                
            else:
                lag1_d, lag2_d = self.quarter_updater(quarter=i)
                final_df = self.stacker(i, final_df, lag1_d, lag2_d)
        
        return final_df
