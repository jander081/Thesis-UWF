import modin.pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# import scikitplot as skplt

#-----------sklearn----------------------------------------------------

from sklearn.model_selection import cross_val_score, train_test_split, KFold,  RepeatedStratifiedKFold
from sklearn.feature_selection import SelectFromModel,RFE, chi2

from sklearn.metrics import (mean_squared_error, r2_score, recall_score, confusion_matrix, classification_report, log_loss, precision_score, accuracy_score,f1_score,roc_auc_score)

from sklearn.pipeline import make_pipeline, FeatureUnion, Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler, scale

#---------------------------------------------------------------

from scipy.stats import chi2_contingency #, boxcox
from skopt import BayesSearchCV
from fancyimpute import SoftImpute
import scikitplot as skplt


from collections import defaultdict
from datetime import datetime
from imblearn.over_sampling import SMOTE
from faker import Faker
from tqdm import tqdm
tqdm.pandas()

#-----------MODELS----------------------------------------------------

# from xgboost import XGBRegressor
from xgboost import XGBClassifier
from xgboost import plot_importance
import xgbfir


#-----------OTHER----------------------------------------------------
import datetime
import calendar 
import sys
import os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'
import sys
import re
import random
import pickle

import warnings
warnings.filterwarnings(action='ignore')

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)


#----------------------------INSTALLS----------------------------------














