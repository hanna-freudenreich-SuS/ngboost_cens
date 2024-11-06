import numpy as np
import pandas as pd
from pyhere import here
import ngboost_module as ngm
# from ngboost_module.ngboost.scores import LogScore, CRPScore
# from ngboost_module.ngboost.distns.normal import Normal as SimpleNormal
from CropLearner import NgBoostLearner
#from ngboost import NGBRegressor, NGBCensored
# from sklearn.datasets import load_boston
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error
#from jax.ops import index_update, index
#from ngboost.censored import CensoredOutcome

# load our data

ngb_path=here('CropPredictor/savedModels/Production/Regressor_GoZero.pkl')
ng= NgBoostLearner(skip_init_from_pickle = True).from_pickle(ngb_path)

# helper function to introduce interval, right, or left censoring
# also demonstrates how to construct a CensoredOutcome object
# how to adapt for custom censoring limits?!

def censor_admin(y, lower=-np.inf, upper=np.inf):
    observed = np.nan*np.zeros(y.shape)
    #ix_obs = (y < lower) | (upper < y)
    ix_obs = (y > lower) & (upper > y)
    #observed = index_update(observed, index[ix_obs], y[ix_obs])
    # rewrite w numpy
    observed[ix_obs] = y[ix_obs]
    
    # if lower and or upper is array-like, must adapt censored
    if isinstance(lower, (list, np.ndarray, pd.Series)) and isinstance(upper, (list, np.ndarray, pd.Series)):
        censored = np.column_stack((lower, upper))
    elif isinstance(lower, (list, np.ndarray, pd.Series)) and isinstance(upper, float):
        upper = upper * np.ones(len(y))
        censored = np.column_stack((lower, upper))
    elif isinstance(upper, (list, np.ndarray, pd.Series)) and isinstance(lower, float):
        lower = lower * np.ones(len(y))
        censored = np.column_stack((lower, upper))
    elif isinstance(lower, float) and isinstance(upper, float):
        censored = np.array([lower, upper])*np.ones((len(y), 2))
    else: 
        raise ValueError('input type not supported')
    return ngm.CensoredOutcome(
        observed, # contains np.nan where observation was censored 
        censored  # for rows where censored, contains the lower and upper bounds of the censoring interval (e.g. [-inf, b] for a row left-censored at b). Rows where `observed` is not np.nan are ignored.
    )  
lower = ng.df['LOQ']
y = ng.y['Residue level']
import pdb; pdb.set_trace()
Y_cens = censor_admin(y,  lower = lower)
import pdb; pdb.set_trace()
Y_cens.observed
Y_cens.observed_all
Y_cens.ix_obs

Y_cens.censored
Y_cens.censored_all
Y_cens.ix_cen



import pdb; pdb.set_trace()

# X, Y = load_boston(return_X_y = True)
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# # fake left-censoring (i.e. all observations must be > 20)
# Y_train_censored = censor_admin(Y_train, upper=20)

# ngb = NGBCensored(Dist=Normal, Score=LogScore).fit(X_train, Y_train_censored)
# Y_preds = ngb.predict(X_test)
# Y_dists = ngb.pred_dist(X_test)