

from ngboost import NGBRegressor as NGBRegressor_
import pandas as pd
import numpy as np
from ngboost.distns.distn import RegressionDistn
from ngboost.scores import CRPScore, LogScore
from CropLearner import NgBoostLearner
from pyhere import here

def score(Y): # how to put censor limit here?!
    # check if y below censor limit
    # need mask for censor limit
    val = Y['Residue level']
    censor_limit = Y['LOQ']
    if val >= censor_limit:
        return -self.dist.logpdf(val)
    else:
        return - np.log(self.dist.cdf(val) + self.eps) 
    
ngb_path=here('CropPredictor/savedModels/Production/Regressor_GoZero.pkl')
ng= NgBoostLearner(skip_init_from_pickle = True).from_pickle(ngb_path)
df = ng.df
X = ng.X
X_test = ng.X_test
self = ng
distns = self.distns
Y = self._y_plain['Residue level']
loq = df['LOQ']
Y_input = pd.DataFrame([Y, loq]).T
import pdb; pdb.set_trace()
score(Y_input.iloc[0])



cens = [np.log(dist.cdf(loq[i]) + 0.001) for i, dist in enumerate(distns[self.censor_mask])]
#cens = (1 - E) * np.log(1 - self.dist.cdf(T) + self.eps)
uncens = [dist.logpdf(y[i]) for i, dist in enumerate(~distns[self.censor_mask])]
#return -(cens + uncens)

from ngboost import NGBRegressor
from ngboost.distns import Normal, LogNormal
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

boston = load_boston()
data = boston.data
X = pd.DataFrame(data, columns = boston.feature_names)
Y = boston.target
import pdb; pdb.set_trace()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
import pdb; pdb.set_trace()
ngb_normal = NGBRegressor()
ngb_lognormal = NGBRegressor(Dist = LogNormal)
ngb = NGBRegressor().fit(X_train, Y_train)
ngb = NGBRegressor(Dist = LogNormal).fit(X_train, Y_train)
Y_preds = ngb.predict(X_test)
Y_dists = ngb.pred_dist(X_test)

# test Mean Squared Error
test_MSE = mean_squared_error(Y_preds, Y_test)
print('Test MSE', test_MSE)

# test Negative Log Likelihood
test_NLL = -Y_dists.logpdf(Y_test).mean()
print('Test NLL', test_NLL)
