from datasets import load_dataset, load_metric
import csv
from sklearn.linear_model import LinearRegression,RidgeCV, LassoCV
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

import sklearn_blend

if __name__ == "__main__":
    # lin_reg = LinearRegression()
    # blending(lin_reg)
    # estimators = [('ridge', RidgeCV()),
    #               ('lasso', LassoCV(random_state=42)),
    #               ('knr', KNeighborsRegressor(n_neighbors=20,metric='euclidean')),
    #               ('svr',SVR()),
    #               ('lin',LinearRegression())
    #               ]
    # final_estimator = GradientBoostingRegressor(
    #             n_estimators=25, subsample=0.5, min_samples_leaf=25, max_features=1,
    #             random_state=42)
    # reg = StackingRegressor(
    #     estimators=estimators,
    #     final_estimator=final_estimator)
    gbr_reg = GradientBoostingRegressor()
    parameters = {
        'n_estimators':[25,50,75,100],
        'min_samples_split':[1,2,3,4,5],
        'min_samples_leaf':[1,2,3,4,5],
        'max_depth':[1,2,3,4,5], 
        'tol':[1e-5,1e-4,1e-3]
        }
    # parameters = {
    #     'kernel':('linear','rbf'), 
    #     'tol':[1e-5,1e-4,1e-3]
    #     }
    spearman_scorer = sklearn_blend.make_scorer(sklearn_blend.spearman_score)
    reg = GridSearchCV(gbr_reg, parameters, scoring=spearman_scorer)
    sklearn_blend.blending(reg,'gbr_spearman_results.csv')

    print(reg.best_params_)

