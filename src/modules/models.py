from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np
import xgboost as xgb
import glob

def xgb_cv(
        data_, test_, y_, max_depth,gamma, reg_lambda , reg_alpha,\
        subsample, scale_pos_weight, min_child_weight, colsample_bytree,
        test_phase=False, stratify=False,
        ):
    """XGBoost cross validation.
    This function will instantiate a XGBoost classifier with parameters
    such as max_depth, subsample etc. Combined with data and
    targets this will in turn be used to perform cross validation. The result
    of cross validation is returned.
    Our goal is to find combinations of parameters that maximizes AUC.
    
    Returns:
    if test_phase (and new data for validators, just change the test_ param
                    to the new data and make sure that the features are processed):
        sub_preds : models prediction to get the hold-out score
    else:
        validation AUC score
    
    Model Notes:
        XGBoost overfits in this dataset, params should be set accordingly.
    Parameter Notes
    
        gamma : Minimum loss reduction required to make a further partition on a leaf \
        node of the tree. The larger gamma is, the more conservative the algorithm will be.
        
        min_child_weight :  The larger min_child_weight is, the more conservative the algorithm will be.
        
        colsample_bytree : The subsample ratio of columns when constructing each tree.
        
        scale_pos_weight : A typical value to consider: sum(negative instances) / sum(positive instances)
    """
    oof_preds = np.zeros(data_.shape[0])
    sub_preds = np.zeros(test_.shape[0])
    if test_phase:
        max_depth = int(np.round(max_depth))
    feats = [f for f in data_.columns if f not in ['bookingid', 'label']]
    
    if stratify:
        folds_ = StratifiedKFold(n_splits=4, shuffle=True, random_state=610)
        splitted = folds_.split(data_, y_)
    else:
        splitted = folds_.split(data_)
    for n_fold, (trn_idx, val_idx) in enumerate(splitted):
        trn_x, trn_y = data_[feats].iloc[trn_idx], y_.iloc[trn_idx]
        val_x, val_y = data_[feats].iloc[val_idx], y_.iloc[val_idx]
        
        xg_train = xgb.DMatrix(
            trn_x.values, label=trn_y.values
                               )
        xg_valid = xgb.DMatrix(
            val_x.values, label=val_y.values
                               )   
        
        watchlist = [(xg_train, 'train'),(xg_valid, 'eval')]
        num_round=10000
        param = {
        'gamma' : gamma, 
        'max_depth':max_depth, 
        'colsample_bytree':colsample_bytree, 
        'subsample':subsample, 
        'min_child_weight':min_child_weight, 
        'objective':'binary:logistic', 
        'random_state':1029,
        'n_jobs':8,
        'eval_metric':'auc',
        'metric': 'auc',
        'scale_pos_weight':scale_pos_weight,
        'eta':0.05,
        'silent':True
        }
        clf = xgb.train(param, xg_train, num_round, watchlist, verbose_eval=100, early_stopping_rounds = 100)
        
        oof_preds[val_idx] = clf.predict(xgb.DMatrix(data_[feats].iloc[val_idx].values), ntree_limit=clf.best_ntree_limit) 
        
        if test_phase:
            
            sub_preds += clf.predict(xgb.DMatrix(test_[feats].values), ntree_limit=clf.best_ntree_limit) / folds_.n_splits
        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(val_y, oof_preds[val_idx])))
        del clf, trn_x, trn_y, val_x, val_y
        gc.collect()
    print('Full AUC score %.6f' % roc_auc_score(y_, oof_preds)) 
    if test_phase:
        return sub_preds
    else:
        return roc_auc_score(y_, oof_preds)
    
    
from bayes_opt import BayesianOptimization
from six.moves import cPickle
import gc
def optimize_xgb(data_, test_, y_,  init_points=2, num_iter=2, num_params=1, stratify=False):
    """
    Bayesian Optimization of XGBoost classifier.
    creates a .dict file that can be used to retrain the model and evaluate results.
    """
    store_fname = 'xgb_BO_res_optimization.dict'
    
    print('Bayesian optimization results will be stored in {} after training...'.format(store_fname))
    """Apply Bayesian Optimization to XGBoost parameters."""
    def xgb_crossval(
        max_depth, reg_alpha, reg_lambda, gamma, subsample, scale_pos_weight, \
                     min_child_weight, colsample_bytree, 
                    ):
        """Wrapper of XGBoost cross validation.
        Casting features on its according data types and using pbounds to cap min and max values
        for each parameter.
        """
        return xgb_cv(
            max_depth=int(max_depth), 
            colsample_bytree=max(min(round(colsample_bytree, 2), 1), 0),
            subsample=max(min(round(subsample, 2), 1), 0),
            min_child_weight=int(min_child_weight), 
            gamma=int(gamma),
            scale_pos_weight=float(scale_pos_weight),
            reg_lambda=float(reg_lambda),
            reg_alpha=float(reg_alpha),
            data_=data_,
            test_=test_,
            y_=y_,
            stratify=stratify
            )

    optimizer = BayesianOptimization(
        f=xgb_crossval,
        pbounds={
            'colsample_bytree': (0.05, 1.0),
            'reg_alpha': (0.2, 1.0),
            'reg_lambda': (0.2, 1.0),
            'max_depth': (2, 6),
            'subsample': (0.05, 1.0),
            'gamma': (1, 10),
            'min_child_weight': (10, 100),
            'scale_pos_weight': (0, 1),
            },
        random_state=1234,
        verbose=2
    )
    optimizer.maximize(init_points=init_points, n_iter=num_iter, acq="poi", xi=0.1) 
    with open('{}'.format(optimizer.max['target']) + '-' + store_fname, 'wb') as fl:
        cPickle.dump(optimizer.res, fl)
    print("Final result:", optimizer.max)
    
###LIGHTGBM
from lightgbm import LGBMClassifier
def lgb_cv(
            data_, test_, y_, max_bin, num_leaves, min_sum_hessian_in_leaf, \
            feature_fraction, bagging_fraction, bagging_freq, min_data_in_leaf,\
            min_gain_to_split, reg_alpha, reg_lambda,
            test_phase=False, stratify=False
        ):
    """lightgbm cross validation.
    This function will instantiate a lightgbm classifier with parameters
    same with XGBoost goal.
    
    Returns:
    if test_phase (and new data for validators, just change the test_ param
                    to the new data and make sure that the features are processed):
        sub_preds : models prediction to get the hold-out score
    else:
        validation AUC score
    Parameter Notes: () are default values
        min_data_in_leaf (20) and min_sum_hessian_in_leaf (1-e3) : it can be used to deal with over-fitting
        
        max_bin (255) : small number of bins may reduce training accuracy 
            but may increase general power (deal with over-fitting)
            
        min_gain_to_split (0) : the minimal gain to perform split
        
        feature_fraction (0) : LightGBM will randomly select part of features on each iteration if feature_fraction smaller than 1.0. 
        For example, if you set it to 0.8, LightGBM will select 80% of features before training each tree.
        
        bagging_fraction (1) : like feature_fraction, but this will randomly select part of data without resampling 
        can be used to speed up training and can be used to deal with over-fitting
    """
    oof_preds = np.zeros(data_.shape[0])
    sub_preds = np.zeros(test_.shape[0])
    if test_phase:
        num_leaves = int(np.round(num_leaves))
        max_bin = int(np.round(max_bin))
        min_data_in_leaf = int(np.round(min_data_in_leaf))
        bagging_freq = int(np.round(bagging_freq))
    feats = [f for f in data_.columns if f not in ['bookingid', 'label']]
    
    if stratify:
        folds_ = StratifiedKFold(n_splits=5, shuffle=True, random_state=610)
        splitted = folds_.split(data_, y_)
    else:
        splitted = folds_.split(data_)
    for n_fold, (trn_idx, val_idx) in enumerate(splitted):
        trn_x, trn_y = data_[feats].iloc[trn_idx], y_.iloc[trn_idx]
        val_x, val_y = data_[feats].iloc[val_idx], y_.iloc[val_idx]
        clf =  LGBMClassifier(
                            num_leaves=num_leaves, 
                            min_sum_hessian_in_leaf=min_sum_hessian_in_leaf, 
                            feature_fraction=feature_fraction, 
                            max_bin=max_bin, 
                            bagging_fraction=bagging_fraction, 
                            bagging_freq=bagging_freq,
                            min_data_in_leaf=min_data_in_leaf,
                            reg_lambda=reg_lambda,
                            reg_alpha=reg_alpha,
                            objective='binary', 
                            metric='auc',
                            random_state=2039281,
                            n_jobs=8,
                            n_estimators=10000,
                            is_unbalance=True, #imbalanced classes
                            eta=0.05
                                )
        clf.fit(trn_x, trn_y, 
                eval_set= [(trn_x, trn_y), (val_x, val_y)], 
                eval_metric='auc', verbose=1000, early_stopping_rounds=500
               )

        oof_preds[val_idx] = clf.predict_proba(val_x, num_iteration=clf.best_iteration_)[:, 1]
        if test_phase:
            sub_preds += clf.predict_proba(test_[feats], num_iteration=clf.best_iteration_)[:, 1] / folds_.n_splits
        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(val_y, oof_preds[val_idx])))
        del clf, trn_x, trn_y, val_x, val_y
        gc.collect()
    print('Full AUC score %.6f' % roc_auc_score(y_, oof_preds)) 
    if test_phase:
        return sub_preds
    else:
        return roc_auc_score(y_, oof_preds)
    
from bayes_opt import BayesianOptimization
from six.moves import cPickle
import gc
def optimize_lgb(data_, test_, y_, init_points=2, num_iter=2, num_params=1, stratify=False):
    
    """
    Bayesian Optimization of lightgbm classifier.
    creates a .dict file that can be used to retrain the model and evaluate results.
    """
    store_fname = 'lgb_BO_res_optimization.dict'
    print('Bayesian optimization results will be stored in {} after training...'.format(store_fname))
    """Apply Bayesian Optimization to lightgbm parameters."""
    def lgb_crossval(max_bin, num_leaves, min_sum_hessian_in_leaf, feature_fraction, bagging_fraction, bagging_freq, min_data_in_leaf,min_gain_to_split, reg_alpha, reg_lambda):
        """
        Wrapper of lightgbm cross validation.
        """
        return lgb_cv(
            max_bin=int(max_bin),
            num_leaves=int(num_leaves),
            min_sum_hessian_in_leaf=int(min_sum_hessian_in_leaf), 
            feature_fraction=max(min(round(feature_fraction, 2), 1), 0),
            bagging_fraction=max(min(round(bagging_fraction, 2), 1), 0),
            reg_alpha=max(min(round(reg_alpha, 2), 1), 0),
            reg_lambda=max(min(round(reg_lambda, 2), 1), 0),
            bagging_freq=int(bagging_freq), 
            min_data_in_leaf=int(min_data_in_leaf), 
            min_gain_to_split=max(min(round(min_gain_to_split, 2), 1), 0),
            data_=data_,
            test_=test_,
            y_=y_,
            stratify=stratify
            )

    optimizer = BayesianOptimization(
        f=lgb_crossval,
        pbounds=
            {
                'max_bin': (25, 150),
                'bagging_fraction': (0.0, 1),
                'num_leaves': (5, 40),
                'min_data_in_leaf' : (1000, 5000),
                'min_sum_hessian_in_leaf' : (0.00001, 0.0005 ),
                'bagging_freq': (0, 1),
                'feature_fraction': (0.05, 0.5),
                'min_gain_to_split': (0.0, 1.0),
                'reg_lambda': (0.0, 1.0),
                'reg_alpha': (0.0, 1.0)
            },
        random_state=1234,
        verbose=2
    )
    optimizer.maximize(init_points=init_points, n_iter=num_iter, acq="poi", xi=0.1)
    with open('{}'.format(optimizer.max['target']) + '-' + store_fname, 'wb') as fl:
        cPickle.dump(optimizer.res, fl)