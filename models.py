import os, json
import numpy as np
import pandas as pd
import seaborn as sns
pd.options.display.precision = 15
import os, gc, time, datetime
from tqdm import tqdm, tqdm_notebook
import warnings
warnings.filterwarnings('ignore')
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.svm import NuSVR,SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold
from sklearn.linear_model import LinearRegression, BayesianRidge, SGDRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

from keras.models import Sequential
from keras.layers import Dense, Dropout


class trainer:
  def __init__(self, train_dict):
    self.train_dict = train_dict
    self.folds = KFold(n_splits=self.train_dict['n_splits'], shuffle=True, random_state=15)
  
  def train_model(self, params, X_train, y_train, X_test, name, force_train=False, model=None):
    oof = np.zeros(len(X_train))
    pred = np.zeros(len(X_test))
    score = np.zeros(self.folds.n_splits)
    features = X_train.columns
    feats = pd.DataFrame()
    for fold_, (trn_idx, val_idx) in enumerate(self.folds.split(X_train.values, y_train.values)):
      print('Fold: ' + str(fold_ + 1) + '/' + str(self.train_dict['n_splits']))
      pth_model = '../models/' + name + '_fold' + str(fold_ + 1) + '_' + self.train_dict['runname'] + '.txt'
      if (name[:4] == 'LGBM') or (name == 'LGBM_final'):
        trn_data = lgb.Dataset(X_train.iloc[trn_idx][features], label=y_train.iloc[trn_idx])
        val_data = lgb.Dataset(X_train.iloc[val_idx][features], label=y_train.iloc[val_idx])
        if (not os.path.exists(pth_model)) or (force_train):
          model = lgb.train(params, trn_data, params['num_round'], valid_sets = [trn_data, val_data], verbose_eval=100, early_stopping_rounds=200)
          model.save_model(pth_model)
        else:
          model = lgb.Booster(model_file=pth_model)
        oof[val_idx] = model.predict(X_train.iloc[val_idx][features], num_iteration=model.best_iteration)
        pred += model.predict(X_test[features], num_iteration=model.best_iteration) / self.folds.n_splits
        df_fold = pd.DataFrame()
        df_fold['feature'] = features
        df_fold['importance'] = model.feature_importance()
        df_fold['fold'] = fold_ + 1
        feats = pd.concat([feats, df_fold], axis=0)
      elif name[:3] == 'XGB':
        trn_data = xgb.DMatrix(data=X_train.iloc[trn_idx][features], label=y_train.iloc[trn_idx])
        val_data = xgb.DMatrix(data=X_train.iloc[val_idx][features], label=y_train.iloc[val_idx])
        watchlist = [(trn_data, 'train'), (val_data, 'valid')]
        if (not os.path.exists(pth_model)) or (force_train):
          model = xgb.train(params, trn_data, params['num_round'], watchlist, 
                                early_stopping_rounds=200, verbose_eval=100)
          pickle.dump(model, open(pth_model, "wb"))
        else:
          model = pickle.load(open(pth_model, "rb"))
        oof[val_idx] = model.predict(xgb.DMatrix(X_train.iloc[val_idx][features]),
                                     ntree_limit=model.best_ntree_limit+50)
        pred += model.predict(xgb.DMatrix(X_test[features]),
                                    ntree_limit=model.best_ntree_limit+50) / self.folds.n_splits
      elif name[:8] == 'CatBoost':
        model = CatBoostRegressor()
        model.set_params(**params)
        if (not os.path.exists(pth_model)) or (force_train):
          model.fit(X_train.iloc[trn_idx][features], y_train.iloc[trn_idx],
                    eval_set=(X_train.iloc[val_idx][features], y_train.iloc[val_idx]))
          model.save_model(pth_model)
        else:
          model.load_model(pth_model)
        oof[val_idx] = model.predict(X_train.iloc[val_idx][features])
        pred += model.predict(X_test) / self.folds.n_splits
      else: #scikit-learn models
        # supply model externally outside the function
        if (not os.path.exists(pth_model)) or (force_train):
          model.fit(X_train.iloc[trn_idx][features], y_train.iloc[trn_idx])
          pickle.dump(model, open(pth_model, "wb"))
        else:
          model = pickle.load(open(pth_model, "rb"))
        oof[val_idx] = model.predict(X_train.iloc[val_idx][features]).reshape(-1,)
        pred += model.predict(X_test).reshape(-1,) / self.folds.n_splits
      score[fold_] = roc_auc_score(y_train.iloc[val_idx], oof[val_idx])
    print('Scores:')
    print(score)
    print('Mean Score:')
    print(np.mean(score))
    params_dump = params.copy()
    params_dump['runname'] = self.train_dict['runname']
    params_dump['name'] = name
    for ii, s in enumerate(score):
      params_dump['score' + str(ii)] = s
    params_dump['mean_score'] = score.mean()
    with open('../params/' + name + '_' + self.train_dict['runname'] + '.txt', 'w') as fp:
      json.dump(params_dump, fp, indent=4)
    if (name[:4] == 'LGBM') or (name == 'LGBM_final'):
      return oof, pred, score, feats
    else:
      return oof, pred, score
  
  
  def train_CatBoost(self, params_CatBoost, X_train, y_train, X_test, features):
    self.oof_CatBoost = np.zeros(len(X_train))
    self.pred_CatBoost = np.zeros(len(X_test))
    model = CatBoostRegressor()
    model.set_params(**params_CatBoost)
    score = np.zeros(self.folds.n_splits)
    for fold_, (trn_idx, val_idx) in enumerate(self.folds.split(X_train.values, y_train.values)):
      print('Fold: ' + str(fold_ + 1) + '/' + str(self.train_dict['n_splits']))
      model.fit(X_train.iloc[trn_idx][features], y_train.iloc[trn_idx],
                eval_set=(X_train.iloc[val_idx][features], y_train.iloc[val_idx]))
      self.oof_CatBoost[val_idx] = model.predict(X_train.iloc[val_idx][features])
      score[fold_] = mean_absolute_error(y_train.iloc[val_idx], self.oof_CatBoost[val_idx])
      self.pred_CatBoost += model.predict(X_test) / self.folds.n_splits
    print('Scores:')
    print(score)
    print('Mean Score:')
    print(np.mean(score))
  
  
  def train_DNN(self, params_DNN, X_train, y_train, X_test, features):
    self.oof_DNN = np.zeros(len(X_train))
    self.pred_DNN = np.zeros(len(X_test))
    score = np.zeros(self.folds.n_splits)
    for fold_, (trn_idx, val_idx) in enumerate(self.folds.split(X_train.values, y_train.values)):
      print('Fold: ' + str(fold_ + 1) + '/' + str(self.train_dict['n_splits']))
      model = Sequential()
      model.add(Dense(units=params_DNN['layers'][0], activation='relu', input_dim=X_train.shape[1]))
      for l in params_DNN['layers'][1:]:
        model.add(Dense(l, activation='relu'))
        model.add(Dropout(params_DNN['dropout']))
      model.add(Dense(1))
      model.compile(loss='mae', optimizer='adam', metrics=['mae'])
      model.summary()
      model.fit(X_train.iloc[trn_idx][features], y_train.iloc[trn_idx], 
                epochs=params_DNN['epochs'], batch_size=params_DNN['batch_size'])
      self.oof_DNN[val_idx] = model.predict(X_train.iloc[val_idx][features]).reshape(-1,)
      score[fold_] = mean_absolute_error(y_train.iloc[val_idx], self.oof_DNN[val_idx])
      self.pred_DNN += model.predict(X_test).reshape(-1,) / self.folds.n_splits
    print('Scores:')
    print(score)
    print('Mean Score:')
    print(np.mean(score))



