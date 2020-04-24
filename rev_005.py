import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from sklearn.svm import NuSVR, SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold
from sklearn.linear_model import LinearRegression, BayesianRidge, SGDRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from catboost import CatBoostRegressor
from models import trainer
from startup_directory import make_startup_directory
msd = make_startup_directory()

drop_cols = ['HRMONTH', 'HRYEAR4']

df_train = pd.read_csv('../input/train_2008.csv')
y_train = df_train['target']
id_train = df_train['id']
df_train.drop(columns=['id', 'target'], inplace=True)
df_train.drop(columns=drop_cols, inplace=True)
df_test = pd.read_csv('../input/test_2008.csv')
id_test = df_test['id']
df_test.drop(columns=['id'], inplace=True)
df_test.drop(columns=drop_cols, inplace=True)

n_train = df_train.shape[0]
df_unique = df_train.nunique()
df_all = df_train.append(df_test)

cat_cols_4hot = df_unique.index[df_unique < 4]
df_all_4hot = pd.get_dummies(df_all, columns=cat_cols_4hot.values)
df_train_4hot = df_all_4hot.iloc[:n_train]
df_test_4hot = df_all_4hot.iloc[n_train:]

cat_cols_8hot = df_unique.index[df_unique < 8]
df_all_8hot = pd.get_dummies(df_all, columns=cat_cols_8hot.values)
df_train_8hot = df_all_8hot.iloc[:n_train]
df_test_8hot = df_all_8hot.iloc[n_train:]

cat_cols_Chot = df_unique.index[df_unique < 12]
df_all_Chot = pd.get_dummies(df_all, columns=cat_cols_Chot.values)
df_train_Chot = df_all_Chot.iloc[:n_train]
df_test_Chot = df_all_Chot.iloc[n_train:]

scaler = StandardScaler()
scaler.fit(df_train)
X_train_scaled_0hot = scaler.transform(df_train)
X_train_scaled_0hot = pd.DataFrame(X_train_scaled_0hot, columns=df_train.columns)
X_test_scaled_0hot = scaler.transform(df_test)
X_test_scaled_0hot = pd.DataFrame(X_test_scaled_0hot, columns=df_test.columns)

scaler_4hot = StandardScaler()
scaler_4hot.fit(df_train_4hot)
X_train_scaled_4hot = scaler_4hot.transform(df_train_4hot)
X_train_scaled_4hot = pd.DataFrame(X_train_scaled_4hot, columns=df_train_4hot.columns)
X_test_scaled_4hot = scaler_4hot.transform(df_test_4hot)
X_test_scaled_4hot = pd.DataFrame(X_test_scaled_4hot, columns=df_test_4hot.columns)

scaler_8hot = StandardScaler()
scaler_8hot.fit(df_train_8hot)
X_train_scaled_8hot = scaler_8hot.transform(df_train_8hot)
X_train_scaled_8hot = pd.DataFrame(X_train_scaled_8hot, columns=df_train_8hot.columns)
X_test_scaled_8hot = scaler_8hot.transform(df_test_8hot)
X_test_scaled_8hot = pd.DataFrame(X_test_scaled_8hot, columns=df_test_8hot.columns)

scaler_Chot = StandardScaler()
scaler_Chot.fit(df_train_Chot)
X_train_scaled_Chot = scaler_Chot.transform(df_train_Chot)
X_train_scaled_Chot = pd.DataFrame(X_train_scaled_Chot, columns=df_train_Chot.columns)
X_test_scaled_Chot = scaler_Chot.transform(df_test_Chot)
X_test_scaled_Chot = pd.DataFrame(X_test_scaled_Chot, columns=df_test_Chot.columns)

runname = 'run009'
train_dict = {
  'runname': runname,
  'n_splits': 5, # K-fold cross-validation
}
trn = trainer(train_dict)

params_LGBM = {
  'num_round': 20000,
  'num_leaves': 20,
  'early_stopping_rounds': 200,
  'objective': 'huber',
  'max_depth': 5,
  'learning_rate': 0.08,
  'boosting': 'gbdt',
  'metric': 'auc',
  'verbosity': -1,
  'num_threads': 4,
}
print('LGBM 0hot')
oof_LGBM_0hot, pred_LGBM_0hot, scores_LGBM_0hot, feats_LGBM_0hot= trn.train_model(
  params_LGBM, X_train_scaled_0hot, y_train, X_test_scaled_0hot, 'LGBM_0hot', force_train=True)
#Scores:
#[0.78596376 0.78878018 0.79128013 0.78751608 0.80262161]
#Mean Score:
#0.791232350975688
gc.collect()
print('LGBM 4hot')
oof_LGBM_4hot, pred_LGBM_4hot, scores_LGBM_4hot, feats_LGBM_4hot = trn.train_model(
  params_LGBM, X_train_scaled_4hot, y_train, X_test_scaled_4hot, 'LGBM_4hot', force_train=False)
#Scores:
#[0.78530753 0.78770225 0.79160739 0.78749262 0.80228948]
#Mean Score:
#0.7908798552680464
gc.collect()
print('LGBM 8hot')
oof_LGBM_8hot, pred_LGBM_8hot, scores_LGBM_8hot, feats_LGBM_8hot = trn.train_model(
  params_LGBM, X_train_scaled_8hot, y_train, X_test_scaled_8hot, 'LGBM_8hot', force_train=False)
gc.collect()
#Scores:
#[0.78480985 0.78935359 0.79059615 0.78743527 0.8018463 ]
#Mean Score:
#0.7908082329550423
print('LGBM Chot')
oof_LGBM_Chot, pred_LGBM_Chot, scores_LGBM_Chot, feats_LGBM_Chot = trn.train_model(
  params_LGBM, X_train_scaled_Chot, y_train, X_test_scaled_Chot, 'LGBM_Chot', force_train=False)
gc.collect()
#Scores:
#[0.78468791 0.78786319 0.79122221 0.78632326 0.80181387]
#Mean Score:
#0.7903820913379158

params_XGB = {
  'num_round': 20000,
  'eta': 0.1,
  'max_depth': 5,
  'gamma': 1,
  'subsample': 0.80,
  'lambda': 2.1,
  'objective': 'reg:logistic',
  'eval_metric': 'auc',
  'silent': True,
  #'nthread': 4,
  'tree_method': 'gpu_exact',
}
print('XGB 0hot')
oof_XGB_0hot, pred_XGB_0hot, scores_XGB_0hot = trn.train_model(
  params_XGB, X_train_scaled_0hot, y_train, X_test_scaled_0hot, 'XGB_0hot', force_train=False)
#Scores:
#[0.78723494 0.78822952 0.79310935 0.78723945 0.80231629]
#Mean Score:
#0.7916259104550913
gc.collect()
print('XGB 4hot')
oof_XGB_4hot, pred_XGB_4hot, scores_XGB_4hot = trn.train_model(
  params_XGB, X_train_scaled_4hot, y_train, X_test_scaled_4hot, 'XGB_4hot', force_train=False)
#Scores:
#[0.78637177 0.78762018 0.79218849 0.78741371 0.80287305]
#Mean Score:
#0.7912934419787528
gc.collect()
print('XGB 8hot')
oof_XGB_8hot, pred_XGB_8hot, scores_XGB_8hot = trn.train_model(
  params_XGB, X_train_scaled_8hot, y_train, X_test_scaled_8hot, 'XGB_8hot', force_train=False)
#Scores:
#[0.78749198 0.78842202 0.7932606  0.78828863 0.80301517]
#Mean Score:
#0.7920956811871989
gc.collect()
print('XGB Chot')
oof_XGB_Chot, pred_XGB_Chot, scores_XGB_Chot = trn.train_model(
  params_XGB, X_train_scaled_Chot, y_train, X_test_scaled_Chot, 'XGB_Chot', force_train=False)
#Scores:
#[0.78787487 0.78888885 0.79148386 0.78727823 0.80271557]
#Mean Score:
#0.7916482755117437
gc.collect()

# CatBoost
params_Cat = {
  'iterations': 10000, #ie num_trees
  'learning_rate': 0.02,
  'l2_leaf_reg': 3,
  'bootstrap_type': 'Bayesian',
  'bagging_temperature': 0.25,
  'custom_metric': 'AUC',
  'eval_metric': 'AUC',
  'loss_function': 'RMSE',
  'task_type': 'GPU',
  'logging_level': 'Verbose',
  'use_best_model': True,
  'best_model_min_trees': 40,
  'max_depth': 8,
  'leaf_estimation_method': 'Gradient',
  'od_type': 'IncToDec',
  'od_pval': 1e-6,
  'metric_period': 1000,
}
print('CatBoost 0hot')
oof_Cat_0hot, pred_Cat_0hot, scores_Cat_0hot = trn.train_model(
  params_Cat, X_train_scaled_0hot, y_train, X_test_scaled_0hot, 'CatBoost_0hot', force_train=False)
#Scores:
#[0.79200969 0.79336355 0.79593285 0.79237173 0.80929327]
#Mean Score:
#0.7965942149673138
gc.collect()
print('CatBoost 4hot')
oof_Cat_4hot, pred_Cat_4hot, scores_Cat_4hot = trn.train_model(
  params_Cat, X_train_scaled_4hot, y_train, X_test_scaled_4hot, 'CatBoost_4hot', force_train=False)
#Scores:
#[0.793739   0.79417526 0.79640292 0.79398864 0.80922958]
#Mean Score:
#0.7975070812077243
gc.collect()
print('CatBoost 8hot')
oof_Cat_8hot, pred_Cat_8hot, scores_Cat_8hot = trn.train_model(
  params_Cat, X_train_scaled_8hot, y_train, X_test_scaled_8hot, 'CatBoost_8hot', force_train=False)
#Scores:
#[0.79203463 0.79376396 0.79652445 0.79369927 0.8096099 ]
#Mean Score:
#0.7971264428292184
gc.collect()
print('CatBoost Chot')
oof_Cat_Chot, pred_Cat_Chot, scores_Cat_Chot = trn.train_model(
  params_Cat, X_train_scaled_Chot, y_train, X_test_scaled_Chot, 'CatBoost_Chot', force_train=False)
gc.collect()
#Scores:
#[0.79203463 0.79376396 0.79652445 0.79369927 0.8096099 ]
#Mean Score:
#0.7971264428292184


stck = trainer(train_dict)
train_stack = np.vstack([
  oof_LGBM_0hot, oof_XGB_0hot, oof_Cat_0hot, 
  oof_LGBM_4hot, oof_XGB_4hot, oof_Cat_4hot, 
  oof_LGBM_8hot, oof_XGB_8hot, oof_Cat_8hot, 
  oof_LGBM_Chot, oof_XGB_Chot, oof_Cat_Chot]).transpose()
train_stack = pd.DataFrame(train_stack, columns = [
  'LGBM_0hot', 'XGB_0hot', 'CatBoost_0hot', 
  'LGBM_4hot', 'XGB_4hot', 'CatBoost_4hot', 
  'LGBM_8hot', 'XGB_8hot', 'CatBoost_8hot', 
  'LGBM_Chot', 'XGB_Chot', 'CatBoost_Chot'])
test_stack = np.vstack([
  pred_LGBM_0hot, pred_XGB_0hot, pred_Cat_0hot, 
  pred_LGBM_4hot, pred_XGB_4hot, pred_Cat_4hot, 
  pred_LGBM_8hot, pred_XGB_8hot, pred_Cat_8hot, 
  pred_LGBM_Chot, pred_XGB_Chot, pred_Cat_Chot]).transpose()
test_stack = pd.DataFrame(test_stack, columns = [
  'LGBM_0hot', 'XGB_0hot', 'CatBoost_0hot', 
  'LGBM_4hot', 'XGB_4hot', 'CatBoost_4hot', 
  'LGBM_8hot', 'XGB_8hot', 'CatBoost_8hot', 
  'LGBM_Chot', 'XGB_Chot', 'CatBoost_Chot'])

params_LGBM = {
  'num_round': 20000,
  'early_stopping_rounds': 200,
  'objective': 'huber',
  'max_depth': 4,
  'learning_rate': 0.1,
  'boosting': 'gbdt',
  'metric': 'auc',
  'verbosity': -1,
  'num_threads': 4,
}
print('LGBM')
oof_final, pred_final, scores_final, feats_final = stck.train_model(
  params_LGBM, train_stack, y_train, test_stack, 'LGBM_final', force_train=False)
#Scores:
#[0.79420672 0.79508908 0.79808125 0.79499113 0.81062626]
#Mean Score:
#0.7985988889930018

df_sub = pd.DataFrame({'id': id_test.values, 'target': pred_final})
df_sub.to_csv('../submissions/Stacking_classifier_' + train_dict['runname'] + '.csv', index=False)
