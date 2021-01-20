# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 20:21:09 2019

@author: ischwartz
"""
import numpy as np
import numba 
from catboost import Pool, CatBoostRegressor

numba.jit()
def fit_surrogate_model(dataset,addFeatures,seed):
	Nfeatures = dataset.shape[1]-1
	traindf = dataset
	traindf_X = traindf[:,0:Nfeatures]
	traindf_X = np.hstack((traindf_X,addFeatures))
	traindf_y = traindf[:,Nfeatures]
	train_pool = Pool(traindf_X,traindf_y)	
	
	# hyperparameters (iterations, l2_leaf_reg, depth and learning_rate) have to be calibrated
	surrogate_model = CatBoostRegressor(loss_function='RMSE', task_type = "GPU", logging_level = 'Silent', random_seed=seed, eval_metric='RMSE',iterations=600, l2_leaf_reg=0, depth=6, learning_rate=0.15)
	surrogate_model.fit(train_pool)
	return surrogate_model
