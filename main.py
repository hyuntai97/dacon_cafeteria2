import argparse
import json 
import pickle

import os
import pandas as pd

import warnings
warnings.filterwarnings(action='ignore')

from load import dataloader
from models import model_selection
from train import training
from utils import make_submission
from utils import version_update
from load2 import dataloader2

if __name__=='__main__':
    # config
    parser = argparse.ArgumentParser(description="Machine Learning Pipeline")
    parser.add_argument('--seed', type=int, default=42, help='Set seed')
    parser.add_argument('--datadir', type=str, default='./data', help='Set data directory')
    parser.add_argument('--logdir', type=str, default='./logs', help='Set log directory')
    parser.add_argument('--paramdir', type=str, default='./params', help='Set parameters directory')
    parser.add_argument('--preprocess', type=int, default=1, help='Select preprocess method')
    parser.add_argument('--load', type=int, default=1, help='Select load type')
    parser.add_argument('--params', type=str, default='default', help='Model hyperparmeter')
    parser.add_argument('--kfold', type=int, default=None, help='Number of cross validation')
    parser.add_argument('--val_size', type=float, default=None, help='Set validation size')
    parser.add_argument('--modelname', type=str, 
                        choices=['OLS','Ridge','Lasso','ElasticNet','DT','RF','ADA','GT','SVM','KNN','LGB','XGB'],
                        help='Choice machine learning model')
    

    args = parser.parse_args()

    # set version and define save directory
    savedir = version_update(args.logdir)

    # save argument
    json.dump(vars(args), open(os.path.join(savedir,'arguments.json'),'w'))

    # 1. load data & preprocessing
    if args.load == 1:
        train_lunch, test_lunch, y_lunch, train_dinner, test_dinner, y_dinner = dataloader(datadir=args.datadir, preprocess=args.preprocess)
    elif args.load == 2:
        train_lunch, test_lunch, y_lunch, train_dinner, test_dinner, y_dinner = dataloader2(datadir=args.datadir, preprocess=args.preprocess)
        
    # 2. model setting
    params = json.load(open(os.path.join(args.paramdir,f'{args.modelname.lower()}_{args.params}.json'),'r'))
    model = model_selection(modelname=args.modelname, params=params, random_state=args.seed)
    
    # 3. training
    
    #lunch
    y_pred_lunch, val_results_lunch = training(model=model,
                                   train=[train_lunch, y_lunch],
                                   test=test_lunch,
                                   val_size=args.val_size,
                                   K=args.kfold)

    #dinner
    y_pred_dinner, val_results_dinner = training(model=model,
                                   train=[train_dinner, y_dinner],
                                   test=test_dinner,
                                   val_size=args.val_size,
                                   K=args.kfold)
                                        
    total_val_results = [val_results_lunch, val_results_dinner]

    # 4. save results
    pickle.dump(total_val_results, open(os.path.join(savedir, f'validation_results.pkl'),'wb'))

    submission = make_submission(y_pred_lunch, y_pred_dinner, args.datadir)
    submission.to_csv(os.path.join(savedir, 'prediction.csv'), index=False)

