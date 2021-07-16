import os 
import pandas as pd 
import numpy as np
import datetime as dt
from workalendar.asia import SouthKorea
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from sklearn.preprocessing import PowerTransformer

def dataloader(datadir, preprocess = 1):
    train = pd.read_csv(os.path.join(datadir, f'train.csv'))
    test = pd.read_csv(os.path.join(datadir, f'test.csv'))

    if preprocess == 1:
        train_lunch, test_lunch, y_lunch, train_dinner, test_dinner, y_dinner = preprocessing1(train, test)
        
    if preprocess == 2:
        train_lunch, test_lunch, y_lunch, train_dinner, test_dinner, y_dinner = preprocessing2(train, test)
    
    if preprocess == 3:
        train_lunch, test_lunch, y_lunch, train_dinner, test_dinner, y_dinner = preprocessing3(train, test)

    if preprocess == 4:
        train_lunch, test_lunch, y_lunch, train_dinner, test_dinner, y_dinner = preprocessing4(train, test)

    if preprocess == 5:
        train_lunch, test_lunch, y_lunch, train_dinner, test_dinner, y_dinner = preprocessing5(train, test)

    return train_lunch, test_lunch, y_lunch, train_dinner, test_dinner, y_dinner


def make_holidays_prev(data):
    # 16,17,18,19,20,21 공휴일
    holidays = pd.concat([pd.Series(np.array(SouthKorea().holidays(2021))[:, 0]),
                        pd.Series(np.array(SouthKorea().holidays(2020))[:, 0]),
                        pd.Series(np.array(SouthKorea().holidays(2019))[:, 0]),
                        pd.Series(np.array(SouthKorea().holidays(2018))[:, 0]),
                        pd.Series(np.array(SouthKorea().holidays(2017))[:, 0]), 
                        pd.Series(np.array(SouthKorea().holidays(2016))[:, 0])
                        ]).reset_index(drop = True)    
                        
    holidays_date_prev = pd.to_datetime(holidays) - timedelta(days = 1)
    data['휴일전날'] = data['일자'].dt.date.isin(holidays_date_prev.dt.date).astype(int)
    data.loc[data['요일']==4, '휴일전날'] = 1

    return data

def detect_fix_outlier(train, test, cols:list):
  for col in cols:
    iqr = train[col].describe()['75%'] - train[col].describe()['25%']
    max_outlier = train[col].describe()['75%'] + (1.5*iqr)
    min_outlier = train[col].describe()['25%'] - (1.5*iqr)
    
    max_outlier_idx_train = []
    min_outlier_idx_train = []
    max_outlier_idx_test = []
    min_outlier_idx_test = []

    # train
    for i in range(len(train[col])):
      if train[col][i] > max_outlier:
        max_outlier_idx_train.append(i)
      elif train[col][i] < min_outlier:
        min_outlier_idx_train.append(i)

    train[col].iloc[max_outlier_idx_train] = max_outlier
    train[col].iloc[min_outlier_idx_train] = min_outlier
  
    # test
    for i in range(len(test[col])):
      if test[col][i] > max_outlier:
        max_outlier_idx_test.append(i)
      elif test[col][i] < min_outlier:
        min_outlier_idx_test.append(i)

    test[col].iloc[max_outlier_idx_test] = max_outlier
    test[col].iloc[min_outlier_idx_test] = min_outlier
        
  return train, test

def min_max_scaling(train, test, cols:list):
    scaler = MinMaxScaler()
    scaler.fit(train[cols])
    #train
    scaled_train = pd.DataFrame(scaler.transform(train[cols]), columns = cols)
    train_non_scaled = train.drop(cols, axis = 1)
    train_new = pd.concat([scaled_train, train_non_scaled], axis = 1)
    #test
    scaled_test = pd.DataFrame(scaler.transform(test[cols]), columns = cols)
    test_non_scaled = test.drop(cols, axis = 1)
    test_new = pd.concat([scaled_test, test_non_scaled], axis = 1)

    return train_new, test_new

def normalize_transform_fit(train, test, cols:list):
    scaler = PowerTransformer(method='yeo-johnson')
    for col in cols:
      t = scaler.fit_transform(np.array(train[col]).reshape(-1,1))
      train[col] = t.reshape(-1)
      t_ = scaler.transform(np.array(test[col]).reshape(-1,1))
      test[col] = t_.reshape(-1)

    return train, test

# no menu
def preprocessing1(train, test):
    train[['현본사소속재택근무자수', '중식계', '석식계']] = train[['현본사소속재택근무자수', '중식계', '석식계']].astype('int')
    test['현본사소속재택근무자수'] = test['현본사소속재택근무자수'].astype('int')

    train['일자'] = pd.to_datetime(train['일자'])
    test['일자'] = pd.to_datetime(test['일자'])

    train['요일'] = train['일자'].dt.weekday
    train['년'] = train['일자'].dt.year
    train['월'] = train['일자'].dt.month
    train['일'] = train['일자'].dt.day
    train['주'] = train['일자'].dt.week
    train['출근'] = train['본사정원수']-(train['본사휴가자수']+train['본사출장자수']+train['현본사소속재택근무자수'])
    train['휴가비율'] = train['본사휴가자수']/train['본사정원수']
    train['출장비율'] = train['본사출장자수']/train['본사정원수']
    train['야근비율'] = train['본사시간외근무명령서승인건수']/train['출근']
    train['재택비율'] = train['현본사소속재택근무자수']/train['본사정원수']

    test['요일'] = test['일자'].dt.weekday
    test['년'] = test['일자'].dt.year
    test['월'] = test['일자'].dt.month
    test['일'] = test['일자'].dt.day
    test['주'] = test['일자'].dt.week
    test['출근'] = test['본사정원수']-(test['본사휴가자수']+test['본사출장자수']+test['현본사소속재택근무자수'])
    test['휴가비율'] = test['본사휴가자수']/test['본사정원수']
    test['출장비율'] = test['본사출장자수']/test['본사정원수']
    test['야근비율'] = test['본사시간외근무명령서승인건수']/test['출근']
    test['재택비율'] = test['현본사소속재택근무자수']/test['본사정원수']

    train = make_holidays_prev(train)
    test = make_holidays_prev(test)

    train_lunch = train[['요일','휴가비율','출장비율','재택비율','출근', '본사출장자수','현본사소속재택근무자수', '본사휴가자수','일', '주', '월', '년', '휴일전날','본사시간외근무명령서승인건수']]
    test_lunch = test[['요일','휴가비율','출장비율','재택비율','출근', '본사출장자수','현본사소속재택근무자수', '본사휴가자수','일', '주', '월', '년','휴일전날','본사시간외근무명령서승인건수']]
    y_lunch = train['중식계']

    train_dinner = train[['요일','휴가비율','출장비율','재택비율','출근', '본사휴가자수','현본사소속재택근무자수', '본사시간외근무명령서승인건수','본사출장자수', '일', '주', '월', '년','휴일전날']]
    test_dinner = test[['요일','휴가비율','출장비율','재택비율', '출근', '본사휴가자수','현본사소속재택근무자수', '본사시간외근무명령서승인건수','본사출장자수','일', '주', '월', '년','휴일전날']]
    y_dinner = train['석식계']

    return train_lunch, test_lunch, y_lunch, train_dinner, test_dinner, y_dinner

def preprocessing2(train, test):
    # 식사 가능 인원
    train["식사가능자수"] = (train["본사정원수"] - train["본사휴가자수"] - train["현본사소속재택근무자수"]).astype(int)
    test["식사가능자수"] = (test["본사정원수"] - test["본사휴가자수"] - test["현본사소속재택근무자수"]).astype(int)

    # 날짜 
    train["일자"] = pd.to_datetime(train["일자"])
    train["년"] = train["일자"].dt.year
    train["월"] = train["일자"].dt.month
    train["주"] = train["일자"].dt.week
    train["일"] = train["일자"].dt.day
    train["요일"] = train["일자"].dt.weekday

    test["일자"] = pd.to_datetime(test["일자"])
    test["년"] = test["일자"].dt.year
    test["월"] = test["일자"].dt.month
    test["주"] = test["일자"].dt.week
    test["일"] = test["일자"].dt.day
    test["요일"] = test["일자"].dt.weekday

    # 휴일 전날 column 생성
    train = make_holidays_prev(train)
    test = make_holidays_prev(test)

    use_col = ["월", "일", "요일", "휴일전날", "식사가능자수", "본사출장자수", "본사시간외근무명령서승인건수"]

    train_lunch = train[use_col]
    test_lunch = test[use_col]
    y_lunch = train['중식계']

    train_dinner = train[use_col]
    test_dinner = test[use_col]
    y_dinner = train['석식계']

    return train_lunch, test_lunch, y_lunch, train_dinner, test_dinner, y_dinner

def preprocessing3(train, test):
    train[['현본사소속재택근무자수', '중식계', '석식계']] = train[['현본사소속재택근무자수', '중식계', '석식계']].astype('int')
    test['현본사소속재택근무자수'] = test['현본사소속재택근무자수'].astype('int')

    train['일자'] = pd.to_datetime(train['일자'])
    test['일자'] = pd.to_datetime(test['일자'])

    train['요일'] = train['일자'].dt.weekday
    train['년'] = train['일자'].dt.year
    train['월'] = train['일자'].dt.month
    train['일'] = train['일자'].dt.day
    train['주'] = train['일자'].dt.week
    train['출근'] = train['본사정원수']-(train['본사휴가자수']+train['본사출장자수']+train['현본사소속재택근무자수'])
    train['휴가비율'] = train['본사휴가자수']/train['본사정원수']
    train['출장비율'] = train['본사출장자수']/train['본사정원수']
    train['야근비율'] = train['본사시간외근무명령서승인건수']/train['출근']
    train['재택비율'] = train['현본사소속재택근무자수']/train['본사정원수']

    test['요일'] = test['일자'].dt.weekday
    test['년'] = test['일자'].dt.year
    test['월'] = test['일자'].dt.month
    test['일'] = test['일자'].dt.day
    test['주'] = test['일자'].dt.week
    test['출근'] = test['본사정원수']-(test['본사휴가자수']+test['본사출장자수']+test['현본사소속재택근무자수'])
    test['휴가비율'] = test['본사휴가자수']/test['본사정원수']
    test['출장비율'] = test['본사출장자수']/test['본사정원수']
    test['야근비율'] = test['본사시간외근무명령서승인건수']/test['출근']
    test['재택비율'] = test['현본사소속재택근무자수']/test['본사정원수']

    train = make_holidays_prev(train)
    test = make_holidays_prev(test)

    # scaling 적용 columns
    scaling_columns = ['본사정원수','본사휴가자수','본사출장자수','본사시간외근무명령서승인건수','현본사소속재택근무자수','출근']
    # train set oulier 제거 
    train,test = detect_fix_outlier(train, test, scaling_columns)
    # train, test min_max_scaling
    train,test = min_max_scaling(train, test, scaling_columns)

    use_col = ["월", "일", "요일", "휴일전날", "출근", "본사출장자수", "본사시간외근무명령서승인건수"]

    train_lunch = train[use_col]
    test_lunch = test[use_col]
    y_lunch = train['중식계']

    train_dinner = train[use_col]
    test_dinner = test[use_col]
    y_dinner = train['석식계']

    return train_lunch, test_lunch, y_lunch, train_dinner, test_dinner, y_dinner

def preprocessing4(train, test):
    train[['현본사소속재택근무자수', '중식계', '석식계']] = train[['현본사소속재택근무자수', '중식계', '석식계']].astype('int')
    test['현본사소속재택근무자수'] = test['현본사소속재택근무자수'].astype('int')

    train['일자'] = pd.to_datetime(train['일자'])
    test['일자'] = pd.to_datetime(test['일자'])

    train['요일'] = train['일자'].dt.weekday
    train['년'] = train['일자'].dt.year
    train['월'] = train['일자'].dt.month
    train['일'] = train['일자'].dt.day
    train['주'] = train['일자'].dt.week
    train['출근'] = train['본사정원수']-(train['본사휴가자수']+train['본사출장자수']+train['현본사소속재택근무자수'])

    test['요일'] = test['일자'].dt.weekday
    test['년'] = test['일자'].dt.year
    test['월'] = test['일자'].dt.month
    test['일'] = test['일자'].dt.day
    test['주'] = test['일자'].dt.week
    test['출근'] = test['본사정원수']-(test['본사휴가자수']+test['본사출장자수']+test['현본사소속재택근무자수'])

    train = make_holidays_prev(train)
    test = make_holidays_prev(test)

    use_col_ = ["월", "일", "요일", "휴일전날", '출근', "본사출장자수", "본사시간외근무명령서승인건수",'석식계']
    use_col = ["월", "일", "요일", "휴일전날", '출근', "본사출장자수", "본사시간외근무명령서승인건수"]

    # 석식계 0 수정하기 
    t_ = train[use_col_]
    t_dn_zero = t_[t_['석식계']==0]
    t_dn_zero.drop('석식계', axis = 1, inplace = True)

    t_dn_nonzero = t_[t_['석식계']!=0]
    t_y = t_dn_nonzero['석식계']
    t_dn_nonzero.drop('석식계', axis = 1, inplace = True)

    xgb = XGBRegressor(objective='reg:squarederror')
    xgb.fit(t_dn_nonzero, t_y)
    pred_t = xgb.predict(t_dn_zero)

    t_dn_zero['석식계'] = pred_t
    t_dn_nonzero['석식계'] = t_y
    
    t_concat = pd.concat([t_dn_nonzero, t_dn_zero]).sort_index()
    ###

    train_lunch = train[use_col]
    test_lunch = test[use_col]
    y_lunch = train['중식계']

    train_dinner = t_concat[use_col]
    test_dinner = test[use_col]
    y_dinner = t_concat['석식계']

    return train_lunch, test_lunch, y_lunch, train_dinner, test_dinner, y_dinner

def preprocessing5(train, test):
    train[['현본사소속재택근무자수', '중식계', '석식계']] = train[['현본사소속재택근무자수', '중식계', '석식계']].astype('int')
    test['현본사소속재택근무자수'] = test['현본사소속재택근무자수'].astype('int')

    train['일자'] = pd.to_datetime(train['일자'])
    test['일자'] = pd.to_datetime(test['일자'])

    train['요일'] = train['일자'].dt.weekday
    train['년'] = train['일자'].dt.year
    train['월'] = train['일자'].dt.month
    train['일'] = train['일자'].dt.day
    train['주'] = train['일자'].dt.week
    train['출근'] = train['본사정원수']-(train['본사휴가자수']+train['본사출장자수']+train['현본사소속재택근무자수'])

    test['요일'] = test['일자'].dt.weekday
    test['년'] = test['일자'].dt.year
    test['월'] = test['일자'].dt.month
    test['일'] = test['일자'].dt.day
    test['주'] = test['일자'].dt.week
    test['출근'] = test['본사정원수']-(test['본사휴가자수']+test['본사출장자수']+test['현본사소속재택근무자수'])
   
    train = make_holidays_prev(train)
    test = make_holidays_prev(test)

    # normalize 적용 columns
    normalize_columns = ['본사휴가자수','본사출장자수','본사시간외근무명령서승인건수','현본사소속재택근무자수','출근']
    # train set oulier 제거 
    train,test = detect_fix_outlier(train, test, normalize_columns)
    # normalize 적용
    train,test = normalize_transform_fit(train, test, normalize_columns)

    use_col = ["월", "일", "요일", "휴일전날", "출근", "본사출장자수", "본사시간외근무명령서승인건수"]

    train_lunch = train[use_col]
    test_lunch = test[use_col]
    y_lunch = train['중식계']

    train_dinner = train[use_col]
    test_dinner = test[use_col]
    y_dinner = train['석식계']

    return train_lunch, test_lunch, y_lunch, train_dinner, test_dinner, y_dinner






