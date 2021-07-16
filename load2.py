# menu 활용 load type

import os
from numpy.core.numeric import True_ 
import pandas as pd 
import numpy as np
import argparse
import datetime as dt
from workalendar.asia import SouthKorea
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from sklearn.preprocessing import PowerTransformer
from load import make_holidays_prev
from gensim.models import FastText

def dataloader2(datadir, preprocess = 11):
    train = pd.read_csv(os.path.join(datadir, f'train.csv'))
    test = pd.read_csv(os.path.join(datadir, f'test.csv'))    
    menu = pd.read_csv(os.path.join(datadir, f'menu.csv'), encoding='cp949')

    if preprocess == 11:
        train_lunch, test_lunch, y_lunch, train_dinner, test_dinner, y_dinner = preprocessing11(train, test, menu)

    if preprocess == 12:
        train_lunch, test_lunch, y_lunch, train_dinner, test_dinner, y_dinner = preprocessing12(train, test, datadir)

    if preprocess == 13:
        train_lunch, test_lunch, y_lunch, train_dinner, test_dinner, y_dinner = preprocessing13(train, test)

    return train_lunch, test_lunch, y_lunch, train_dinner, test_dinner, y_dinner

def menu_split(x):
    # 중식과 석식이 없던 날의 메뉴명(자기계발의날, * 등)을 길이로 판별하여 제외
    # if len(x) < 21:
    #     return ""
    
    menu_list = []
    x = x.split(" ")
    
    for i in x:
        if "new" in i.lower():
            menu_list.append(i.split(")")[1])
            continue
        if "(" in i and ")" in i and (":" in i or "," in i):
            continue
        if "(" not in i and ")" not in i and "/" in i:
            menu_list.extend(i.split("/"))
        else:
            if "," in i and "(" not in i and ")" not in i:
                menu_list.extend(i.split("/"))
            elif len(i) > 0 and (i[0] == "(" or i[-1] == ")"):
                continue
            else:
                menu_list.append(i)
            
    menu_list = list(set(menu_list))
    menu_list.remove("")
    
    return menu_list

def menu_embedding(x, model, total_menu_list, vec_size=100):
    # 중식과 석식이 없던 날의 메뉴명(자기계발의날, * 등)을 길이로 판별하여 제외
    if len(x) < 21:
        vec_exp = np.zeros(vec_size)
    
    menu_list = []
    x = x.split(" ")
    
    for i in x:
        if "new" in i.lower():
            menu_list.append(i.split(")")[1])
            continue
        if "(" in i and ")" in i and (":" in i or "," in i):
            continue
        if "(" not in i and ")" not in i and "/" in i:
            menu_list.extend(i.split("/"))
        else:
            if "," in i and "(" not in i and ")" not in i:
                menu_list.extend(i.split("/"))
            elif len(i) > 0 and (i[0] == "(" or i[-1] == ")"):
                continue
            else:
                menu_list.append(i)
            
    menu_list = list(set(menu_list))
    menu_list.remove("")
    
    vec_exp = np.zeros(vec_size)
    for i in menu_list:
        if not i in total_menu_list:
            alt = model.wv.most_similar(i)[0][0]
            vec = model.wv.get_vector(alt)
        else:
            vec = model.wv.get_vector(i)
        vec_exp += vec
    vec_exp /= len(menu_list) 
    
    return vec_exp

def get_emb_vec(train, test):
    food_combinations = []
    total_menu_set = set()

    for i in ["조식메뉴", "중식메뉴", "석식메뉴"]:
        food_combinations += train[i].apply(lambda x: menu_split(x)).to_list()

    for menu_list in food_combinations:
        for menu in menu_list:
            total_menu_set.add(menu)
            
    total_menu_list = list(total_menu_set)
    total_menu_list.remove("이연복의") # 이연복의 청경채찜

    TRAIN_W2V = True
    try:
        model = FastText.load("food_embedding.model")
        print("Model loaded")
    except:
        if TRAIN_W2V:
            print("Training w2v")
            model = FastText(sentences = food_combinations, vector_size = 100, window = 7, min_count = 0, workers = 4, sg = 0, epochs = 1000)
            model.save("food_embedding.model")
        else:
            print("Model loading failed. Do not train.")

    train["중식임베딩"] = train["중식메뉴"].apply(lambda x: menu_embedding(x, model, total_menu_list))
    train["석식임베딩"] = train["석식메뉴"].apply(lambda x: menu_embedding(x, model, total_menu_list))
    test["중식임베딩"] = test["중식메뉴"].apply(lambda x: menu_embedding(x, model, total_menu_list))
    test["석식임베딩"] = test["석식메뉴"].apply(lambda x: menu_embedding(x, model, total_menu_list))

    train_emb_l = np.array(train["중식임베딩"].to_numpy().tolist())
    train_emb_d = np.array(train["석식임베딩"].to_numpy().tolist())
    test_emb_l = np.array(test["중식임베딩"].to_numpy().tolist())
    test_emb_d = np.array(test["석식임베딩"].to_numpy().tolist())

    return train_emb_l, train_emb_d, test_emb_l, test_emb_d


def get_food_embedding(data):
    data_ = []
    data = data.split(' ')
    for i in data:
        if '(' in i and ':' in i and ')' in i:
            continue
        if '/' in i:
            data_.extend(i.split('/'))
        else:
            data_.append(i)
    data_ = list(set(data_))
    data_.remove('')
    return data_

def get_menu_encoding(df, result_lst:list, menu, food):
    for k in range(len(df)):
        trigger = np.zeros(34, dtype = 'int')
        for i in range(len(df.iloc[k][0])):
            if df.iloc[k][0][i] in food:
                trigger += np.array(menu[menu['메뉴'] == df.iloc[k][0][i]].iloc[0][1:-1], dtype='int')
        trigger = trigger / trigger.sum()
        result_lst.append(trigger)

    return result_lst
    
def gg(x):
  if x >= 80:
    return 3
  elif x >= 75 and x < 80:
    return 2
  elif x >= 68 and x < 75:
    return 1
  else:
    return 0

# 기상청 데이터 전처리    
def weather_preprocess(df):
    df.fillna(0 , inplace = True)
    df['일시'] = pd.to_datetime(df['일시'])
    
    df_lunch = df[df['일시'].dt.hour == 12]
    df_dinner = df[df['일시'].dt.hour == 18]

    columns = ['일시','기온(°C)','강수량(mm)','풍속(m/s)','습도(%)','적설(cm)']
    df_lunch = df_lunch[columns].reset_index(drop=True)
    df_dinner = df_dinner[columns].reset_index(drop=True)

    df_lunch['일시'] = df_lunch['일시'].dt.strftime('%Y-%m-%d')
    df_dinner['일시'] = df_dinner['일시'].dt.strftime('%Y-%m-%d')

    # 불쾌지수 , 폭염 , 체감온도 추가 
    df_lunch['불쾌지수'] = 1.8*df_lunch['기온(°C)']-0.55*(1-df_lunch['습도(%)']/100)*(1.8*df_lunch['기온(°C)']-26)+32
    df_dinner['불쾌지수'] = 1.8*df_dinner['기온(°C)']-0.55*(1-df_dinner['습도(%)']/100)*(1.8*df_dinner['기온(°C)']-26)+32
    df_lunch['불쾌지수'] = df_lunch['불쾌지수'].apply(lambda x: gg(x) )
    df_dinner['불쾌지수'] = df_dinner['불쾌지수'].apply(lambda x: gg(x) )
    df_lunch['체감온도'] = 13.12+0.6215*df_lunch['기온(°C)']-11.37*df_lunch['풍속(m/s)']**0.16+0.3965*df_lunch['풍속(m/s)']**0.16*df_lunch['기온(°C)']
    df_dinner['체감온도'] = 13.12+0.6215*df_dinner['기온(°C)']-11.37*df_dinner['풍속(m/s)']**0.16+0.3965*df_dinner['풍속(m/s)']**0.16*df_dinner['기온(°C)']
    df_lunch['폭염'] = df_lunch['기온(°C)'].apply(lambda x: 1 if x>=30  else 0)
    df_dinner['폭염'] = df_dinner['기온(°C)'].apply(lambda x: 1 if x>=30  else 0)
    df_lunch['비'] = df_lunch['강수량(mm)'].apply(lambda x: 1 if x==0 else 0)
    df_dinner['비'] = df_dinner['강수량(mm)'].apply(lambda x: 1 if x==0 else 0)
    
    return df_lunch, df_dinner


# menu.csv 활용 menu columns 생성
def preprocessing11(train, test, menu):
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

    train['중식메뉴_split'] = train['중식메뉴'].apply(lambda x: get_food_embedding(x))
    train['석식메뉴_split'] = train['석식메뉴'].apply(lambda x: get_food_embedding(x))

    test['중식메뉴_split'] = test['중식메뉴'].apply(lambda x: get_food_embedding(x))
    test['석식메뉴_split'] = test['석식메뉴'].apply(lambda x: get_food_embedding(x))

    lunch_lst = []
    dinner_lst = []
    lunch_lst_test = []
    dinner_lst_test = []
    
    for i in range(1205):
        lunch_lst.append(train['중식메뉴_split'][i])
        dinner_lst.append(train['석식메뉴_split'][i])

    for i in range(50):
        lunch_lst_test.append(test['중식메뉴_split'][i])
        dinner_lst_test.append(test['석식메뉴_split'][i])

    lunch_df = pd.DataFrame({'중식메뉴':lunch_lst})
    dinner_df = pd.DataFrame({'석식메뉴':dinner_lst})
    lunch_df_t = pd.DataFrame({'중식메뉴':lunch_lst_test})
    dinner_df_t = pd.DataFrame({'석식메뉴':dinner_lst_test})

    food = list(menu['메뉴'])

    result_lunch = []
    result_dinner = []
    result_lunch_t = []
    result_dinner_t = []

    result_lunch = get_menu_encoding(lunch_df, result_lunch, menu, food)
    result_dinner = get_menu_encoding(dinner_df, result_dinner, menu, food)
    result_lunch_t = get_menu_encoding(lunch_df_t, result_lunch_t, menu, food)
    result_dinner_t = get_menu_encoding(dinner_df_t, result_dinner_t, menu, food)

    column = list(menu.columns[1:-1])
    lunch_sort = pd.DataFrame(result_lunch ,columns=column)
    dinner_sort = pd.DataFrame(result_dinner, columns = column)
    lunch_sort_t = pd.DataFrame(result_lunch_t, columns=column)
    dinner_sort_t = pd.DataFrame(result_dinner_t, columns = column)

    lunch_sort.fillna(0, inplace = True)
    dinner_sort.fillna(0, inplace = True)
    dinner_sort_t.fillna(0, inplace = True)

    use_col = ["월", "일", "요일", "휴일전날", "출근", "본사출장자수", "본사시간외근무명령서승인건수"]

    train_lunch = train[use_col]
    test_lunch = test[use_col]
    y_lunch = train['중식계']

    train_dinner = train[use_col]
    test_dinner = test[use_col]
    y_dinner = train['석식계']

    # menu columns 추가
    train_lunch = pd.concat([train_lunch, lunch_sort], axis = 1)
    train_dinner = pd.concat([train_dinner, dinner_sort], axis = 1)
    test_lunch = pd.concat([test_lunch, lunch_sort_t], axis = 1)
    test_dinner = pd.concat([test_dinner, dinner_sort_t], axis = 1)

    return train_lunch, test_lunch, y_lunch, train_dinner, test_dinner, y_dinner

# 기상청 데이터 활용
def preprocessing12(train, test, datadir):
    # 기상청 데이터 
    weather2016 = pd.read_csv(os.path.join(datadir, f'경상남도2016.csv'), encoding='cp949')
    weather2017 = pd.read_csv(os.path.join(datadir, f'경상남도2017.csv'), encoding='cp949')
    weather2018 = pd.read_csv(os.path.join(datadir, f'경상남도2018.csv'), encoding='cp949')
    weather2019 = pd.read_csv(os.path.join(datadir, f'경상남도2019.csv'), encoding='cp949')
    weather2020 = pd.read_csv(os.path.join(datadir, f'경상남도2020.csv'), encoding='cp949')
    weather2021 = pd.read_csv(os.path.join(datadir, f'경상남도2021.csv'), encoding='cp949')

    wt_lunch_2016, wt_dinner_2016 = weather_preprocess(weather2016)
    wt_lunch_2017, wt_dinner_2017 = weather_preprocess(weather2017)
    wt_lunch_2018, wt_dinner_2018 = weather_preprocess(weather2018)
    wt_lunch_2019, wt_dinner_2019 = weather_preprocess(weather2019)
    wt_lunch_2020, wt_dinner_2020 = weather_preprocess(weather2020)
    wt_lunch_2021, wt_dinner_2021 = weather_preprocess(weather2021)

    wt_lunch_lst = [wt_lunch_2016, wt_lunch_2017, wt_lunch_2018, wt_lunch_2019,wt_lunch_2020, wt_lunch_2021]
    wt_dinner_lst = [wt_dinner_2016, wt_dinner_2017, wt_dinner_2018, wt_dinner_2019,wt_dinner_2020, wt_dinner_2021]

    wt_lunch = pd.concat(wt_lunch_lst, axis = 0)
    wt_dinner = pd.concat(wt_dinner_lst, axis = 0)

    wt_lunch.rename(columns = {'일시' : '일자'}, inplace = True)
    wt_dinner.rename(columns = {'일시' : '일자'}, inplace = True)

    # train / test
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

    train['일자'] = train['일자'].dt.strftime('%Y-%m-%d')
    test['일자'] = test['일자'].dt.strftime('%Y-%m-%d')

    # 기상청 데이터와 합치기 
    # 기상청 데이터 2018-10-19 12시에 하나의 결측치가 있음
    train_lunch = pd.merge(wt_lunch, train, how='right')
    train_lunch.fillna(0, inplace=True)
    train_dinner = pd.merge(wt_dinner, train)  

    test_lunch = pd.merge(wt_lunch, test)
    test_dinner = pd.merge(wt_dinner, test)

    use_col = ["월", "일", "요일", "휴일전날", "출근", "본사출장자수", "본사시간외근무명령서승인건수", "체감온도", "비"]

    train_lunch_wt = train_lunch[use_col]
    test_lunch_wt = test_lunch[use_col]
    y_lunch = train['중식계']

    train_dinner_wt = train_dinner[use_col]
    test_dinner_wt = test_dinner[use_col]
    y_dinner = train['석식계']

    return train_lunch_wt, test_lunch_wt, y_lunch, train_dinner_wt, test_dinner_wt, y_dinner

# FastText 활용 word embedding
def preprocessing13(train, test):
    # train / test
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

    # train/test menu word_embedding 
    train_emb_l, train_emb_d, test_emb_l, test_emb_d = get_emb_vec(train, test)

    use_col = ["월", "일", "요일", "휴일전날", "출근", "본사출장자수", "본사시간외근무명령서승인건수"]

    x1_train = train[use_col]
    x1_train = np.concatenate((x1_train.to_numpy(), train_emb_l), axis = 1)
    x1_train = pd.DataFrame(x1_train)
    x1_test = test[use_col]
    x1_test = np.concatenate((x1_test.to_numpy(), test_emb_l), axis = 1)
    x1_test = pd.DataFrame(x1_test)
    y1_train = train["중식계"]

    x2_train = train[use_col]
    x2_train = np.concatenate((x2_train.to_numpy(), train_emb_d), axis = 1)
    x2_train = pd.DataFrame(x2_train)
    x2_test = test[use_col]
    x2_test = np.concatenate((x2_test.to_numpy(), test_emb_d), axis = 1)
    x1_test = pd.DataFrame(x2_test)
    y2_train = train["석식계"]

    return x1_train, x1_test, y1_train, x2_train, x2_test, y2_train


