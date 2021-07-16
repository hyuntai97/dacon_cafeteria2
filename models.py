from sklearn import linear_model, tree, ensemble, svm, neighbors
# import lightgbm as lgb
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error

def model_selection(modelname, params, random_state=42):
    if modelname == 'XGB':
        return XGB(params=params, random_state=random_state)
    else:
        return SKlearnModels(modelname=modelname, params=params, random_state=random_state)
        
class XGB:
    def __init__(self, params, random_state = 42):
        self.params = params
        self.random_state = random_state

    def build(self):
        pass

    def fit(self, X, y, validation_set=None, validation_size=None):
        if (validation_size!=None) or (validation_set!=None):
            if validation_size:
                # split train and validation set
                x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=validation_size, random_state=self.random_state)

            elif validation_set:
                x_train, y_train = X, y 
                x_val, y_val = validation_set 
            
            xgb = XGBRegressor()
            grid_search = GridSearchCV(xgb, self.params, n_jobs=1, scoring = 'neg_mean_absolute_error')
            grid_search.fit(x_train, y_train)
            best_model = grid_search.best_estimator_

            self.model = best_model

            val_results = self.eval(X=x_val, y=y_val)

            return val_results

        else:  # no validation
            xgb = XGBRegressor()
            grid_search = GridSearchCV(xgb, self.params, n_jobs=1, scoring = 'neg_mean_absolute_error')
            grid_search.fit(X,y)
            best_model = grid_search.best_estimator_
            
            self.model = best_model

    def eval(self, X, y):
        # prediction
        y_pred = self.predict(X)

        # eval
        results = mean_absolute_error(y, y_pred)

        return results

    def predict(self, X):
        y_pred = self.model.predict(X)
        return y_pred


class SKlearnModels:
    def __init__(self, modelname, params, random_state=42):
        self.modelname = modelname
        self.random_state = random_state
        self.params = params

    def build(self):
        # regression
        if self.modelname == 'OLS':
            # Ordinary Least Square
            self.model = linear_model.LinearRegression(**self.params)
        elif self.modelname == 'Ridge':
            # Ridge
            self.params['random_state'] = self.random_state
            self.model = linear_model.Ridge(**self.params)
        elif self.modelname == 'Lasso':
            # Lasso 
            self.params['random_state'] = self.random_state
            self.model = linear_model.Lasso(**self.params)
        elif self.modelname == 'ElasticNet':
            # Elastic-Net
            self.params['random_state'] = self.random_state
            self.model = linear_model.ElasticNet(**self.params)
        elif self.modelname == 'DT':
            # Decision Tree
            self.params['random_state'] = self.random_state
            self.model = tree.DecisionTreeRegressor(**self.params)
        elif self.modelname == 'RF':
            # Random Forest
            self.params['random_state'] = self.random_state
            self.model = ensemble.RandomForestRegressor(**self.params)
        elif self.modelname == 'ADA':
            # Adaboost
            self.params['random_state'] = self.random_state
            self.model = ensemble.AdaBoostRegressor(**self.params)
        elif self.modelname =='GT':
            # Gradient Tree Boosting
            self.params['random_state'] = self.random_state
            self.model = ensemble.GradientBoostingRegressor(**self.params)
        elif self.modelname == 'SVM':
            # Support Vector Machine
            self.model = svm.SVR(**self.params)
        elif self.modelname == 'KNN':
            # K-Nearest Neighbors
            self.model = neighbors.KNeighborsRegressor(**self.params)

    def fit(self, X, y, validation_set=None, validation_size=None):

        if (validation_size!=None) or (validation_set!=None):
            if validation_size:
                # split train and validation set
                x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=validation_size, random_state=self.random_state)

            elif validation_set:
                x_train, y_train = X, y 
                x_val, y_val = validation_set 

            # model training
            self.model.fit(X=x_train, y=y_train)
            
            # evaluate validation set
            val_results = self.eval(X=x_val, y=y_val)

            return val_results
        else:
            # model training
            self.model.fit(X=X, y=y)


    def eval(self, X, y):
        # prediction
        y_pred = self.predict(X)

        # eval
        results = mean_absolute_error(y, y_pred)

        return results

    def predict(self, X):
        y_pred = self.model.predict(X)
        
        return y_pred


    
        


