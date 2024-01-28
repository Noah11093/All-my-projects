import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
import datetime
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.utils import resample
from sklearn.model_selection import StratifiedShuffleSplit

class SupervisedLearning:
    
    def __init__(self):
        plt.ioff()
        self.x_train_transformed = 0
        self.df = pd.read_csv('./KSI.csv')
        self.template = {}
        self.result = []
        
    def exploration(self):
        
        # Load and describe data elements (columns), provide descriptions & types, ranges and values of elements as appropriate. 
        df = pd.read_csv('./KSI.csv')
        print('Descriptions: ')
        print(df.describe(), '\n')
        print('Types: ')
        print(df.info(), '\n')
        df_stats = df.describe()
        df_stats.loc['Range'] = df_stats.loc['max'] - df_stats.loc['min']
        print('Range:')
        print(df_stats.loc['Range'], '\n')
        
        # Statistical assessments including means, averages, correlations
        print('Means:')
        print(df_stats.loc['mean'], '\n')
        print('Median:')
        print(df_stats.loc['50%'], '\n')
        print('Correlation:')
        print(df.corr(), '\n')

        # Missing data evaluations – use pandas, numpy and any other python packages
        print(df.shape)

        print('Missing Values')
        print(df.isnull().sum(), '\n')

        print('Unknown Values of INVAGE column')
        print(df['INVAGE'].isin(["unknown"]).sum())

        print(df.shape)
        
    def preprocessing(self):
        df = pd.read_csv('./KSI.csv')
        df_clean = df.drop_duplicates('ACCNUM')
        df_clean.replace(' ', np.nan, regex=False, inplace=True)
        
        for col in df_clean:
            df_clean = df_clean[df_clean[col] != 'unknown']

        # Convert integer column to time column
        df_clean['TIME'] = df_clean['TIME'].apply(lambda x: datetime.time(x // 100, x % 100))

        # Define time intervals with a given frequency
        freq = datetime.timedelta(hours=3)
        start_time = datetime.time(0, 0)
        end_time = datetime.time(23, 59)
        today = datetime.date.today()
        period = (datetime.datetime.combine(today, end_time) - datetime.datetime.combine(today, start_time)) // freq + 1
        intervals = [(datetime.datetime.combine(today, start_time) + i*freq).time() for i in range(period)]

        # Classify time into a specific time interval
        def classify_time(time):
            for i, interval in enumerate(intervals):
                if time < interval:
                    return f"{intervals[i-1].strftime('%H%M')}-{interval.strftime('%H%M')}"
            return f"{intervals[-2].strftime('%H%M')}-{intervals[-1].strftime('%H%M')}"

        df_clean['INTERVAL'] = df_clean['TIME'].apply(classify_time)
        df_clean.drop(['TIME','DATE'], axis=1, inplace=True)

        df_clean['ACCLASS'] = np.where(df_clean['ACCLASS'] == 'Property Damage Only', 'Non-Fatal', df_clean['ACCLASS'])
        df_clean['ACCLASS'] = np.where(df_clean['ACCLASS'] == 'Non-Fatal Injury', 'Non-Fatal', df_clean['ACCLASS'])
        df_clean = df_clean[df_clean['ACCLASS'].notna()]
        
        for col in df_clean.select_dtypes(include=object):
            if len(df_clean[col].unique()) > 2 and len(df_clean[col].unique()) < 20:
                plt.figure()
                df_clean[col].value_counts().plot(kind='bar', color=list('rgbkmc'))
                plt.xlabel(col)
                plt.ylabel('Count')
                # plt.show()
            
            if len(df_clean[col].unique()) == 2:
                df_clean[col].fillna('No', inplace=True)
                combo_counts = df_clean.groupby([col]).size().reset_index(name='Count')
                combo_counts.plot(kind='bar', x=col, y='Count', stacked=True, title=col)
                # plt.show()
        
        return df_clean
    
    def modelling(self):
        # select the columns that is related to 'ACCLASS' and do not have too much unique value
        df_clean = self.preprocessing()
        df_final = df_clean[[
            'INTERVAL', 'LATITUDE', 'LONGITUDE', 'DISTRICT', 
            'VISIBILITY', 'LIGHT', 'RDSFCOND', 'PEDESTRIAN', 'CYCLIST', 
            'AUTOMOBILE', 'MOTORCYCLE', 'TRUCK', 'TRSN_CITY_VEH', 'EMERG_VEH',
            'PASSENGER', 'SPEEDING', 'AG_DRIV', 'REDLIGHT', 'ALCOHOL', 
            'DISABILITY', 'ACCLASS'
        ]]
        df_final['ACCLASS'] = df_final['ACCLASS'].replace({'Non-Fatal':0, 'Fatal':1})
        
        X = df_final.drop('ACCLASS', axis=1)
        y = df_final['ACCLASS']
        
        # Data transformations
        imputer = SimpleImputer(strategy="constant",fill_value='missing')
        data_tr = imputer.fit_transform(X)
        data_tr = pd.DataFrame(data_tr, columns=X.columns)
        print(data_tr.isna().sum())

        num_data = X.select_dtypes(include=[np.number]).columns
        print(num_data)
        data_num = data_tr[num_data]
        #standardize
        scaler = MinMaxScaler() #define the instance

        scaled = scaler.fit_transform(data_num)
        data_num_scaled = pd.DataFrame(scaled, columns=num_data)

        cat_data = X.select_dtypes(exclude=[np.number]).columns
        categoricalData = data_tr[cat_data]
        data_cat = pd.get_dummies(categoricalData, columns=cat_data, drop_first=True)
        X_train_prepared = pd.concat([data_num_scaled, data_cat], axis=1)
        
        # Feature Selection
        bestfeatures = SelectKBest(score_func=chi2, k=10)

        fit = bestfeatures.fit(X_train_prepared, y)
        # dfscores = pd.DataFrame(fit.scores_)
        # dfcolumns = pd.DataFrame(X_train_prepared.columns)
        # #concat two dataframes for better visualization 
        # featureScores = pd.concat([dfcolumns,dfscores],axis=1)
        # featureScores.columns = ['Specs','Score']  #naming the dataframe columns
        # bestfea = featureScores.nlargest(10,'Score')  # best 10 features
        # print(bestfea)
        
        # best_features = bestfea['Specs'].values.flatten().tolist()
        # best_X = data_cat[data_cat.columns.intersection(best_features)]
        
        # Managing imbalanced classes
        df_majority = df_final[df_final.ACCLASS==0]
        df_minority = df_final[df_final.ACCLASS==1]

        df_minority_upsampled = resample(df_minority, replace=True, n_samples=12808, random_state=44)

        df_upsampled = pd.concat([df_majority, df_minority_upsampled])
        
        df_final = df_upsampled
        
        X = df_final.drop('ACCLASS', axis=1)
        y = df_final['ACCLASS']
        
        # Use pipelines class to streamline
        cat_pipeline = Pipeline(
            [
                ('imputer', SimpleImputer(strategy="constant",fill_value='missing')),
                ('one_hot', OneHotEncoder(drop='first'))
            ]
        )

        num_pipeline = Pipeline(
            [
                ('imputer', SimpleImputer(strategy="median")),
                ('scaler', MinMaxScaler())
            ]
        )

        num_attribs = X.select_dtypes(exclude=object).columns
        cat_attribs = X.select_dtypes(include=object).columns
        # all_attribs = X.columns

        transformer = ColumnTransformer(
            [
                ("num", num_pipeline, num_attribs),
                ("cat", cat_pipeline, cat_attribs)
            ]
        )
        
        return df_final, transformer
    
    def data_split(self, X, y):
        # Train, Test data splitting

        split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=44)

        # Split the data into training and testing sets
        for train_index, test_index in split.split(X, y):
            X_train = X.iloc[train_index]
            X_test = X.iloc[test_index]
            y_train = y.iloc[train_index]
            y_test = y.iloc[test_index]
        
        return X_train, X_test, y_train, y_test
    
    def evaluation(self):
        # Predictive model building
        # Use logistic regression, decision trees, SVM, Random forest and neural networks algorithms as a minimum– use scikit learn
        df_final, transformer = self.modelling()
        
        X = df_final.drop('ACCLASS', axis=1)
        y = df_final['ACCLASS']
        
        # Train, Test data splitting
        X_train, X_test, y_train, y_test = self.data_split(X, y)
        
        X_train_prepared = transformer.fit_transform(X_train, y_train)
        X_test_prepared = transformer.fit_transform(X_test)

        X_train_df = pd.DataFrame(X_train_prepared.toarray())
        X_test_df = pd.DataFrame(X_test_prepared.toarray())

        # num_attribs_prepared = list(transformer.named_transformers_['num'].get_feature_names_out(num_attribs))
        # cat_attribs_prepared = list(transformer.named_transformers_['cat'].get_feature_names_out(cat_attribs))
        # all_attribs = num_attribs_prepared + cat_attribs_prepared

        selector_train = SelectKBest(score_func=chi2,k=10).fit(X_train_prepared, y_train)
        selector_test = SelectKBest(score_func=chi2,k=10).fit(X_test_prepared, y_test)

        cols_idxs = selector_train.get_support(indices=True)
        X_train_final = X_train_df.iloc[:,cols_idxs]

        cols_idxs = selector_test.get_support(indices=True)
        X_test_final = X_test_df.iloc[:,cols_idxs]

        lr = LogisticRegression(random_state=42)
        dt = DecisionTreeClassifier(criterion="entropy", max_depth=5)
        # svc = SVC(probability=True)
        rf = RandomForestClassifier(n_estimators=10, random_state=42)
        nn = MLPClassifier(hidden_layer_sizes=(5, 2), solver='lbfgs', random_state=42)

        # models = [lr, dt, svc, rf, nn]
        models = [lr, dt, rf, nn]
        piplines = []
        for mod in models:
            full_pipeline = Pipeline(
                [
                    # ('transformer', transformer),
                    ('clf', mod)
                ]
            )
            piplines.append(full_pipeline)
        
        # Fine tune the models using Grid search and randomized grid search. 
        # Modify the penalty values to include only 'l2' or 'none'
        param_grid_lr = {
            'clf__penalty': ['l2', 'none'],
            'clf__C': [0.1, 1, 10, 100],
            'clf__solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
        }


        param_grid_dt = {
            'clf__min_samples_split': range(10, 300, 30),
            'clf__max_depth': range(1, 30, 3),
            'clf__min_samples_leaf': range(1, 15, 3),
            'clf__criterion': ['gini', 'entropy']
        }

        # param_grid_svm = {
        #     'clf__kernel': ['linear', 'rbf', 'poly'],
        #     'clf__C': [0.01, 0.1, 1, 10, 100],
        #     'clf__gamma': [0.01, 0.03, 0.1, 0.3, 1.0, 3.0],
        #     'clf__degree': [2, 3]
        # }

        param_grid_rf = {
            'clf__n_estimators': [100, 200, 300],
            'clf__max_depth': [3, 5, 7, None],
            'clf__min_samples_split': [2, 5, 10],
            'clf__min_samples_leaf': [1, 2, 4],
            'clf__criterion': ['gini', 'entropy']
        }

        param_grid_nn = {
            'clf__hidden_layer_sizes': [(10,), (20,), (10, 5), (20, 10)],
            'clf__activation': ['relu', 'tanh', 'sigmoid'],
            'clf__learning_rate': ['constant', 'invscaling', 'adaptive'],
            'clf__alpha': [0.0001, 0.001, 0.01, 0.1]
        }
        
        # param_list = [param_grid_lr, param_grid_dt, param_grid_svm, param_grid_rf, param_grid_nn]
        param_list = [param_grid_lr, param_grid_dt, param_grid_rf, param_grid_nn]
        best_model = []
        for i in range(len(param_list)):
            rand = RandomizedSearchCV(
                estimator=piplines[i], 
                param_distributions=param_list[i], 
                scoring='accuracy', cv=3,
                n_iter=3, refit=True, 
                verbose=3)
            # @@
            search = rand.fit(X_train_final, y_train)
            best_model.append(search.best_estimator_)
            # print("Best Params:", search.best_params_)
            # print("Best Score:",search.best_score_)
            # print("Best Estimator:",best_model)
        
        modeltitle = ['LogisticRegression','DecisionTree','RandomForest','NeuralNetwork']
        for i in range(len(best_model)):
            best_model[i].fit(X_train_final, y_train)
            y_test_pred = best_model[i].predict(X_test_final)

            # create ROC curve
            y_pred_proba = best_model[i].predict_proba(X_test_final)[::,1]
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            plt.figure(figsize=(12,9))
            plt.plot(fpr,tpr)
            
            plt.title(f'ROC - {models[i]}')
            plt.ylabel('True Positive Rate')
            plt.xlabel('False Positive Rate')

            plt.savefig(f'./static/roc_{i}.png')

            self.template = {}
            self.template["a_name"] = modeltitle[i]
            self.template["accuracy"] = accuracy_score(y_test, y_test_pred)
            self.template["precision"] = precision_score(y_test, y_test_pred)
            self.template["recall"] = recall_score(y_test, y_test_pred)
            self.template["f1_score"] = f1_score(y_test, y_test_pred)
            # self.template[f"accuracy_{str(i)}"] = accuracy_score(y_test, y_test_pred)
            # self.template[f"precision_{str(i)}"] = precision_score(y_test, y_test_pred)
            # self.template[f"recall_{str(i)}"] = recall_score(y_test, y_test_pred)
            # self.template[f"f1_score_{str(i)}"] = f1_score(y_test, y_test_pred)
            
            
            con_mat = '[ '
            for i in confusion_matrix(y_test, y_test_pred):
                for j in i:
                    con_mat = con_mat + str(j) + ' '
            con_mat = con_mat + ']'
            self.template["confusion_matrix"] = con_mat
            self.result.append(self.template)
            
    def final_data(self):
        df_final, transformer = self.modelling()
    
        X = df_final.drop('ACCLASS', axis=1)
        y = df_final['ACCLASS']
        
        # Train, Test data splitting
        X_train, X_test, y_train, y_test = self.data_split(X, y)
        
        return X_train, X_test, y_train, y_test

    def get_result(self, a):
        print()
        print()
        print(self.result[a])
        print()
        print()
        return self.result[a]
    
# sl = SupervisedLearning()
# sl.evaluation()
# print(sl.result)