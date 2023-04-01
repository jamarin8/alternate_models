import yaml, pickle
import subprocess
from IPython.display import clear_output
from collections import defaultdict, Counter

pipInstall = "pip install smart_open"
process = subprocess.Popen(pipInstall.split(), stdout=subprocess.PIPE)
output, error = process.communicate()
from smart_open import smart_open

pipInstall = "pip install pandas"
process = subprocess.Popen(pipInstall.split(), stdout=subprocess.PIPE)
output, error = process.communicate()
import pandas as pd
import numpy as np

pd.options.mode.chained_assignment = None

pipInstall = "pip install pyod"
process = subprocess.Popen(pipInstall.split(), stdout=subprocess.PIPE)
output, error = process.communicate()

from pyod.models.lscp import LSCP
from pyod.models.lof import LOF
from pyod.models.knn import KNN
from pyod.models.pca import PCA
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest

# from pyod.models.suod import SUOD

import matplotlib.pyplot as plt
from datetime import datetime
from categorical_features import categorical_vars, all_date_vars, numeric_vars
from sklearn.preprocessing import StandardScaler
from IPython.display import clear_output
from sklearn.ensemble import IsolationForest
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from itertools import product, combinations

import string
from urllib.parse import unquote


def parse_url_seps(string):
    try:
        value = unquote(string)
    except TypeError:
        value = np.nan
    return value


def main():
    pass


if __name__ == '__main__':

    with open("input_params.yaml", 'r') as stream:
        params_loaded = yaml.safe_load(stream)

    dependant_variable = params_loaded['dependant_variable']
    df_dvs = pd.read_csv(smart_open(dependant_variable), low_memory=False)
    df_dvs['ca_id'] = df_dvs['ca_id'].astype('int')
    df_dvs['customer_id'] = df_dvs['customer_id'].astype('int')
    assert 'any_fraud' in df_dvs.columns
    print('dependant variable loaded...')

    key_vars_giact = params_loaded['key_vars_giact']
    df_03_21 = pd.read_csv(smart_open(key_vars_giact), low_memory=False)
    df_03_21['ca_id'] = df_03_21['ca_id'].astype('int')
    df_03_21['customer_id'] = df_03_21['customer_id'].astype('int')

    df1 = df_dvs.merge(df_03_21, how='inner', on=['ca_id', 'customer_id'])
    print('key variables loaded and merged...')

    df1 = df1.drop('ca_created_at_x', axis=1)
    df1['ca_created_at'] = df1['ca_created_at_y']
    df1 = df1.drop('ca_created_at_y', axis=1)

    fraud_scores = params_loaded['fraud_scores']
    fm5 = pd.read_csv(smart_open(fraud_scores), low_memory=False)
    fm5['ca_id'] = fm5['ca_id'].astype('int')
    fm5['customer_id'] = fm5['customer_id'].astype('int')
    df1_fm5 = df1.merge(fm5, how='inner', on=['ca_id', 'customer_id'])
    print('FM5.0 fraud scores loaded and merged...', df1_fm5.shape)

    #     PRODUCT FILTER

    df2 = df1_fm5.copy()
    df2 = df2.query('product_type == "installment"')

    #     INDEX CREATION

    df2['ca_created_at'] = pd.to_datetime(df2['ca_created_at'])
    df2 = df2.set_index('ca_id').sort_values(by=['ca_created_at'])
    print('index set to ca_id...')

    #    RETENTION OF SPECIFIC ROWS

    df2 = df2[~df2.fm_5_score.isna()]
    fm_5 = df2.fm_5_score

    ca_created_at = df2.ca_created_at
    print('dataset timeframe: ', (ca_created_at.max() - ca_created_at.min()).days, 'DAYS')

    features = ['ca_created_at', 'requested_loan_amount', 'claimed_mni', 'time_spent_on_rates_terms', 'fsl_source',
                'tmx_raw_true_ip_organization_type', 'tcr_phone_type_description', 'gr_verification_response',
                'gr_account_response_code', 'gr_customer_response_code', 'gr_account_added_date']
    features = [f.strip(" ").strip(string.punctuation) for f in features]


    #    DESIGNATION OF FRAUD TYPE

    def fraud_selection(df, type_of_fraud):

        if type_of_fraud == 'general':
            df['fraud_specified'] = ((df['dep_var'] == 1) | (df['first_party_fraud'] == 1)).astype(int).to_list()
        elif type_of_fraud == 'first':
            df['fraud_specified'] = (df['first_party_fraud'] == 1).astype(int).to_list()
        elif type_of_fraud == 'third':
            df['fraud_specified'] = (df['dep_var'] == 1).astype(int).to_list()
        elif type_of_fraud == 'any':
            df['fraud_specified'] = (df['any_fraud'] == 1).astype(int).to_list()
        else:
            print('please choose one of our three choices above')
        return df2[['fraud_specified']]


    fraud_column = fraud_selection(df2, params_loaded['type_of_fraud'])
    print('fraud designation: ', params_loaded['type_of_fraud'])
    avg_fraud = np.round(np.mean(fraud_column.values), 5)
    print('fraud rate: ', avg_fraud)

    df2 = df2[features]
    #     print (df2.columns)

    df2['fsl_source'] = df2['fsl_source'].str.replace(r'google.*', 'google', regex=True)
    df2['fsl_source'] = np.where(np.in1d(df2['fsl_source'], ['google', 'organic']), df2['fsl_source'], 'OTHER')

    df2.loc[:, 'time_spent_on_rates_terms'] = df2.loc[:, 'time_spent_on_rates_terms'].apply(pd.to_timedelta).fillna(
        pd.Timedelta(seconds=0))
    df2.loc[:, 'time_spent_on_rates_terms'] = df2.time_spent_on_rates_terms.dt.total_seconds()

    account_age = (df2['ca_created_at'] - pd.to_datetime(df2['gr_account_added_date'])).dt.days
    df2.loc[:, 'gr_account_added_date'] = account_age.fillna(0).astype(int)

    ca_created_at_ = pd.DataFrame(ca_created_at.index, columns=['ca_id'], index=ca_created_at.values)

    '''
        test_features = ['A','B']
    for n in range(1, len(test_features)+1):   # start at 1 to avoid empty set; end at +1 for full set
        for q in combinations(range(len(test_features)), n):
    #         pass
            print (pd.concat([df2[base_features], df2[np.take(test_features, list(q))]], axis=1))
    '''

    categorical_vars_ = set(features).intersection(categorical_vars)
    print("Categorical Features to be Transformed:\n", list(categorical_vars_))
    #     all_date_vars_ = set(features).intersection(all_date_vars)
    numeric_vars_ = set(features).intersection(numeric_vars)
    print("Numeric Features to be Transformed:\n", list(numeric_vars_))
    all_vars = categorical_vars_ | numeric_vars_
    #     print ("Total Features to be Transformed:\n", list(all_vars))
    to_remove = list(set(features).difference(all_vars))
    print("Features to be Removed pre-Standardization:\n", to_remove)
    df2.drop(to_remove, inplace=True, axis=1)


    def numeric_handling(df):

        if len(numeric_vars_) > 0:
            for numeric_col in numeric_vars_:
                null_values = df.loc[:, numeric_col].isnull()
                df[numeric_col].fillna(-1, inplace=True)
                df.loc[~null_values, [numeric_col]] = StandardScaler().fit_transform(
                    df.loc[~null_values, [numeric_col]])

        return df


    def categorical_handling(df):

        if len(categorical_vars_) > 0:
            df = pd.get_dummies(df, columns=categorical_vars_, drop_first=True)

        dummy_cols = df.select_dtypes(['uint8', 'bool']).columns

        def sum_product_division(a, b):
            if b != 0:
                return a / b
            else:
                return 0

        '''
        application of FAMD algorithm applied for both categorical and numerical variables below
        '''

        df.loc[:, dummy_cols] = (df.loc[:, dummy_cols]
                                 .applymap(
            lambda x: sum_product_division(x, (np.sum(x) / df.shape[0]) ** .5) - (np.sum(x) / df.shape[0])))

        '''
        although numeric handling is handled above in the eponymous function, there may have been features rendered numeric 
        by our transformations such as datetime transformations to counts, thus the extra step below
        '''
        #         adjusted_numerics = list(set(df.select_dtypes(['float', 'int']).columns).difference(dummy_cols))

        #         print ('adjusted_numerics', adjusted_numerics)

        #         scalar = StandardScaler()
        #         for adj in adjusted_numerics:
        #             df.loc[:, adj] = scalar.fit_transform(df.loc[:, adj].values.reshape(-1,1))

        print("\nZ-Scored and Feature-Engineered DataFrame:", df.shape)

        return df


    dayz = str(params_loaded['Window'])
    testdate = pd.date_range(start=ca_created_at.min(), end=ca_created_at.max(), freq=dayz + 'D', closed='right')[0]


    #     print (testdate.date())

    def create_repeat(df2, batch_size=dayz):

        out = defaultdict(dict)

        for dayz in list(map(str, [batch_size])):

            for ix, d in enumerate(
                    pd.date_range(start=ca_created_at.min(), end=ca_created_at.max(), freq=dayz + 'D', closed='right')):

                print(ix, dayz, d.date())
                clear_output(wait=True)

                if ix == 0:
                    out[dayz][d.date()] = categorical_handling(
                        numeric_handling(df2.iloc[np.where(np.in1d(df2.index, ca_created_at_[ca_created_at.min():d]))]))
                    last = d
                else:
                    out[dayz][d.date()] = categorical_handling(
                        numeric_handling(df2.iloc[np.where(np.in1d(df2.index, ca_created_at_[last:d]))]))
                    last = d
        return out


    import os.path

    normalized_repeat_dict = None
    file_exists = os.path.exists('normalized_repeat_dict_x.pkl')
    if file_exists:
        with open('normalized_repeat_dict_x.pkl', 'rb') as f:
            normalized_repeat_dict = pickle.load(f)
    else:
        normalized_repeat_dict = create_repeat(df2)
        with open('normalized_repeat_dict_x.pkl', 'wb') as f:
            pickle.dump(normalized_repeat_dict, f)
    #     print (normalized_repeat_dict[dayz][testdate.date()])

    contam = avg_fraud
    rnd_ = 42

    detector_list_0 = [PCA(n_components=n_component) for n_component in np.arange(1, min(len(df2.columns) + 1, 20), 2)]

    detector_list_1 = [IForest(n_estimators=128), HBOS()]

    detector_list_2 = [LOF(n_neighbors=n_neighbor) for n_neighbor in np.arange(5, 21, 5)] + \
                      [KNN(method='mean', n_neighbors=n_neighbor) for n_neighbor in [1, 3, 5, 10, 15, 20]] + \
                      [IForest(n_estimators=128), HBOS()]

    classifier_dict = {'IForest (128)': IForest(n_estimators=128),
                       'Histogram Based': HBOS(contamination=contam),
                       'Local Outlier Factor LOF (15)': LOF(n_neighbors=15),
                       'K Nearest Neighbors KNN (15)': KNN(method='mean', n_neighbors=15),
                       'PCA (1-19)': LSCP(detector_list_0, contamination=contam, random_state=rnd_),
                       'IForest/HBOS Ensembles': LSCP(detector_list_1, contamination=contam, random_state=rnd_),
                       'IForest/HBOS/LOF/KNN Ensembles': LSCP(detector_list_2, contamination=contam, random_state=rnd_)}


    def get_df_name(df):
        name = [x for x in globals() if globals()[x] is df][0]
        return name


    relevant_columns = list(normalized_repeat_dict['90'].keys())

    metrics_df = pd.DataFrame(np.zeros((len(classifier_dict.keys()), len(relevant_columns))), columns=relevant_columns,
                              index=[classifier_name for classifier_name in classifier_dict.keys()])

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

    for d in list(normalized_repeat_dict['90'].keys()):

        X_train = normalized_repeat_dict['90'][d]

        '''
        Dimension Reduction
        '''
        df = X_train.join(fraud_column).join(fm_5)
        labels = df.fraud_specified
        fm_5_scores = df.fm_5_score
        fm_5_scores = (fm_5_scores - 0.00197736842965041) / 1.22643393399164
        fm_pred = fm_5_scores > 0.05
        df_ = df.drop(['fraud_specified', 'fm_5_score'], axis=1)

        pca = PCA(n_components=20)
        design_mtrx = pca.fit_transform(X_train)

        #         model_reg.fit(scaled_x_train.values, y_train[vp].values)
        #         data_pred = model_reg.predict(scaled_x_test.values)

        for name, classifier in classifier_dict.items():

            metrics_dict = Counter()

            if (('KNN' in name) or ('LOF' in name)) and (X_train.shape[1] > 20):
                #                 print (name, classifier.__class__.__name__, get_df_name(design_mtrx))
                classifier.fit(design_mtrx)
                classifier_proba = classifier.predict_proba(design_mtrx)
                classifier_pred = classifier_proba[:, 1] > 0.7
                augmented_pred = (fm_pred | classifier_pred)

                tcf = float("{0:.4f}".format(((fm_pred == 0) & (labels == 1)).sum() * 7000))
                augmented_tcf = float("{0:.4f}".format(((augmented_pred == 0) & (labels == 1)).sum() * 7000))
                tcol = float("{0:.4f}".format(((fm_pred == 1) & (labels == 0)).sum() * 0.22 * 0.10 * 5000))
                augmented_tcol = float(
                    "{0:.4f}".format(((augmented_pred == 1) & (labels == 0)).sum() * 0.22 * 0.10 * 5000))
                metrics_dict[d] = (metrics_dict['tcf'] + metrics_dict['tcol']) - (
                            metrics_dict['augmented_tcf'] + metrics_dict['augmented_tcol'])
                metrics_df.loc[name] = metrics_dict

            else:
                #                 print (name, classifier.__class__.__name__, get_df_name(X_train))
                print(X_train)
                classifier.fit(X_train)
                classifier_proba = classifier.predict_proba(X_train)
                classifier_pred = classifier_proba[:, 1] > 0.7
                augmented_pred = (fm_pred | classifier_pred)

                tcf = float("{0:.4f}".format(((fm_pred == 0) & (labels == 1)).sum() * 7000))
                augmented_tcf = float("{0:.4f}".format(((augmented_pred == 0) & (labels == 1)).sum() * 7000))
                tcol = float("{0:.4f}".format(((fm_pred == 1) & (labels == 0)).sum() * 0.22 * 0.10 * 5000))
                augmented_tcol = float(
                    "{0:.4f}".format(((augmented_pred == 1) & (labels == 0)).sum() * 0.22 * 0.10 * 5000))
                metrics_dict[d] = (metrics_dict['tcf'] + metrics_dict['tcol']) - (
                            metrics_dict['augmented_tcf'] + metrics_dict['augmented_tcol'])
                metrics_df.loc[name] = metrics_dict

            print(metrics_df)

    metrics_df.to_excel('metrics_df.xlsx', index=False)
    main()