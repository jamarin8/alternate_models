import json
from difflib import SequenceMatcher
import numpy as np
import pandas as pd
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
from urllib.parse import unquote
from pyzipcode import ZipCodeDatabase
zcdb = ZipCodeDatabase()
from Levenshtein import ratio
from fincen_dict import fincen_vars


"""
misc
"""

# Extract necessary raw model inputs
def extract_all_raw_model_inputs(df):
    ''' Selecting all raw model inputs to be used '''
    df_columns = df.columns
    keep_col_list = features_needed_for_preprocessing
    keep_col_list = df_columns.intersection(keep_col_list).tolist()
    df = df[keep_col_list]
    return df

# tmx_raw data parsing
def tmx_raw_data_parsing(df):
    ''' Parse out tmx_raw data '''
    tmx_raw_df = df.loc[:,['tmx_raw']]
    tmx_raw_df.tmx_raw = tmx_raw_df.tmx_raw.str.split('&')
    tmx_dict = {}
    tmx_dict_list = []
    for row in tmx_raw_df.itertuples():
        try:
            for i in tmx_raw_df['tmx_raw'][row.Index]:
                if i.split('=')[0] not in tmx_dict:
                    tmx_dict[i.split('=')[0]] = i.split('=')[1]
                else:
                    tmx_dict[i.split('=')[0]] += ', ' + (i.split('=')[1])
            tmx_dict_list.append(tmx_dict)
            tmx_dict = {}
        except TypeError:
            tmx_dict_list.append("{}")
    tmx_raw_df['tmx_raw_parsed'] = np.array(tmx_dict_list)
    tmx_raw_df.drop(columns=['tmx_raw'],axis=1,inplace=True)
    df = df.drop(columns=['tmx_raw'],axis=1)
    tmx_raw_df = tmx_raw_df.tmx_raw_parsed.apply(pd.Series).add_prefix('tmx_raw_')
    tmx_raw_df_cols = tmx_raw_df.columns
    keep_tmx_raw_col_list = tmx_raw_df_cols.intersection(tmx_raw_features_needed).tolist()
    df2 = pd.concat([df,tmx_raw_df[keep_tmx_raw_col_list]],axis=1)
    return df2
"""
misc
"""
# Access zip database to make timezone adjustments for app created_at time
def apply_tz_by_zip(x):
    '''Return value to create offset ca_created_at field/Ignore zips outside scope of zip database'''
    try:
        value = zcdb[x].timezone
    except KeyError:
        value = np.nan
    return value

# Create timezone adjusted ca_created_at field from applicant zipcode
def generate_tz_adj_date(df2):
    ''' Create adjusted customer app created at timezone based on applicant zip '''
    df2.a_zip = df2.a_zip.fillna(0)
    df2['tz_offset'] = df2['a_zip'].apply(lambda x: apply_tz_by_zip(x))
    df2.tz_offset = df2.tz_offset.fillna(0)
    df2.tz_offset = df2.tz_offset.astype('timedelta64[h]')
    df2['ca_ft_created_at_tz_adj'] = df2.apply(lambda x: pd.to_datetime(x['ca_created_at']) + x['tz_offset'],axis=1)
    df2['ca_ft_created_at_tz_adj'] = df2['ca_ft_created_at_tz_adj'].apply(lambda x: str(x))
    df2.drop(['ca_created_at','tz_offset'], axis=1, inplace=True)
    return df2


# Levenshtein ratio distance: TeleSign & Customer Application First Name Match Score
def generate_lev_first_name_match_score(df):
    ''' Create fuzzy first name match score using Levenshtein ratio distance '''
    name_match_list = []
    df['tcr_contact_firstname'].replace(np.nan, None, inplace=True)
    for i in range(0, len(df)):
        if pd.isnull(df['tcr_contact_firstname'][i]) or pd.isnull(df['p_first_name'][i]):
            name_match_list.append(-1)
        elif len(df['tcr_contact_firstname'][i]) > 0:
            name_match_list.append(
                ratio(str(df['p_first_name'][i].lower()), str(df['tcr_contact_firstname'][i].lower())))
        else:
            name_match_list.append(-1)
    df['p_tcr_ft_first_name_match'] = name_match_list
    return df


# Levenshtein ratio distance: TeleSign & Customer Application Last Name Match Score
def generate_lev_last_name_match_score(df):
    ''' Create fuzzy last name match score using Levenshtein ratio distance '''
    name_match_list = []
    df['tcr_contact_lastname'].replace(np.nan, None, inplace=True)
    for i in range(0, len(df)):
        if pd.isnull(df['tcr_contact_lastname'][i]) or pd.isnull(df['p_last_name'][i]):
            name_match_list.append(-1)
        elif len(df['tcr_contact_lastname'][i]) > 0:
            name_match_list.append(ratio(str(df['p_last_name'][i].lower()), str(df['tcr_contact_lastname'][i].lower())))
        else:
            name_match_list.append(-1)
    df['p_tcr_ft_last_name_match'] = name_match_list
    return df


# Difflib sequence matcher: TeleSign & Customer Application First Name Match Score
def generate_difflib_first_name_match_score(df2):
    ''' Create fuzzy first name match score using Difflib sequence mathcer distance score '''
    diff_first_name_match = []
    for i in range(0, len(df2)):
        if pd.isnull(df2['tcr_contact_firstname'][i]) or pd.isnull(df2['p_first_name'][i]):
            diff_first_name_match.append(-1)
        elif (len(df2['tcr_contact_firstname'][i]) > 0):
            diff_first_name_match.append(SequenceMatcher(None, str(df2['p_first_name'][i].lower()),
                                                         str(df2['tcr_contact_firstname'][i].lower())).ratio())
        else:
            diff_first_name_match.append(-1)
    df2['p_tcr_ft_difflib_first_name_match'] = diff_first_name_match
    return df2


# Difflib sequence matcher: TeleSign & Customer Application Last Name Match Score
def generate_difflib_last_name_match_score(df2):
    ''' Create fuzzy last name match score using Difflib sequence mathcer distance score '''
    diff_last_name_match = []
    for i in range(0, len(df2)):
        if pd.isnull(df2['tcr_contact_lastname'][i]) or pd.isnull(df2['p_last_name'][i]):
            diff_last_name_match.append(-1)
        elif (len(df2['tcr_contact_lastname'][i]) > 0):
            diff_last_name_match.append(SequenceMatcher(None, str(df2['p_last_name'][i].lower()),
                                                        str(df2['tcr_contact_lastname'][i].lower())).ratio())
        else:
            diff_last_name_match.append(-1)
    df2['p_tcr_ft_difflib_last_name_match'] = diff_last_name_match
    return df2


# Levenshtein ratio distance: TeleSign & Customer Application Address Match Score
def generate_lev_address_match_score(df2):
    ''' Create fuzzy address match score using Levenshtein ratio distance '''
    df2 = df2.replace('\"', '', regex=True)
    df2 = df2.replace("\'", "", regex=True)
    address_match = []
    for i in range(0, len(df2)):
        if pd.isnull(df2['a_address_1'][i]) or pd.isnull(df2['tcr_contact_address1'][i]):
            address_match.append(-1)
        elif (len(df2['tcr_contact_address1'][i]) > 0):
            address_match.append(ratio(str(df2['a_address_1'][i].lower()), str(df2['tcr_contact_address1'][i].lower())))
        else:
            address_match.append(-1)
    df2['a_tcr_ft_address_match'] = address_match
    return df2


# Number of digits before @ in email
def generate_email_len_var(df2):
    ''' Create length of email variable '''
    df2['ea_ft_digits'] = df2['ea_email'].apply(lambda x: None if pd.isnull(x) else len(x.split('@')[0]))
    return df2


# Create TMX/TCR/SS distance features
def Haversine(lat1, lon1, lat2, lon2, **kwarg):
    """
    This uses the ‘haversine’ formula to calculate the great-circle distance between two points – that is,
    the shortest distance over the earth’s surface – giving an ‘as-the-crow-flies’ distance between the points
    (ignoring any hills they fly over, of course!).
    Haversine
    formula:    a = sin²(Δφ/2) + cos φ1 ⋅ cos φ2 ⋅ sin²(Δλ/2)
    c = 2 ⋅ atan2( √a, √(1−a) )
    d = R ⋅ c
    where   φ is latitude, λ is longitude, R is earth’s radius (mean radius = 6,371km);
    note that angles need to be in radians to pass to trig functions!
    """
    R = 6371.0088
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(a ** 0.5, (1 - a) ** 0.5)
    d = R * c

    return d


def clean_coordinate_attributes(df, col_list):
    ''' encode values from feature list & cast to numeric '''
    for i in col_list:
        if i in df.columns:
            if 'tmx_raw' in i:
                df[i] = df[i].apply(lambda row: parse_url_seps(row))
                df[i] = df[i].astype(float)
            else:
                df[i] = df[i].replace("'", '', regex=True)
                df[i] = pd.to_numeric(df[i], errors='coerce')
    return df


def generate_input_true_ip_distance_vars(df):
    ''' Create IP distance features using Haversine Distance Function '''
    if ('tmx_raw_input_ip_latitude') in df.columns:
        df['tmx_ft_input_ip_true_ip_distance'] = df.apply(
            lambda row: Haversine(row.tmx_raw_input_ip_latitude, row.tmx_raw_input_ip_longitude,
                                  row.tmx_raw_true_ip_latitude, row.tmx_raw_true_ip_longitude) if (
                        pd.notnull(row.tmx_raw_input_ip_longitude) & pd.notnull(
                    row.tmx_raw_true_ip_longitude)) else None, axis=1)
    else:
        df['tmx_ft_input_ip_true_ip_distance'] = np.nan
    return df


def generate_input_ip_address_distance_vars(df):
    ''' Create Input IP to Address distance features using Haversine Distance Function '''
    if 'tmx_raw_input_ip_latitude' in df.columns:
        df['tmx_ss_ft_input_ip_address_distance'] = df.apply(
            lambda row: Haversine(row.ss_lat, row.ss_long, row.tmx_raw_input_ip_latitude,
                                  row.tmx_raw_input_ip_longitude) if (
                        pd.notnull(row.ss_long) & pd.notnull(row.tmx_raw_input_ip_longitude)) else None, axis=1)
    else:
        df['tmx_ss_ft_input_ip_address_distance'] = np.nan
    return df


def generate_true_ip_address_distance_vars(df):
    ''' Create True IP to Address distance features using Haversine Distance Function '''
    if 'tmx_raw_true_ip_longitude' in df.columns:
        df['tmx_ss_ft_true_ip_address_distance'] = df.apply(
            lambda row: Haversine(row.ss_lat, row.ss_long, row.tmx_raw_true_ip_latitude,
                                  row.tmx_raw_true_ip_longitude) if (
                        pd.notnull(row.ss_long) & pd.notnull(row.tmx_raw_true_ip_longitude)) else None, axis=1)
    else:
        df['tmx_ss_ft_true_ip_address_distance'] = np.nan
    return df


def generate_phone_address_distance_vars(df):
    ''' Create Phone to Address distance features using Haversine Distance Function '''
    if 'tcr_location_coordinates_longitude' in df.columns:
        df['ss_tcr_ft_address_phone_distance'] = df.apply(
            lambda row: Haversine(row.ss_lat, row.ss_long, row.tcr_location_coordinates_latitude,
                                  row.tcr_location_coordinates_longitude) if (
                        pd.notnull(row.ss_long) & pd.notnull(row.tcr_location_coordinates_longitude)) else None, axis=1)
    else:
        df['ss_tcr_ft_address_phone_distance'] = np.nan
    return df


def generate_true_ip_phone_distance_vars(df):
    ''' Create True IP to Phone distance features using Haversine Distance Function '''
    if 'tmx_raw_true_ip_latitude' in df.columns:
        df['tcr_tmx_ft_phone_true_ip_distance'] = df.apply(
            lambda row: Haversine(row.tcr_location_coordinates_latitude, row.tcr_location_coordinates_longitude,
                                  row.tmx_raw_true_ip_latitude, row.tmx_raw_true_ip_longitude) if (
                        pd.notnull(row.tcr_location_coordinates_longitude) & pd.notnull(
                    row.tmx_raw_true_ip_longitude)) else None, axis=1)
    else:
        df['tcr_tmx_ft_phone_true_ip_distance'] = np.nan
    return df


def generate_dns_ip_address_distance_vars(df):
    ''' Create DNS IP to Address distance features using Haversine Distance Function '''
    if 'tmx_raw_dns_ip_longitude' in df.columns:
        df['ss_tmx_ft_address_dns_ip_distance'] = df.apply(
            lambda row: Haversine(row.ss_lat, row.ss_long, row.tmx_raw_dns_ip_latitude,
                                  row.tmx_raw_dns_ip_longitude) if (
                        pd.notnull(row.ss_long) & pd.notnull(row.tmx_raw_dns_ip_longitude)) else None, axis=1)
    else:
        df['ss_tmx_ft_address_dns_ip_distance'] = np.nan
    return df


def generate_proxy_ip_dns_ip_distance_vars(df):
    ''' Create Proxy IP to DNS IP distance features using Haversine Distance Function '''
    if 'tmx_raw_proxy_ip_latitude' in df.columns:
        df['tmx_ft_proxy_ip_dns_ip_distance'] = df.apply(
            lambda row: Haversine(row.tmx_raw_proxy_ip_latitude, row.tmx_raw_proxy_ip_longitude,
                                  row.tmx_raw_dns_ip_latitude, row.tmx_raw_dns_ip_longitude) if (
                        pd.notnull(row.tmx_raw_proxy_ip_longitude) & pd.notnull(
                    row.tmx_raw_dns_ip_longitude)) else None, axis=1)
    else:
        df['tmx_ft_proxy_ip_dns_ip_distance'] = np.nan
    return df


def generate_input_ip_proxy_ip_distance_vars(df):
    ''' Create Proxy IP to Input IP distance features using Haversine Distance Function '''
    if 'tmx_raw_proxy_ip_latitude' in df.columns:
        df['tmx_ft_proxy_ip_input_ip_distance'] = df.apply(
            lambda row: Haversine(row.tmx_raw_proxy_ip_latitude, row.tmx_raw_proxy_ip_longitude,
                                  row.tmx_raw_input_ip_latitude, row.tmx_raw_input_ip_longitude) if (
                        pd.notnull(row.tmx_raw_proxy_ip_longitude) & pd.notnull(
                    row.tmx_raw_input_ip_longitude)) else None, axis=1)
    else:
        df['tmx_ft_proxy_ip_input_ip_distance'] = np.nan
    return df


def generate_input_ip_phone_distance_vars(df):
    ''' Create Input IP to Phone distance features using Haversine Distance Function '''
    if 'tcr_location_coordinates_latitude' in df.columns:
        df['tcr_tmx_ft_phone_input_ip_distance'] = df.apply(
            lambda row: Haversine(row.tcr_location_coordinates_latitude, row.tcr_location_coordinates_longitude,
                                  row.tmx_raw_input_ip_longitude, row.tmx_raw_input_ip_latitude) if (
                        pd.notnull(row.tcr_location_coordinates_longitude) & pd.notnull(
                    row.tmx_raw_input_ip_longitude)) else None, axis=1)
    else:
        df['tcr_tmx_ft_phone_input_ip_distance'] = np.nan
    return df


def generate_proxy_ip_address_distance_vars(df):
    ''' Create Proxy IP to Address distance features using Haversine Distance Function '''
    if 'tmx_raw_proxy_ip_latitude' in df.columns:
        df['ss_tmx_ft_address_proxy_ip_distance'] = df.apply(
            lambda row: Haversine(row.ss_lat, row.ss_long, row.tmx_raw_proxy_ip_latitude,
                                  row.tmx_raw_proxy_ip_longitude) if (
                        pd.notnull(row.ss_long) & pd.notnull(row.tmx_raw_proxy_ip_longitude)) else None, axis=1)
    else:
        df['ss_tmx_ft_address_proxy_ip_distance'] = np.nan
    return df


def generate_proxy_ip_phone_distance_vars(df):
    ''' Create Proxy IP to Phone distance features using Haversine Distance Function '''
    if 'tmx_raw_proxy_ip_latitude' in df.columns:
        df['tcr_tmx_ft_phone_proxy_ip_distance'] = df.apply(
            lambda row: Haversine(row.tcr_location_coordinates_latitude, row.tcr_location_coordinates_longitude,
                                  row.tmx_raw_proxy_ip_latitude, row.tmx_raw_proxy_ip_longitude) if (
                        pd.notnull(row.tcr_location_coordinates_longitude) & pd.notnull(
                    row.tmx_raw_proxy_ip_longitude)) else None, axis=1)
    else:
        df['tcr_tmx_ft_phone_proxy_ip_distance'] = np.nan
    return df


def generate_dns_ip_phone_distance_vars(df):
    ''' Create DNS IP to Phone distance features using Haversine Distance Function '''
    if 'tmx_raw_dns_ip_longitude' in df.columns:
        df['tcr_tmx_ft_phone_dns_ip_distance'] = df.apply(
            lambda row: Haversine(row.tcr_location_coordinates_latitude, row.tcr_location_coordinates_longitude,
                                  row.tmx_raw_dns_ip_longitude, row.tmx_raw_dns_ip_latitude) if (
                        pd.notnull(row.tcr_location_coordinates_longitude) & pd.notnull(
                    row.tmx_raw_dns_ip_longitude)) else None, axis=1)
    else:
        df['tcr_tmx_ft_phone_dns_ip_distance'] = np.nan
    return df


def generate_null_features(df):
    ''' Populate DF with all features that were not present containing null values '''
    df_columns = df.columns
    col_not_present = list(set(final_raw_features) - set(df_columns))
    for i in col_not_present:
        df[i] = ""
    return df


# Extract necessary sms model inputs
def extract_sms_model_inputs(df):
    ''' Selecting all model inputs to be passed to SageMaker '''
    '''removal of PII'''
    df_columns = df.columns
    keep_col_list = final_raw_features
    keep_col_list = df_columns.intersection(final_raw_features).tolist()
    df = df[keep_col_list]
    return df


####################################################################
# End of PII Field Transformations & Extraction
####################################################################

# apply all column names as lower case
def apply_cols_as_lower_case(df2):
    df2.columns = df2.columns.str.lower()
    return df2


def run_all(df, distance_feature_list=distance_feature_list):
    ''' Invoke all parsing, cleaning & feature engineering functions, Returns JSON Value '''
    df = pd.DataFrame.from_dict(df, orient='index').T
    df = extract_all_raw_model_inputs(df)
    df = tmx_raw_data_parsing(df)
    df = generate_fincen_high_risk_var(df)
    df = generate_lev_first_name_match_score(df)
    df = generate_lev_last_name_match_score(df)
    df = generate_difflib_first_name_match_score(df)
    df = generate_difflib_last_name_match_score(df)
    df = generate_lev_address_match_score(df)
    df = generate_tz_adj_date(df)
    df = generate_email_len_var(df)
    df = clean_coordinate_attributes(df, distance_feature_list)
    df = generate_input_true_ip_distance_vars(df)
    df = generate_input_ip_address_distance_vars(df)
    df = generate_true_ip_address_distance_vars(df)
    df = generate_phone_address_distance_vars(df)
    df = generate_true_ip_phone_distance_vars(df)
    df = generate_dns_ip_address_distance_vars(df)
    df = generate_proxy_ip_dns_ip_distance_vars(df)
    df = generate_input_ip_proxy_ip_distance_vars(df)
    df = generate_input_ip_phone_distance_vars(df)
    df = generate_proxy_ip_address_distance_vars(df)
    df = generate_proxy_ip_phone_distance_vars(df)
    df = generate_dns_ip_phone_distance_vars(df)
    df = generate_null_features(df)
    df = apply_cols_as_lower_case(df)
    df = extract_sms_model_inputs(df)
    json_value = df.to_json()
    return json_value


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")

    with open('payload.txt', 'r') as file:
        tempdict = json.load(file)
        df0_payload = pd.DataFrame.from_dict(tempdict, orient='index').T
        lambda_out = run_all(df0_payload, distance_feature_list)
        data = json.loads(json.dumps(lambda_out))
        print(data)

