#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

kddcup_names = open("../datastore/kddcup.names").read().split('\n')
features = []
for feature in kddcup_names[1:]:
    features.append(feature.split(':')[0].strip())

features.append("connection_type")

# data = pd.read_csv('../datastore/kddcup.data.csv', skiprows=0, nrows=494021)
# data = data.drop("label", axis=1)
# data.to_csv("../datastore/kddcup.data_demo_0.csv", index=False, header=None)
# data = pd.read_csv("../datastore/kddcup.data_demo_0.csv", names=features, index_col=False, header=None)
# data.to_csv("../datastore/kddcup.data_demo_0.csv", index=False)
# data=pd.read_csv("../datastore/kddcup.data_demo.csv")
data = pd.read_csv("../datastore/kddcup.data_10_percent_with_header.csv")

print(data)


def one_hot_encoder(data, features):
    """
    one hot encoding text values to dummy variables

    :param data:
    :param feature:
    :return:
    """
    for feature in features:
        dummies = pd.get_dummies(data[feature])
        for x in dummies.columns:
            dummy_feature = f"{feature}-{x}"
            data[dummy_feature] = dummies[x]
        data.drop(feature, axis=1, inplace=True)


def zscore_numeric_encoder(data, features, mean=None, sd=None):
    """
    Encode numerical columns as z-scores

    :param data:
    :param feature:
    :param mean:
    :param sd:
    :return:
    """
    for feature in features:
        if mean is None:
            mean = data[feature].mean()

        if sd is None:
            sd = data[feature].std()

        data[feature] = (data[feature] - mean) / sd


transformer = make_column_transformer(

    (MinMaxScaler(),
     ['duration', 'src_bytes', 'dst_bytes', 'wrong_fragment', 'urgent',
      'hot', 'num_failed_logins', 'num_compromised', 'root_shell',
      'su_attempted',
      'num_root', 'num_file_creations', 'num_shells', 'num_access_files',
      'num_outbound_cmds',
      'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate',
      'srv_rerror_rate',
      'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
      'dst_host_srv_count',
      'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
      'dst_host_same_src_port_rate',
      'dst_host_srv_diff_host_rate',
      'dst_host_serror_rate', 'dst_host_srv_serror_rate',
      'dst_host_rerror_rate',
      'dst_host_srv_rerror_rate'
      ]
     ),

    (OneHotEncoder(handle_unknown="ignore"),
     ['protocol_type', 'service', 'flag', 'land', 'logged_in', 'is_host_login', 'is_guest_login'])

)

transformer.fit(data)
transformer.transform(data)

# def transform_data(data):
#     categorical_features = ['protocol_type', 'service', 'flag', 'land', 'logged_in', 'is_host_login', 'is_guest_login']
#
#     one_hot_encoder(data, categorical_features)
#     non_categorical_features = ['duration', 'src_bytes', 'dst_bytes', 'wrong_fragment', 'urgent',
#                                 'hot', 'num_failed_logins', 'num_compromised', 'root_shell', 'su_attempted',
#                                 'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
#                                 'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate',
#                                 'srv_rerror_rate',
#                                 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
#                                 'dst_host_srv_count',
#                                 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
#                                 'dst_host_srv_diff_host_rate',
#                                 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
#                                 'dst_host_srv_rerror_rate'
#                                 ]
#
#     zscore_numeric_encoder(data, non_categorical_features)
#     print(data.head())
#
#     return data


# data = transform_data(data)

print(data)


