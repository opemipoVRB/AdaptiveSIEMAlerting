#!/usr/bin/env python
# -*- coding: utf-8 -*-
from sklearn import decomposition
from tensorflow.keras.models import load_model

model = load_model("nid_siem_model.h5", compile=True)


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


non_categorical_features = ['duration', 'src_bytes', 'dst_bytes', 'wrong_fragment', 'urgent',
                            'hot', 'num_failed_logins', 'num_compromised', 'root_shell', 'su_attempted',
                            'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
                            'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
                            'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
                            'dst_host_srv_count',
                            'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
                            'dst_host_srv_diff_host_rate',
                            'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
                            'dst_host_srv_rerror_rate'
                            ]

categorical_features = ['protocol_type', 'service', 'flag', 'land', 'logged_in', 'is_host_login', 'is_guest_login']


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


def transform(X_samp):
    one_hot_encoder(X_samp, categorical_features)
    zscore_numeric_encoder(X_samp, non_categorical_features)
    pca = decomposition.PCA(n_components=7)
    pca.fit(X_samp)
    X_samp = pca.transform(X_samp)
    X_samp = X_samp.reshape(X_samp.shape[0], X_samp.shape[1], 1)

    return X_samp
