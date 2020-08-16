
protocol = list(X_train['protocol_type'].values)

protocol = list(set(protocol))

print('Protocol types are:', protocol)

from sklearn.feature_extraction.text import CountVectorizer

one_hot = CountVectorizer(vocabulary=protocol, binary=True)

train_protocol = one_hot.fit_transform(X_train['protocol_type'].values)

test_protocol = one_hot.transform(X_test['protocol_type'].values)

print(train_protocol[1].toarray())


service = list(X_train['service'].values)

service = list(set(service))

print('Service types are:\n', service)



from sklearn.feature_extraction.text import CountVectorizer



one_hot = CountVectorizer(vocabulary=service, binary=True)

train_service = one_hot.fit_transform(X_train['service'].values)

test_service = one_hot.transform(X_test['service'].values)

print(train_service[100].toarray())

train_service.shape

(116468, 65)

flag = list(X_train['flag'].values)

flag = list(set(flag))

print('flag types are:', flag)


from sklearn.feature_extraction.text import CountVectorizer

one_hot = CountVectorizer(binary=True)

one_hot.fit(X_train['flag'].values)

train_flag = one_hot.transform(X_train['flag'].values)

test_flag = one_hot.transform(X_test['flag'].values)

