'''
Created on 2018年1月26日

@author: Administrator
'''
import pandas as pd
import numpy as np
import jieba
 
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.ensemble.tests.test_forest import check_min_samples_leaf

df = pd.read_csv('./data/result_process02', sep =',')
# print(df.head(5))
df.dropna(axis = 0, how ='any', inplace = 'True') 
print(df.head(5))
print(df.info())

x_train, x_test, y_train, y_test = train_test_split(df[['has_date','jieba_cut_content',\
                                                        'content_length_sema']],df['label'],\
                                                    test_size = 0.2, random_state = 0)
print("训练集大小%d" % x_train.shape[0])
print("测试集大小%d" % x_test.shape[0])
print(x_train.head(1000))
print(x_test.head(10)) 
#================================================================================================
print('='*30 + '开始训练集的特征工程' + '='*30)
transformer = TfidfVectorizer(norm = 'l2', use_idf = True)
svd = TruncatedSVD(n_components=20)
jieba_cut_content = list(x_train['jieba_cut_content'].astype('str'))
transformer_model = transformer.fit(jieba_cut_content)
df1 = transformer_model.transform(jieba_cut_content)
svd_model = svd.fit(df1)
df2 = svd_model.transform(df1)
data = pd.DataFrame(df2)
print(data.head(10))
print(data.info())

data['has_date'] = list(x_train['has_date'])
data['content_length_sema'] = list(x_train['content_length_sema'])
print(data.head(10))
print(data.info())

knn = KNeighborsClassifier(n_neighbors=5)
model = knn.fit(data, y_train)

jieba_cut_content_test = list(x_test['jieba_cut_content'].astype('str'))
data_test = pd.DataFrame(svd_model.transform(transformer_model.transform(jieba_cut_content_test)))
data_test['has_date'] = list(x_test['has_date'])
data_test['content_length_sema'] = list(x_test['content_length_sema'])
print(data_test.head(10))
print(data_test.info())
#��ʼԤ��
y_predict = model.predict(data_test)

precision = precision_score(y_test, y_predict)
recall = recall_score(y_test, y_predict)
f1mean = f1_score(y_test, y_predict)

print('精确率为：%0.5f' % precision)
print('召回率为：%0.5f' % recall)
print('F1均值为：%0.5f' % f1mean)







