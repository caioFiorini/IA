from cProfile import label
from dataclasses import replace
from math import remainder
import pandas as pd
import pickle as pk
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB 
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE 


dataSet = pd.read_csv("breast-cancer 1.csv", sep=',')
print(dataSet)

dataSet.replace(to_replace = "?", value = np.nan, inplace = True)


#usei a moda como método de imputação pois não consegui encontrar dentre os outros meios de imputação
#uma forma de fazer o labelEncoder ou o OneHot, sem ignorar os NaN e sem isso os algoritmos não conseguem rodar.
dataSet['node-caps'].fillna(dataSet['node-caps'].mode(), inplace=True)
dataSet['breast-quad'].fillna(dataSet['breast-quad'].mode(), inplace=True)

# Verificando se temos valores Missing
print(dataSet.isna().sum())

# Retirando valores missing
dataSet.dropna(inplace=True)

# Verificando como está a nossa classe
print(dataSet['Class'].value_counts())

# O que foi retornado:
# no-recurrence-events    196
# recurrence-events        81
#temos um desbalanceamento grande entre essas classes, precisamos rebalancear

dadosSemResposta = dataSet.iloc[:, 0:9].values
# print(dadosSemResposta)
dadosComResposta = dataSet.iloc[:, 9].values
# print(dadosComResposta)

labelEncoder_NodeCaps = LabelEncoder() #coluna e
labelEncoder_Breast = LabelEncoder() #coluna g
labelEncoder_Irradiat = LabelEncoder() #coluna i
labelEncoder_Class = LabelEncoder() #coluna j
dadosSemResposta[:, 4] = labelEncoder_NodeCaps.fit_transform(dadosSemResposta[:, 4])
dadosSemResposta[:, 6] = labelEncoder_Breast.fit_transform(dadosSemResposta[:, 6])
dadosSemResposta[:, 8] = labelEncoder_Irradiat.fit_transform(dadosSemResposta[:,8])
dadosComResposta = labelEncoder_Class.fit_transform(dadosComResposta)
# np.savetxt("verifica1.csv", dadosSemResposta, delimiter=" ", fmt='% s')

# OneHot
#        0  1  2  3  7
#Coluna: a, b, c, d, h,  

#labelEncoder
#        4  6  8  9
#Coluna: e, g, i, j

oneHot = ColumnTransformer(transformers = [('OneHot', OneHotEncoder(), [3,7])], remainder='passthrough') 
dadosSemResposta = oneHot.fit_transform(dadosSemResposta)
oneHot = ColumnTransformer(transformers = [('OneHot', OneHotEncoder(), [9])], remainder='passthrough') 
dadosSemResposta = oneHot.fit_transform(dadosSemResposta)
oneHot = ColumnTransformer(transformers = [('OneHot', OneHotEncoder(), [13,14,15])], remainder='passthrough') 
dadosSemResposta = oneHot.fit_transform(dadosSemResposta)
# print(dados)

# Gerando um csv para verificar se as instâncias estão corretas
np.savetxt("breast-cancer-numerico.csv", dadosSemResposta, delimiter=",", fmt='% s')


dadosTreino, dadosTeste, respostaTreino, respostasTeste = train_test_split(dadosSemResposta, dadosComResposta, test_size = 0.20, random_state = 0)

# Fazendo o balanceamento

smt = SMOTE(sampling_strategy='auto')
x_treino , y_treino = smt.fit_resample(dadosTreino , respostaTreino)

# verificando se correu tudo certo no balanceamento 
print(x_treino.shape)
print(y_treino.shape)

# Rodando os algoritmos:

naiveBayes = GaussianNB()
naiveBayes.fit(x_treino,y_treino)
previsao = naiveBayes.predict(dadosTeste)
print("Naive Bayes")
print(accuracy_score(respostasTeste,previsao))
print(confusion_matrix(respostasTeste, previsao))
print(classification_report(respostasTeste, previsao))


randomForest = RandomForestClassifier(n_estimators=100, max_features=7,criterion='gini', random_state = 0)
randomForest.fit(x_treino, y_treino)

previsao = randomForest.predict(dadosTeste)
print("Random Forest")
print(accuracy_score(respostasTeste,previsao))
print(confusion_matrix(respostasTeste, previsao))
print(classification_report(respostasTeste, previsao))

decisionTree = DecisionTreeClassifier(criterion='gini')
decisionTree.fit(x_treino,y_treino)

previsao = decisionTree.predict(dadosTeste)
print("Árvore de Decisão")
print(accuracy_score(respostasTeste,previsao))
print(confusion_matrix(respostasTeste, previsao))
print(classification_report(respostasTeste, previsao))