from cProfile import label
from dataclasses import replace
from math import remainder
import pandas as pd
import pickle as pk
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split


dataSet = pd.read_csv("breast-cancer 1.csv", sep=',')
print(dataSet)

dataSet.replace(to_replace = "?", value = np.nan, inplace = True)


#usei a moda como método de imputação pois não consegui encontrar dentre os outros meios de imputação
#uma forma de fazer o labelEncoder ou o OneHot, sem ignorar os NaN e sem isso os algoritmos não conseguem rodar.
dataSet['node-caps'].fillna(dataSet['node-caps'].mode(), inplace=True)
dataSet['breast-quad'].fillna(dataSet['breast-quad'].mode(), inplace=True)

# dataSet.to_csv("cancer.csv", sep=',')

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
oneHot = ColumnTransformer(transformers = [('OneHot', OneHotEncoder(), [13,14,15])], remainder='passthrough') 
dadosSemResposta = oneHot.fit_transform(dadosSemResposta)


# np.savetxt("verifica.csv", dadosSemResposta, delimiter=" ", fmt='% s')

dadosTreino, dadosTeste, respostaTreino, respostaTeste = train_test_split(dadosSemResposta, dadosComResposta, test_size = 0.20, random_state = 0)

with open("lista_05.pkl", mode = "wb") as f:
    pk.dump([dadosTreino, dadosTeste, respostaTreino, respostaTeste], f)
