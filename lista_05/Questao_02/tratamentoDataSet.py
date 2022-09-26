from cProfile import label
from dataclasses import replace
from math import remainder
from statistics import mode
import string
import pandas as pd
import pickle as pk
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split


dataSet = pd.read_csv("labor.csv", sep=',')
# print(dataSet)

# Verificando se temos valores Missing
dataSet.replace(to_replace = "?", value = np.nan, inplace = True)

print(dataSet.isnull().sum())

# print(dataSet.dtypes)

# age TSH T3 TT4 T4U FTI

dataSet = dataSet.convert_dtypes()
dataSet['age'] = pd.to_numeric(dataSet['age'], errors='coerce')
dataSet['TSH'] = pd.to_numeric(dataSet['TSH'], errors='coerce')
dataSet['T3'] = pd.to_numeric(dataSet['T3'], errors='coerce')
dataSet['TT4'] = pd.to_numeric(dataSet['TT4'], errors='coerce')
dataSet['FTI'] = pd.to_numeric(dataSet['FTI'], errors='coerce')


# usei a moda como método de imputação pois não consegui encontrar dentre os outros meios de imputação
# uma forma de fazer o labelEncoder ou o OneHot, sem ignorar os NaN e sem isso os algoritmos não conseguem rodar.
dataSet['age'] = dataSet['age'].fillna(dataSet['age'].mode(), inplace=False)
dataSet['sex'] = dataSet['sex'].fillna(dataSet['sex'].mode(), inplace=False)
dataSet['TSH'] = dataSet['TSH'].fillna(dataSet['TSH'].mode(), inplace=False)
dataSet['T3'] = dataSet['T3'].fillna(dataSet['T3'].mode(), inplace=False)
dataSet['TT4'] = dataSet['TT4'].fillna(dataSet['TT4'].mode(), inplace=False)
dataSet['T4U'] = dataSet['T4U'].fillna(dataSet['T4U'].mode(), inplace=False)
dataSet['FTI'] = dataSet['FTI'].fillna(dataSet['FTI'].mode(), inplace=False)

# print(dataSet.dtypes)

print(dataSet.isnull().sum())

# dataSet.to_csv("verifica.csv", sep=',')

# dadosSemResposta = dataSet.iloc[:, 0:29].values
# print(dadosSemResposta)
# dadosComResposta = dataSet.iloc[:, 29].values
# print(dadosComResposta)
# # 1 a 16 
# labelEncoder = LabelEncoder()
# dadosSemResposta[:, 1] = labelEncoder.fit_transform(dadosSemResposta[:, 1])
# dadosSemResposta[:, 2] = labelEncoder.fit_transform(dadosSemResposta[:, 2])
# dadosSemResposta[:, 3] = labelEncoder.fit_transform(dadosSemResposta[:, 3])
# dadosSemResposta[:, 4] = labelEncoder.fit_transform(dadosSemResposta[:, 4])
# dadosSemResposta[:, 5] = labelEncoder.fit_transform(dadosSemResposta[:, 5])
# dadosSemResposta[:, 6] = labelEncoder.fit_transform(dadosSemResposta[:, 6])
# dadosSemResposta[:, 7] = labelEncoder.fit_transform(dadosSemResposta[:, 7])
# dadosSemResposta[:, 8] = labelEncoder.fit_transform(dadosSemResposta[:, 8])
# dadosSemResposta[:, 9] = labelEncoder.fit_transform(dadosSemResposta[:, 9])
# dadosSemResposta[:, 10] = labelEncoder.fit_transform(dadosSemResposta[:, 10])
# dadosSemResposta[:, 11] = labelEncoder.fit_transform(dadosSemResposta[:, 11])
# dadosSemResposta[:, 12] = labelEncoder.fit_transform(dadosSemResposta[:, 12])
# dadosSemResposta[:, 13] = labelEncoder.fit_transform(dadosSemResposta[:, 13])
# dadosSemResposta[:, 14] = labelEncoder.fit_transform(dadosSemResposta[:, 14])
# dadosSemResposta[:, 15] = labelEncoder.fit_transform(dadosSemResposta[:, 15])
# dadosSemResposta[:, 16] = labelEncoder.fit_transform(dadosSemResposta[:, 16])
# dadosSemResposta[:, 18] = labelEncoder.fit_transform(dadosSemResposta[:, 18])
# dadosSemResposta[:, 20] = labelEncoder.fit_transform(dadosSemResposta[:, 20])
# dadosSemResposta[:, 22] = labelEncoder.fit_transform(dadosSemResposta[:, 22])
# dadosSemResposta[:, 24] = labelEncoder.fit_transform(dadosSemResposta[:, 24])
# dadosSemResposta[:, 26] = labelEncoder.fit_transform(dadosSemResposta[:, 26])
# dadosComResposta = labelEncoder.fit_transform(dadosComResposta)
# np.savetxt("verifica1.csv", dadosSemResposta, delimiter=" ", fmt='% s')

# OneHot
#        0  1  2  3  7
#Coluna: a, b, c, d, h,  

#labelEncoder
#        4  6  8  9
#Coluna: e, g, i, j

# oneHot = ColumnTransformer(transformers = [('OneHot', OneHotEncoder(), [28])], remainder='passthrough') 
# dadosSemResposta = oneHot.fit_transform(dadosSemResposta)
# oneHot = ColumnTransformer(transformers = [('OneHot', OneHotEncoder(), [1])], remainder='passthrough') 
# dadosComResposta = oneHot.fit_transform(dadosComResposta)

# np.savetxt("verifica.csv", dadosSemResposta, delimiter=" ", fmt='% s')

# dadosTreino, dadosTeste, respostaTreino, respostaTeste = train_test_split(dadosSemResposta, dadosComResposta, test_size = 0.20, random_state = 0)

# with open("lista_05.pkl", mode = "wb") as f:
#     pk.dump([dadosTreino, dadosTeste, respostaTreino, respostaTeste], f)
