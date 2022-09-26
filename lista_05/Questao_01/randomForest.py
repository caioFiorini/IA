from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle as pk

with open("lista_05.pkl", mode="rb") as f:
    dadosTreino, dadosTeste, respostasTreino, respostasTeste = pk.load(f)

randomForest = RandomForestClassifier(n_estimators=100, max_features=7,criterion='gini', random_state = 0)
randomForest.fit(dadosTreino,respostasTreino)

previsao = randomForest.predict(dadosTeste)

print(randomForest.feature_importances_)

print(accuracy_score(respostasTeste,previsao))
print(confusion_matrix(respostasTeste, previsao))
print(classification_report(respostasTeste, previsao))