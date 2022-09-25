from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle as pk

with open("lista_05.pkl", mode="rb") as f:
    dadosTreino, dadosTeste, respostasTreino, respostasTeste = pk.load(f)

decisionTree = DecisionTreeClassifier(criterion='gini')
decisionTree.fit(dadosTreino,respostasTreino)

previsao = decisionTree.predict(dadosTeste)
print(accuracy_score(respostasTeste,previsao))
print(confusion_matrix(respostasTeste, previsao))
print(classification_report(respostasTeste, previsao))