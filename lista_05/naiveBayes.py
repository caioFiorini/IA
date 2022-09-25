from sklearn.naive_bayes import GaussianNB 
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle as pk

with open("lista_05.pkl", mode="rb") as f:
    dadosTreino, dadosTeste, respostasTreino, respostasTeste = pk.load(f)

naiveBayes = GaussianNB()
naiveBayes.fit(dadosTreino,respostasTreino)

previsao = naiveBayes.predict(dadosTeste)
print(accuracy_score(respostasTeste,previsao))
print(confusion_matrix(respostasTeste, previsao))
print(classification_report(respostasTeste, previsao))