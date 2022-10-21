from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier

#com Kfold

X, y = datasets.load_iris(return_X_y=True)
clf = DecisionTreeClassifier(criterion = "gini")
k_folds = KFold(n_splits = 10)
scores = cross_val_score(clf, X, y, cv = k_folds)

print("Árvore de Decisão")
print("Cross Validation Scores >>", scores)
print("Pontuação média da cross validação >>", scores.mean())
print("Número de Cross validações usadas na média >>", len(scores))

clf = RandomForestClassifier(n_estimators=100, max_features=3,criterion='gini', random_state = 0)
k_folds = KFold(n_splits = 10)
scores = cross_val_score(clf, X, y, cv = k_folds)

print("Random Forest")
print("Cross Validation Scores >>", scores)
print("Pontuação média da cross validação >>", scores.mean())
print("Número de Cross validações usadas na média >>", len(scores))
