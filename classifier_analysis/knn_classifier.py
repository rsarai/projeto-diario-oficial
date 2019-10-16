import pandas as pd
import numpy as np
from nltk.corpus import stopwords


pasta = '../base-djma/'
arquivo = 'VARA'


#Leitura da Planilha
df = pd.read_csv(pasta + arquivo + '.csv', encoding='utf-8')
X, y = df.conteudo, df.saida


import re
from nltk.stem import WordNetLemmatizer

documents = []
stemmer = WordNetLemmatizer()
for sen in range(0, len(X)):
    # Remove all the special characters
    document = re.sub(r'\W', ' ', str(X[sen]))
    # remove all single characters
    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
    # Remove single characters from the start
    document = re.sub(r'\^[a-zA-Z]\s+', ' ', document) 
    # Substituting multiple spaces with single space
    document = re.sub(r'\s+', ' ', document, flags=re.I)
    # Removing prefixed 'b'
    document = re.sub(r'^b\s+', '', document)
    # Converting to Lowercase
    document = document.lower()
    documents.append(document)

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('portuguese'))
X = vectorizer.fit_transform(documents).toarray()





def single_round(classifier,X_train, y_train, X_test, y_test):
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    classifier.fit(X_train, y_train) 
    y_pred = classifier.predict(X_test)
    print(confusion_matrix(y_test,y_pred))
    print(classification_report(y_test,y_pred))
    print(accuracy_score(y_test, y_pred))

#Cross Validation utilizando KFOLD estratificado
def cross_validation(clf, X,y):
    from sklearn.model_selection import cross_validate
    from sklearn.metrics import recall_score
    scoring = ['precision_macro', 'recall_macro', 'accuracy']
    scores = cross_validate(clf, X, y, cv=10, scoring=scoring)
    print("Recall: %0.2f (+/- %0.2f)" % (scores['test_recall_macro'] .mean(), scores['test_recall_macro'] .std() * 2))
    print("Precission: %0.2f (+/- %0.2f)" % (scores['test_precision_macro'] .mean(), scores['test_precision_macro'] .std() * 2))
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores['test_accuracy'] .mean(), scores['test_accuracy'] .std() * 2))

#Divide o dataset em treino e teste, mantendo a proporção de cada classe.
def stratifiedShuffleSplit(X,y,classifier):
    from sklearn.model_selection import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.25)
    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        #Chama o classificador gerando as métricas padrão para cada SPLIT.
        single_round(classifier,X_train,y_train,X_test,y_test)


from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5)
# classifier = MLPClassifier(solver='lbfgs', alpha=1e-5,
#                     hidden_layer_sizes=(5, 2), random_state=1)


stratifiedShuffleSplit(X,y,classifier)
cross_validation(classifier,X,y)  
print('CLASSE: ' + arquivo)
print('Configuração dos dados: Sem stopwords')
print('Classificador: Rede Neural (5,2) e configuração padrão de inicialização do scikitlearn')