import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, fbeta_score
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier

############################# Preprocessing the dataset for easier use in analysis ########################

df_train = pd.read_csv('/local/scratch/shared-directories/ssanet/swati_folder/nlp_project/medical_tc_train.csv')
df_test = pd.read_csv('/local/scratch/shared-directories/ssanet/swati_folder/nlp_project/medical_tc_test.csv')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    filtered = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and word.isalpha()]
    return ' '.join(filtered)

df_train['medical_abstract'] = df_train['medical_abstract'].apply(clean)
df_test['medical_abstract'] = df_test['medical_abstract'].apply(clean)
df_train = pd.read_csv('/local/scratch/shared-directories/ssanet/swati_folder/nlp_project/medical_tc_train_preprocessed.csv')
df_test= pd.read_csv('/local/scratch/shared-directories/ssanet/swati_folder/nlp_project/medical_tc_test_preprocessed.csv')

################################ Using the preprocessed dataset ###############################################
df_train = pd.read_csv('/local/scratch/shared-directories/ssanet/swati_folder/nlp_project/medical_tc_train_preprocessed.csv')
df_test= pd.read_csv('/local/scratch/shared-directories/ssanet/swati_folder/nlp_project/medical_tc_test_preprocessed.csv')

print(df_train.shape, flush=True)
df_train,_ = train_test_split(df_train,train_size=0.4,stratify=df_train['condition_label'])
print(df_train.shape,flush=True)

X_train = df_train['medical_abstract']
y_train = df_train['condition_label']-1

X_test = df_test['medical_abstract']
y_test = df_test['condition_label']-1

vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=5000)
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)


############### Hyperparameters tuning ####################
models = {
    'KNN': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'GradientBoosting': GradientBoostingClassifier(),
    'Naive Bayes': MultinomialNB(),
    'Logistic Regression': LogisticRegression(max_iter=10000),
    'SVM': SVC()
}

param_grids = {
    'Naive Bayes': {'alpha': [0.01, 1, 100]},

    'Logistic Regression': {'C': [0.1, 1, 10],
                            'penalty': ['l2'],
                            'solver': ['liblinear']},

    'SVM': {'C': [0.1, 1, 10]},

    'KNN': {'n_neighbors': [5, 7]},

    'Decision Tree': {'max_depth': [None, 10]},
    
    'Random Forest': {'n_estimators': [50, 100],
                      'max_depth': [None, 10, 30],
                      'min_samples_split': [2, 5],
                      'min_samples_leaf': [1, 2]},

    
    'GradientBoosting': {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1]},
}


best_models = {}


for classifier_name, classifier in models.items():
    grid_search = GridSearchCV(classifier, param_grids[classifier_name], cv=5)
    grid_search.fit(X_train, y_train)
    best_models[classifier_name] = grid_search.best_estimator_
    models[classifier_name].set_params(**grid_search.best_params_)
    predictions = grid_search.predict(X_test)
    print(f"Model: {classifier_name}",flush=True)
    print(f"Best Parameters: {grid_search.best_params_}", flush=True)
    print(f"Accuracy: {accuracy_score(y_test, predictions)}",flush=True)
    print(f"Micro F1: {f1_score(y_test, predictions, average='micro')}", flush=True)
    print(f"Macro F1: {f1_score(y_test, predictions, average='macro')}", flush=True)
    print(f"Micro F2: {fbeta_score(y_test, predictions, beta=2, average='micro')}", flush=True)
    print(f"Macro F2: {fbeta_score(y_test, predictions, beta=2, average='macro')}", flush=True)
    print("--------------------------------")
print("\n\n Best models parameters")

print(best_models)