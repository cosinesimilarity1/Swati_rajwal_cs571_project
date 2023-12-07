import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, fbeta_score, precision_score, recall_score, roc_auc_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import label_binarize
import joblib

df_train = pd.read_csv('/local/scratch/shared-directories/ssanet/swati_folder/nlp_project/medical_tc_train_preprocessed.csv')
df_test= pd.read_csv('/local/scratch/shared-directories/ssanet/swati_folder/nlp_project/medical_tc_test_preprocessed.csv')

X_train = df_train['medical_abstract']
y_train = df_train['condition_label']

X_test = df_test['medical_abstract']
y_test = df_test['condition_label']

vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=5000)
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Adjusting the class labels in the target variable
y_train = y_train - 1
y_test = y_test - 1
n_classes = df_train['condition_label'].nunique()
y_train_bin = label_binarize(y_train, classes=range(n_classes))
y_test_bin = label_binarize(y_test, classes=range(n_classes))

# Define classifiers with hyperparameters
classifiers = {
    'Naive Bayes': MultinomialNB(alpha=0.01),
    'Logistic Regression': LogisticRegression(C=1, penalty='l2', solver='liblinear'),
    'Decision Tree': DecisionTreeClassifier(max_depth=10),
    'Random Forest': RandomForestClassifier(min_samples_split= 5,n_estimators=100),
    'LinearSVM': SGDClassifier(loss='hinge',penalty='l2',alpha=1e-3, random_state=42, max_iter=5, tol=None),
    'KNN': KNeighborsClassifier(n_neighbors=7),
    'GradientBoosting': GradientBoostingClassifier(n_estimators=200, learning_rate=0.01)
}

# Train classifiers and compute metrics
results = {}
for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)

    if name == 'LinearSVM':
        scores = clf.decision_function(X_test)  # Get decision scores for LinearSVM
    else:
        probas = clf.predict_proba(X_test)  # Get probabilities for other classifiers

    accuracy = accuracy_score(y_test, predictions)
    balanced_accuracy = balanced_accuracy_score(y_test, predictions)
    f1_micro = f1_score(y_test, predictions, average='micro')
    f1_macro = f1_score(y_test, predictions, average='macro')
    f2_micro = fbeta_score(y_test, predictions, beta=2, average='micro')
    f2_macro = fbeta_score(y_test, predictions, beta=2, average='macro')
    precision = precision_score(y_test, predictions, average='macro')
    recall = recall_score(y_test, predictions, average='macro')
    if name == 'LinearSVM':
        roc_auc = roc_auc_score(y_test_bin, scores, multi_class='ovr')  # Use scores for ROC AUC calculation for LinearSVM
    else:
        roc_auc = roc_auc_score(y_test_bin, probas, multi_class='ovr')  # Use probabilities for other classifiers

    results[name] = (accuracy, balanced_accuracy,f1_micro, f1_macro, f2_micro, f2_macro, precision, recall, roc_auc)

    # Generate and save classification report as an image
    print(f"\nClassification report for {name}")
    print(classification_report(y_test, predictions, target_names=[str(i) for i in range(n_classes)]))

    print(f"{name} - Accuracy: {accuracy},Balanced Accuracye:{balanced_accuracy}, F1 Micro: {f1_micro}, F1 Macro: {f1_macro}") 
    print(f"F2 Micro: {f2_micro}, F2 Macro: {f2_macro}, Precision: {precision}, Recall: {recall}, ROC_AUC: {roc_auc}", flush=True)
    joblib.dump(clf, f"/local/scratch/shared-directories/ssanet/swati_folder/nlp_project/trained_models/{name}_model.pkl")

# Save results to CSV
results_df = pd.DataFrame(results, index=['Accuracy','Balanced_accuracy' ,'F1 Micro', 'F1 Macro', 'F2 Micro', 'F2 Macro','Precision','Recall', 'ROC']).T
results_df.to_csv('/local/scratch/shared-directories/ssanet/swati_folder/nlp_project/model_results_5_december.csv')