#!/usr/bin/env python
# coding: utf-8

# # Necessary Intstallations

# In[1]:


get_ipython().system('pip install torch')
get_ipython().system('pip install transformers')
get_ipython().system('pip install numpy')
get_ipython().system('pip install pandas')


# In[2]:


# the code works in Python 3.6.
get_ipython().system('pip install simpletransformers')


# In[5]:


get_ipython().system('pip uninstall pytorch')


# # BERT based Classification

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

# Load the training data
df= pd.read_csv('/labs/sarkerlab/srajwal/medical_tc_train_preprocessed.csv',usecols=['medical_abstract','condition_label'])
df['condition_label'] = df['condition_label']-1  # Making categories range between [0,4]

df_train, df_val = train_test_split(df, test_size=0.35, stratify=df['condition_label'], random_state=45)
df_train.reset_index(inplace=True)
df_val.reset_index(inplace=True)

# Load the test data
df_test = pd.read_csv('/labs/sarkerlab/srajwal/medical_tc_test_preprocessed.csv',usecols=['medical_abstract','condition_label'])
df_test['condition_label'] = df_test['condition_label']-1
df_test.reset_index(inplace=True)


# Drop additional columns
df_test.drop('index',axis=1,inplace=True)
df_train.drop('index',axis=1,inplace=True)
df_val.drop('index',axis=1,inplace=True)

df_train.rename(columns={'medical_abstract': 'text', 'condition_label': 'labels'}, inplace=True)
df_test.rename(columns={'medical_abstract': 'text', 'condition_label': 'labels'}, inplace=True)
df_val.rename(columns={'medical_abstract': 'text', 'condition_label': 'labels'}, inplace=True)


print(f'train set shape:{df_train.shape}, Validation set shape:{df_val.shape}, Test set shape:{df_test.shape}')
print("\nTraining set sample:")
print(df_train.sample(3))


# In[5]:


df_train['text'][:10]


# In[6]:


df_train.sample(2)


# In[7]:


df_test.sample(3)


# In[8]:


df_val.sample(3)


# ## Run a BERT-based classification model
# ### Training

# In[5]:


from simpletransformers.classification import ClassificationModel
# Initialize the model
model = ClassificationModel(
    "bert", 
    "bert-base-uncased", 
    num_labels=5, 
    use_cuda=True, 
    args={
        'max_seq_length': 512,
        'train_batch_size': 16,
        'overwrite_output_dir': True,
        'num_train_epochs':5,
        'evaluate_during_training': True,
        'evaluate_during_training_steps': 1000,
        'evaluate_during_training_verbose': True,
        'use_early_stopping': True,  # Enable early stopping
        'early_stopping_patience': 2,  # Number of epochs to wait for improvement
        'early_stopping_metric': 'eval_loss',  # Metric to monitor for early stopping
        'early_stopping_metric_minimize': True  # Set to True to minimize the metric
    }
)

# Train the model
model.train_model(df_train, eval_df=df_val)


# In[11]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, balanced_accuracy_score, classification_report, f1_score, log_loss, fbeta_score, roc_auc_score, roc_curve, auc, confusion_matrix
from sklearn.preprocessing import label_binarize

result, model_outputs, wrong_predictions = model.eval_model(df_test)
# Extract true labels and predicted labels
true_labels = df_test['labels']
predicted_labels = model_outputs.argmax(axis=1)

# Compute metrics
class_names = ['Neoplasms','Digestive System Diseases','Nervous System Diseases','Cardiovascular Diseases','General Pathological Conditions']

accuracy = accuracy_score(true_labels, predicted_labels)
balanced_accuracy = balanced_accuracy_score(true_labels, predicted_labels)
f1_micro = f1_score(true_labels, predicted_labels, average='micro')
f1_macro = f1_score(true_labels, predicted_labels, average='macro')
f2_micro = fbeta_score(true_labels, predicted_labels, beta=2, average='micro')
f2_macro = fbeta_score(true_labels, predicted_labels, beta=2, average='macro')
precision = precision_score(true_labels, predicted_labels, average='macro')
recall = recall_score(true_labels, predicted_labels, average='macro')
y_true_binarized = label_binarize(true_labels, classes=range(len(class_names)))
roc_auc = roc_auc_score(y_true_binarized, model_outputs, multi_class='ovr')
results = {'accuracy': accuracy,'balanced_accuracy': balanced_accuracy,
    'f1_micro': f1_micro,'f1_macro': f1_macro,
    'f2_micro': f2_micro,'f2_macro': f2_macro,'precision': precision,
    'recall': recall,'roc_auc': roc_auc}
print(results)

print("confusion matrix: ",confusion_matrix(true_labels,predicted_labels))
print('\tClassification Report for BERT:\n\n',classification_report(true_labels,predicted_labels,target_names=class_names))


# # ROBERTA

# In[ ]:                                                                                        'num_train_epochs': 10})


from simpletransformers.classification import ClassificationModel
model = ClassificationModel(
    "roberta", "roberta-base", 
    num_labels=5, 
    use_cuda=True, 
    args={
        'max_seq_length': 512,
        'train_batch_size': 16,
        'overwrite_output_dir': True,
        'output_dir':"outputs_roberta/",
        'num_train_epochs': 5,
        'evaluate_during_training': True,
        'evaluate_during_training_steps': 1000,
        'evaluate_during_training_verbose': True,
        'use_early_stopping': True,  
        'early_stopping_patience': 2, 
        'early_stopping_metric': 'eval_loss',
        'early_stopping_metric_minimize': True
    }
)

# Train the model
model.train_model(df_train, eval_df=df_val)


# In[13]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, balanced_accuracy_score, classification_report, f1_score, log_loss, fbeta_score, roc_auc_score, roc_curve, auc, confusion_matrix
from sklearn.preprocessing import label_binarize

result, model_outputs, wrong_predictions = model.eval_model(df_test)
# Extract true labels and predicted labels
true_labels = df_test['labels']
predicted_labels = model_outputs.argmax(axis=1)

# Compute metrics
class_names = ['Neoplasms','Digestive System Diseases','Nervous System Diseases','Cardiovascular Diseases','General Pathological Conditions']

accuracy = accuracy_score(true_labels, predicted_labels)
balanced_accuracy = balanced_accuracy_score(true_labels, predicted_labels)
f1_micro = f1_score(true_labels, predicted_labels, average='micro')
f1_macro = f1_score(true_labels, predicted_labels, average='macro')
f2_micro = fbeta_score(true_labels, predicted_labels, beta=2, average='micro')
f2_macro = fbeta_score(true_labels, predicted_labels, beta=2, average='macro')
precision = precision_score(true_labels, predicted_labels, average='macro')
recall = recall_score(true_labels, predicted_labels, average='macro')
y_true_binarized = label_binarize(true_labels, classes=range(len(class_names)))
roc_auc = roc_auc_score(y_true_binarized, model_outputs, multi_class='ovr')
results = {'accuracy': accuracy,'balanced_accuracy': balanced_accuracy,
    'f1_micro': f1_micro,'f1_macro': f1_macro,
    'f2_micro': f2_micro,'f2_macro': f2_macro,'precision': precision,
    'recall': recall,'roc_auc': roc_auc}
print(results)

print("confusion matrix: ",confusion_matrix(true_labels,predicted_labels))
print('\tClassification Report for ROBERTA:\n\n',classification_report(true_labels,predicted_labels,target_names=class_names))


# In[15]:


df_test['roberta_predictions'] = predicted_labels
df_test.to_csv('roberta_predictions.csv',index=False)