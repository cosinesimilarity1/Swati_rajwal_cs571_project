import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import label_binarize
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, fbeta_score, precision_score, recall_score, roc_auc_score
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

df = pd.read_csv('/local/scratch/shared-directories/ssanet/swati_folder/nlp_project/medical_tc_train_preprocessed.csv')
test_df = pd.read_csv('/local/scratch/shared-directories/ssanet/swati_folder/nlp_project/medical_tc_test_preprocessed.csv')

train_df, val_df = train_test_split(df, test_size=0.3, random_state=42)
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)

# Convert abstract text lines into lists
train_sentences = train_df["medical_abstract"].tolist()
test_sentences = test_df["medical_abstract"].tolist()
val_sentences = val_df["medical_abstract"].tolist()

one_hot_encoder = OneHotEncoder(sparse=False)
train_labels_one_hot = one_hot_encoder.fit_transform(train_df["condition_label"].to_numpy().reshape(-1, 1))
joblib.dump(one_hot_encoder, '/local/scratch/shared-directories/ssanet/swati_folder/nlp_project/trained_models/one_hot_encoder.pkl')

val_labels_one_hot = one_hot_encoder.transform(val_df["condition_label"].to_numpy().reshape(-1, 1))
test_labels_one_hot = one_hot_encoder.transform(test_df["condition_label"].to_numpy().reshape(-1, 1))
# print(train_df["condition_label"][0])
# print(train_labels_one_hot)

num_classes = 5
class_names = ['Neoplasms','Digestive System Diseases','Nervous System Diseases','Cardiovascular Diseases','General Pathological Conditions']

def create_pipeline(features, labels, batch_size=32, shuffle=False, cache=False, prefetch=False) -> tf.data.Dataset:
    AUTOTUNE = tf.data.AUTOTUNE
    ds = tf.data.Dataset.from_tensor_slices((features, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=1000)
    ds = ds.batch(batch_size)
    if cache:
        ds = ds.cache(buffer_size=AUTOTUNE)
    if prefetch:
        ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds

BATCH_SIZE = 32

################ Creating each dataset's Input pipeline ###################
train_ds = create_pipeline(train_sentences, train_labels_one_hot.astype(np.float32),batch_size=BATCH_SIZE, shuffle=True,cache=False, prefetch=True)
val_ds = create_pipeline(val_sentences, val_labels_one_hot.astype(np.float32),batch_size=BATCH_SIZE, shuffle=False,cache=False, prefetch=True)
test_ds = create_pipeline(test_sentences, test_labels_one_hot.astype(np.float32), batch_size=BATCH_SIZE, shuffle=False,cache=False, prefetch=True)

encoder = hub.KerasLayer('https://tfhub.dev/google/universal-sentence-encoder/4',trainable=False,name='universal_sentence_encoder')

class SelfAttentionBlock(layers.Layer):
    def __init__(self, units: int, activation='relu', kernel_initializer='GlorotNormal', **kwargs):
        super(SelfAttentionBlock, self).__init__(**kwargs)
        self.units = units
        self.activation = activation
        self.kernel_initializer = tf.keras.initializers.deserialize(kernel_initializer)
        self.query = layers.LSTM(self.units, activation=self.activation,kernel_initializer=self.kernel_initializer,return_sequences=True)
        self.value = layers.LSTM(self.units, activation=self.activation,kernel_initializer=self.kernel_initializer, go_backwards=True,return_sequences=True)
        self.attention = layers.AdditiveAttention()
        self.average_pooler = layers.GlobalAveragePooling1D()
        self.query_batch_norm = layers.BatchNormalization()
        self.attention_batch_norm = layers.BatchNormalization()
        self.residual = layers.Add()

    def __call__(self, x):
        dim_expand_layer = layers.Lambda(lambda embedding: tf.expand_dims(embedding, axis=1))
        x_expanded = dim_expand_layer(x)
        block_query = self.query(x_expanded)
        block_value = self.value(x_expanded)
        block_attention = self.attention([block_query, block_value])
        block_query_pooling = self.average_pooler(block_query)
        block_query_batch_norm = self.query_batch_norm(block_query_pooling)
        block_attention_pooling = self.average_pooler(block_attention)
        block_attention_batch_norm = self.attention_batch_norm(block_attention_pooling)
        block_residual = self.residual([block_query_batch_norm, block_attention_batch_norm])
        return block_residual

def build_model():
    abstract_input = layers.Input(shape=[], dtype=tf.string)
    initializer = tf.keras.initializers.GlorotNormal()
    abstract_embedding = encoder(abstract_input)
    add_attention_block = SelfAttentionBlock(64)(abstract_embedding)
    abstract_dense = layers.Dense(64, activation='relu', kernel_initializer=initializer)(abstract_embedding)
    attention_residual = layers.Multiply(name='mul_residual')([add_attention_block, abstract_dense])
    output_layer = layers.Dense(64, activation='relu', kernel_initializer=initializer)(attention_residual)
    output_layer = layers.Dense(5, activation='softmax', kernel_initializer=initializer)(output_layer)
    return tf.keras.Model(inputs=[abstract_input], outputs=[output_layer], name="swati_attention_model")

model = build_model()
model.summary()
# plot_model(model,show_shapes=True,expand_nested=True)

def train_model(model, num_epochs, callbacks_list, tf_train_data, tf_valid_data=None, shuffling=False):
    model_history = {}
    if tf_valid_data != None:
        model_history = model.fit(tf_train_data,
                                  epochs=num_epochs,
                                  validation_data=tf_valid_data,
                                  validation_steps=int(len(tf_valid_data)),
                                  callbacks=callbacks_list,
                                  shuffle=shuffling)

    if tf_valid_data == None:
        model_history = model.fit(tf_train_data,
                                  epochs=num_epochs,
                                  callbacks=callbacks_list,
                                  shuffle=shuffling)
    return model_history

EPOCHS = 50
CALLBACKS = [tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=9,restore_best_weights=True), 
             tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',patience=6,factor=0.1,verbose=1)]
METRICS = ['accuracy']

tf.random.set_seed(42)

model.compile(loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              metrics=['accuracy'])

print(f'Training {model.name} started!!', flush=True)

model_history = train_model(model, EPOCHS, CALLBACKS,train_ds, val_ds,shuffling=False)

model.save('/local/scratch/shared-directories/ssanet/swati_folder/nlp_project/trained_models/trained_nn_model.h5')
print("MODEL (.h5) SAVED SUCCESSFULLY!!")
# model = tf.keras.models.load_model('/local/scratch/shared-directories/ssanet/swati_folder/nlp_project/trained_models/trained_nn_model.h5')
# model evaluation
print(f"Model evaluation on test set: {model.evaluate(test_ds)}", flush=True)

# generate validation probabilities
val_probabilities = model.predict(val_ds, verbose=1)
val_probabilities

# generate validation predictions with argmax
val_predictions = tf.argmax(val_probabilities, axis=1)
val_predictions

# generate test probabilities
test_probabilities = model.predict(test_ds, verbose=1)
# test_probabilities
test_predictions = tf.argmax(test_probabilities, axis=1).numpy()

# Add prediction results to test_df
test_df['NN_predicted_label'] = test_predictions
test_df['predicted_label_name'] = [class_names[label] for label in test_predictions]

# Save the modified test_df to a new CSV file
test_df.to_csv('/local/scratch/shared-directories/ssanet/swati_folder/nlp_project/medical_tc_test_predictions.csv', index=False)
print("Test predictions saved to CSV file.")

# generate test predictions with argmax
test_predictions = tf.argmax(test_probabilities, axis=1)
test_predictions

loss = np.array(model_history.history['loss'])
val_loss = np.array(model_history.history['val_loss'])
accuracy = np.array(model_history.history['accuracy'])
val_accuracy = np.array(model_history.history['val_accuracy'])
epochs = range(len(model_history.history['loss']))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
# Plot loss
ax1.plot(epochs, loss, label='training_loss', marker='o')
ax1.plot(epochs, val_loss, label='val_loss', marker='o')
ax1.fill_between(epochs, loss, val_loss, where=(loss > val_loss), color='C0', alpha=0.3, interpolate=True)
ax1.fill_between(epochs, loss, val_loss, where=(loss < val_loss), color='C1', alpha=0.3, interpolate=True)
ax1.set_title('Loss (Lower Means Better)', fontsize=16)
ax1.set_xlabel('Epochs', fontsize=12)
ax1.grid()
ax1.legend()
# Plot accuracy
ax2.plot(epochs, accuracy, label='training_accuracy', marker='o')
ax2.plot(epochs, val_accuracy, label='val_accuracy', marker='o')
ax2.fill_between(epochs, accuracy, val_accuracy, where=(accuracy > val_accuracy), color='C0', alpha=0.3, interpolate=True)
ax2.fill_between(epochs, accuracy, val_accuracy, where=(accuracy < val_accuracy), color='C1', alpha=0.3, interpolate=True)
ax2.set_title('Accuracy (Higher Means Better)', fontsize=16)
ax2.set_xlabel('Epochs', fontsize=12)
ax2.grid()
ax2.legend()
fig.savefig('/local/scratch/shared-directories/ssanet/swati_folder/nlp_project/figures/LSTM_results.png')


# Generate validation classification report
label_encoder = LabelEncoder()
train_labels_encoded = label_encoder.fit_transform(train_df["condition_label"].to_numpy())
val_labels_encoded = label_encoder.transform(val_df["condition_label"].to_numpy())
test_labels_encoded = label_encoder.transform(test_df["condition_label"].to_numpy())
print("Classification report for validation dataset:")
print(classification_report(val_labels_encoded, val_predictions, target_names=class_names))
print("Classification report for Testing dataset:")
print(classification_report(test_labels_encoded, test_predictions, target_names=class_names))

############ VALidation########
accuracy = accuracy_score(val_labels_encoded, val_predictions)
balanced_accuracy = balanced_accuracy_score(val_labels_encoded, val_predictions)
f1_micro = f1_score(val_labels_encoded, val_predictions, average='micro')
f1_macro = f1_score(val_labels_encoded, val_predictions, average='macro')
f2_micro = fbeta_score(val_labels_encoded, val_predictions, beta=2, average='micro')
f2_macro = fbeta_score(val_labels_encoded, val_predictions, beta=2, average='macro')
precision = precision_score(val_labels_encoded, val_predictions, average='macro')
recall = recall_score(val_labels_encoded, val_predictions, average='macro')
y_true_binarized = label_binarize(val_labels_encoded, classes=range(5))
roc_auc = roc_auc_score(y_true_binarized, val_probabilities, multi_class='ovr')
results = {'accuracy': accuracy,'balanced_accuracy': balanced_accuracy,
    'f1_micro': f1_micro,'f1_macro': f1_macro,
    'f2_micro': f2_micro,'f2_macro': f2_macro,'precision': precision,
    'recall': recall,'roc_auc': roc_auc}
print(results)

########### Test set ####################
accuracy = accuracy_score(test_labels_encoded, test_predictions)
balanced_accuracy = balanced_accuracy_score(test_labels_encoded, test_predictions)
f1_micro = f1_score(test_labels_encoded, test_predictions, average='micro')
f1_macro = f1_score(test_labels_encoded, test_predictions, average='macro')
f2_micro = fbeta_score(test_labels_encoded, test_predictions, beta=2, average='micro')
f2_macro = fbeta_score(test_labels_encoded, test_predictions, beta=2, average='macro')
precision = precision_score(test_labels_encoded, test_predictions, average='macro')
recall = recall_score(test_labels_encoded, test_predictions, average='macro')
y_true_binarized = label_binarize(test_labels_encoded, classes=range(5))
roc_auc = roc_auc_score(y_true_binarized, test_probabilities, multi_class='ovr')
results = {'accuracy': accuracy,'balanced_accuracy': balanced_accuracy,
    'f1_micro': f1_micro,'f1_macro': f1_macro,
    'f2_micro': f2_micro,'f2_macro': f2_macro,'precision': precision,
    'recall': recall,'roc_auc': roc_auc}
print(results)

cm = confusion_matrix(test_labels_encoded, test_predictions)
plt.figure(figsize=(10, 10))
disp = sns.heatmap(cm, annot=True, cmap='Blues',annot_kws={"size": 13},
    fmt='g',linewidths=0.5, linecolor='black', clip_on=False,
    xticklabels=list(class_names) if list(class_names) != 'auto' else ["Class " + str(i) for i in range(cm.shape[0])],
    yticklabels=list(class_names) if list(class_names) != 'auto' else ["Class " + str(i) for i in range(cm.shape[0])]
)

# Set title and axis labels
plt.title('Confusion Matrix', fontsize=24, pad=20)
plt.xlabel('Predicted Label', fontsize=20)
plt.ylabel('True Label', fontsize=20)

# Adjust the position of the labels
plt.xticks(rotation=45, ha='right', fontsize=13)
plt.yticks(rotation=0, fontsize=13)
plt.savefig('/local/scratch/shared-directories/ssanet/swati_folder/nlp_project/figures/LSTM_Test_confusion_matrix.png')
print("Confusion matrix saved!!")