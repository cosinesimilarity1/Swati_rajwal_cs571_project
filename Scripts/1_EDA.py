import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from tqdm import trange
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

df_train = pd.read_csv('/local/scratch/shared-directories/ssanet/swati_folder/nlp_project/medical_tc_train.csv')
df_test = pd.read_csv('/local/scratch/shared-directories/ssanet/swati_folder/nlp_project/medical_tc_test.csv')

print(f"Train dataset with {df_train.shape} shape has {df_train.columns.values} columns")

print("Sample train dataframe:\n",df_train.sample(8))

print("Frequency count for each target label/class:\n")
print(df_train['condition_label'].value_counts())

# Creating a new column with the name of numeric labels for easy visulization
df_train['Condition name'] = df_train['condition_label']
df_train['Condition name'].replace({1: 'Neoplasms',
        2: 'Digestive System Diseases',
        3: 'Nervous System Diseases',
        4: 'Cardiovascular Diseases',
        5: 'General Pathological Conditions'},inplace=True)

df_test['Condition name'] = df_test['condition_label']
df_test['Condition name'].replace({1: 'Neoplasms',
        2: 'Digestive System Diseases',
        3: 'Nervous System Diseases',
        4: 'Cardiovascular Diseases',
        5: 'General Pathological Conditions'},inplace=True)

print("Frequency count for each target label/class:\n")
print(df_train['Condition name'].value_counts())

labels = df_train['Condition name'].unique().tolist()
sizes = df_train['Condition name'].value_counts()

# Creating a pie chart with enhanced styling
plt.figure(figsize=(8, 8))  # Set the figure size
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, explode=[0.1] * len(labels), shadow=True)

# Adding a title and customizing the font
plt.title('Distribution of Condition Labels', fontdict={'fontsize': 16, 'fontweight': 'bold'})
plt.savefig('/local/scratch/shared-directories/ssanet/swati_folder/nlp_project/figures/condition_labels_distributions.png')

# Display the chart
plt.show()


fig, (ax1, ax2) = plt.subplots(2, figsize=(10, 8))

# Set the spacing between subplots
fig.tight_layout(pad=6.0)
sns.despine()
# Plot Train Labels Distribution
ax1.set_title('Train Labels Distribution', fontsize=15)
train_distribution = df_train['Condition name'].value_counts().sort_values()
bar_plot = sns.barplot(x=train_distribution.values,
            y=list(train_distribution.keys()),
            orient="h", palette=sns.color_palette("winter", 5),
            ax=ax1)

for container in bar_plot.containers:
    bar_plot.bar_label(container, fmt='%.0f', fontsize=14);
    
# Plot Validation Labels Distribution
ax2.set_title('Test Labels Distribution', fontsize=15)
val_distribution = df_test['Condition name'].value_counts().sort_values()
bar_plot = sns.barplot(x=val_distribution.values,
            y=list(val_distribution.keys()),
            orient="h", palette=sns.color_palette("winter", 5),
            ax=ax2)
for container in bar_plot.containers:
    bar_plot.bar_label(container, fmt='%.0f', fontsize=14)

for container in bar_plot.containers:
    bar_plot.bar_label(container, fmt='%.0f', fontsize=14)

'''
Generating new features to understand the dataset
'''
# Length of abstract
df_train['Length'] = df_train['medical_abstract'].str.len()

# Number of words in an abstract
def word_count(review):
    review_list = review.split()
    return len(review_list)

df_train['Word_count'] = df_train['medical_abstract'].apply(word_count)

df_train.sample(3)

def visualize(col):
    plt.figure(figsize=(15, 6))  # Set the size of the figure

    # Boxplot
    plt.subplot(1, 2, 1)
    sns.boxplot(y=df_train[col], x=df_train['Condition name'])
    plt.ylabel(col, labelpad=12.5)
    plt.xticks(rotation=90)  # Rotate x-axis labels
    plt.grid()

    # KDE plot
    plt.subplot(1, 2, 2)
    sns.kdeplot(data=df_train, x=col, hue='Condition name')
    plt.xlabel(col, labelpad=12.5)
    plt.ylabel('Distribution')
    plt.xticks(rotation=90)  # Rotate x-axis labels
    plt.grid()

    # Adjust layout
    plt.tight_layout()

    # Save the figure before showing it
    plt.savefig(f'/local/scratch/shared-directories/ssanet/swati_folder/nlp_project/figures/{col}_visualization.png')

    plt.show()

# Example usage
features = ['Length', 'Word_count']
for feature in features:
    visualize(feature)

print(df_train.info())

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    filtered = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and word.isalpha()]
    return ' '.join(filtered)

df_train['medical_abstract'] = df_train['medical_abstract'].apply(clean)



def corpus(text):
    text_list = text.split()
    return text_list
df_train['medical_abstract_list'] = df_train['medical_abstract'].apply(corpus)

# Length of abstract
df_train['Length_after_preprocess'] = df_train['medical_abstract'].str.len()
# Number of words in an abstract
def word_count(review):
    review_list = review.split()
    return len(review_list)

df_train['Word_count_after_preprocess'] = df_train['medical_abstract'].apply(word_count)


# Example usage
features = ['Length_after_preprocess', 'Word_count_after_preprocess']
for feature in features:
    visualize(feature)

corpus = []
for i in trange(df_train.shape[0], ncols=150, nrows=10, colour='red', smoothing=0.8):
    corpus += df_train['medical_abstract_list'][i]
print("\nLength of the corpus based on medical abstracts in train dataset: ",len(corpus))

mostCommon = Counter(corpus).most_common(10)
words = []
freq = []
for word, count in mostCommon:
    words.append(word)
    freq.append(count)

sns.barplot(x=freq, y=words)
plt.title('Top 10 Most Frequently Occuring Words')
plt.grid()
plt.savefig('top_10_frequent_words_unigrams.png')
plt.show()

cv = CountVectorizer(ngram_range=(2,3))
bigrams = cv.fit_transform(df_train['medical_abstract'])
count_values = bigrams.toarray().sum(axis=0)
ngram_freq = pd.DataFrame(sorted([(count_values[i], k) for k, i in cv.vocabulary_.items()], reverse = True))
ngram_freq.columns = ["frequency", "ngram"]

sns.barplot(x=ngram_freq['frequency'][:10], y=ngram_freq['ngram'][:10])
plt.title('Top 10 Most Frequently Occuring Bigrams')
plt.grid()
plt.show()

cv1 = CountVectorizer(ngram_range=(3,3))
trigrams = cv1.fit_transform(df_train['medical_abstract'])
count_values = trigrams.toarray().sum(axis=0)
ngram_freq = pd.DataFrame(sorted([(count_values[i], k) for k, i in cv1.vocabulary_.items()], reverse = True))
ngram_freq.columns = ["frequency", "ngram"]

sns.barplot(x=ngram_freq['frequency'][:10], y=ngram_freq['ngram'][:10])
plt.title('Top 10 Most Frequently Occuring Trigrams')
plt.grid()
plt.savefig('top_10_frequent_words_bigrams.png')
plt.show()
