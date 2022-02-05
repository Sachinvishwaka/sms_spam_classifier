#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df=pd.read_csv("spam.csv")
df.head()


# In[3]:


df.shape


# In[4]:


# 1.Data cleaning
# 2.EDA
# 3.Text preprocessing
# 4.MOdel Building
# 5.Evaluation
# 6.Improvement
# 7.Website
# 8.Deploy


# # 1.Data Cleaning

# In[5]:


df.info()


# In[6]:


# drop last 3 columns
df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace=True)


# In[7]:


df.sample(5)


# In[8]:


# rename the columns
df.rename(columns={'v1':'target','v2': 'text'},inplace=True)
df.sample(5)


# In[9]:


from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()


# In[10]:


df['target']=encoder.fit_transform(df['target'])


# In[11]:


df.head()


# In[12]:


# missing values
df.isnull().sum()


# In[13]:


# check for duplicate values
df.duplicated().sum()


# In[14]:


# remove duplicates
df=df.drop_duplicates(keep='first')


# In[15]:


df.duplicated().sum()


# In[16]:


df.shape


# # 2.EDA

# In[17]:


df.head()


# In[18]:


df['target'].value_counts()


# In[19]:


import matplotlib.pyplot as plt
plt.pie(df['target'].value_counts(),labels=['ham','spam'],autopct="%.02f")
plt.show()


# In[20]:


# Data is Imbalanced


# In[21]:


import nltk
# if it is not in you library then u have to download this library from the following code
#pip install nltk


# In[22]:


nltk.download('punkt')


# In[23]:


df['num_character']=df['text'].apply(len)


# In[24]:


df.head()


# In[25]:


# num of words
df['num_words']=df['text'].apply(lambda x:len(nltk.word_tokenize(x)))


# In[26]:


df.head()


# In[27]:


df['num_sentences']=df['text'].apply(lambda x:len(nltk.sent_tokenize(x)))


# In[28]:


df.head()


# In[29]:


df[['num_character','num_words','num_sentences']].describe()


# In[30]:


# hamm
df[df['target']==0][['num_character','num_words','num_sentences']].describe()


# In[31]:


# spam
df[df['target']==1][['num_character','num_words','num_sentences']].describe()


# In[32]:


import seaborn as sns


# In[33]:


plt.figure(figsize=(12,6))
sns.histplot(df[df['target']==0]['num_character'])
sns.histplot(df[df['target']==1]['num_character'],color='red')


# In[34]:


plt.figure(figsize=(12,6))
sns.histplot(df[df['target']==0]['num_words'])
sns.histplot(df[df['target']==1]['num_words'],color='red')


# In[35]:


sns.pairplot(df,hue='target')


# In[36]:


sns.heatmap(df.corr(),annot=True)


# # 3. Data Preprocessing
(1).Lower Case
(2).Tokenization
(3).Removing Special Charaters
(4).Removing stop words and punctuations
(5).Stemming
# In[37]:


def transform_text(text):
    text=text.lower()
    text=nltk.word_tokenize(text)
    return text


# In[38]:


transform_text("Hi How Are You")


# In[39]:


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    
    text = y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
            
    text = y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))
    
            
    return " ".join(y)


# In[50]:


transform_text("I'm gonna be home soon and i don't want to talk about this stuff anymore tonight, k? I've cried enough today.")


# In[48]:


nltk.download('stopwords')


# In[49]:


from nltk.corpus import stopwords
stopwords.words('english')


# In[41]:


import string 
string.punctuation


# In[42]:


df['text'][10]


# In[43]:


from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
ps.stem('loving')


# In[51]:


df['text'].apply(transform_text)


# In[52]:


df['transformed_text'] = df['text'].apply(transform_text)


# In[53]:


df.head()


# In[54]:


from wordcloud import WordCloud
wc = WordCloud(width=500,height=500,min_font_size=10,background_color='white')


# In[55]:


spam_wc = wc.generate(df[df['target'] == 1]['transformed_text'].str.cat(sep=" "))


# In[56]:


plt.figure(figsize=(15,6))
plt.imshow(spam_wc)


# In[57]:


ham_wc = wc.generate(df[df['target'] == 0]['transformed_text'].str.cat(sep=" "))


# In[58]:


plt.figure(figsize=(15,6))
plt.imshow(ham_wc)


# In[59]:


df.head()


# In[60]:


spam_corpus = []
for msg in df[df['target'] == 1]['transformed_text'].tolist():
    for word in msg.split():
        spam_corpus.append(word)
        


# In[61]:


len(spam_corpus)


# In[62]:


from collections import Counter
sns.barplot(pd.DataFrame(Counter(spam_corpus).most_common(30))[0],pd.DataFrame(Counter(spam_corpus).most_common(30))[1])
plt.xticks(rotation='vertical')
plt.show()


# In[63]:


ham_corpus = []
for msg in df[df['target'] == 0]['transformed_text'].tolist():
    for word in msg.split():
        ham_corpus.append(word)


# In[64]:


len(ham_corpus)


# In[65]:


from collections import Counter
sns.barplot(pd.DataFrame(Counter(ham_corpus).most_common(30))[0],pd.DataFrame(Counter(ham_corpus).most_common(30))[1])
plt.xticks(rotation='vertical')
plt.show()


# In[66]:


# Text Vectorization
# using Bag of Words
df.head()


# # 4. Model Building

# In[67]:


get_ipython().system('pip install sklearn')


# In[68]:


from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
cv = CountVectorizer()
tfidf = TfidfVectorizer(max_features=3000)


# In[69]:


#from sklearn.preprocessing import MinMaxScaler
#scaler = MinMaxScaler()
#X = scaler.fit_transform(X)


# In[70]:


X = tfidf.fit_transform(df['transformed_text']).toarray()


# In[71]:


# appending the num_character col to X
#X = np.hstack((X,df['num_characters'].values.reshape(-1,1)))


# In[72]:


X.shape


# In[73]:


y = df['target'].values


# In[74]:


from sklearn.model_selection import train_test_split


# In[75]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)


# In[76]:


from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score


# In[77]:


gnb = GaussianNB()
mnb = MultinomialNB()
bnb = BernoulliNB()


# In[80]:


gnb.fit(X_train,y_train)
y_pred1 = gnb.predict(X_test)
print(accuracy_score(y_test,y_pred1))
print(confusion_matrix(y_test,y_pred1))
print(precision_score(y_test,y_pred1))


# In[81]:


mnb.fit(X_train,y_train)
y_pred2 = mnb.predict(X_test)
print(accuracy_score(y_test,y_pred2))
print(confusion_matrix(y_test,y_pred2))
print(precision_score(y_test,y_pred2))


# In[82]:


bnb.fit(X_train,y_train)
y_pred3 = bnb.predict(X_test)
print(accuracy_score(y_test,y_pred3))
print(confusion_matrix(y_test,y_pred3))
print(precision_score(y_test,y_pred3))


# In[83]:


# tfidf --> MNB


# In[84]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier


# In[85]:


svc = SVC(kernel='sigmoid', gamma=1.0)
knc = KNeighborsClassifier()
mnb = MultinomialNB()
dtc = DecisionTreeClassifier(max_depth=5)
lrc = LogisticRegression(solver='liblinear', penalty='l1')
rfc = RandomForestClassifier(n_estimators=50, random_state=2)
abc = AdaBoostClassifier(n_estimators=50, random_state=2)
bc = BaggingClassifier(n_estimators=50, random_state=2)
etc = ExtraTreesClassifier(n_estimators=50, random_state=2)
gbdt = GradientBoostingClassifier(n_estimators=50,random_state=2)
xgb = XGBClassifier(n_estimators=50,random_state=2)


# In[86]:


clfs = {
    'SVC' : svc,
    'KN' : knc, 
    'NB': mnb, 
    'DT': dtc, 
    'LR': lrc, 
    'RF': rfc, 
    'AdaBoost': abc, 
    'BgC': bc, 
    'ETC': etc,
    'GBDT':gbdt,
    'xgb':xgb
}


# In[87]:


def train_classifier(clf,X_train,y_train,X_test,y_test):
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred)
    precision = precision_score(y_test,y_pred)
    
    return accuracy,precision


# In[88]:


train_classifier(svc,X_train,y_train,X_test,y_test)


# In[91]:


accuracy_scores = []
precision_scores = []

for name,clf in clfs.items():
    
    current_accuracy,current_precision = train_classifier(clf, X_train,y_train,X_test,y_test)
    
    print("For ",name)
    print("Accuracy - ",current_accuracy)
    print("Precision - ",current_precision)
    
    accuracy_scores.append(current_accuracy)
    precision_scores.append(current_precision)


# In[92]:


performance_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy':accuracy_scores,'Precision':precision_scores}).sort_values('Precision',ascending=False)


# In[93]:


performance_df


# In[94]:


performance_df1 = pd.melt(performance_df, id_vars="Algorithm")


# In[95]:


performance_df1


# In[96]:


sns.catplot(x = 'Algorithm', y='value', 
               hue = 'variable',data=performance_df1, kind='bar',height=5)
plt.ylim(0.5,1.0)
plt.xticks(rotation='vertical')
plt.show()


# In[97]:


# model improve
# 1. Change the max_features parameter of TfIdf


# In[98]:


temp_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy_max_ft_3000':accuracy_scores,'Precision_max_ft_3000':precision_scores}).sort_values('Precision_max_ft_3000')


# In[99]:


temp_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy_scaling':accuracy_scores,'Precision_scaling':precision_scores}).sort_values('Precision_scaling',ascending=False)


# In[100]:


new_df = performance_df.merge(temp_df,on='Algorithm')


# In[101]:


new_df_scaled = new_df.merge(temp_df,on='Algorithm')


# In[102]:


temp_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy_num_chars':accuracy_scores,'Precision_num_chars':precision_scores}).sort_values('Precision_num_chars',ascending=False)


# In[103]:


new_df_scaled.merge(temp_df,on='Algorithm')


# In[104]:


# Voting Classifier
svc = SVC(kernel='sigmoid', gamma=1.0,probability=True)
mnb = MultinomialNB()
etc = ExtraTreesClassifier(n_estimators=50, random_state=2)

from sklearn.ensemble import VotingClassifier


# In[105]:


from sklearn.naive_bayes import MultinomialNB


# In[106]:


voting = VotingClassifier(estimators=[('svm', svc), ('nb', mnb), ('et', etc)],voting='soft')


# In[107]:


voting.fit(X_train,y_train)


# In[108]:


y_pred = voting.predict(X_test)
print("Accuracy",accuracy_score(y_test,y_pred))
print("Precision",precision_score(y_test,y_pred))


# In[109]:


# Applying stacking
estimators=[('svm', svc), ('nb', mnb), ('et', etc)]
final_estimator=RandomForestClassifier()


# In[110]:


from sklearn.ensemble import StackingClassifier


# In[111]:


clf = StackingClassifier(estimators=estimators, final_estimator=final_estimator)


# In[115]:


clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Accuracy",
      accuracy_score(y_test,y_pred))
print("Precision",precision_score(y_test,y_pred))


# In[116]:


import pickle
pickle.dump(tfidf,open('vectorizer.pkl','wb'))
pickle.dump(mnb,open('model.pkl','wb'))


# In[ ]:




