
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

data = pd.read_csv("../input/mercari/train.tsv", sep='\t')
import matplotlib.pyplot as plt
import seaborn as sns

# # Exploratory Data Analysis (EDA)
##################commented EDA in this script
"""
stats = []
for col in data.columns:
    stats.append((col,data[col].nunique(), data[col].isnull().sum(), data[col].isnull().sum() * 100 / data.shape[0], data[col].dtype))
    
stats_df = pd.DataFrame(stats, columns=['Feature','Unique_values','null count', 'Percentage of missing values', 'type'])
# gives a table containg 'Unique_values','null count', 'Percentage of missing values', 'type' for each feature
stats_df

data['category_name'].fillna('missing', inplace = True) # filling the missing values

data['brand_name'].fillna('missing', inplace = True)

data['item_description'].fillna('missing', inplace = True)
# we found that coulms category_name brand name and description have null values so we have filled those with missing

plt.hist(data.price, bins = 100)
plt.show() #### highly skewed


data['log']=np.log(data['price']+1)
plt.hist(data.log, bins = 100)
plt.show()  ####removing skewness in price


shipping = data[data['shipping']==1]['log']
no_shipping = data[data['shipping']==0]['log']

plt.figure(figsize=(12,7))
plt.hist(shipping, bins=50, range=[0,10], alpha=0.7, label='Price With Shipping')
plt.hist(no_shipping, bins=50, range=[0,10], alpha=0.7, label='Price With No Shipping')
plt.title('Price Distrubtion With/Without Shipping', fontsize=15)
plt.xlabel('Price')
plt.ylabel('Normalized Samples')
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(fontsize=15)
plt.show()

plt.hist(data.shipping, bins = 100)
plt.xlabel('shipping')
plt.show()

# this is how price varies accourding to shipping

sns.barplot(x = data.item_condition_id, y = data.price )
# as expected best condition has better price.

plt.hist(data.item_condition_id, bins = 100)
plt.show()
"""
#function to split the category into 3 sub categorys
def catsp(col):
    try:
        inp=col
        out1,out2,out3=inp.split('/')
        return out1,out2,out3
    except:
        return("Not given","Not given","Not given")

data['sub1'],data['sub2'],data['sub3']=zip(*data['category_name'].apply(lambda x:catsp(x)))

# we have seen in the data.describe() that catogery_name has 3 sub catogrys so we have split them and filling the empty with not given.

"""
data.groupby('sub1')['price'].mean().plot(kind='bar')
plt.show() # we can see how price is varying in this sub category.


data.groupby("brand_name")["price"].sum().sort_values(ascending=False)


data.groupby("brand_name")["price"].mean().sort_values(ascending=False)
# we have observed that nike has the higest sales. but it isnt in the top 5 of the mean price. 


b20 = data['brand_name'].value_counts()[0:20].reset_index().rename(columns={'index': 'brand_name', 'brand_name':'count'})
ax = sns.barplot(x="brand_name", y="count", data=b20)
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
ax.set_title('Top 20 Brand Distribution', fontsize=15)
plt.show()

# Display Top 20 Expensive Brands By Mean Price
top20_brand = data.groupby('brand_name', axis=0).median()
df_expPrice = pd.DataFrame(top20_brand.sort_values('price', ascending = False)['price'][0:20].reset_index())
ax = sns.barplot(x="brand_name", y="price", data=df_expPrice)
ax.set_xticklabels(ax.get_xticklabels(),rotation=90, fontsize=15)
ax.set_title('Top 20 Expensive Brand', fontsize=15)
plt.show()

# graphical representation of above data
"""
# # Sentiment Analyzer

# for the item decription we are using a function SentimentIntensityAnalyzer() which gives a sentimantal analysis of the description. so that we get to know which descripition is good then it must be a good product and will have a better price. this function will analyize the text in four differnt formats namely negitive, possitive,neutral, compound.

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from tqdm import tnrange,tqdm_notebook

def generate_sentiment_scores(data):  
    sid = SentimentIntensityAnalyzer()
    scores = []
    for sentence in tqdm_notebook(data):
        for_senti=sentence
        sentence_sentiment_score = sid.polarity_scores(for_senti)
        scores.append(sentence_sentiment_score)
    return scores
# the above function analyses the given text and return 4 values as a dictionary

sentimental_names1 = generate_sentiment_scores(data['item_description'])#calling the sentiment analyzer

data['sentiment'] = sentimental_names1 #this coulum has a dicitonary we need to split them to different colums

## converting dictionary into diff coulums
temp=data['sentiment']
dt0 = list(temp.items())
an_array0 = np.array(dt0)
aa20={}
for i in range(0,1037774):
    aa20[i]=an_array0[i][1]

aan20=list(range(0,1037774))
aap20=list(range(0,1037774))
aanu20=list(range(0,1037774))
aac20=list(range(0,1037774))

for i in range(0,1037774):
    aan20[i]=aa20[i]['neg']         #splitting the dictonary into four different lists
    aap20[i]=aa20[i]['pos']
    aanu20[i]=aa20[i]['neu']
    aac20[i]=aa20[i]['compound']

data['negg']=aan20   #assigning lists to the respective colums 
data['poss']=aap20
data['comp']=aac20
data['neu']=aanu20


# # Text Preprocessing

from string import punctuation
punctuation_symbols = []
for symbol in punctuation: # intializing punctuation symbols
    punctuation_symbols.append((symbol, ''))


import re
def decontracted(phrase):  # function that  decontracting contracted strings
 
    # specific
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase

def text_preprocess(text):
    text = decontracted(text)
    text = re.sub("[\-\\\n\t]", " ", text)  #Regex to remove all \n, \t, - and \
    text = re.sub("[^A-Za-z0-9]", " ", text)  #Regex to remove all the words except A-Za-z0-9
    text = re.sub('\s\s+', ' ', str(text))  #Regex to remove all the extra spaces
    text = text.lower() #Converts everything to lower case
    return text


from nltk.stem.porter import PorterStemmer
porter = PorterStemmer() #function used for stemming

import string #function to rempove punctuation
def remove_punctuation(sentence: str) -> str:
    return sentence.translate(str.maketrans('', '', string.punctuation))

from nltk.corpus import stopwords
stop = stopwords.words('english')
def remove_stop_words(x):   #function to remove stopwords
    x = ' '.join([i for i in x.lower().split(' ') if i not in stop])
    return x

#text preprocessing for item_description 
data['item_description'] = data['item_description'].apply(remove_stop_words)
data['item_description'] = data['item_description'].apply(remove_punctuation) 
data['item_description'] = data['item_description'].apply(porter.stem)
data['item_desc_preprocess'] = data['item_description'].apply(lambda x : text_preprocess(x))

# text preporcessing for name
data['name']=data['name'].apply(remove_stop_words)
data['name']=data['name'].apply(remove_punctuation)
data['name'] = data['name'].apply(porter.stem)
data['name_preprocess'] = data['name'].apply(lambda x : text_preprocess(x))

# we also felt that length of the item decription and name plays a role in price hence we calculated it.

no_desc_string = 'missing'  
def text_length(text, no_desc_string): # function to calculate length of the text ie, no fo words
    try:
        if text in no_desc_string:
            return 0
        else:
            return len(text.split())
    except:
        return 0

#calcualting length of the pre processed text of item_description and name 
data['item_pre_length'] = data['item_desc_preprocess'].apply(lambda x : text_length(x, no_desc_string))
data['namepre_length'] = data['name_preprocess'].apply(lambda x : text_length(x, no_desc_string))


"""
##below pic shows which are the words with high freqency. bigger size means higher its freqency.

from wordcloud import WordCloud
comment_words = '' 
for val in data.item_desc_preprocess: 
      
    # typecaste each val to string 
    val = str(val) 
  
    # split the value 
    tokens = val.split() 
      
    # Converts each token into lowercase 
    for i in range(len(tokens)): 
        tokens[i] = tokens[i].lower() 
      
    comment_words += " ".join(tokens)+" "
  
wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                min_font_size = 10).generate(comment_words) 
  
# plot the WordCloud image                        
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
  
plt.show()  
"""


# # Feature Extraction

# below is the code for how we implimented label encoding for sub category 1

initial_sub1 = data.groupby('sub1', axis=0).median()
sub1_inorder1 = pd.DataFrame(initial_sub1.sort_values('price', ascending = False)['price'].reset_index())
sf1=sub1_inorder1["sub1"] #sf1 is a list that contains them in desending median price

valsub1=list(range(0,11))
dicts1=dict(zip(sf1, valsub1)) ### forms a dictionary for list in the same order

def funbs1(x): # function that impliments label encoding
    try:
        return dicts1[x]
    except:
        xxx=5
        return xxx 
data['sub1v']=data['sub1'].apply(lambda x : funbs1(x)) # label encoding

# same process will be followed for sub2 sub3 and brand_name. as the results are not upto the mark we choose to shift to one-hot encoding.


# # One-hot encoding for *sub1,sub2,sub3,brand_name*

from sklearn import preprocessing
bbb=np.reshape(np.array(data.brand_name), (-1,1))
sss1=np.reshape(np.array(data.sub1), (-1,1))
sss2=np.reshape(np.array(data.sub2), (-1,1))
sss3=np.reshape(np.array(data.sub3), (-1,1))

onehot=preprocessing.OneHotEncoder(handle_unknown='ignore')
bran=onehot.fit_transform(bbb)  #one hot encoding for brand name

onehot1=preprocessing.OneHotEncoder(handle_unknown='ignore')
su1=onehot1.fit_transform(sss1)    # #one hot encoding for sub1

onehot2=preprocessing.OneHotEncoder(handle_unknown='ignore')
su2=onehot2.fit_transform(sss2)   ##one hot encoding for sub2

onehot3=preprocessing.OneHotEncoder(handle_unknown='ignore')
su3=onehot3.fit_transform(sss3) #one hot encoding for sub3


# # tfidf and bag of words

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer1 = TfidfVectorizer(ngram_range=(1,3),min_df=10,max_features=500000).fit(data['item_desc_preprocess'])
data_desc_tfidf = vectorizer1.transform(data['item_desc_preprocess']) #tfidf 


from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(ngram_range=(1,2),max_features=150000).fit(data['name_preprocess'])
data_name = vectorizer.transform(data['name_preprocess']) # bag of words


feature_cols=['item_condition_id','shipping','item_pre_length', 'namepre_length', 'negg', 'poss', 'comp', 'neu']

teemp=data[feature_cols] # finding corealtion between features
plt.figure(figsize=(20,20))
sns.heatmap(teemp.corr(), vmin=-1, cmap="coolwarm", annot=True)

from scipy.sparse import vstack, hstack, csr_matrix
xt = csr_matrix(pd.get_dummies(data[feature_cols], sparse=True).values) # converting normal data into matrix format
sparse_merge = hstack((xt,data_name, data_desc_tfidf,bran,su1,su2,su3)).tocsr() # combinind it with leftout features
# this is our x_trian


y_train=data['log'] #target variable
y_train=np.reshape(np.array(data.log), (-1,1))
#y_train


# # Test data
testdata = pd.read_csv("../input/mercari/test.tsv", sep='\t')

""" EDA
stats = []
for col in testdata.columns:
    stats.append((col,testdata[col].nunique(), testdata[col].isnull().sum(), testdata[col].isnull().sum() * 100 / testdata.shape[0], testdata[col].dtype))
    
stats_df = pd.DataFrame(stats, columns=['Feature','Unique_values','null count', 'Percentage of missing values', 'type'])
stats_df
# gives a table containg 'Unique_values','null count', 'Percentage of missing values', 'type' for each feature
"""

testdata['category_name'].fillna('missing', inplace = True) #filling null values
testdata['item_description'].fillna('missing', inplace = True)
testdata['brand_name'].fillna('missing', inplace = True)

#splliting category_name
testdata['sub1'],testdata['sub2'],testdata['sub3']=zip(*testdata['category_name'].apply(lambda x:catsp(x)))


# # Sentiment Analyzer for test data
sentimental_names2 = generate_sentiment_scores(testdata['item_description'])#calling the sentiment analyzer

testdata['sentiment'] = sentimental_names2 #this coulum has a dicitonary we need to split them to different colums
tempo=testdata['sentiment']
dt = list(tempo.items())
an_array = np.array(dt)
aa2={}
for i in range(0,444761):
    aa2[i]=an_array[i][1]

aan2=list(range(0,444761))
aap2=list(range(0,444761))
aanu2=list(range(0,444761))
aac2=list(range(0,444761))

for i in range(0,444761):
    aan2[i]=aa2[i]['neg']         #splitting the dictonary into four different lists
    aap2[i]=aa2[i]['pos']
    aanu2[i]=aa2[i]['neu']
    aac2[i]=aa2[i]['compound']

testdata['negg']=aan2   #assigning lists to the respective colums 
testdata['poss']=aap2
testdata['comp']=aac2
testdata['neu']=aanu2

# # Text Preprocessing for test data

#text preprocessing for item_description of testdata
testdata['item_description'] = testdata['item_description'].apply(remove_stop_words)
testdata['item_description'] = testdata['item_description'].apply(remove_punctuation) 
testdata['item_description'] = testdata['item_description'].apply(porter.stem)
testdata['item_desc_preprocess'] = testdata['item_description'].apply(lambda x : text_preprocess(x))

# text preporcessing for name of testdata
testdata['name']=testdata['name'].apply(remove_stop_words)
testdata['name']=testdata['name'].apply(remove_punctuation)
testdata['name'] = testdata['name'].apply(porter.stem)
testdata['name_preprocess'] = testdata['name'].apply(lambda x : text_preprocess(x))

#calcualting length of the pre processed text of item_description and name of testdata
testdata['item_pre_length'] = testdata['item_desc_preprocess'].apply(lambda x : text_length(x, no_desc_string))
testdata['namepre_length'] = testdata['name_preprocess'].apply(lambda x : text_length(x, no_desc_string))

# # Feature Extraction

### one hot encoding for catogorical features
tbbb=np.reshape(np.array(testdata.brand_name), (-1,1))
tsss1=np.reshape(np.array(testdata.sub1), (-1,1))
tsss2=np.reshape(np.array(testdata.sub2), (-1,1))
tsss3=np.reshape(np.array(testdata.sub3), (-1,1))


tbran=onehot.transform(tbbb) # one hot encoding of brand
tsu1=onehot1.transform(tsss1)  # one hot encoding of sub1
tsu2=onehot2.transform(tsss2) # one hot encoding of sub2
tsu3=onehot3.transform(tsss3)  # one hot encoding of sub3


testdata_desc_tfidf = vectorizer1.transform(testdata['item_desc_preprocess'])# tfidf on item_description
testdata_name = vectorizer.transform(testdata['name_preprocess'])# bag of words on name

txt = csr_matrix(pd.get_dummies(testdata[feature_cols], sparse=True).values)  # converting normal data into matrix format

tsparse_merge = hstack((txt,testdata_name, testdata_desc_tfidf,tbran,tsu1,tsu2,tsu3)).tocsr() # combinind it with leftout features
#this is our x_test


# # Regression

########### code used for linear regression ########

#import sklearn
#from sklearn.linear_model import LinearRegression
#lm = LinearRegression()
#lm.fit(sparse_merge, y_train)
#c_pred = lm.intercept_
#c_pred = lm.intercept_

#y_pred = lm.predict(tsparse_merge)
#y_pred


#####code used to find the best alpha for ridge regression##########

#from sklearn.linear_model import Ridge 
#from sklearn.model_selection import GridSearchCV

#ridge=Ridge() 
#parameters={'alpha':[0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7]} 
#ridge_regressor=GridSearchCV(ridge,parameters,scoring='neg_mean_squared_error',cv=5) 
#ridge_regressor.fit(sparse_merge, y_train)

#print(ridge_regressor.best_params_) 
#print(ridge_regressor.best_score_)

##### we got alpha=5.5 for normal ridge regression #### 

#we got alpha=2 for ensembling with lgbm 

# code for ridge regression
from sklearn.linear_model import Ridge
model = Ridge(alpha =2)  ### for ensembling we got best acuuracy for alpha=2
model.fit(sparse_merge, y_train)
preds= model.predict(tsparse_merge)
preds


###########code used for Xgboost#######
#from xgboost import XGBRegressor
#from sklearn.model_selection import GridSearchCV

#params = { 
#          'gamma':[i/10.0 for i in range(3,8,2)],  
#          'max_depth': [4,8,16]}

#xgb = XGBRegressor() 

#grid = GridSearchCV(estimator=xgb, param_grid=params, n_jobs=-1, cv=2, verbose=3)
#grid.fit(sparse_merge, y_train)
#print("Best estimator : ", grid.best_estimator_)
#print("Best Score : ", grid.best_score_)

#xgb = grid.best_estimator_

#print("Fitting Model 1")
#xgb.fit(sparse_merge, y_train)
#y_predxx = xgb.predict(tsparse_merge)

##### As xgboost didnt produce good accuracies we shifted to lgbm   ####


# code for lgbm
import lightgbm as lgbmm
lgReg = lgbmm.LGBMRegressor(n_estimators=4000,learning_rate=0.4,
        max_depth= 9,
        num_leaves=64) 
lgReg.fit(sparse_merge, y_train)
sip=lgReg.predict(tsparse_merge)


testdata['ridge']=preds
testdata['lgbm']=sip # assigning the predicted values to a column

testdata['price3']=(0.55*testdata['lgbm'])+(0.45*testdata['ridge']) # combining both predictions

testdata['price']= np.exp(testdata['price3']) -1 # as we have predicted log value of the price now we are chancging to normal price


testdata[['id','price']].to_csv('output.csv', index=False) # creating a output in csv format

out1=pd.read_csv('./output.csv') out1.describe() # checking the output
out1.head()

# # finding weights for ensembling

y1train=data['log']

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

#splitting the train data randomly to find the best weights for ensembling
X0_train, X0_test, y0_train, y0_test = train_test_split(sparse_merge, y1train,test_size=0.33, random_state=42)

#fitting lgbm with the same hyper parameters used for the test data of the computation.
import lightgbm as lgbmm
lgReg = lgbmm.LGBMRegressor(n_estimators=4000,learning_rate=0.4,max_depth= 9,num_leaves=64) 
lgReg.fit(X0_train,y0_train,eval_set=[(X0_test,y0_test)],early_stopping_rounds=100,eval_metric='rmse')
sp=lgReg.predict(X0_test)

print(r2_score(y0_test,sp)) 


#fitting ridge regression with the same hyper parameters used for the test data of the competetion.
from sklearn.linear_model import Ridge
model = Ridge(alpha =2)
model.fit(X0_train,y0_train)
y11=model.predict(X0_test)

print(r2_score(y0_test,y11))

#output of this loop we will be the best possible weight combination with precession upto second decimal  
for i in range (1,20):
    tp=(i*y11)+((20-i)*sp2)
    tpp=tp/20   
    print(i/20)
    print(r2_score(y0_test,tpp))
# the obatined best weights for the given trianing data set and hyperparameters for the models is 0.55 for lgbm and 0.45 for ridge


