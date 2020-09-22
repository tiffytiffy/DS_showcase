
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


#Q1 - 
# read in csv file

ins= pd.read_csv('insurance.csv', delimiter=",")
pd.set_option('display.max_columns',None)

#Q1(a) - 

#create conditions and choices
conditions = [
    ins['age'].between (18, 24, inclusive=True),
    ins['age'].between (25, 55, inclusive=True),
    ins['age']>= 56
   ]

choices = [
    'Young',
    'Adult',
    'Elder']

# apply the new column to my dataframe using numpy 'select' 
ins['age_cat'] = np.select(conditions, choices, default='NA')

#check how many per category 
ins.age_cat.value_counts()

#Q1(b) - 

#create conditions and choices

conditions2 = [
    ins['bmi']<18.5,
    ins['bmi'].between (18.5, 24.99, inclusive=True),
    ins['bmi'].between (25, 29.99, inclusive = True),
    ins['bmi']>=30
   ]

choices2 = ['Under Weight', 'Normal Weight', 'Over Weight', 'Obese']

ins['weight'] = np.select(conditions2,choices2,default='NA')

ins.weight.value_counts()

#Q1(b) - no of smokers by genders
 
xtab=pd.crosstab(ins.sex, ins.smoker, dropna=False)
xtab.loc["Total"] = xtab.loc['female'] + xtab.loc['male']
xtab

#calculate column % 
pd.crosstab(ins.sex,ins.smoker).apply(lambda r: r/r.sum(), axis=0)

#Q1(c) - relationships between charges vs other features


#1.correlation
### Set figure size
plt.figure(figsize=(12,10))
cor = ins.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()


#2.scatterplot

# for continuous variables
sns.pairplot(ins, kind="scatter")
plt.show()

#for categorical variables
#a. - charges vs sex 
sex_charge = ins.groupby('sex')['charges'].sum().reset_index()
sex_charge.head()

fig, axs = plt.subplots(ncols=2)
sns.boxplot(x='sex', y='charges',  data=ins, ax=axs[0])
sns.barplot(x='sex', y='charges', data=sex_charge,  ax=axs[1])

#b. - charges vs smoker
smoker_charge = ins.groupby('smoker')['charges'].sum().reset_index()
smoker_charge.head()

fig, axs = plt.subplots(ncols=2)
sns.boxplot(x='smoker', y='charges',data=ins, order=['no','yes'],ax=axs[0])
sns.barplot(x='smoker', y='charges', data=smoker_charge, order=['no','yes'],  ax=axs[1])



#c. - charges vs region
region_charge = ins.groupby('region')['charges'].sum().reset_index()
region_charge.head()

fig, axs = plt.subplots(ncols=2)
sns.boxplot(x='region', y='charges', data=ins, order=['northeast','northwest','southeast','southwest'],ax=axs[0])
sns.barplot(x='region', y='charges', data=region_charge, order=['northeast','northwest','southeast','southwest'] ,ax=axs[1])


#d. - charges vs age categories
agecat_charge = ins.groupby('age_cat')['charges'].sum().reset_index()
agecat_charge.head()

fig, axs = plt.subplots(ncols=2)
plt.ticklabel_format(style='plain', axis='y')
sns.boxplot(x='age_cat', y='charges', data=ins, order=['Young','Adult','Elder'], ax=axs[0])
sns.barplot(x='age_cat', y='charges', data=agecat_charge, order=['Young','Adult','Elder'], ax=axs[1])


#e. - charges vs weight
weight_charge = ins.groupby('weight')['charges'].sum().reset_index()
weight_charge.head()

fig, axs = plt.subplots(ncols=2)
plt.ticklabel_format(style='plain', axis='y')
sns.boxplot(x='weight', y='charges', data=ins, order=['Under Weight', 'Normal Weight', 'Over Weight', 'Obese'],ax=axs[0])
sns.barplot(x='weight', y='charges', data=weight_charge, order=['Under Weight', 'Normal Weight', 'Over Weight', 'Obese'] , ax=axs[1])






#Q1(d)- 


#check if there are any missing in dataset
ins.isnull().sum()


#check all variable types 
ins.dtypes

#descriptive stats to check all columns
ins.describe(include='all')

#check how many rows have unexpected 0 values
print("rows in dataframe {0}".format(len(ins)))
print("rows in age {0}".format(len(ins.loc[ins['age']==0])))
print("rows in bmi {0}".format(len(ins.loc[ins['bmi']==0])))
print("rows in sex {0}".format(len(ins.loc[ins['sex']==0])))
print("rows in smoker {0}".format(len(ins.loc[ins['smoker']==0])))
print("rows in region {0}".format(len(ins.loc[ins['region']==0])))
print("rows in charges {0}".format(len(ins.loc[ins['charges']==0])))
print("rows in age_cat {0}".format(len(ins.loc[ins['age_cat']==0])))
print("rows in weight {0}".format(len(ins.loc[ins['weight']==0])))


#dummy out categorical variables for modelling
ins=pd.get_dummies(ins, columns =['sex','smoker', 'region', 'age_cat', 'weight', 'children'])

ins.head()

#check again if there's any missing in dataset -- empty dataframe indicates none missing
ins[ins.isnull().any(axis=1)]



from sklearn.model_selection import train_test_split
#define features 
x =ins.drop(['charges'],axis=1)

#define target variable
y=ins['charges']


#split into 80% for traning and 20% for test dataset
x_train, x_test, y_train,y_test = train_test_split(x,y, test_size=0.2, random_state=0)


from sklearn.linear_model import LinearRegression

linear_model = LinearRegression()
linear_model.fit(x_train, y_train)

#display coefficient of linear coefficient --show all the weights for all features 
print(linear_model.coef_)

#rearrange the coefficients by sorting them from most important to less important
predictors =x_train.columns
coef=pd.Series(linear_model.coef_, predictors).sort_values(ascending=False)
print(coef)

####################################### LR - testing data  #################################

#create prediction based on testing data
y_pred_test = linear_model.predict(x_test)


##plot to show how close predict vs testing data
%pylab inline
pylab.rcParams['figure.figsize'] =(15,6)

plt.plot(y_pred_test, label ='Predict_test')
plt.plot(y_test.values, label ='Actual')
plt.ylabel('charges')

plt.legend()
plt.show()

#calculate R-square for testing dataset
linear_model.score(x_test, y_test)


#calculate mean squared error for predict vs testing
from sklearn.metrics import mean_squared_error

linear_model_mse = mean_squared_error(y_pred_test, y_test)
print(linear_model_mse)

#calculate root mean square error (RMSE) - standard deviation of residuals
import math
math.sqrt(linear_model_mse)

#on average, predict charges is around 5968 away from actual charge



####################################### LR - training data  #################################

#create prediction based on training data
y_pred_train = linear_model.predict(x_train)


##plot to show how close predict vs training data
%pylab inline
pylab.rcParams['figure.figsize'] =(15,6)

plt.plot(y_pred_train, label ='Predict_train')
plt.plot(y_train.values, label ='Actual')
plt.ylabel('charges')

plt.legend()
plt.show()

#calculate R-square for training dataset
linear_model.score(x_train, y_train)

#calculate mean squared error for predict vs training
from sklearn.metrics import mean_squared_error

linear_model_mse = mean_squared_error(y_pred_train, y_train)
print(linear_model_mse)

#calculate root mean square error (RMSE) - standard deviation of residuals
import math
math.sqrt(linear_model_mse)











#second estimator --Lasso 
  

from sklearn.linear_model  import Lasso

#find best alpha for the model

alphas = [0.0001, 0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 1]

for a in alphas:
   lasso_model = Lasso(alpha=a,normalize=True).fit(x,y)   
   r2 = lasso_model.score(x, y) 
   y_pred = lasso_model.predict(x)
   mse = mean_squared_error(y, y_pred)  
   rmse=math.sqrt(mse)
   print("Alpha:{0:.4f}, r2:{1:.2f}, MSE:{2:.2f}, RMSE:{3:.2f}"
       .format(a, r2, mse, rmse))




lasso = Lasso(alpha =0.0001, normalize=True).fit(x_train,y_train)


#print the coefficients by sorting them from most important to less important
predictors =x_train.columns
coef=pd.Series(lasso.coef_, predictors).sort_values(ascending=False)
print(coef)


####################################### Lasso - testing data  #################################


#create prediction based on testing data
y_pred_test = lasso.predict(x_test)


##plot to show how close predict vs testing data
%pylab inline
pylab.rcParams['figure.figsize'] =(15,6)

plt.plot(y_pred_test, label ='Predict_test')
plt.plot(y_test.values, label ='Actual')
plt.ylabel('charges')

plt.legend()
plt.show()

#calculate R-square for testing dataset
lasso.score(x_test, y_test)


#calculate mean squared error for predict vs testing
from sklearn.metrics import mean_squared_error

lasso_mse = mean_squared_error(y_pred_test, y_test)
print(lasso_mse)

#calculate root mean square error (RMSE) - standard deviation of residuals
import math
math.sqrt(lasso_mse)

    
    
    
    
####################################### Lasso - training data  #################################

#create prediction based on training data
y_pred_train = lasso.predict(x_train)


##plot to show how close predict vs training data
%pylab inline
pylab.rcParams['figure.figsize'] =(15,6)

plt.plot(y_pred_train, label ='Predict_train')
plt.plot(y_train.values, label ='Actual')
plt.ylabel('charges')

plt.legend()
plt.show()

#calculate R-square for training dataset
lasso.score(x_train, y_train)

#calculate mean squared error for predict vs training
from sklearn.metrics import mean_squared_error

lasso_mse = mean_squared_error(y_pred_train, y_train)
print(lasso_mse)

#calculate root mean square error (RMSE) - standard deviation of residuals
import math
math.sqrt(lasso_model_mse)



    
    
    
    
    
#Third estimator --Ridge

from sklearn.linear_model import Ridge


alphas = [0.0001, 0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 1]

for a in alphas:
   ridge_model = Ridge(alpha=a,normalize=True).fit(x,y)   
   r2 = ridge_model.score(x, y) 
   y_pred = ridge_model.predict(x)
   mse = mean_squared_error(y, y_pred)  
   rmse=math.sqrt(mse)
   print("Alpha:{0:.4f}, r2:{1:.2f}, MSE:{2:.2f}, RMSE:{3:.2f}"
       .format(a, r2, mse, rmse))

#The result shows that we can use alpha=0.0001 value for our model.


ridgemodel=Ridge(alpha=0.0001, normalize=True).fit(x_train, y_train)
ypred_test= ridgemodel.predict(x_test)

#coefficients by sorting them from most important to less important
coef = pd.Series(ridgemodel.coef_,predictors).sort_values(ascending=False)
print(coef)

 
####################################### Ridge - testing data  #################################

#create prediction based on testing data
y_pred_test = ridgemodel.predict(x_test)


##plot to show how close predict vs testing data
%pylab inline
pylab.rcParams['figure.figsize'] =(15,6)

plt.plot(y_pred_test, label ='Predict_test')
plt.plot(y_test.values, label ='Actual')
plt.ylabel('charges')

plt.legend()
plt.show()

#calculate R-square for testing dataset
ridgemodel.score(x_test, y_test)


#calculate mean squared error for predict vs testing
from sklearn.metrics import mean_squared_error

ridge_model_mse = mean_squared_error(y_pred_test, y_test)
print(linear_model_mse)

#calculate root mean square error (RMSE) - standard deviation of residuals
import math
math.sqrt(ridge_model_mse)


  
####################################### Ridge - training data  #################################

#create prediction based on training data
y_pred_train = ridgemodel.predict(x_train)


##plot to show how close predict vs training data
%pylab inline
pylab.rcParams['figure.figsize'] =(15,6)

plt.plot(y_pred_train, label ='Predict_train')
plt.plot(y_train.values, label ='Actual')
plt.ylabel('charges')

plt.legend()
plt.show()

#calculate R-square for training dataset
ridgemodel.score(x_train, y_train)

#calculate mean squared error for predict vs training
from sklearn.metrics import mean_squared_error

ridge_model_mse = mean_squared_error(y_pred_train, y_train)
print(ridge_model_mse)

#calculate root mean square error (RMSE) - standard deviation of residuals
import math
math.sqrt(ridge_model_mse)






#fourth estimator -- Elastic Net

#finding best alpha for the model by testing on original data
from sklearn.linear_model import ElasticNet

alphas = [0.0001, 0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 1]

for a in alphas:
    elastic_model = ElasticNet(alpha=a,normalize=True).fit(x,y)   
    r2 = elastic_model.score(x, y) 
    y_pred = elastic_model.predict(x)
    mse = mean_squared_error(y, y_pred)  
    rmse=math.sqrt(mse)
    print("Alpha:{0:.4f}, r2:{1:.2f}, MSE:{2:.2f}, RMSE:{3:.2f}"
       .format(a, r2, mse, rmse))

#The result shows that we can use alpha=0.0001 value for our model.


elastic=ElasticNet(alpha=0.0001, normalize=True).fit(x_train, y_train)

#coefficients by sorting them from most important to less important
coef = pd.Series(elastic.coef_,predictors).sort_values(ascending=False)
print(coef)


####################################### Elastic - testing data  #################################

#create prediction based on testing data
y_pred_test = elastic.predict(x_test)


##plot to show how close predict vs testing data
%pylab inline
pylab.rcParams['figure.figsize'] =(15,6)

plt.plot(y_pred_test, label ='Predict_test')
plt.plot(y_test.values, label ='Actual')
plt.ylabel('charges')

plt.legend()
plt.show()

#calculate R-square for testing dataset
elastic.score(x_test, y_test)


#calculate mean squared error for predict vs testing
from sklearn.metrics import mean_squared_error

elastic_mse = mean_squared_error(y_pred_test, y_test)
print(elastic_mse)

#calculate root mean square error (RMSE) - standard deviation of residuals
import math
math.sqrt(elastic_mse)





####################################### elastic - training data  #################################

#create prediction based on training data
y_pred_train = elastic.predict(x_train)


##plot to show how close predict vs training data
%pylab inline
pylab.rcParams['figure.figsize'] =(15,6)

plt.plot(y_pred_train, label ='Predict_train')
plt.plot(y_train.values, label ='Actual')
plt.ylabel('charges')

plt.legend()
plt.show()

#calculate R-square for training dataset
elastic.score(x_train, y_train)

#calculate mean squared error for predict vs training
from sklearn.metrics import mean_squared_error

elastic_mse = mean_squared_error(y_pred_train, y_train)
print(elastic_mse)

#calculate root mean square error (RMSE) - standard deviation of residuals
import math
math.sqrt(elastic_mse)


#saving the chosen model

from sklearn.externals import joblib

ridge_final=Ridge(alpha=0.0001, normalize=True).fit(x_train, y_train)
joblib.dump(ridge_final, 'Q1_ins_ridge.pkl')

#load model back into spyder
ins_model = joblib.load('Q1_ins_ridge.pkl')

#create original file
ins_x_train=x_train.copy()
ins_x_test=x_test.copy()
ins_y_test=y_test.copy()

# 4 - create prediction and convert it into dataframe
df_pred = pd.DataFrame(ins_model.predict(x_test),columns = ['prediction'])

#5 merge predction back to test dataset
final_df = pd.concat([ins_x_test.reset_index(drop='True'),ins_y_test.reset_index(drop='True'),
                      df_pred.reset_index(drop='Tru‌​e')],axis=1)


##Export data as CSV
final_df.to_csv('Q1_ins_ridge.csv', index=False)





###############################################################################
###############################################################################
##############################   Q2  ###########################################
###############################################################################
###############################################################################


#Q2 - 
# read in the file


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns',None)

mydata1 = pd.read_csv("adult.data", header = None,names=[ 'age', 'workclass', 'fnlwgt',
                      'education','education_num','marital_status','occupation',
                      'relationship','race','sex','capital_gain','capital_loss',
                      'hrs_per_week','native_ctry','label'])
mydata1['flag'] = 'train'


mydata2 = pd.read_csv("adult.test",header = None, skiprows=1, names=[ 'age', 'workclass', 
                    'fnlwgt', 'education','education_num','marital_status','occupation',
                      'relationship','race','sex','capital_gain','capital_loss',
                      'hrs_per_week','native_ctry','label'])
mydata2['flag'] = 'test'


adult = pd.concat([mydata1, mydata2],axis=0)



#check if there are any missing in dataset
adult.isnull().sum()


#check all variable types 
adult.dtypes

#descriptive stats to check all columns
adult.describe(include='all')

#check individual category 
adult.workclass.value_counts()
adult.education.value_counts()
adult.marital_status.value_counts()
adult.occupation.value_counts()
adult.relationship.value_counts()
adult.race.value_counts()
adult.sex.value_counts()
adult.native_ctry.value_counts()
adult.label.value_counts()
adult.flag.value_counts()



#replace question mark by missing for workclass, occupation, native country
adult['workclass'].replace(' ?', np.NaN, inplace=True)
adult['occupation'].replace(' ?', np.NaN, inplace=True)
adult['native_ctry'].replace(' ?', np.NaN, inplace=True)



#check them again to ensure ? being replaced by NaN
adult.workclass.value_counts()
adult.occupation.value_counts()
adult.native_ctry.value_counts()

#rename various label together 
adult['label'].replace(' >50K.', ' >50K', inplace=True)
adult['label'].replace(' <=50K.', ' <=50K', inplace=True)

adult.label.value_counts()


#Q2a - check relationships between age group and label
 
#create age group 


age_conditions = [
    adult['age'].between (16, 20, inclusive=True),
    adult['age'].between (21, 30, inclusive=True),
    adult['age'].between (31, 40, inclusive = True),
    adult['age'].between (41, 50, inclusive = True),
    adult['age'].between (51, 60, inclusive = True),
    adult['age']>60

   ]

age_choice = ['16-20', '21-30', '31-40', '41-50','51-60','60+']

adult['age_group'] = np.select(age_conditions,age_choice,default='NA')

adult.age_group.value_counts()


#a. - income vs age group

#calculate row % 
age_xtab_perc=pd.crosstab(adult.label, adult.age_group).apply(lambda r: r/r.sum()*100, axis=1)
age_xtab_perc
#stacked bar chart for income vs age group (%)
age_xtab_perc.plot.bar(stacked=True)

#horizontal chart of income vs age group
sns.countplot(y="age_group", hue="label", 
              order=('16-20', '21-30', '31-40', '41-50','51-60','60+'),data=adult)


#a. - income vs gender group

#calculate row % 
sex_xtab_perc1=pd.crosstab(adult.label, adult.sex).apply(lambda r: r/r.sum()*100, axis=1)
sex_xtab_perc1

sex_xtab_perc2=pd.crosstab(adult.label, adult.sex).apply(lambda r: r/r.sum()*100, axis=0)
sex_xtab_perc2

#stacked bar chart for income vs age group (%)
sex_xtab_perc1.plot.bar(stacked=True)

sex_xtab_perc2.plot.bar()

#horizontal chart of income vs age group
sns.countplot(y="sex", hue="label",data=adult)


#a. - income vs occupation group


#calculate row % 
occ_xtab_perc=pd.crosstab(adult.label, adult.occupation).apply(lambda r: r/r.sum()*100, axis=1)
occ_xtab_perc
#stacked bar chart for income vs age group (%)
occ_xtab_perc.plot.bar(stacked=True)

#horizontal chart of income vs age group
sns.countplot(y="occupation", hue="label",data=adult)




#dummy out features for modelling
adult=pd.get_dummies(adult, columns =['workclass', 'education', 'marital_status', 'occupation','relationship', 'race',
                                      'sex','native_ctry', 'age_group'])

#encoding target variable
from sklearn.preprocessing import LabelEncoder
gle = LabelEncoder()
labels = gle.fit_transform(adult['label'])
print(gle.classes_)  #0 represent <=50K and 1 represents >50K
#update the label column
adult['label'] = labels



#split out train and test  for model
train=adult[adult['flag']=='train']
y_train =adult['label'][(adult['flag']=='train')]
x_train = train.drop(["label","flag"],axis=1) 

test=adult[adult['flag']=='test']
y_test = test.label
x_test = test.drop(["label","flag"],axis=1) 


from sklearn.neighbors import KNeighborsClassifier

#train KNN model with kfold cross validation - initial try
knn=KNeighborsClassifier()
knn.fit(x_train,y_train)
#predicted value from test
y_pred_test=knn.predict(x_test)

from sklearn.metrics import accuracy_score
#evaluate accuracy
print(accuracy_score(y_test,y_pred_test)) #0.7768564584484983


#standardized features
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(x_train)
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)

knn_scaled = KNeighborsClassifier()
knn_scaled.fit(x_train_scaled, y_train)
#testing
y_preds_scaled = knn_scaled.predict(x_test_scaled)
# Evaluate accuracy
print(accuracy_score(y_test,y_preds_scaled)) #0.8244579571279406 - better to standardised data before modelling

#Try 5 classifications with kfold CV

#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.svm import SVC
#from sklearn.linear_model import LogisticRegression
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.ensemble import RandomForestClassifier

#from sklearn import model_selection
#from sklearn.model_selection import cross_val_score
#from sklearn.metrics import accuracy_score
#from sklearn.metrics import mean_squared_error
#import math

#models=[]
#models.append(('KNN', KNeighborsClassifier()))
#models.append(('SVC', SVC(probability=True)))
#models.append(('LR', LogisticRegression()))
#models.append(('DT', DecisionTreeClassifier()))
#models.append(('RF', RandomForestClassifier()))

#results = []
#names = []
#scoring = 'accuracy'
#for name, model in models:
#	kfold = model_selection.KFold(n_splits=5)
#	cv_results = model_selection.cross_val_score(model, x_train_scaled, y_train, cv=kfold, scoring=scoring)
#	results.append(cv_results)
 #   y_pred = model.predict(x_test_scaled)
	#names.append(name)
	m#sg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())


#elastic_mse = mean_squared_error(y_pred_train, y_train)
#print(elastic_mse)

#calculate root mean square error (RMSE) - standard deviation of residuals

#math.sqrt(elastic_mse)
#	print(msg)
    
    

#Q2b -
#########################       KNN  #############################################
from sklearn.neighbors import KNeighborsClassifier
from sklearn import model_selection
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
import math

knn = KNeighborsClassifier(n_neighbors = 5, leaf_size=10)
knn.fit(x_train_scaled, y_train)

####################################### KNN - testing data  #################################

#create prediction based on testing data
y_pred_test = knn.predict(x_test_scaled)

test_scores = cross_val_score(knn, x_test_scaled, y_test, cv=5, scoring='accuracy')

# print all 5 times scores 
print(test_scores)
#get average scores R-square
print(test_scores.mean())


#calculate mean squared error for predict vs testing

knn_test_mse = mean_squared_error(y_pred_test, y_test)
print(knn_test_mse)

#calculate root mean square error (RMSE) - standard deviation of residuals
import math
math.sqrt(knn_test_mse)


####################################### KNN - training data  #################################

#create prediction based on training data
y_pred_train= knn.predict(x_train_scaled)

train_scores = cross_val_score(knn, x_train_scaled, y_train, cv=5, scoring='accuracy')

# print all 5 times scores 
print(train_scores)
#get average scores R-square
print(train_scores.mean())


#calculate mean squared error for predict vs training

knn_train_mse = mean_squared_error(y_pred_train, y_train)
print(knn_train_mse)

#calculate root mean square error (RMSE) - standard deviation of residuals
import math
math.sqrt(knn_train_mse)




###########################   SVC  ####################################

from sklearn import svm
from sklearn import model_selection
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
import math

clf = svm.SVC(kernel='linear', C = 1.0)
clf.fit(x_train_scaled, y_train)


####################################### SVC - testing data  #################################

#create prediction based on testing data
y_pred_test = clf.predict(x_test_scaled)

test_scores = cross_val_score(clf, x_test_scaled, y_test, cv=5, scoring='accuracy')

# print all 5 times scores 
print(test_scores)
#get average scores R-square
print(test_scores.mean())


#calculate mean squared error for predict vs testing

svc_test_mse = mean_squared_error(y_pred_test, y_test)
print(svc_test_mse)

#calculate root mean square error (RMSE) - standard deviation of residuals
import math
math.sqrt(svc_test_mse)

####################################### SVC - training data  #################################

#create prediction based on training data
y_pred_train= clf.predict(x_train_scaled)

train_scores = cross_val_score(clf, x_train_scaled, y_train, cv=5, scoring='accuracy')

# print all 5 times scores 
print(train_scores)
#get average scores R-square
print(train_scores.mean())


#calculate mean squared error for predict vs training

svc_train_mse = mean_squared_error(y_pred_train, y_train)
print(svc_train_mse)

#calculate root mean square error (RMSE) - standard deviation of residuals
import math
math.sqrt(svc_train_mse)



###########################  LR  ####################################

from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
import math


lr=LogisticRegression(solver='liblinear', penalty='l2', C=1.0)
lr.fit(x_train_scaled, y_train)


####################################### LR - testing data  #################################

#create prediction based on testing data
y_pred_test = lr.predict(x_test_scaled)

test_scores = cross_val_score(lr, x_test_scaled, y_test, cv=5, scoring='accuracy')

# print all 5 times scores 
print(test_scores)
#get average scores R-square
print(test_scores.mean())


#calculate mean squared error for predict vs testing

lr_test_mse = mean_squared_error(y_pred_test, y_test)
print(lr_test_mse)

#calculate root mean square error (RMSE) - standard deviation of residuals
import math
math.sqrt(lr_test_mse)



####################################### LR - training data  #################################

#create prediction based on training data
y_pred_train= lr.predict(x_train_scaled)

train_scores = cross_val_score(lr, x_train_scaled, y_train, cv=5, scoring='accuracy')

# print all 5 times scores 
print(train_scores)
#get average scores R-square
print(train_scores.mean())


#calculate mean squared error for predict vs training

lr_train_mse = mean_squared_error(y_pred_train, y_train)
print(lr_train_mse)

#calculate root mean square error (RMSE) - standard deviation of residuals
import math
math.sqrt(lr_train_mse)




################################  decision tree #################################


from sklearn.tree import DecisionTreeClassifier
from sklearn import model_selection
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
import math

dt =  DecisionTreeClassifier()
dt.fit(x_train_scaled, y_train)

####################################### DT - testing data  #################################

#create prediction based on testing data
y_pred_test = dt.predict(x_test_scaled)

test_scores = cross_val_score(dt, x_test_scaled, y_test, cv=5, scoring='accuracy')

# print all 5 times scores 
print(test_scores)
#get average scores R-square
print(test_scores.mean())


#calculate mean squared error for predict vs testing

dt_test_mse = mean_squared_error(y_pred_test, y_test)
print(dt_test_mse)

#calculate root mean square error (RMSE) - standard deviation of residuals
import math
math.sqrt(dt_test_mse)


####################################### DT - training data  #################################

#create prediction based on training data
y_pred_train= dt.predict(x_train_scaled)

train_scores = cross_val_score(dt, x_train_scaled, y_train, cv=5, scoring='accuracy')

# print all 5 times scores 
print(train_scores)
#get average scores R-square
print(train_scores.mean())


#calculate mean squared error for predict vs training

dt_train_mse = mean_squared_error(y_pred_train, y_train)
print(dt_train_mse)

#calculate root mean square error (RMSE) - standard deviation of residuals
import math
math.sqrt(dt_train_mse)


############################# Random Forrest  ##############################


from sklearn.ensemble import RandomForestClassifier

# Create the model with 100 trees
rf = RandomForestClassifier(n_estimators = 100, random_state=0)
# Fit on training data
rf.fit(x_train_scaled, y_train)


####################################### RF - testing data  #################################

#create prediction based on testing data
y_pred_test = rf.predict(x_test_scaled)

test_scores = cross_val_score(rf, x_test_scaled, y_test, cv=5, scoring='accuracy')

# print all 5 times scores 
print(test_scores)
#get average scores R-square
print(test_scores.mean())


#calculate mean squared error for predict vs testing

rf_test_mse = mean_squared_error(y_pred_test, y_test)
print(rf_test_mse)

#calculate root mean square error (RMSE) - standard deviation of residuals
import math
math.sqrt(rf_test_mse)



####################################### RF - training data  #################################

#create prediction based on training data
y_pred_train= rf.predict(x_train_scaled)

train_scores = cross_val_score(rf, x_train_scaled, y_train, cv=5, scoring='accuracy')

# print all 5 times scores 
print(train_scores)
#get average scores R-square
print(train_scores.mean())


#calculate mean squared error for predict vs training

rf_train_mse = mean_squared_error(y_pred_train, y_train)
print(rf_train_mse)

#calculate root mean square error (RMSE) - standard deviation of residuals
import math
math.sqrt(rf_train_mse)



#saving the chosen model

#1. prepare testing file for merging

adult_test = mydata2.copy()


#replace question mark by missing for workclass, occupation, native country
adult_test['workclass'].replace(' ?', np.NaN, inplace=True)
adult_test['occupation'].replace(' ?', np.NaN, inplace=True)
adult_test['native_ctry'].replace(' ?', np.NaN, inplace=True)


#rename various label together 
adult_test['label'].replace(' >50K.', ' >50K', inplace=True)
adult_test['label'].replace(' <=50K.', ' <=50K', inplace=True)


age_conditions = [
    adult_test['age'].between (16, 20, inclusive=True),
    adult_test['age'].between (21, 30, inclusive=True),
    adult_test['age'].between (31, 40, inclusive = True),
    adult_test['age'].between (41, 50, inclusive = True),
    adult_test['age'].between (51, 60, inclusive = True),
    adult_test['age']>60

   ]

age_choice = ['16-20', '21-30', '31-40', '41-50','51-60','60+']

adult_test['age_group'] = np.select(age_conditions,age_choice,default='NA')

#2 - encoding target variable
from sklearn.preprocessing import LabelEncoder
gle = LabelEncoder()
labels = gle.fit_transform(adult_test['label'])
print(gle.classes_)  #0 represent <=50K and 1 represents >50K
#update the label column
adult_test['label'] = labels


#3 - save model for future use
from sklearn.externals import joblib

rf_kfold = RandomForestClassifier(n_estimators = 100, random_state=0).fit(x_train_scaled, 
                                                                          y_train)
joblib.dump(rf_kfold, 'q2b_rf_kfold.pkl')

#load model back into spyder
rf_kfold = joblib.load('q2b_rf_kfold.pkl')

# 4 - create prediction and convert it into dataframe
df_pred = pd.DataFrame(rf_kfold.predict(x_test_scaled),columns = ['prediction'])

#5 merge predction back to test dataset
final_df = pd.merge(adult_test,df_pred,how = 'left',left_index = True, right_index = True)

##Export data as CSV
final_df.to_csv('q2b_rf_kfold.csv', index=False)




########################Q2c - fine tune hyperparameters




#########################       KNN  #############################################
from sklearn.neighbors import KNeighborsClassifier
from sklearn import model_selection
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
import math
from sklearn.model_selection import GridSearchCV



#making the instance
knn = KNeighborsClassifier(n_jobs=-1)
#Hyper Parameters Set
params = {'n_neighbors':[4,5,6,7,8],
          'leaf_size':[1,2,3,4,5],
          'algorithm':['auto', 'kd_tree'],
          'n_jobs':[-1]}
#Making models with hyper parameters sets
knn_gscv = GridSearchCV(knn, param_grid=params, n_jobs=1, cv=3)
#Learning
knn_gscv.fit(x_train_scaled, y_train)

#The best hyper parameters set
print("Best Hyper Parameters:\n",knn_gscv.best_params_)
#Prediction
y_pred=knn_gscv.predict(x_test_scaled)
#importing the metrics module
from sklearn import metrics
#evaluation(Accuracy)
print("Accuracy:",metrics.accuracy_score(y_pred,y_test))
#evaluation(Confusion Metrix)
print("Confusion Metrix:\n",metrics.confusion_matrix(y_pred,y_test))







###########################   SVC  ####################################

from sklearn import svm
from sklearn import model_selection
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
import math
from sklearn.model_selection import GridSearchCV

#making the instance
model=svm.SVC()

#Hyper Parameters Set
svc_para = [{'kernel': ['linear','rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 50, 100]}]

#Making models with hyper parameters sets
svc_gscv = GridSearchCV(model, param_grid=svc_para, n_jobs=-1, cv=3)

#Learning
svc_gscv.fit(x_train_scaled, y_train)

#The best hyper parameters set
print("Best Hyper Parameters:\n",svc_gscv.best_params_)

#Prediction
y_pred=svc_gscv.predict(x_test_scaled)

#importing the metrics module
from sklearn import metrics
#evaluation(Accuracy)
print("Accuracy:",metrics.accuracy_score(y_pred,y_test))
#evaluation(Confusion Metrix)
print("Confusion Metrix:\n",metrics.confusion_matrix(y_pred,y_test))




################### logistic regression #################################

from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
import math
from sklearn.model_selection import GridSearchCV

#make the instance
lr = LogisticRegression()

#hyperparameter set
grid_parameters = {'penalty': ['l1', 'l2'],
'C':[0.001,0.009,0.01,0.09,1,5,10,25]}

#create model with hyper parameter set
lr_gscv = GridSearchCV(lr, param_grid=grid_parameters, cv = 3, n_jobs=-1)

#learning
lr_gscv.fit(x_train_scaled, y_train)

#The best hyper parameters set
print("Best Hyper Parameters:",lr_gscv.best_params_)
#Prediction
y_pred=lr_gscv.predict(x_test_scaled)
#importing the metrics module
from sklearn import metrics
#evaluation(Accuracy)
print("Accuracy:",metrics.accuracy_score(y_pred,y_test))
#evaluation(Confusion Metrix)
print("Confusion Metrix:\n",metrics.confusion_matrix(y_pred,y_test))



########################  Decision Tree ####################################

from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
#making the instance
dt= DecisionTreeClassifier()
#Hyper Parameters Set
params = {'max_features': ['auto', 'sqrt', 'log2'],
          'min_samples_split': [6,8,10,12,14], 
          'min_samples_leaf':[4,5,6,7,8],
          'random_state':[123]}
#Making models with hyper parameters sets
dt_gscv = GridSearchCV(dt, param_grid=params, n_jobs=-1, cv=3)
#Learning
dt_gscv.fit(x_train_scaled, y_train)
#The best hyper parameters set
print("Best Hyper Parameters:",dt_gscv.best_params_)
#Prediction
y_pred=dt_gscv.predict(x_test_scaled)
#importing the metrics module
from sklearn import metrics
#evaluation(Accuracy)
print("Accuracy:",metrics.accuracy_score(y_pred,y_test))
#evaluation(Confusion Metrix)
print("Confusion Metrix:\n",metrics.confusion_matrix(y_pred,y_test))



########################## Random Forrest ################################


from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
#making the instance
rf=RandomForestClassifier()
#hyper parameters set
params = {'criterion':['gini','entropy'],
          'n_estimators':[10,15,20,25,30],
          'min_samples_leaf':[1,2,3],
          'min_samples_split':[3,4,5,6,7], 
          'random_state':[123],
          'n_jobs':[-1]}
#Making models with hyper parameters sets
rf_gscv = GridSearchCV(rf, param_grid=params, n_jobs=-1, cv=3)
#learning
rf_gscv.fit(x_train_scaled, y_train)
#The best hyper parameters set
print("Best Hyper Parameters:\n",rf_gscv.best_params_)
#Prediction
y_pred=rf_gscv.predict(x_test_scaled)
#importing the metrics module
from sklearn import metrics
#evaluation(Accuracy)
print("Accuracy:",metrics.accuracy_score(y_pred,y_test))
#evaluation(Confusion Metrix)
print("Confusion Metrix:\n",metrics.confusion_matrix(y_pred,y_test))











#feature selections - Recursive Feature Elimination

##############  C1 - KNN
#below doesn't work

#from sklearn.feature_selection import RFE
#from sklearn.neighbors import KNeighborsClassifier


#knn = KNeighborsClassifier()
#knn_sel=RFE(knn,20,step=1).fit(x_train, y_train)


#to see which features are important in the model
#knn_sel.get_support()

#create a list of selected feature

#selected_feat= x_train.columns[(knn_sel.get_support())]
#len(selected_feat)

#print(selected_feat)

#train model with selected features
#knn_sel_fea = KNeighborsClassifier()
#knn_sel_fea.fit(x_train[selected_feat],y_train)
#knn_sel_fea.score(x_test[selected_feat],y_test)



########### C1 - KNN try using PCA


from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

# normalize data
from sklearn import preprocessing
x_train_scaled = pd.DataFrame(preprocessing.scale(x_train),columns = x_train.columns) 
x_test_scaled = pd.DataFrame(preprocessing.scale(x_test),columns = x_test.columns) 

pca = PCA(n_components=10).fit(x_train_scaled)
x_train_reduce = pca.transform(x_train_scaled)
x_test_reduce = pca.transform(x_test_scaled)

knn_sel_fea = KNeighborsClassifier()
knn_sel_fea.fit(x_train_reduce,y_train)
knn_sel_fea.score(x_test_reduce,y_test)



############## C2 - SVM 

#from sklearn.feature_selection import RFE
#from sklearn import svm

#clf=svm.SVC(kernel="linear")
#svm_sel=RFE(clf, 20, step=1).fit(x_train, y_train)
#print (svm_sel.support_)
#print (svm_sel.ranking_)


#to see which features are important in the model
#svm_sel.get_support()

#create a list of selected feature

#selected_feat= x_train.columns[(svm_sel.get_support())]
#len(selected_feat)

#print(selected_feat)

#train model with selected features
#svm_sel_fea = svm.SVC()
#svm_sel_fea.fit(x_train[selected_feat],y_train)
#svm_sel_fea.score(x_test[selected_feat],y_test)




########### C2 - SVM try using PCA


from sklearn.decomposition import PCA
from sklearn import svm

pca = PCA(n_components=10).fit(x_train_scaled)
x_train_reduce = pca.transform(x_train_scaled)
x_test_reduce = pca.transform(x_test_scaled)

svm_sel_fea = svm.SVC()
svm_sel_fea.fit(x_train_reduce,y_train)
svm_sel_fea.score(x_test_reduce,y_test)



################# C3-  Logistic regression

from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(max_iter=2000,dual=False)
rfe_lr = RFE(lr, 10) # we have selected here 10 features
rfe_lr.fit(x_train_scaled, y_train)

#to see which features are important in the model
rfe_lr.get_support()

#create a list of selected feature

selected_feat= x_train.columns[(rfe_lr.get_support())]
len(selected_feat)

print(selected_feat)



#train model with selected features
rfe_lr_fea=LogisticRegression()
rfe_lr.fit(x_train_scaled[selected_feat],y_train)
rfe_lr.score(x_test_scaled[selected_feat],y_test)




################### C4-  decision tree
from sklearn.feature_selection import SelectFromModel
#use select from model to select those features which importance is greater than the mean importance of all the features by default
dt_sel = SelectFromModel(DecisionTreeClassifier())
dt_sel.fit(x_train_scaled, y_train)

#to see which features are important in the model
dt_sel.get_support()

#create a list of selected feature

selected_feat= x_train.columns[(dt_sel.get_support())]
len(selected_feat)

print(selected_feat)

#train model with selected features
dt_sel_fea = DecisionTreeClassifier()
dt_sel_fea.fit(x_train[selected_feat],y_train)
dt_sel_fea.score(x_test[selected_feat],y_test)


#Tree-based feature selection
#################   C5-   Random Forrest
from sklearn.feature_selection import SelectFromModel
#use select from model to select those features which importance is greater than the mean importance of all the features by default
rf_sel = SelectFromModel(RandomForestClassifier(n_estimators = 20))
rf_sel.fit(x_train_scaled, y_train)

#to see which features are important in the model
rf_sel.get_support()

#create a list of selected feature

selected_feat= x_train.columns[(rf_sel.get_support())]
len(selected_feat)

print(selected_feat)

#train model with selected features
rf_sel_fea = RandomForestClassifier()
rf_sel_fea.fit(x_train[selected_feat],y_train)
rf_sel_fea.score(x_test[selected_feat],y_test)





#################### feature selection + GSCV




from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

#initialize the classifier models with their default parameters and add them to a model list.
models = []
models.append(('KNN', KNeighborsClassifier(
    algorithm='auto', leaf_size=1, n_jobs=-1, n_neighbors=8 )))
models.append(('SVC', SVC( C=10, gamma=0.001, kernel = 'rbf',max_iter=20000)))
models.append(('LR', LogisticRegression(C=0.09, penalty='l2',max_iter=20000,dual=False)))
models.append(('DT', DecisionTreeClassifier(
    max_features='auto', min_samples_leaf=4, min_samples_split = 12, random_state=123)))
models.append(('RF', RandomForestClassifier(
    criterion='entropy', min_samples_leaf=3, min_samples_split=7, n_estimators=30, n_jobs=-1, random_state=123)))


pca = PCA(n_components=10).fit(x_train_scaled)
x_train_reduce = pca.transform(x_train_scaled)
x_test_reduce = pca.transform(x_test_scaled)


names = []
scores = []
for name, model in models:
    
    model.fit(x_train_reduce, y_train)
    y_pred = model.predict(x_test_reduce)
    scores.append(accuracy_score(y_test, y_pred))
    names.append(name)
    tr_split = pd.DataFrame({'Name': names, 'Score': scores})
    print(tr_split)
    





#saving the chosen model


#Tree-based feature selection
#################   C5-   Random Forrest
from sklearn.feature_selection import SelectFromModel
#use select from model to select those features which importance is greater than the mean importance of all the features by default
rf_sel = RandomForestClassifier(n_estimators = 20).fit(x_train[selected_feat], y_train)

#save model
joblib.dump(rf_sel, 'q2c_rf_fea_sel.pkl')

#load model back into spyder
rf_sel = joblib.load('q2c_rf_fea_sel.pkl')

# 4 - create prediction and convert it into dataframe
df_pred = pd.DataFrame(rf_sel.predict(x_test[selected_feat]),columns = ['prediction'])

#5 merge predction back to test dataset
final_df = pd.merge(adult_test,df_pred,how = 'left',left_index = True, right_index = True)

##Export data as CSV
final_df.to_csv('q2c_rf_fea_sel.csv', index=False)



#### Q2d - ensemble method
## voting

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier

#initialize the classifier models with their default parameters and add them to a model list.
models = []
models.append(('KNN', KNeighborsClassifier()))
models.append(('SVC', SVC(probability=True)))
models.append(('LR', LogisticRegression(max_iter=20000)))
models.append(('DT', DecisionTreeClassifier()))
models.append(('RF', RandomForestClassifier()))

names = []
scores = []
for name, model in models:
    model.fit(x_train_scaled, y_train)
    y_pred = model.predict(x_test_scaled)
    scores.append(accuracy_score(y_test, y_pred))
    names.append(name)
    tr_split = pd.DataFrame({'Name': names, 'Score': scores})
    print(tr_split)

#create a dictionary of our models
estimators=[ m for m in models]
#create our voting classifier, inputting our models
ensemble = VotingClassifier(estimators, voting='soft')
#fit model to training data
ensemble.fit(x_train_scaled, y_train)
#test our model on the test data
ensemble.score(x_test_scaled, y_test)

### stacking


from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import StackingClassifier

#initialize the classifier models with their default parameters and add them to a model list.
models = []
models.append(('KNN', KNeighborsClassifier()))
models.append(('SVC', SVC(probability=True)))
models.append(('LR', LogisticRegression(max_iter=20000)))
models.append(('DT', DecisionTreeClassifier()))
models.append(('RF', RandomForestClassifier()))

names = []
scores = []
for name, model in models:
    model.fit(x_train_scaled, y_train)
    y_pred = model.predict(x_test_scaled)
    scores.append(accuracy_score(y_test, y_pred))
    names.append(name)
    tr_split = pd.DataFrame({'Name': names, 'Score': scores})
    print(tr_split)
    
from sklearn.ensemble import StackingClassifier
estimators = [ m for m in models]
lr = LogisticRegression()
sclf = StackingClassifier(estimators=estimators, final_estimator=lr)
sclf.fit(x_train, y_train)
sclf.score(x_test, y_test)



#saving the chosen model

from sklearn.externals import joblib

rf_gscv = RandomForestClassifier(criterion='entropy', min_samples_leaf=3, 
                                  min_samples_split=7, n_estimators=30, n_jobs=-1, 
                                  random_state=123).fit(x_train_scaled, y_train)
joblib.dump(rf_gscv, 'q2d_rf_GSCV.pkl')


#Make Predictions using the saved model
rf_gscv = joblib.load("q2d_rf_GSCV.pkl")


# 4 - create prediction and convert it into dataframe
df_pred = pd.DataFrame(rf_gscv.predict(x_test_scaled),columns = ['prediction'])

#5 merge predction back to test dataset
final_df = pd.merge(adult_test,df_pred,how = 'left',left_index = True, right_index = True)

##Export data as CSV
final_df.to_csv('q2d_rf_GSCV.csv', index=False)

