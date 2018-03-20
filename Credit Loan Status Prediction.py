#Predicting Loan Status of credit card

#import data
import pandas as pd
train = pd.read_csv('./data/samples/Credit Loan Status/credit_train.csv')

#Delete Empty rows
train = train.head(100000)

#Delete Duplicate rows
train = train_data.drop_duplicates(subset='Loan ID')

#Formating 'years in current job' cells
train['Years in current job'] = train_data['Years in current job'].replace('8 years', 8)
train['Years in current job'] = train_data['Years in current job'].replace('<1 year', 0.5)
train['Years in current job'] = train_data['Years in current job'].replace('1 year', 1)
train['Years in current job'] = train_data['Years in current job'].replace('2 years', 2)
train['Years in current job'] = train_data['Years in current job'].replace('3 years', 3)
train['Years in current job'] = train_data['Years in current job'].replace('4 years', 4)
train['Years in current job'] = train_data['Years in current job'].replace('5 years', 5)
train['Years in current job'] = train_data['Years in current job'].replace('6 years', 6)
train['Years in current job'] = train_data['Years in current job'].replace('7 years', 7)
train['Years in current job'] = train_data['Years in current job'].replace('9 years', 9)
train['Years in current job'] = train_data['Years in current job'].replace('10+ years', 11)
train['Years in current job'] = train_data['Years in current job'].replace('n/a', train['Years in current job'].mode())

#Finding types of Vsriables
train.dtypes

#Encoding  variables from object to categorical variables
train['Loan Status'] = train['Loan Status'].astype('category')
train['Term'] = train['Term'].astype('category')
train['Years in current job'] = train['Years in current job'].astype('category')
train['Home Ownership'] = train['Home Ownership'].astype('category')
train['Purpose'] = train['Purpose'].astype('category')

#Converting Categorical into integers
cat_columns = train.select_dtypes(['category']).columns
train[cat_columns] = train[cat_columns].apply(lambda x: x.cat.codes)

#Changing Incorrect Credit Score
for i in train_data['Credit Score'].loc[train_data['Credit Score'] > 1000]:
     train_data['Credit Score'].loc[train_data['Credit Score'] > 1000] = i/10

#Finding Missing Values
print(train_data.isnull().sum())

#Imputing Missing Values
train_data['Credit Score'] = train_data['Credit Score'].fillna(train_data['Credit Score'].mean())
train_data['Annual Income'] = train_data['Annual Income'].fillna(train_data['Annual Income'].mean())
train_data['Maximum Open Credit'] = train_data['Maximum Open Credit'].fillna(train_data['Maximum Open Credit'].mean())
#For Categorical Variables
train_data['Bankruptcies'] = train_data['Bankruptcies'].fillna(train_data['Bankruptcies'].value_counts().index[0])
train_data['Tax Liens'] = train_data['Tax Liens'].fillna(train_data['Tax Liens'].value_counts().index[0])

#Ordering the data table
train_data.columns.tolist()
train_data = train_data[[
 'Current Loan Amount',
 'Term',
 'Credit Score',
 'Annual Income',
 'Years in current job',
 'Home Ownership',
 'Purpose',
 'Monthly Debt',
 'Years of Credit History',
 'Number of Open Accounts',
 'Number of Credit Problems',
 'Current Credit Balance',
 'Maximum Open Credit',
 'Bankruptcies',
 'Tax Liens',
'Loan Status']]

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
train_data.iloc[:,:15] = sc.fit_transform(train_data.iloc[:,:15])

#Splitting Data
from sklearn.cross_validation import train_test_split
d = train_data.values
x_train, x_test, y_train, y_test = train_test_split(d[:,:15], d[:,15:], test_size = 0.25, random_state = 0)

#Model Preparation

#Fitting model to KNN
from sklearn.neighbors import KNeighborsClassifier
knn_classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
knn_classifier.fit(x_train, y_train)

#Fitting model to Random Forest
from sklearn.ensemble import RandomForestClassifier
rf_classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
rf_classifier.fit(x_train, y_train)

#Predicting the data
knn_pred = knn_classifier.predict(x_test)
rf_pred = rf_classifier.predict(x_test)

#Confusion Matrix
from sklearn.metrics import confusion_matrix
knn_cm = confusion_matrix(y_test, knn_pred) #69
rf_cm = confusion_matrix(y_test, rf_pred)   #74
