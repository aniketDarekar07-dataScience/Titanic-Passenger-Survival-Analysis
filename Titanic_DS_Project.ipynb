import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# For Excel files, you need to install openpyxl
# Run: pip install openpyxl


df = pd.read_csv("Titanic-Dataset.csv")
df
df.shape

df.info()
df.columns
# ALL COLUMNS - MEAN, MEDIAN, MODE



# ============================================
# NUMERICAL COLUMNS
# ============================================

#age
df['Age'].mean()
df['Age'].median()
df['Age'].mode()
df['Age'].isnull().sum()

#fare
df['Fare'].mean()
df['Fare'].median()
df['Fare'].mode()
df['Fare'].isnull().sum()

# SIBSP
df['SibSp'].mean()
df['SibSp'].median()
df['SibSp'].mode()
df['SibSp'].isnull().sum()

#parch
df['Parch'].mean()
df['Parch'].median()
df['Parch'].mode()
df['Parch'].isnull().sum()

#PASSENGERID 
df['PassengerId'].mean()
df['PassengerId'].median()
df['PassengerId'].mode()
df['PassengerId'].isnull().sum()

# SURVIVED 
df['Survived'].mean()
df['Survived'].median()
df['Survived'].mode()
df['Survived'].isnull().sum()

#PCLASS 
df['Pclass'].mean()
df['Pclass'].median()
df['Pclass'].mode()
df['Pclass'].isnull().sum()

# ============================================
# CATEGORICAL COLUMNS
# ============================================

#SEX
df['Sex'].mode()
df['Sex'].isnull().sum()

#EMBARKED 
df['Embarked'].mode()
df['Embarked'].isnull().sum()

# TICKET 
df['Ticket'].mode()
df['Ticket'].isnull().sum()

# NAME
df['Name'].mode()
df['Name'].isnull().sum()

df.head()
df.tail()
# Missing Values
# 3. CHECK MISSING VALUES

df.isnull().sum()


# Visualize missing values
df.isnull().sum().plot(kind='bar')
plt.title("Missing Values")
plt.show()
# DATA CLEANING
# Drop Cabin (too many missing values)
df.drop('Cabin', axis=1, inplace=True)
# Fill Age with median
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Check again
df.isnull().sum()
# UNIVARIATE ANALYSIS
df['Survived'].value_counts()
df['Survived'].value_counts().plot(kind='bar')
plt.title("Survival Distribution")
plt.show()
#Gender
df['Sex'].value_counts()
df['Sex'].value_counts().plot(kind='bar')
plt.title("Gender Distribution")
plt.show()


# Pclass
df['Pclass'].value_counts()
df['Pclass'].value_counts().plot(kind='bar')
plt.title("Passenger Class Distribution")
plt.show()


# Age
df['Age'].plot(kind='hist',bins=20)
plt.title("Age Distribution")
plt.show()

df['Age'].plot(kind='box')
plt.title("Age Box Plot")
plt.show()


# Fare
df['Fare'].plot(kind='hist', bins=20)
plt.title("Fare Distribution")
plt.show()
# BIVARIATE ANALYSIS
# Survival by Gender
pd.crosstab(df['Sex'], df['Survived'])
pd.crosstab(df['Sex'], df['Survived']).plot(kind='bar')
plt.title("Survival by Gender")
plt.legend(['Died', 'Survived'])
plt.show()
# Survival by Pclass
pd.crosstab(df['Pclass'], df['Survived'])
pd.crosstab(df['Pclass'], df['Survived']).plot(kind='bar')
plt.title("Survival by Pclass")
plt.legend(['Died', 'Survived'])
plt.show()
# Survival by Embarked
pd.crosstab(df['Embarked'], df['Survived'])
pd.crosstab(df['Embarked'], df['Survived']).plot(kind='bar')
plt.title("Survival by Embarked")
plt.legend(['Died', 'Survived'])
plt.show()
# Age vs Survival
df.boxplot(column='Age', by='Survived')
plt.title("Age by Survival")
plt.show()
# Fare vs Survival
df.boxplot(column='Fare', by='Survived')
plt.title("Fare by Survival")
plt.show()
# MULTIVARIATE ANALYSIS
#Survival by Gender and Pclass

pd.crosstab([df['Sex'], df['Pclass']], df['Survived'])
pd.crosstab([df['Sex'], df['Pclass']], df['Survived']).plot(kind='bar', figsize=(10,6))
plt.title("Survival by Gender and Pclass")
plt.legend(['Died', 'Survived'])
plt.show()
# Correlation Matrix
numeric_cols = ['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
df[numeric_cols].corr()
# Heatmap
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()
# loc-iloc
# Female passengers who survived
df.loc[(df['Sex'] == 'female') & (df['Survived'] == 1)]


# Male passengers who died
df.loc[(df['Sex'] == 'male') & (df['Survived'] == 0)]


# First class female survivors
df.loc[(df['Pclass'] == 1) & (df['Sex'] == 'female') & (df['Survived'] == 1)]


# Passengers with Age > 60
df.loc[df['Age'] > 60]
# ILOC - uses numbers (position)
df.iloc[0:5, 0:3]    # First 5 rows, first 3 columns

# LOC - uses labels (names)
df.loc[0:5, ['Name', 'Age', 'Sex']]  # Rows 0-5, specific columns
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

df['Title'] = df['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())

df['AgeGroup'] = pd.cut(df['Age'], bins=[0,12,18,35,60,100], labels=['Child','Teen','Young','Adult','Senior'])
# Age Group:

def age_group(age):
    if age <= 12:
        return 'Child'
    elif age <= 18:
        return 'Teen'
    elif age <= 35:
        return 'Young Adult'
    elif age <= 60:
        return 'Adult'
    else:
        return 'Senior'

df['AgeGroup'] = df['Age'].apply(age_group)
print("\n5. AGE GROUP:")
print(df[['Age', 'AgeGroup']].head(10))
#fare group

def fare_group(fare):
    if fare <= 10:
        return 'Low'
    elif fare <= 50:
        return 'Medium'
    elif fare <= 100:
        return 'High'
    else:
        return 'Very High'

df['FareGroup'] = df['Fare'].apply(fare_group)
print("\n6. FARE GROUP:")
print(df[['Fare', 'FareGroup']].head(10))
df

# GROUP BY FUNCTION
1. GROUPBY 'SEX' 

df.groupby('Sex')['Age'].mean()
df.groupby('Sex')['Age'].median()
df.groupby('Sex')['Age'].min()
df.groupby('Sex')['Age'].max()
df.groupby('Sex')['Age'].count()
df.groupby('Sex')['Age'].std()


df.groupby('Sex')['Fare'].mean()
df.groupby('Sex')['Fare'].median()
df.groupby('Sex')['Fare'].min()
df.groupby('Sex')['Fare'].max()
df.groupby('Sex')['Fare'].count()
df.groupby('Sex')['Fare'].std()


df.groupby('Sex')['SibSp'].mean()
df.groupby('Sex')['SibSp'].median()
df.groupby('Sex')['SibSp'].min()
df.groupby('Sex')['SibSp'].max()
df.groupby('Sex')['SibSp'].count()


df.groupby('Sex')['Parch'].mean()
df.groupby('Sex')['Parch'].median()
df.groupby('Sex')['Parch'].min()
df.groupby('Sex')['Parch'].max()
df.groupby('Sex')['Parch'].count()


df.groupby('Sex')['Survived'].mean()
df.groupby('Sex')['Survived'].count()
df.groupby('Sex')['Survived'].sum()

2. GROUPBY 'PCLASS'
df.groupby('Pclass')['Age'].mean()
df.groupby('Pclass')['Age'].median()
df.groupby('Pclass')['Age'].min()
df.groupby('Pclass')['Age'].max()
df.groupby('Pclass')['Age'].count()
df.groupby('Pclass')['Fare'].mean()
df.groupby('Pclass')['Fare'].median()
df.groupby('Pclass')['Fare'].min()
df.groupby('Pclass')['Fare'].max()
df.groupby('Pclass')['Fare'].count()
df.groupby('Pclass')['SibSp'].mean()
df.groupby('Pclass')['SibSp'].median()
df.groupby('Pclass')['SibSp'].min()
df.groupby('Pclass')['SibSp'].max()
df.groupby('Pclass')['SibSp'].count()
df.groupby('Pclass')['Parch'].mean()
df.groupby('Pclass')['Parch'].median()
df.groupby('Pclass')['Parch'].min()
df.groupby('Pclass')['Parch'].max()
df.groupby('Pclass')['Parch'].count()
df.groupby('Pclass')['Survived'].mean()
df.groupby('Pclass')['Survived'].count()
df.groupby('Pclass')['Survived'].sum()

#3. GROUPBY 'EMBARKED' 
df.groupby('Embarked')['Age'].mean()
df.groupby('Embarked')['Age'].median()
df.groupby('Embarked')['Age'].min()
df.groupby('Embarked')['Age'].max()
df.groupby('Embarked')['Age'].count()

df.groupby('Embarked')['Fare'].mean()
df.groupby('Embarked')['Fare'].median()
df.groupby('Embarked')['Fare'].min()
df.groupby('Embarked')['Fare'].max()
df.groupby('Embarked')['Fare'].count()

df.groupby('Embarked')['SibSp'].mean()
df.groupby('Embarked')['SibSp'].median()
df.groupby('Embarked')['SibSp'].min()
df.groupby('Embarked')['SibSp'].max()
df.groupby('Embarked')['SibSp'].count()

df.groupby('Embarked')['Parch'].mean()
df.groupby('Embarked')['Parch'].median()
df.groupby('Embarked')['Parch'].min()
df.groupby('Embarked')['Parch'].max()
df.groupby('Embarked')['Parch'].count()

df.groupby('Embarked')['Survived'].mean()
df.groupby('Embarked')['Survived'].count()
df.groupby('Embarked')['Survived'].sum()

#4. GROUPBY 'SURVIVED'
df.groupby('Survived')['Age'].mean()
df.groupby('Survived')['Age'].median()
df.groupby('Survived')['Age'].min()
df.groupby('Survived')['Age'].max()
df.groupby('Survived')['Age'].count()
df.groupby('Survived')['Fare'].mean()
df.groupby('Survived')['Fare'].median()
df.groupby('Survived')['Fare'].min()
df.groupby('Survived')['Fare'].max()
df.groupby('Survived')['Fare'].count()
df.groupby('Survived')['SibSp'].mean()
df.groupby('Survived')['SibSp'].median()
df.groupby('Survived')['SibSp'].min()
df.groupby('Survived')['SibSp'].max()
df.groupby('Survived')['SibSp'].count()
df.groupby('Survived')['Parch'].mean()
df.groupby('Survived')['Parch'].median()
df.groupby('Survived')['Parch'].min()
df.groupby('Survived')['Parch'].max()
df.groupby('Survived')['Parch'].count()
