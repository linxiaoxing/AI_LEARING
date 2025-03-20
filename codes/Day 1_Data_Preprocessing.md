# Data preprocessing
Data preprocessing is accomplished in 6 step

This example uses the [Data](https://github.com/linxiaoxing/AI_LEARING/blob/main/datasets%20/Data.csv) and the [code]()

## Step 1: Importing the library
```Python
import numpy as np
import pandas as pd
```

## Step 2: Importing the data sets
```python
//The last column is label
dataset = pd.read_csv('Data.csv')//Reading csv files
X = dataset.iloc[ : , :-1].values//.iloc[Row，Column]
Y = dataset.iloc[ : , 3].values  // : All rows or Column；[a]Row a or column
                                 // [a,b,c]Row or column a,b,c
```

## Step 3: Processing of lost data
```python
from sklearn.impute import SimpleImputer
import numpy as np

# データXを仮定
# X = np.array([[1, 2, np.nan], [4, np.nan, 6], [7, 8, 9]])

# SimpleImputerを使用
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")

# 1列目から2列目の欠損値を補完
X[:, 1:3] = imputer.fit_transform(X[:, 1:3])

print(X)
```
## Step 4:Parsing categorized data
```python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[ : , 0] = labelencoder_X.fit_transform(X[ : , 0])
```

Use ColumnTransformer
```python
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer

# OneHotEncoderを適用
ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(), [0])],  # 0列目をOne-Hotエンコーディング
    remainder='passthrough'  # 他の列はそのまま
)
```

### Step 5:Creating dummy variables
```python
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
labelencoder_Y = LabelEncoder()
Y =  labelencoder_Y.fit_transform(Y)
```

Use ColumnTransformer
```python
X = ct.fit_transform(X)

# Encoding the target variable Y using LabelEncoder
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)

# output result
print("X:\n", X)
print("Y:\n", Y)
```

## Step 6:Split the dataset into a training set and a test set
```python
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split( X , Y , test_size = 0.2, random_state = 0)
```

## Step 7:feature quantization
```python
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
# Calculate standardized parameters (mean and standard deviation) using training set data and transform the training set
X_train = sc_X.fit_transform(X_train)
# Transform the test set using the same standardized parameters (mean and standard deviation)
X_test = sc_X.transform(X_test)
```
