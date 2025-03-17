# Data preprocessing
Data preprocessing is accomplished in 6 step

This example uses the [Data](https://github.com/linxiaoxing/AI_LEARING/blob/main/datasets%20/Data.csv) and the [code]()

## Step 1: Importing the library
```Python
import numpy as np
import pandas as pd
```

## Step 1: Importing the data sets
```python
//The last column is label
dataset = pd.read_csv('Data.csv')//Reading csv files
X = dataset.iloc[ : , :-1].values//.iloc[Row，Column]
Y = dataset.iloc[ : , 3].values  // : All rows or Column；[a]Row a or column
                                 // [a,b,c]Row or column a,b,c
```
