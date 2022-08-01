import pandas as pd

from sklearn import tree

# package imports
from . import get_global


# --------------------------------------------------------------------------------
# Excel FIles
# --------------------------------------------------------------------------------
def load_credit(filename:str=None) -> pd.DataFrame:
    if filename is None:
        filename = f"{get_global('PATH_DATA')}/credit.csv"

    df = pd.read_csv(filename)
    
    df["default"] -= 1
    
    return df


def group_credit_variables(data, min_samples=0.15, yvar="default"):
    xvars = [col for col in data.columns if col != yvar]

    for xvar in xvars:
        if data[xvar].dtype != object and data[xvar].nunique() > 5:
            continue
        
        X = pd.get_dummies(data[xvar])
        y = data[yvar]

        tree_classifier = tree.DecisionTreeClassifier(
            min_samples_leaf=min_samples, random_state=42
        )    
        tree_classifier.fit(X, y)

        buckets = pd.concat(
            [data[xvar], pd.Series(tree_classifier.apply(X), name="bucket")],
            axis=1
        ).groupby("bucket")[xvar].unique()
        buckets.reset_index(drop=True, inplace=True)
        
        if len(buckets) < data[xvar].nunique():
            data[xvar] = data[xvar].replace({
                value:bucket 
                for bucket, values in buckets.items()
                for value in values
            })

    return data
    

def get_ordinal_dummies(ser, values):
    df = pd.DataFrame()
    
    missing_values = [x for x in ser.unique() if x not in values]
    if missing_values:
        raise ValueError([f"The value {missing_values[0]} is not included in the input values."])
    
    cum_values = []
    for x in values[:-1]:
        cum_values.append(x)
        
        df[f"{ser.name} ({x})"] = ser.isin(cum_values).astype(int)

    return df
