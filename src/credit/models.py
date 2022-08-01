import itertools
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from imblearn.over_sampling import RandomOverSampler
from scipy import stats
from sklearn import metrics, model_selection
from tqdm import tqdm

# Errors
from scipy.linalg import LinAlgError
from statsmodels.tools.sm_exceptions import ConvergenceWarning


# --------------------------------------------------------------------------------
# Model Exploration
# --------------------------------------------------------------------------------
def explore_logit_models(data, possible_variables, yvar="default", oversample=False):
    idx = 0
    df_res = pd.DataFrame()
    
    best_score = 0
    best_model = None
    for xvars in tqdm(possible_variables):
        xvars = list(xvars)

        if len(xvars) > 1:
            pvalue_chi2 = max([0] + [
                stats.chi2_contingency(pd.crosstab(data[x1], data[x2]))[1]
                for x1, x2 in list(itertools.combinations(xvars, 2))
                if (data[x1].dtype == object or data[x1].nunique() < 5)
                    and (data[x2].dtype == object or data[x2].nunique() < 5)
            ])
            if pvalue_chi2 > 0.05:
                continue

        df = get_modelling_data(data, xvars, yvar)

        y = df[yvar].copy()
        X = df[[col for col in df.columns if col != yvar]].copy()
        X.insert(0, "const", 1)

        X_train, X_test, y_train, y_test = model_selection.train_test_split(
            X, y, random_state=42, test_size=0.20
        )

        if oversample:
            ros = RandomOverSampler(random_state=42)
            X_over, y_over = ros.fit_resample(X_train, y_train)

            try:
                model = sm.Logit(y_over, X_over).fit(disp=False)
            except (ConvergenceWarning, RuntimeWarning, LinAlgError) as e:
                continue

            p = y_train.mean()
            q = y_over.mean()
            model.params["const"] = model.params["const"] \
                - np.log(((1.0 - p) * q) / (p * (1.0 - q)))
        else:
            try:
                model = sm.Logit(y_train, X_train).fit(disp=False)
            except (ConvergenceWarning, RuntimeWarning, LinAlgError) as e:
                continue
        
        if any(pd.isna(model.pvalues.values)):
            continue

        for xvar in xvars:
            pvalues = model.pvalues[[
                x for x in model.pvalues.index if x.startswith(xvar)
            ]]
            
            if all(pvalues > 0.05):
                continue
        
        # calculate training scores
        yp_train = model.predict(X_train)
        ks_train = stats.ks_2samp(yp_train[y_train == 0], yp_train[y_train == 1])[0]
        roc_auc_train = metrics.roc_auc_score(y_train, yp_train)
        f1_train = metrics.f1_score(y_train, (yp_train > 0.5).astype(int))
        
        # calculate test scores
        yp_test = model.predict(X_test)
        ks_test = stats.ks_2samp(yp_test[y_test == 0], yp_test[y_test == 1])[0]
        roc_auc_test = metrics.roc_auc_score(y_test, yp_test)
        f1_test = metrics.f1_score(
            y_test, (yp_test > 0.5).astype(int)
        )
        
        if ks_train < 0.9*ks_test or roc_auc_train < 0.9*roc_auc_test:
            continue
        
        df_res.loc[idx, "nvars"] = len(xvars)
        df_res.loc[idx, "yvar"] = yvar
        df_res.loc[idx, "xvars"] = str(xvars)
        df_res.loc[idx, "Coefficients"] = str(model.params.values.tolist())
        df_res.loc[idx, "p-values"] = str(model.pvalues.values.tolist())
        
        df_res.loc[idx, "KS Score (train)"] = ks_train
        df_res.loc[idx, "ROC AUC Score (train)"] = roc_auc_train
        df_res.loc[idx, "F1 Score (train)"] = f1_train
        
        df_res.loc[idx, "KS Score (test)"] = ks_test
        df_res.loc[idx, "ROC AUC Score (test)"] = roc_auc_test
        df_res.loc[idx, "F1 Score (test)"] = f1_test
        
        if best_score < ks_train:
            best_score = ks_train
            best_model = model

        idx += 1
        
    df_res.sort_values("KS Score (train)", ascending=False, inplace=True)
    
    return df_res, best_model


# --------------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------------
def plot_probabilities(y_true, y_prob, how=None):
    bins = np.linspace(0, 1, num=50)
    xarr = np.linspace(0, 1, num=1000)
    
    fig = plt.figure(figsize=(16,5))
    ax = fig.add_subplot(111)

    for y in np.sort(np.unique(y_true)):
        if how is None or how in ["hist", "histogram"]:
            ax.hist(y_prob[y_true == y], bins=bins, edgecolor="black", alpha=0.5, label=y)
        elif how == "kde":
            kde = stats.gaussian_kde(y_prob[y_true == y])
            yarr = kde(xarr)
            
            ax.fill_between(xarr, yarr, alpha=0.5, label=y)
            ax.plot(xarr, yarr, linewidth=3, linestyle='--')
    
    ax.legend()

    plt.show()
    
    
# --------------------------------------------------------------------------------
# Other Functions
# --------------------------------------------------------------------------------
def get_possible_variables(data, min_nvars=1, max_nvars=3, yvar="default"):
    variables_list = [
        col for col in data.columns
        if col != yvar and data[col].nunique() > 1
    ]

    n = len(variables_list)
    total = sum([
        math.factorial(n) // (math.factorial(k) * math.factorial(n-k))
        for k in range(min_nvars, max_nvars+1)
    ])

    possible_variables = []
    with tqdm(total=total) as pbar:
        for nvars in range(min_nvars, max_nvars+1):
            combinations = list(itertools.combinations(variables_list, nvars))

            for xvars in combinations:
                pbar.update(1)

                if len(xvars) > 1:
                    pvalue_chi2 = max([0] + [
                        stats.chi2_contingency(pd.crosstab(data[x1], data[x2]))[1]
                        for x1, x2 in list(itertools.combinations(xvars, 2))
                        if (data[x1].dtype == object or data[x1].nunique() < 5)
                            and (data[x2].dtype == object or data[x2].nunique() < 5)
                    ])
                    if pvalue_chi2 > 0.05:
                        continue
                
                possible_variables.append(list(xvars))
    
    return possible_variables


def get_modelling_data(data, xvars, yvar):
    df = data[yvar].copy()
    
    for xvar in xvars:
        if data[xvar].dtype == object or data[xvar].nunique() < 5:
            df = pd.concat(
                [
                    df,
                    pd.get_dummies(data[xvar], prefix=xvar, drop_first=True)
                ], axis=1
            )
        else:
            df = pd.concat([df, data[xvar]], axis=1)
            
    return df.copy()
