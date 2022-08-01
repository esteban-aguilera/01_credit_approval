import itertools
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import warnings
import xgboost as xgb

from imblearn.over_sampling import RandomOverSampler
from scipy import stats
from sklearn import ensemble, linear_model, metrics, model_selection, tree
from tqdm import tqdm

# package imports
from . import get_global
from .utils import paths, strings


# --------------------------------------------------------------------------------
# Model (scikit-learns)
# --------------------------------------------------------------------------------
def explore_models(
        data, possible_variables, yvar="default", model_type="Logit",
        oversample=False, col_score=None
    ):
    if col_score is None:
        col_score = "KS Score (train)"
    
    idx = 0
    df_res = pd.DataFrame()
    
    best_score = 0
    best_model = None
    for xvars in tqdm(possible_variables, desc=strings.capitalize(model_type, sep="_")):
        (X_train, y_train), (X_test, y_test) = get_train_test(data, xvars, yvar)
        
        model = fit_model(X_train, y_train, model_type=model_type, oversample=oversample)
            
        # calculate model scores
        ks_train, roc_auc_train, f1_train = calculate_scores(
            y_train, model.predict_proba(X_train)[:,1]
        )
        ks_test, roc_auc_test, f1_test = calculate_scores(
            y_test, model.predict_proba(X_test)[:,1]
        )
        
        # skip overfitted models and underfitted models
        if (
            abs(ks_test / ks_train - 1.0) > 0.15
            or abs(roc_auc_test / roc_auc_train - 1.0) > 0.15
        ):
            continue
        
        df_res.loc[idx, "model_type"] = model_type
        df_res.loc[idx, "oversample"] = oversample
        df_res.loc[idx, "nvars"] = len(xvars)
        df_res.loc[idx, "xvars"] = str(xvars)
        df_res.loc[idx, "yvar"] = yvar
        
        df_res.loc[idx, "KS Score (train)"] = ks_train
        df_res.loc[idx, "ROC AUC Score (train)"] = roc_auc_train
        df_res.loc[idx, "F1 Score (train)"] = f1_train
        
        df_res.loc[idx, "KS Score (test)"] = ks_test
        df_res.loc[idx, "ROC AUC Score (test)"] = roc_auc_test
        df_res.loc[idx, "F1 Score (test)"] = f1_test
        
        if best_score < df_res.loc[idx, col_score]:
            best_score = ks_train
            best_model = model

        idx += 1
        
    df_res.sort_values(col_score, ascending=False, inplace=True)
    
    return df_res, best_model
    
    
def fit_model(X_train, y_train, model_type="Logit", oversample=False):
    model_type = strings.snake_case(model_type)
    if model_type in ["logit", "logistic"]:
        model = linear_model.LogisticRegression(
            random_state=get_global("SEED_MODEL")
        )
    elif model_type in ["tree", "decision_tree"]:
        model = tree.DecisionTreeClassifier(
            min_samples_leaf=0.05, random_state=get_global("SEED_MODEL")
        )
    elif model_type in ["random_forest"]:
        model = ensemble.RandomForestClassifier(
            n_estimators=10, random_state=get_global("SEED_MODEL")
        )
    elif model_type == "xgboost":
        model = xgb.XGBClassifier(
            random_state=get_global("SEED_MODEL")
        )
    else:
        raise ValueError(f"The model_type '{model_type}' is invalid.")
    
    if oversample:
        ros = RandomOverSampler(random_state=get_global("SEED_RANDOM_OVER_SAMPLER"))
        X_train, y_train = ros.fit_resample(X_train, y_train)

    model.fit(X_train, y_train)
            
    return model


# --------------------------------------------------------------------------------
# Logit (statsmodels)
# --------------------------------------------------------------------------------
def explore_logit_models(
        data, possible_variables, yvar="default",
        oversample=False, col_score=None
    ):
    if col_score is None:
        col_score = "KS Score (train)"
    
    idx = 0
    df_res = pd.DataFrame()
    
    best_score = 0
    best_model = None
    for xvars in tqdm(possible_variables, desc="Logit"):
        (X_train, y_train), (X_test, y_test) = get_train_test(data, xvars, yvar, const=True)
        
        model = fit_logit_model(X_train, y_train, oversample=oversample)
        
        if any(pd.isna(model.pvalues.values)):
            continue

        for xvar in xvars:
            pvalues = model.pvalues[[
                x for x in model.pvalues.index if x.startswith(xvar)
            ]]
            
            if all(pvalues > 0.05):
                continue
            
        # calculate model scores
        ks_train, roc_auc_train, f1_train = calculate_scores(
            y_train, model.predict(X_train)
        )
        ks_test, roc_auc_test, f1_test = calculate_scores(
            y_test, model.predict(X_test)
        )
        
        # skip overfitted models
        if (
            abs(ks_test / ks_train - 1.0) > 0.15
            or abs(roc_auc_test / roc_auc_train - 1.0) > 0.15
        ):
            continue
        
        df_res.loc[idx, "model"] = "Logit"
        df_res.loc[idx, "oversample"] = oversample
        df_res.loc[idx, "nvars"] = len(xvars)
        df_res.loc[idx, "xvars"] = str(xvars)
        df_res.loc[idx, "yvar"] = yvar
        df_res.loc[idx, "Coefficients"] = str(model.params.values.tolist())
        df_res.loc[idx, "p-values"] = str(model.pvalues.values.tolist())
        
        df_res.loc[idx, "KS Score (train)"] = ks_train
        df_res.loc[idx, "ROC AUC Score (train)"] = roc_auc_train
        df_res.loc[idx, "F1 Score (train)"] = f1_train
        
        df_res.loc[idx, "KS Score (test)"] = ks_test
        df_res.loc[idx, "ROC AUC Score (test)"] = roc_auc_test
        df_res.loc[idx, "F1 Score (test)"] = f1_test
        
        if best_score < df_res.loc[idx, col_score]:
            best_score = ks_train
            best_model = model

        idx += 1
        
    df_res.sort_values(col_score, ascending=False, inplace=True)
    
    return df_res, best_model
    
    
def fit_logit_model(X_train, y_train, oversample=False):
    if oversample:
        ros = RandomOverSampler(random_state=get_global("SEED_RANDOM_OVER_SAMPLER"))
        X_over, y_over = ros.fit_resample(X_train, y_train)

        model = sm.Logit(y_over, X_over).fit(disp=False)

        p = y_train.mean()
        q = y_over.mean()
        model.params["const"] = model.params["const"] \
            - np.log(((1.0 - p) * q) / (p * (1.0 - q)))
    else:
        model = sm.Logit(y_train, X_train).fit(disp=False)
            
    return model


# --------------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------------
def plot_probabilities(y_true, y_prob, how=None, figax=None, bins=None, xarr=None,
                       model_type=None, filename=None):
    if how is None or how in ["hist", "histogram"]:
        if xarr is not None:
            warnings.warn(f"xarr is not used with how='{how}'")
            
        return plot_probabilities_hist(
            y_true, y_prob, figax=figax, bins=bins, density=True,
            filename=filename, model_type=model_type
        )
    elif how == "kde":
        if bins is not None:
            warnings.warn(f"bins is not used with how='{how}'")
            
        return plot_probabilities_kde(
            y_true, y_prob, figax=figax, xarr=xarr,
            filename=filename, model_type=model_type
        )
    else:
        raise ValueError(f"how must have values 'hist' or 'kde', not '{how}'.")

    
def plot_probabilities_hist(y_true, y_prob, figax=None, bins=None,
                            model_type=None, filename=None):
    if figax is None:
        fig = plt.figure(figsize=(16,5))
        ax = fig.add_subplot(111)
    
    if bins is None:
        bins = np.linspace(0, 1, num=50)
        
    if filename is not None and "/" in filename:
        paths.mkdir("/".join(filename.split("/")[:-1]))

    for y in np.sort(np.unique(y_true)):
        ax.hist(y_prob[y_true == y], bins=bins, edgecolor="black", alpha=0.5, label=f"y = {y}")
    
    ax.legend()
    
    if model_type is not None:
        ax.set_title(model_type)
    
    if filename is None:
        return fig, ax

    fig.tight_layout()
    fig.savefig(filename)


def plot_probabilities_kde(y_true, y_prob, figax=None, xarr=None,
                           model_type=None, filename=None):
    if figax is None:
        fig = plt.figure(figsize=(16,5))
        ax = fig.add_subplot(111)
    
    if xarr is None:
        xarr = np.linspace(0, 1, num=1000)
        
    if filename is not None and "/" in filename:
        paths.mkdir("/".join(filename.split("/")[:-1]))
    
    for y in np.sort(np.unique(y_true)):
        kde = stats.gaussian_kde(y_prob[y_true == y])
        yarr = kde(xarr)
        
        ax.fill_between(xarr, yarr, alpha=0.5, label=f"y = {y}")
        ax.plot(xarr, yarr, linewidth=3, linestyle='--')
    
    ax.legend()
    
    if model_type is not None:
        ax.set_title(model_type)

    if filename is None:
        return fig, ax

    fig.tight_layout()
    fig.savefig(filename)
        
    
# --------------------------------------------------------------------------------
# General Functions
# --------------------------------------------------------------------------------
def calculate_scores(y_true, y_prob):
    ks = stats.ks_2samp(y_prob[y_true == 0], y_prob[y_true == 1])[0]
    roc_auc = metrics.roc_auc_score(y_true, y_prob)
    f1_score = metrics.f1_score(y_true, (y_prob > 0.5).astype(int))
    
    return ks, roc_auc, f1_score


def get_train_test(data, xvars, yvar="default", const=False):
    df = get_modelling_data(data, xvars, yvar)

    X = df[[col for col in df.columns if col != yvar]].copy()
    if const:
        X.insert(0, "const", 1)

    y = df[yvar].copy()

    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, random_state=get_global("SEED_TRAIN_TEST_SPLIT"), test_size=0.20
    )
    return (X_train, y_train), (X_test, y_test)


def get_modelling_data(data, xvars, yvar="default"):
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
    with tqdm(total=total, desc="Analyzing Possible Variables") as pbar:
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
