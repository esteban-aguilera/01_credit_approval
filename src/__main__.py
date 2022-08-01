import pandas as pd

# package imports
from credit import preprocess, models
from credit.utils import paths, excel


# --------------------------------------------------------------------------------
# Parameters
# --------------------------------------------------------------------------------


# --------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------
def main():
    paths.mkdir("output/img")
    
    yvar = "default"

    # preprocess
    data = preprocess.load_credit()
    data = preprocess.group_credit_variables(data.copy())
    possible_variables = models.get_possible_variables(data)

    best_score = 0
    df_res = pd.DataFrame()
    for model_type in ["logit", "decision_tree", "random_forest", "xgboost"]:
        df_model, model = models.explore_models(
            data, possible_variables, model_type=model_type, oversample=True
        )

        df = models.get_modelling_data(data, eval(df_model["xvars"].values[0]), yvar)
        
        X = df[[col for col in df.columns if col != yvar]].copy()
        y = df[yvar]
        for how in ["hist", "kde"]:
            models.plot_probabilities(
                y, model.predict_proba(X)[:,1], how=how,
                model_type=model_type,
                filename=f"output/img/{model_type}_{how}.png"
            )
        
        df_res = pd.concat([df_res, df_model], ignore_index=True)
        df_res.sort_values("KS Score (train)", ascending=False, inplace=True)
        
        if best_score < df_res["KS Score (train)"].values[0]:
            best_score = df_res["KS Score (train)"].values[0]

    df_res.reset_index(drop=True, inplace=True)
    excel.save_dataframe(df_res, "output/Fitted Models.xlsx", "Results")


# --------------------------------------------------------------------------------
# Functions
# --------------------------------------------------------------------------------


# --------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
