import logging
from typing import List
from src.data.analysis import clean_and_preprocess_data, load_data
from src.utils.helper import get_root_path, setup_logger
from src.model.train import train_model,prepare_data
from src.model.evualate import evaluate_model
from src.model.explian import explain_model

setup_logger("main")
logger = logging.getLogger()

root_path = get_root_path()
data_path = root_path / "data" / "raw"

def load_all():
    credit_df = load_data(data_path / "creditcard.csv")
    fraud_df = load_data(data_path / "Fraud_Data.csv")
    ip_df = load_data(data_path / "IpAddress_to_Country.csv")

    return credit_df,fraud_df,ip_df

def main():
    #preprocess data
    credit_df,fraud_df,ip_df=load_all()
    clean_creadit_df=clean_and_preprocess_data(credit_df)
    clean_fraud_df=clean_and_preprocess_data(fraud_df)
    cleam_ip_df=clean_and_preprocess_data(ip_df)
    #train model
    X_train, X_test, y_train, y_test, preprocessor, num_cols, cat_cols=prepare_data(df=fraud_df,target_col="class")
    logit_model, rf_model=train_model(X_train,y_train,preprocessor,num_cols,cat_cols)
    results=evaluate_model(models=[logit_model,rf_model],X_test=X_test,y_test=y_test,names=["Logistic Regression", "Random Forest"])
    logger.info(f"Evualtion results:{results}")
    shap_values=explain_model(model=rf_model,X_train=X_train,preprocessor=preprocessor,num_cols=num_cols,cat_cols=cat_cols,max_display=15)
    logger.info(shap_values)


if __name__ == "__main__":
    main()