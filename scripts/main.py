import logging
from typing import List
from src.data.analysis import clean_and_preprocess_data, load_data
from src.utils.helper import get_root_path, setup_logger

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
    
    credit_df,fraud_df,ip_df=load_all()
    creadit_df=clean_and_preprocess_data(credit_df)
    clean_fraud_df=clean_and_preprocess_data(fraud_df)
    cleam_ip_df=clean_and_preprocess_data(ip_df)
    