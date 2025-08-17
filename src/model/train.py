from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from src.data.analysis import load_data
from src.utils.helper import get_root_path

root_path=get_root_path()
data_path=root_path/"data"/"processed"

cridit_data=load_data(data_path/"creditcard.csv")
fraud_data=load_data(data_path/"fraud.csv")
ip_data=load_data(data_path/"ip.csv")

def prepare_data(df, target_col="class"):
    """
    Split data into train/test and set up preprocessing.
    Stratified split
    Standardize numeric cols
    One-hot encode categorical cols
    Returns: X_train, X_test, y_train, y_test, preprocessor, num_cols, cat_cols
    """
    # Identify categorical and numeric features
    cat_cols = [c for c in ["source","browser","sex","country","device_id"] if c in df.columns]
    num_cols = [c for c in ["purchase_value","age","hour_of_day","day_of_week",
                            "time_since_signup_min","time_since_prev_min","txn_ct_24h"] if c in df.columns]

    # Drop rows with missing values in selected cols
    model_df = df[cat_cols + num_cols + [target_col]].dropna().copy()

    # Features/target
    X = model_df.drop(columns=[target_col])
    y = model_df[target_col].astype(int)

    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Preprocessor
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # older sklearn
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", ohe, cat_cols),
    ])

    return X_train, X_test, y_train, y_test, preprocessor, num_cols, cat_cols


def train_model(X_train, y_train, preprocessor, num_cols, cat_cols):
    """
    Train two models on training data (with SMOTE).
    Logistic Regression (baseline)
    Random Forest (ensemble)
    Returns: trained logistic model, trained RF model
    """

    # Logistic Regression pipeline
    logit_model = ImbPipeline([
        ("pre", preprocessor),
        ("smote", SMOTE(random_state=42)),
        ("clf", LogisticRegression(max_iter=2000))
    ])

    # Random Forest pipeline
    rf_model = ImbPipeline([
        ("pre", preprocessor),
        ("smote", SMOTE(random_state=42)),
        ("clf", RandomForestClassifier(
            n_estimators=600, random_state=42, n_jobs=-1
        ))
    ])

    logit_model.fit(X_train, y_train)
    rf_model.fit(X_train, y_train)

    return logit_model, rf_model
