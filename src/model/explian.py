import shap

def explain_model(model, X_train, preprocessor, num_cols, cat_cols, max_display=15):
    """
    Explain the best model with SHAP.
    Transforms training data with preprocessor
    Runs SHAP summary plot
    model: trained pipeline (e.g., rf_model)
    """
    # Transform training data
    X_train_transformed = preprocessor.fit_transform(X_train)

    # Get feature names
    num_features = preprocessor.named_transformers_["num"].get_feature_names_out(num_cols).tolist()
    cat_features = preprocessor.named_transformers_["cat"].get_feature_names_out(cat_cols).tolist()
    feature_names = num_features + cat_features

    # Choose the classifier inside the pipeline
    clf = model.named_steps["clf"]

    # Create SHAP explainer
    explainer = shap.Explainer(clf.predict_proba, X_train_transformed, feature_names=feature_names)
    shap_values = explainer(X_train_transformed)

    # Plot summary
    shap.summary_plot(shap_values[:,:,1], X_train_transformed, feature_names=feature_names, max_display=max_display)
    
    return shap_values
