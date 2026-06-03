import shap
import lime.lime_tabular
from alibi.explainers import AnchorTabular

def apply_shap(model, X_train, X_test):
    # SHAP (Global and Local): Calculates exact marginal contributions of features
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    # Generate a Summary Plot (Graph)
    shap.summary_plot(shap_values, X_test, show=False)
    return shap_values

def apply_lime(model, X_train, X_test_instance, feature_names):
    # LIME (Local): Fits a simple linear model around a single prediction
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=feature_names,
        mode='regression'
    )
    # Explain one specific prediction
    exp = explainer.explain_instance(X_test_instance.values, model.predict)
    exp.show_in_notebook(show_table=True)
    return exp

def apply_anchor(predict_fn, X_train, feature_names):
    # Anchor (Local): Finds "If-Then" rules that firmly anchor a prediction
    # Note: Anchors are natively designed for classification. 
    # For regression, you typically bin the target variable first.
    explainer = AnchorTabular(predict_fn, feature_names)
    explainer.fit(X_train.values)
    # Returns rules like: "IF Volatility > 1.2 AND Volume < 1000 THEN Prediction = High"
    return explainer
