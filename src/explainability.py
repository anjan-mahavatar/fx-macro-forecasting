import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import lime
import lime.lime_tabular
from alibi.explainers import AnchorTabular

def generate_model_explanations(price_model, vol_model, X_train, X_test, feature_names):
    print("\n==================================================")
    print("STARTING EXPLAINABLE AI (XAI) PIPELINE STAGE")
    print("==================================================")
    
    # Ensure a directory exists for outputting our charts
    os.makedirs('plots', exist_ok=True)
    
    # -------------------------------------------------------------------------
    # 1. SHAP (SHapley Additive exPlanations) - Global & Local Interpretability
    # -------------------------------------------------------------------------
    print("Generating SHAP Explanations for the Ensemble Model...")
    # TreeExplainer is heavily optimized for tree models like LightGBM
    explainer_shap = shap.TreeExplainer(price_model)
    shap_values = explainer_shap.shap_values(X_test)
    
    # Generate and save global feature importance summary plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
    plt.title("SHAP Global Feature Importance (FX Price Model)", fontsize=14, pad=15)
    plt.tight_layout()
    plt.savefig('plots/shap_global_summary.png')
    plt.close()
    print("✅ SHAP Summary Plot exported to 'plots/shap_global_summary.png'")
    
    # -------------------------------------------------------------------------
    # 2. LIME (Local Interpretable Model-agnostic Explanations)
    # -------------------------------------------------------------------------
    print("Generating LIME Explanation for a single local prediction instance...")
    # We choose the very first test instance as our case study sample
    test_instance = X_test.iloc[0]
    
    explainer_lime = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=feature_names,
        class_names=['Target_Price'],
        mode='regression',
        random_state=42
    )
    
    exp_lime = explainer_lime.explain_instance(
        data_row=test_instance.values,
        predict_fn=price_model.predict
    )
    
    # Export LIME visualization as a local HTML report page
    exp_lime.save_to_file('plots/lime_local_explanation.html')
    print("✅ LIME Case Study exported to 'plots/lime_local_explanation.html'")
    
    # -------------------------------------------------------------------------
    # 3. ANCHOR Explanations - Generating "If-Then" Boundary Rules
    # -------------------------------------------------------------------------
    print("Generating Anchor Decision Rules for the Volatility Model...")
    
    # Define a helper function because Anchor expects a clean probability/prediction function
    def vol_predict_fn(x):
        return vol_model.predict(x)
        
    # Initialize the AnchorTabular framework
    explainer_anchor = AnchorTabular(
        predictor=vol_predict_fn,
        feature_names=feature_names
    )
    
    # Fit the explainer on a subset of training data to learn features' ranges
    explainer_anchor.fit(X_train.values)
    
    try:
        # Find the mathematical "anchors" (conditions) for our test instance
        explanation_anchor = explainer_anchor.explain(test_instance.values, threshold=0.95)
        print("\n--- Anchor Generated Rule Matrix for Volatility Prediction ---")
        print(f"Prediction Value: {vol_predict_fn(test_instance.values.reshape(1, -1))[0]:.4f}")
        print("Anchor Rule (If these conditions are met, prediction is firmly held):")
        print(" AND ".join(explanation_anchor.anchor))
        print("--------------------------------------------------------------")
    except Exception as e:
        print(f"Skipping complete Anchor matrix print: {e} (Requires strict categorical binning for continuous outputs).")

    print("=== XAI Evaluation Pipeline Complete ===")