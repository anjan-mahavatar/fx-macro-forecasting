from transformers import pipeline

def get_macro_risk_score(text_headlines):
    # Load a pre-trained financial LLM
    analyzer = pipeline("text-classification", model="ProsusAI/finbert")
    results = analyzer(text_headlines)
    
    # Convert sentiment to a numerical risk score (0 to 1)
    risk_scores = [1.0 if res['label'] == 'negative' else 0.0 for res in results]
    return risk_scores