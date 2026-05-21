from rouge_score import rouge_scorer
import pandas as pd

def calculate_rouge(reference_summary, generated_summary):
    """
    Calculates ROUGE-1, ROUGE-2, and ROUGE-L scores for a generated summary.
    
    :param reference_summary: The ground-truth/human-written summary (string).
    :param generated_summary: The summary outputted by your model (string).
    :return: A dictionary containing the F1, Precision, and Recall scores.
    """
    # Initialize the scorer for ROUGE-1, ROUGE-2, and ROUGE-L
    # use_stemmer=True removes word suffixes (e.g., "running" -> "run") for fairer matching
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    # Calculate scores
    scores = scorer.score(reference_summary, generated_summary)
    
    # Format the results into a clean dictionary
    results = {}
    for metric in ['rouge1', 'rouge2', 'rougeL']:
        results[metric] = {
            'Precision': round(scores[metric].precision, 4),
            'Recall': round(scores[metric].recall, 4),
            'F1-Score': round(scores[metric].fmeasure, 4)
        }
        
    return results

def display_rouge_scores(reference, generated, model_name="Model"):
    """
    Helper function to calculate and print ROUGE scores in a readable format.
    """
    print(f"--- ROUGE Evaluation for {model_name} ---")
    scores = calculate_rouge(reference, generated)
    
    # Convert to pandas DataFrame for clean notebook rendering
    df = pd.DataFrame(scores).T
    print(df.to_string())
    print("-" * 40)
    return df

# ==========================================
# Example Usage inside your demo.ipynb
# ==========================================
if __name__ == "__main__":
    # Example reference (ground truth)
    human_summary = "The stock market experienced a significant drop today due to rising inflation fears. Tech stocks were hit the hardest."
    
    # Example model outputs (pretend these came from your functions)
    bart_output = "Stock markets fell sharply today amid inflation concerns, with tech companies seeing the biggest losses."
    tfidf_output = "The stock market experienced a significant drop today. Tech stocks were hit the hardest."
    
    # Evaluate BART
    display_rouge_scores(human_summary, bart_output, model_name="Hybrid BART")
    
    # Evaluate TF-IDF alone
    display_rouge_scores(human_summary, tfidf_output, model_name="Statistical (TF-IDF)")