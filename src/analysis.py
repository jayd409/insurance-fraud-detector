import numpy as np
import pandas as pd

def fraud_by_policy(df):
    """Fraud rate by policy type."""
    return df.groupby('policy_type')['is_fraud'].agg(['sum', 'count', 'mean']).round(4)

def claim_distribution(df):
    """Statistics for claim amounts by fraud status."""
    return df.groupby('is_fraud')['claim_amount'].describe()

def confusion_matrix(y_true, y_pred_binary):
    """Calculate TP, FP, TN, FN."""
    tp = np.sum((y_pred_binary == 1) & (y_true == 1))
    fp = np.sum((y_pred_binary == 1) & (y_true == 0))
    tn = np.sum((y_pred_binary == 0) & (y_true == 0))
    fn = np.sum((y_pred_binary == 0) & (y_true == 1))
    return {'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn}

def metrics(y_true, y_pred_proba, threshold=0.5):
    """Calculate accuracy, precision, recall."""
    y_pred_binary = (y_pred_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred_binary)
    tp, fp, tn, fn = cm['TP'], cm['FP'], cm['TN'], cm['FN']

    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'confusion_matrix': cm
    }

def prior_claims_analysis(df):
    """Fraud rate by number of prior claims."""
    return df.groupby('num_prior_claims')['is_fraud'].agg(['sum', 'count', 'mean']).reset_index()

def days_to_report_analysis(df, bins=6):
    """Fraud rate binned by days to report."""
    df['report_bin'] = pd.cut(df['days_to_report'], bins=bins)
    result = df.groupby('report_bin', observed=True)['is_fraud'].agg(['sum', 'count', 'mean']).reset_index()
    result['report_bin'] = result['report_bin'].astype(str)
    return result
