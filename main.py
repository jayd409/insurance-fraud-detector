import sys
sys.path.insert(0, 'src')

from fraud_data import generate_claims
from model import train, score, normalize, importance
from fraud_analysis import metrics
from fraud_charts import (fraud_by_policy_chart, claim_amount_distribution,
                    feature_importance_chart, prior_claims_chart,
                    days_to_report_chart, risk_score_distribution)
from utils import save_html
from database import save_to_db, query
import numpy as np

df = generate_claims(8000)
# Will save to db after calculating risk_score

feature_cols = ['claim_amount', 'days_to_report', 'num_prior_claims',
                'policy_tenure_yrs', 'claimant_age', 'witnesses',
                'repair_estimate_ratio']
X = df[feature_cols].values
y = df['is_fraud'].values

X_norm, mean, std = normalize(X)
w = train(X_norm, y, lr=0.05, epochs=500)

risk_scores = score(X_norm, w)
df['risk_score'] = risk_scores

save_to_db(df, 'claims')

m = metrics(y, risk_scores, threshold=0.5)
accuracy = m['accuracy']
precision = m['precision']
recall = m['recall']

charts = [
    ('Fraud Rate by Policy Type', fraud_by_policy_chart(df)),
    ('Claim Amount Distribution', claim_amount_distribution(df)),
    ('Feature Importance', feature_importance_chart(importance(w, feature_cols))),
    ('Prior Claims vs Fraud', prior_claims_chart(df)),
    ('Days to Report vs Fraud', days_to_report_chart(df)),
    ('Risk Score Distribution', risk_score_distribution(risk_scores))
]

kpis = [
    ('Fraud Rate', f"{df['is_fraud'].mean()*100:.1f}%"),
    ('Model Accuracy', f"{accuracy*100:.1f}%"),
    ('Precision', f"{precision*100:.1f}%"),
    ('Recall', f"{recall*100:.1f}%"),
]

save_html(charts, 'Insurance Fraud Detection', kpis, 'outputs/fraud_dashboard.html')

print("\n--- SQL Analytics (SQLite) ---")
fraud_by_policy = query("""
    SELECT policy_type,
           COUNT(*) as total_claims,
           SUM(CASE WHEN is_fraud = 1 THEN 1 ELSE 0 END) as fraud_count,
           ROUND(100.0 * SUM(CASE WHEN is_fraud = 1 THEN 1 ELSE 0 END) / COUNT(*), 1) as fraud_rate
    FROM claims
    GROUP BY policy_type
    ORDER BY fraud_rate DESC
""")
print("\nFraud Rate by Policy Type:")
print(fraud_by_policy.to_string(index=False))

avg_claim_by_fraud = query("""
    SELECT CASE WHEN is_fraud = 1 THEN 'Fraudulent' ELSE 'Legitimate' END as claim_type,
           COUNT(*) as count,
           ROUND(AVG(claim_amount), 2) as avg_amount,
           ROUND(MIN(claim_amount), 2) as min_amount,
           ROUND(MAX(claim_amount), 2) as max_amount
    FROM claims
    GROUP BY is_fraud
""")
print("\nAverage Claim Amount by Fraud Status:")
print(avg_claim_by_fraud.to_string(index=False))

risk_dist = query("""
    SELECT
        CASE
            WHEN risk_score < 0.25 THEN 'Low'
            WHEN risk_score < 0.50 THEN 'Medium'
            WHEN risk_score < 0.75 THEN 'High'
            ELSE 'Very High'
        END as risk_level,
        COUNT(*) as count,
        ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM claims), 1) as pct,
        ROUND(100.0 * AVG(CASE WHEN is_fraud = 1 THEN 1 ELSE 0 END), 1) as actual_fraud_rate
    FROM claims
    GROUP BY risk_level
    ORDER BY risk_score
""")
print("\nRisk Score Distribution:")
print(risk_dist.to_string(index=False))

fraud_rate = df['is_fraud'].mean() * 100
print(f"\nFraud Rate: {fraud_rate:.1f}% | Model Accuracy: {accuracy*100:.1f}% | Precision: {precision*100:.1f}% | Recall: {recall*100:.1f}%")
