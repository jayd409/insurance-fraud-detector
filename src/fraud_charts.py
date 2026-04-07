import matplotlib.pyplot as plt
import pandas as pd
from analysis import fraud_by_policy, prior_claims_analysis, days_to_report_analysis

def fraud_by_policy_chart(df):
    """Chart 1: Fraud rate by policy type."""
    data = fraud_by_policy(df)
    fig, ax = plt.subplots(figsize=(8, 5))
    data['mean'].plot(kind='bar', ax=ax, color='steelblue')
    ax.set_title('Fraud Rate by Policy Type', fontsize=12, fontweight='bold')
    ax.set_xlabel('Policy Type')
    ax.set_ylabel('Fraud Rate')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    plt.tight_layout()
    return fig

def claim_amount_distribution(df):
    """Chart 2: Claim amount distribution (fraud vs legit)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    fraud = df[df['is_fraud'] == 1]['claim_amount']
    legit = df[df['is_fraud'] == 0]['claim_amount']
    ax.hist(fraud, bins=40, alpha=0.6, label='Fraud', color='red')
    ax.hist(legit, bins=40, alpha=0.6, label='Legitimate', color='green')
    ax.set_title('Claim Amount Distribution', fontsize=12, fontweight='bold')
    ax.set_xlabel('Claim Amount ($)')
    ax.set_ylabel('Frequency')
    ax.legend()
    return fig

def feature_importance_chart(importance_series):
    """Chart 3: Feature importance (horizontal bar)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    importance_series.sort_values().plot(kind='barh', ax=ax, color='coral')
    ax.set_title('Feature Importance for Fraud Detection', fontsize=12, fontweight='bold')
    ax.set_xlabel('Absolute Weight')
    plt.tight_layout()
    return fig

def prior_claims_chart(df):
    """Chart 4: Prior claims vs fraud rate."""
    data = prior_claims_analysis(df)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(data['num_prior_claims'], data['mean'], color='teal')
    ax.set_title('Fraud Rate by Prior Claims', fontsize=12, fontweight='bold')
    ax.set_xlabel('Number of Prior Claims')
    ax.set_ylabel('Fraud Rate')
    return fig

def days_to_report_chart(df):
    """Chart 5: Days to report vs fraud rate."""
    data = days_to_report_analysis(df)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(len(data)), data['mean'], color='mediumpurple')
    ax.set_title('Fraud Rate by Days to Report', fontsize=12, fontweight='bold')
    ax.set_xlabel('Report Delay Bin')
    ax.set_ylabel('Fraud Rate')
    ax.set_xticks(range(len(data)))
    ax.set_xticklabels([str(x)[:10] for x in data['report_bin']], rotation=45, fontsize=8)
    plt.tight_layout()
    return fig

def risk_score_distribution(scores):
    """Chart 6: Risk score distribution."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(scores, bins=50, color='orange', edgecolor='black', alpha=0.7)
    ax.set_title('Fraud Risk Score Distribution', fontsize=12, fontweight='bold')
    ax.set_xlabel('Risk Score')
    ax.set_ylabel('Frequency')
    return fig
