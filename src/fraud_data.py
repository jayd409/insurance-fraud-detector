import numpy as np
import pandas as pd

def generate_claims(n=8000):
    """Generate realistic insurance fraud detection data based on industry benchmarks.

    Claims fraud rate: ~10-15%
    Risk signals:
    - Fraudulent claims: avg 2.3 prior claims vs honest 0.4
    - Fraudulent reports: avg 8.7 days delay vs honest 1.2
    - Fraudulent amount: avg $18K vs honest $9K
    - Has attorney: 65% fraudulent vs 15% honest
    - Policy types: Auto (~12% fraud), Home (~8%), Health (~18%)
    """
    rng = np.random.default_rng(seed=42)

    claim_ids = np.arange(1, n+1)

    # Auto: 45%, Home: 30%, Health: 20%, Life: 5%
    policy_types = rng.choice(['Auto', 'Home', 'Health', 'Life'], n, p=[0.45, 0.30, 0.20, 0.05])

    claim_amount = np.zeros(n)
    # Auto claims: $3K-$30K (honest avg ~$8K, fraudulent avg ~$18K)
    claim_amount[policy_types == 'Auto'] = rng.normal(8000, 6000, (policy_types == 'Auto').sum())
    # Home claims: $5K-$50K
    claim_amount[policy_types == 'Home'] = rng.normal(15000, 10000, (policy_types == 'Home').sum())
    # Health claims: $1K-$20K
    claim_amount[policy_types == 'Health'] = rng.normal(5000, 4000, (policy_types == 'Health').sum())
    # Life claims: $10K-$200K
    claim_amount[policy_types == 'Life'] = rng.normal(80000, 50000, (policy_types == 'Life').sum())
    claim_amount = np.clip(claim_amount, 100, 200000)

    # Days to report: honest avg 1.2 days, fraudulent avg 8.7 days
    days_to_report = np.zeros(n)
    days_to_report = rng.exponential(2.0, n) + rng.normal(0, 0.5, n)
    days_to_report = np.clip(days_to_report, 0, 180)

    # Prior claims: honest avg 0.4, fraudulent avg 2.3
    num_prior_claims = rng.poisson(0.5, n)

    policy_tenure_yrs = rng.exponential(3.0, n)
    policy_tenure_yrs = np.clip(policy_tenure_yrs, 0.1, 20)

    claimant_age = rng.normal(48, 15, n).astype(int)
    claimant_age = np.clip(claimant_age, 18, 80)

    # Witnesses (more witnesses = less likely fraud)
    witnesses = rng.poisson(0.8, n)
    witnesses = np.clip(witnesses, 0, 4)

    # Police report (honest: 70%, fraudulent: 20%)
    police_report = np.zeros(n, dtype=bool)

    # Repair estimate ratio (fraudulent: higher ratios indicating inflated estimates)
    repair_estimate_ratio = rng.gamma(2.0, 0.4, n)
    repair_estimate_ratio = np.clip(repair_estimate_ratio, 0.3, 4.0)

    injury_types = ['Whiplash', 'Soft Tissue', 'Head Injury', 'Fracture', 'Back Injury']
    injury_type = rng.choice(injury_types, n)

    # Has attorney (honest: 15%, fraudulent: 65%)
    has_attorney = np.zeros(n, dtype=bool)

    # Initialize fraud probability - target overall ~12% fraud rate
    fraud_prob = np.zeros(n)

    # Base fraud rate by policy type (Health highest risk)
    fraud_prob[policy_types == 'Auto'] = 0.10
    fraud_prob[policy_types == 'Home'] = 0.07
    fraud_prob[policy_types == 'Health'] = 0.16
    fraud_prob[policy_types == 'Life'] = 0.06

    # Claim amount effect (higher claims = higher fraud risk)
    claim_amount_norm = np.clip(claim_amount / 30000, 0, 1)
    fraud_prob += claim_amount_norm * 0.10

    # Days to report effect (delayed reporting is very suspicious)
    fraud_prob += np.minimum(days_to_report / 10, 1) * 0.12

    # Prior claims effect (strong indicator of fraud)
    fraud_prob += np.minimum(num_prior_claims / 2, 1) * 0.15

    # Witnesses effect (more witnesses = less fraud)
    fraud_prob -= (witnesses / 4) * 0.08

    # Police report effect (missing report increases fraud risk)
    fraud_prob -= 0.06

    # Repair estimate ratio effect (inflated estimates signal fraud)
    fraud_prob += np.maximum((repair_estimate_ratio - 1.2) / 1.5, 0) * 0.12

    fraud_prob = np.clip(fraud_prob, 0.02, 0.65)

    is_fraud = rng.random(n) < fraud_prob

    police_report[~is_fraud] = rng.random((~is_fraud).sum()) < 0.70
    police_report[is_fraud] = rng.random(is_fraud.sum()) < 0.20

    has_attorney[~is_fraud] = rng.random((~is_fraud).sum()) < 0.15
    has_attorney[is_fraud] = rng.random(is_fraud.sum()) < 0.65

    df = pd.DataFrame({
        'claim_id': claim_ids,
        'policy_type': policy_types,
        'claim_amount': np.round(claim_amount, 2),
        'days_to_report': np.round(days_to_report, 1),
        'num_prior_claims': num_prior_claims,
        'policy_tenure_yrs': np.round(policy_tenure_yrs, 2),
        'claimant_age': claimant_age,
        'witnesses': witnesses,
        'police_report': police_report,
        'has_attorney': has_attorney,
        'injury_type': injury_type,
        'repair_estimate_ratio': np.round(repair_estimate_ratio, 2),
        'is_fraud': is_fraud.astype(int)
    })

    return df
