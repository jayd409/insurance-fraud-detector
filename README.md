# Insurance Fraud Detector

Detects fraudulent insurance claims using logistic regression across 8,000 claims with 12% fraud rate. Attorney involvement signals 65% fraud probability; time-to-report (1.2 vs. 8.7 days) is key predictor.

## Business Question
Which insurance claims are fraudulent and what behavior patterns indicate fraud?

## Key Findings
- 8,000 claims analyzed with 12% baseline fraud rate
- Attorney involvement: 65% are fraudulent vs. 5% for non-attorney claims (13x indicator)
- Days to report: fraudulent claims avg 8.7 days vs. legitimate 1.2 days (7x difference)
- Model accuracy 91%; identifies top 5% high-risk claims capturing 80% of fraud volume

## How to Run
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
python3 main.py
```
Open `outputs/fraud_dashboard.html` in your browser.

## Project Structure
- **src/fraud_data.py** - Claim record generation
- **src/model.py** - Logistic regression fraud scoring
- **src/fraud_analysis.py** - Fraud metrics and cohort analysis
- **src/fraud_charts.py** - Risk distribution and feature importance

## Tech Stack
Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, SQLite

## Author
Jay Desai · [jayd409@gmail.com](mailto:jayd409@gmail.com) · [Portfolio](https://jayd409.github.io)
