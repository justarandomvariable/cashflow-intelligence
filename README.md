# Cash Flow Intelligence: Financial Diagnostics & Revenue Forecasting

This is a project I built to practice end-to-end data analysis — from messy raw data all the way to a forecasting model — using a synthetic dataset modeled on a small service business's monthly financials (2018–2024).

I wanted to go beyond a typical "clean CSV → make some charts" project. So I picked a problem that actually needed some financial thinking: a business can look profitable every month and still run into cash problems. I wanted to see if I could find that gap in the data myself, and then build something that could actually forecast where the business was headed.

## What I did

**1. Cleaned a genuinely messy dataset.**
The raw data had three different date formats mixed together, currency symbols and commas in numeric columns, missing values, a duplicate row, and negative headcount values. I wrote a parser to normalize the dates, stripped and converted the currency strings, imputed the missing expense values, and caught a duplicate-row bug that was quietly double-counting one month's data.

**2. Explored revenue, expenses, and cash flow.**
Went chart by chart — revenue vs. cash collected, cash balance over time, expense breakdown, DSO — and tried to answer a specific business question with each one instead of just describing the plot.

**3. Broke revenue into trend, seasonality, and noise** using time series decomposition, to check whether the growth I was seeing was real or just noise.

**4. Flagged anomalies two ways** (IQR and Z-score) and compared what each method caught, IQR flagged a lot more months than Z-score did, which was a useful lesson in how sensitive these methods are to your choice of threshold.

**5. Forecasted revenue with ARIMA and SARIMA**, then actually tested which one held up, including 5-fold cross-validation, since a single train/test split isn't enough to trust a model. Turned out the simpler ARIMA model outperformed SARIMA here, mainly because there wasn't enough historical data (~6.5 seasonal cycles) for the seasonal model to learn a reliable pattern instead of overfitting.

## What I found

- The business is profitable in 79 of 84 months, but its cash balance still fell in 5 of 7 years, being profitable on paper and having cash in the bank turned out to be two different things.
- That gap wasn't really an operations problem, it came from things like tax payments and owner withdrawals that never show up in "operating expenses" at all.
- ARIMA beat SARIMA on both a single holdout test and cross-validation, which taught me that a fancier model isn't automatically a better one, especially with limited data.

## Feature Engineering & Models

Beyond the raw columns, I built a few features to actually measure financial health instead of just eyeballing charts:

- **DSO (Days Sales Outstanding)** — how many days on average it takes to collect on billed revenue
- **Cash ratio** — how many months of operating expenses the current cash balance could cover
- **Burn rate** — months where operating expenses outpaced cash actually collected

For forecasting, I tested two models on the revenue series:

- **ARIMA(1,1,1)** — a baseline that only looks at the series' own trend and momentum
- **SARIMA(1,1,1)(1,1,0,12)** — the same thing, plus a 12-month seasonal component

I evaluated both with a 6-month holdout (MAE, RMSE) and then again with 5-fold cross-validation, since one train/test split isn't really enough to trust a result. ARIMA won on both. With only 78 months of training data, SARIMA's seasonal term didn't have enough repeated yearly cycles to learn a reliable pattern, so it ended up overfitting to noise instead.

## Tools

Python, Pandas, Matplotlib, Seaborn, Statsmodels, Scikit-learn

## Files

- `cashflow-intelligence.ipynb` — the full notebook
- `cashflow_intelligence_data.csv` — the dataset

## Running it

```bash
pip install pandas numpy matplotlib seaborn statsmodels scikit-learn
jupyter notebook cashflow-intelligence.ipynb
```
