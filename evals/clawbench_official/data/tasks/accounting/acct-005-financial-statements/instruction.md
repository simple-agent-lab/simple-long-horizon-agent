You are given `workspace/trial_balance.csv` (account, type, debit_balance, credit_balance) where type is one of: asset, liability, equity, revenue, expense.

**Your task:**
1. Generate Income Statement: Revenue - Expenses = Net Income. Write to `workspace/income_statement.csv`
2. Generate Balance Sheet: Assets = Liabilities + Equity + Retained Earnings (net income). Write to `workspace/balance_sheet.csv`
3. Verify accounting equation holds
4. Write `workspace/financial_summary.json` with total_revenue, total_expenses, net_income, total_assets, total_liabilities, total_equity, equation_balanced
