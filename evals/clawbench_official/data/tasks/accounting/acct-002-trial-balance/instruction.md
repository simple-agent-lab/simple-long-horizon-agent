You are given `workspace/ledger.csv` with general ledger entries (date, account, debit, credit).

**Your task:**
1. Read the ledger entries
2. Aggregate by account: sum all debits and credits for each account
3. Calculate net balance for each account (total_debit - total_credit)
4. Generate trial balance: `workspace/trial_balance.csv` (account, total_debit, total_credit, balance)
5. Check if total debits == total credits (within 0.01 tolerance)
6. Write `workspace/validation_report.json`:
```json
{
  "total_accounts": 10,
  "total_debits": 50000.0,
  "total_credits": 50000.0,
  "is_balanced": true,
  "imbalance_amount": 0.0,
  "accounts_with_issues": []
}
```
