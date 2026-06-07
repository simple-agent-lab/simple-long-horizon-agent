You are given `workspace/bank_statement.csv` and `workspace/company_ledger.csv`.

**Your task:**
1. Match transactions between bank and ledger by amount and date (within 2 days tolerance)
2. Identify: outstanding checks (in ledger but not bank), deposits in transit (in ledger but not bank), bank charges (in bank but not ledger)
3. Calculate adjusted bank balance and adjusted book balance
4. Write `workspace/reconciliation.json` with matched_count, outstanding_checks, deposits_in_transit, bank_charges, adjusted_bank_balance, adjusted_book_balance, is_reconciled
