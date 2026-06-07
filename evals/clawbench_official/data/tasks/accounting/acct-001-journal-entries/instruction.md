You are given `workspace/transactions.csv` with raw business transactions (date, description, amount, type).

**Your task:**
1. Read the transactions
2. Convert each transaction into proper double-entry journal entries (debit and credit accounts)
3. Transaction types map to accounts: "sale" -> Debit: Cash/AR, Credit: Revenue; "purchase" -> Debit: Inventory/Expense, Credit: Cash/AP; "salary" -> Debit: Salary Expense, Credit: Cash; "rent" -> Debit: Rent Expense, Credit: Cash; "loan" -> Debit: Cash, Credit: Loan Payable
4. Write journal entries to `workspace/journal.csv` with columns: date, entry_id, account, debit, credit
5. Write `workspace/journal_summary.json` with total debits, total credits, and entry count. Debits must equal credits.
