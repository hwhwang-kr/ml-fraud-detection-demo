import cml.data_v1 as cmldata

CONNECTION_NAME = "default-impala-aws"
conn = cmldata.get_connection(CONNECTION_NAME)

## Sample Usage to get pandas data frame
EXAMPLE_SQL_QUERY = "show databases"
dataframe = conn.get_pandas_dataframe(EXAMPLE_SQL_QUERY)
print(dataframe)
# Closing the connection

query = """
SELECT transaction_hour,
       COUNT(*) AS total_tx,
       SUM(is_fraud) AS fraud_tx,
       ROUND(100 * SUM(is_fraud)/COUNT(*), 2) AS fraud_rate
FROM hwhwang_finance_db.credit_card_transactions
GROUP BY transaction_hour
ORDER BY transaction_hour
"""
df = conn.get_pandas_dataframe(query)
df.head()

conn.close()
