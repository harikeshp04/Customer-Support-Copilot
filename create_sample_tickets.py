# create_sample_tickets.py
import pandas as pd

sample = [
    {"id":1, "channel":"Email", "subject":"How to connect Snowflake connector", "text":"Hi, can you guide me how to set up the Snowflake connector for ingestion?"},
    {"id":2, "channel":"Chat", "subject":"SSO login failing", "text":"We can't login with our SSO. It throws an error 'Invalid token'. Please help, this is blocking access."},
    {"id":3, "channel":"WhatsApp", "subject":"Data lineage missing", "text":"Why is lineage not showing for my tables created last week?"},
    {"id":4, "channel":"Email", "subject":"Product outage", "text":"Our workspace appears down and we get 500 errors on dashboards."},
    {"id":5, "channel":"Chat", "subject":"API returns 401", "text":"When I call the API with token X, I get HTTP 401 Unauthorized."},
    {"id":6, "channel":"Email", "subject":"Glossary sync question", "text":"How does glossary sync work across regions?"},
    {"id":7, "channel":"Chat", "subject":"Sensitive data leak", "text":"I think confidential customer data was exposed in a dataset preview."},
    {"id":8, "channel":"WhatsApp", "subject":"Best practices for Tags", "text":"What are recommended best practices for tagging columns and assets?"},
    {"id":9, "channel":"Email", "subject":"Connector bug", "text":"The BigQuery connector fails on schema change."},
    {"id":10, "channel":"Chat", "subject":"How to call SDK", "text":"Example to create an asset using Atlan SDK in Python?"}
]

df = pd.DataFrame(sample)
df.to_csv("sample_tickets.csv", index=False)
print("sample_tickets.csv created with", len(df), "rows.")
