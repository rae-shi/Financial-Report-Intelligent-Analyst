import psycopg2
try:
    conn = psycopg2.connect(
        host="127.0.0.1", # Use IP to avoid IPv6 issues
        database="financial_db",
        user="myuser",
        password="mypassword",
        port="5432"
    )
    print("Connection Successful!")
    conn.close()
except Exception as e:
    print(f"Connection Failed: {e}")