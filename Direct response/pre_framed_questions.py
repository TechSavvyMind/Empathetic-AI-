import sqlite3
import yaml
file_path = f"pre_framed_questions\\questions.yaml"
# Load YAML file
with open(file_path, "r") as file:
    query_map = yaml.safe_load(file)

# Function to execute SQL query from YAML
def execute_query(customer_id, query_key):
    db_path="NextGen1.db"
    if query_key not in query_map:
        return f"Query key '{query_key}' not found in YAML."

    sql_query = query_map[query_key]["sql"]

    # Connect to SQLite
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    try:
        # Execute query with customer_id
        cursor.execute(sql_query, (customer_id,))
        result = cursor.fetchone()
        if not result:
            return "No data found."
        return dict(result) 
    except Exception as e:
        return f"Error executing query: {e}"
    finally:
        conn.close()

if __name__ == "__main__":
    customer_id = 1
    query_key = input()
    result = execute_query(customer_id, query_key)
    print(result)


