import sqlite3
import yaml
file_path = f"direct_questions\\question_response.yaml"
# Load YAML file
with open(file_path, "r") as file:
    query_map = yaml.safe_load(file)

# def generate_chatbot_response(sql_key: str, result_dict: dict):
#     """Format result using YAML template."""

#     template = queries[sql_key]["response_template"]

#     try:
#         return template.format(**result_dict)

# Function to execute SQL query from YAML
def execute_query(query_key):
    db_path="NextGen1.db"
    if query_key not in query_map:
        return f"Query key '{query_key}' not found in YAML."

    sql_query = query_map[query_key]["sql"]
    template = query_map[query_key]["response_template"]

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    try:
        cursor.execute(sql_query)
        result = cursor.fetchone()
        if not result:
            return {customer_id: "No tickets found for this customer."}

        result_dict = dict(result)
        formatted = template.format(**result_dict)

        return {query_key: formatted}

    except Exception as e:
        return {query_key: f"Error executing query: {e}"}

    finally:
        conn.close()


if __name__ == "__main__":

    query_key = int(input("Enter query key: "))

    result = execute_query(query_key)
    print(result)