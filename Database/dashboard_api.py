import sqlite3
from fastapi import FastAPI

app = FastAPI()
DB_PATH = "C:\\Users\\GenAICHNSIRUSR31\\Desktop\\avengers_team\\DB\\NextGen1.db"

def run_query(sql):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(sql)
    data = cursor.fetchall()
    conn.close()
    return data

@app.get("/dashboard/sessions")
def get_sessions():
    return run_query("SELECT metric_date, total_sessions FROM daily_chatbot_metrics;")

@app.get("/dashboard/resolution")
def get_resolution_stats():
    return run_query("""
        SELECT SUM(bot_resolved_count), SUM(agent_resolved_count)
        FROM daily_chatbot_metrics;
    """)

@app.get("/dashboard/issue_resolution")
def get_issue_resolution():
    return run_query("""
        SELECT it.name,
               SUM(CASE WHEN cir.resolved_by='bot' THEN 1 END),
               SUM(CASE WHEN cir.resolved_by='agent' THEN 1 END)
        FROM chatbot_issue_resolution cir
        JOIN issue_types it ON cir.issue_type_id = it.issue_type_id
        GROUP BY it.name;
    """)

@app.get("/dashboard/feedback")
def get_feedback():
    return run_query("""
        SELECT 
          SUM(CASE WHEN feedback_rating >= 4 THEN 1 END),
          SUM(CASE WHEN feedback_rating <= 2 THEN 1 END)
        FROM chatbot_sessions;
    """)

@app.get("/dashboard/sentiment")
def get_sentiment():
    return run_query("""
        SELECT DATE(timestamp), sentiment, COUNT(*)
        FROM chatbot_interactions
        GROUP BY DATE(timestamp), sentiment;
    """)
