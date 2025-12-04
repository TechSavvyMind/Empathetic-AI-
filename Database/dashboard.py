import pandas as pd
import streamlit as st
import requests
import matplotlib.pyplot as plt

API = "http://127.0.0.1:8000/dashboard"

st.title("📊 Telecom Chatbot Performance Dashboard")

# --- Total Sessions Chart ---
sessions = requests.get(f"{API}/sessions").json()
df_sessions = pd.DataFrame(sessions, columns=["date", "sessions"])
st.line_chart(df_sessions.set_index("date"))

# --- Bot vs Agent Resolution Pie ---
res = requests.get(f"{API}/resolution").json()
bot_resolved, agent_resolved = res[0]

st.subheader("Bot vs Agent Resolution")
plt.pie([bot_resolved, agent_resolved], labels=["Bot", "Agent"], autopct="%1.1f%%")
st.pyplot()

# --- Issue-wise Resolution ---
issue_data = requests.get(f"{API}/issue_resolution").json()
df_issue = pd.DataFrame(issue_data, columns=["issue", "bot", "agent"])
st.bar_chart(df_issue.set_index("issue"))

# --- Feedback Chart ---
feedback = requests.get(f"{API}/feedback").json()
positive, negative = feedback[0]
st.subheader("Customer Feedback")
plt.bar(["Positive", "Negative"], [positive, negative])
st.pyplot()

# --- Sentiment Trend ---
sentiment = requests.get(f"{API}/sentiment").json()
df_sent = pd.DataFrame(sentiment, columns=["date", "sentiment", "count"])
st.area_chart(df_sent.pivot(index="date", columns="sentiment", values="count"))
