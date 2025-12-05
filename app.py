from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import sqlite3
import os
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash
from  Orchestrator import orchestrator_app
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

app = Flask(__name__)
app.secret_key = os.urandom(32) 
app.config['SESSION_PERMANENT'] = False
DB_NAME = "./Database/NextGen1.db"

# --- Database Helper Functions ---
def init_db():
    if not os.path.exists(DB_NAME):
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT NOT NULL,
                password TEXT NOT NULL
            )
        ''')
        # Create default admin user
        admin_pass = generate_password_hash("admin")
        try:
            cursor.execute("INSERT INTO users (username, email, password) VALUES (?, ?, ?)", 
                           ("admin", "admin@telia.com", admin_pass))
            print("Default admin user created.")
        except sqlite3.IntegrityError:
            pass
            
        conn.commit()
        conn.close()

def get_db_connection():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn

# --- Decorators ---
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('You need to log in to access this page.', 'error')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# --- Routes ---
@app.before_request
def before_request():
    init_db()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        conn = get_db_connection()
        user = conn.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
        conn.close()
        
        if user and check_password_hash(user['password'], password):
            session['user_id'] = user['id']
            session['username'] = user['username']
            session['customer_id'] = user['customer_id']  # 🔴 ADD THIS
            flash(f'Welcome back, {user["username"]}!', 'success')
            return redirect(url_for('dashboard'))

        else:
            flash('Invalid username or password.', 'error')
            
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        
        if password != confirm_password:
            flash('Passwords do not match!', 'error')
            return redirect(url_for('register'))
            
        hashed_password = generate_password_hash(password)
        
        try:
            conn = get_db_connection()
            conn.execute('INSERT INTO users (username, email, password) VALUES (?, ?, ?)',
                         (username, email, hashed_password))
            conn.commit()
            conn.close()
            flash('Registration successful! Please log in.', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Username already exists.', 'error')
            
    return render_template('register.html')

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template(
        'dashboard.html', username=session['username'], 
        customer_id=session.get("customer_id"))
@app.route('/services')
@login_required
def services():
    # Placeholder for the services page content
    return render_template('services.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))


# --- API for future Chatbot Logic ---
@app.route('/api/chat', methods=['POST'])
@login_required
def chat_api():
    data = request.json
    user_message = data.get("message")
    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    # Retrieve or initialize chat history from session (as dicts)
    raw_history = session.get('chat_history', [])
    chat_history = []
    for msg in raw_history:
        if msg["type"] == "human":
            chat_history.append(HumanMessage(content=msg["content"]))
        elif msg["type"] == "ai":
            chat_history.append(AIMessage(content=msg["content"]))

    chat_history.append(HumanMessage(content=user_message))

    response = orchestrator_app.invoke({
        "messages": chat_history,
        "customer_id": session.get("customer_id")
    })
    bot_response = response["final_response"]
    print(f"Orchestrator: {response}")
    chat_history.append(AIMessage(content=bot_response))

    # Trim history if too long
    if len(chat_history) > 6:
        chat_history = chat_history[-2:]

    # Store as serializable dicts
    session['chat_history'] = [
        {"type": "human", "content": m.content} if isinstance(m, HumanMessage) else {"type": "ai", "content": m.content}
        for m in chat_history
    ]
    return jsonify({"response": bot_response})

if __name__ == '__main__':
    # init_db()
    app.run(port=5873, debug=True)