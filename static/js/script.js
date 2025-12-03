// Global state variables passed from Flask
// const currentUser; // Defined in layout.html
// const isLoggedIn; // Defined in layout.html

let isChatOpen = false;
let hasProactivelyGreeted = false; // Tracks if the proactive login message was shown

// --- Helper Functions ---

function toggleChat() {
    const chatWindow = document.getElementById('chatWindow');
    const chatNotification = document.getElementById('chatNotification');
    isChatOpen = !isChatOpen;
    
    if (isChatOpen) {
        chatWindow.classList.add('active');
        // Clear any notifications when opening
        chatNotification.classList.remove('active');
        // Ensure status is correctly set
        document.getElementById('chat-status').innerText = isLoggedIn ? `Logged in as ${currentUser}` : 'Guest Mode';

    } else {
        chatWindow.classList.remove('active');
    }
}

/**
 * Adds a standard bot message (text only) to the chat body.
 * @param {string} text - The message text.
 */
function addBotMessage(text) {
    const chatBody = document.getElementById('chatBody');
    const msgDiv = document.createElement('div');
    msgDiv.className = 'message bot-msg';
    msgDiv.innerText = text;
    chatBody.appendChild(msgDiv);
    chatBody.scrollTop = chatBody.scrollHeight;
}

/**
 * Adds a bot message containing text and the Login/Register buttons.
 * @param {string} text - The message text.
 */
function addLoginRequiredMessage(text) {
    const chatBody = document.getElementById('chatBody');
    const msgDiv = document.createElement('div');
    msgDiv.className = 'message bot-msg';
    
    // Create text node
    const textNode = document.createTextNode(text);
    msgDiv.appendChild(textNode);
    
    // Create button container (chat-actions)
    const actionsDiv = document.createElement('div');
    actionsDiv.className = 'chat-actions';
    
    // Create Login button
    const loginBtn = document.createElement('a');
    loginBtn.href = '/login';
    loginBtn.className = 'chat-btn chat-login-btn';
    loginBtn.innerText = 'Login';
    
    // Create Register button
    const registerBtn = document.createElement('a');
    registerBtn.href = '/register';
    registerBtn.className = 'chat-btn chat-register-btn';
    registerBtn.innerText = 'Register';

    actionsDiv.appendChild(loginBtn);
    actionsDiv.appendChild(registerBtn);
    
    msgDiv.appendChild(actionsDiv);

    chatBody.appendChild(msgDiv);
    chatBody.scrollTop = chatBody.scrollHeight;
}


function addUserMessage(text) {
    const chatBody = document.getElementById('chatBody');
    const msgDiv = document.createElement('div');
    msgDiv.className = 'message user-msg';
    msgDiv.innerText = text;
    chatBody.appendChild(msgDiv);
    chatBody.scrollTop = chatBody.scrollHeight;
}

function sendMessage() {
    const input = document.getElementById('userInput');
    const text = input.value.trim();
    
    if (text) {
        addUserMessage(text);
        input.value = '';
        
        // --- Gatekeeping Logic: Check Login Status ---
        if (!isLoggedIn) {
            // If the user is a guest, display the login message with buttons and stop.
            setTimeout(() => {
                addLoginRequiredMessage(
                    "I'm your dedicated Telia Assistant. To provide personalized support and access your account, please login or signup to continue our chat."
                );
            }, 500);
            return; // STOP execution for guests
        }
        // --- End Gatekeeping Logic ---
        
        // Logic for LOGGED-IN users (future API calls)
        
        // Show typing indicator or simple processing
        setTimeout(() => {
            // Placeholder response - in future connect to API
            addBotMessage("I've received your account-related query. Our smart systems will process this soon!");
        }, 1000);
        
        // Example of calling the backend API (commented out until fully implemented)
        /*
        fetch('/api/chat', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({message: text})
        })
        .then(r => r.json())
        .then(data => addBotMessage(data.response));
        */
    }
}

/**
 * Initializes the chat status display on page load.
 */
function initChatbot() {
    document.getElementById('chat-status').innerText = isLoggedIn ? `Logged in as ${currentUser}` : 'Guest Mode';
    
    // Ensure the proactive guest message (with buttons) or logged-in greeting 
    // is rendered by Flask in layout.html, so no need for JS to add the initial message.
    // However, if the user is logged in, we must ensure the initial message disappears
    // upon interaction (not needed now, but good practice).
}

// Allow Enter key to send message when input is focused
document.addEventListener('DOMContentLoaded', () => {
    const input = document.getElementById('userInput');
    if (input) {
        input.addEventListener('keyup', (event) => {
            if (event.key === 'Enter') {
                sendMessage();
            }
        });
    }
});