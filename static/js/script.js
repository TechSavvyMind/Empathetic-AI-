// Chatbot Logic
let isChatOpen = false;
let isMaximized = false;
let hasGreeted = false;

// Static messages for non-logged-in users
const LOGIN_PROMPT = `I am happy to assist you, but for security and account-specific queries, you need to be logged in. 
Please <a href="/login" style="color: var(--accent-color); font-weight: bold;">Login</a> or <a href="/register" style="color: var(--accent-color); font-weight: bold;">Register</a> to continue the conversation!`;

// --- UI Toggles ---

// Toggle Chat Window & Notification Popup
function toggleChat() {
    const chatWindow = document.getElementById('chatWindow');
    const chatPopup = document.getElementById('chatPopup');

    isChatOpen = !isChatOpen;

    if (isChatOpen) {
        chatWindow.classList.add('active');
        // Hide the notification popup when chat is opened
        if (chatPopup) chatPopup.style.display = 'none';

        // Initial welcome/login prompt
        if (!hasGreeted) {
            setTimeout(() => {
                if (isLoggedIn) {
                    addBotMessage(`Hi ${currentUser}, how can I help you today?`);
                } else {
                    // Initial prompt for Guest users
                    addBotMessage(`Welcome to Tata Teleservices Support! Please <a href="/login" style="color: var(--accent-color); font-weight: bold;">Login</a> or <a href="/register" style="color: var(--accent-color); font-weight: bold;">Register</a> to access account-specific help.`);
                }
                hasGreeted = true;
            }, 500);
        }
    } else {
        chatWindow.classList.remove('active');
    }
}

// Feature 1: Maximize/Minimize Logic (Full Device)
function toggleMaximize(event) {
    if (event) event.stopPropagation();

    const chatWindow = document.getElementById('chatWindow');
    const maximizeBtn = document.getElementById('maximizeBtn');
    const toggleBtn = document.querySelector('.chat-toggle');

    isMaximized = !isMaximized;

    if (isMaximized) {
        chatWindow.classList.add('maximized');
        maximizeBtn.classList.remove('fa-expand-arrows-alt');
        maximizeBtn.classList.add('fa-compress-arrows-alt'); // Change icon to compress
        maximizeBtn.title = "Minimize";
        // Hide the toggle button when maximized for full-screen effect
        if (toggleBtn) toggleBtn.style.display = 'none';
    } else {
        chatWindow.classList.remove('maximized');
        maximizeBtn.classList.remove('fa-compress-arrows-alt');
        maximizeBtn.classList.add('fa-expand-arrows-alt'); // Change icon back
        maximizeBtn.title = "Maximize";
        // Show the toggle button again
        if (toggleBtn) toggleBtn.style.display = 'flex';
    }
}

// --- Message Handling ---

// Feature 2: Send FAQ Function
function sendFAQ(question, staticAnswer) {
    // Check login status first
    if (!isLoggedIn) {
        addUserMessage(question);
        addBotMessage(LOGIN_PROMPT);
        return;
    }

    // 1. Display the User's click as a message
    addUserMessage(question);

    // 2. Display the Static Answer after short delay
    setTimeout(() => {
        addBotMessage(staticAnswer);
    }, 600);
}

function addBotMessage(text) {
    const chatBody = document.getElementById('chatBody');
    const msgDiv = document.createElement('div');
    msgDiv.className = 'message bot-msg';
    // Use innerHTML to allow for the login/register links
    msgDiv.innerHTML = text;
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

        // Non-logged-in user gate
        if (!isLoggedIn) {
            addBotMessage(LOGIN_PROMPT);
            return;
        }

        // Logged-in user: Placeholder for API call
        setTimeout(() => {
            addBotMessage("I've received your query. Our advanced AI is analyzing it...");
        }, 1000);
    }
}

// Allow Enter key to send
document.getElementById('userInput')?.addEventListener('keypress', function (e) {
    if (e.key === 'Enter') {
        sendMessage();
    }
});

// --- Initialization ---
document.addEventListener('DOMContentLoaded', () => {
    const faqContainer = document.getElementById('faqContainer');

    // Hide FAQs for non-logged-in users and display a prompt instead
    if (!isLoggedIn) {
        if (faqContainer) {
            faqContainer.innerHTML = `<p class="faq-label">Please Login to view account-specific quick actions.</p>`;
        }
    }
});