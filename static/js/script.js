// ============ SAFE GLOBAL CONFIG ============
// These may or may not be defined by the template.
// We safely read them from window and normalize types.

const CURRENT_USER =
  typeof window !== "undefined" && window.currentUser
    ? String(window.currentUser)
    : "";

const IS_LOGGED_IN =
  typeof window !== "undefined" &&
  (window.isLoggedIn === true || window.isLoggedIn === "true");

// ============ GLOBAL STATE ============

let isChatOpen = false;
let isMaximized = false;
let hasGreeted = false;

const CHAT_KEY = `chatHistory:${CURRENT_USER || "guest"}`;
let chatHistory = [];

// Static message for non-logged-in users
const LOGIN_PROMPT = `
I am happy to assist you, but for security and account-specific queries, you need to be logged in. 
Please <a href="/login" style="color: var(--accent-color); font-weight: bold;">Login</a> or 
<a href="/register" style="color: var(--accent-color); font-weight: bold;">Register</a> to continue the conversation!
`;

// ============ STORAGE HELPERS ============

function saveChatHistory() {
  try {
    sessionStorage.setItem(CHAT_KEY, JSON.stringify(chatHistory));
  } catch (e) {
    console.warn("Failed to save chat history:", e);
  }
}

function loadChatHistory() {
  try {
    const saved = sessionStorage.getItem(CHAT_KEY);
    if (!saved) {
      chatHistory = [];
      return;
    }

    chatHistory = JSON.parse(saved);
    const chatBody = document.getElementById("chatBody");
    if (!chatBody) return;

    chatHistory.forEach((msg) => {
      const msgDiv = document.createElement("div");
      msgDiv.className =
        msg.role === "user" ? "message user-msg" : "message bot-msg";
      msgDiv.innerHTML =
        msg.role === "bot" ? renderMarkdown(msg.content) : msg.content;
      chatBody.appendChild(msgDiv);
    });
    chatBody.scrollTop = chatBody.scrollHeight;
  } catch (e) {
    console.warn("Failed to load chat history:", e);
    chatHistory = [];
  }
}

function clearChatHistory() {
  try {
    sessionStorage.removeItem(CHAT_KEY);
  } catch (e) {
    console.warn("Failed to clear chat history:", e);
  }
  chatHistory = [];
}

// ============ MARKDOWN RENDERING ============

function renderMarkdown(text) {
  let html = text.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
  html = html.replace(/\n/g, "<br>");
  return html;
}

// ============ CHAT WINDOW TOGGLES ============

function toggleChat() {
  const chatWindow = document.getElementById("chatWindow");
  const chatPopup = document.getElementById("chatPopup");

  if (!chatWindow) return;

  isChatOpen = !isChatOpen;

  if (isChatOpen) {
    chatWindow.classList.add("active");
    if (chatPopup) chatPopup.style.display = "none";

    if (!hasGreeted) {
      setTimeout(() => {
        if (IS_LOGGED_IN) {
          addBotMessageInstant(
            `Hi ${CURRENT_USER || "there"}, how can I help you today?`
          );
        } else {
          addBotMessageInstant(
            `Welcome to Tata Teleservices Support! Please 
            <a href="/login" style="color: var(--accent-color); font-weight: bold;">Login</a> or 
            <a href="/register" style="color: var(--accent-color); font-weight: bold;">Register</a> 
            to access account-specific help.`
          );
        }
        hasGreeted = true;
      }, 400);
    }
  } else {
    chatWindow.classList.remove("active");
  }
}

function toggleMaximize(event) {
  if (event) event.stopPropagation();

  const chatWindow = document.getElementById("chatWindow");
  const maximizeBtn = document.getElementById("maximizeBtn");
  const toggleBtn = document.querySelector(".chat-toggle");

  if (!chatWindow || !maximizeBtn) return;

  isMaximized = !isMaximized;

  if (isMaximized) {
    chatWindow.classList.add("maximized");
    maximizeBtn.innerHTML = '<i class="fas fa-compress"></i>';
    maximizeBtn.title = "Minimize";
    if (toggleBtn) toggleBtn.style.display = "none";
  } else {
    chatWindow.classList.remove("maximized");
    maximizeBtn.innerHTML = '<i class="fas fa-expand"></i>';
    maximizeBtn.title = "Maximize";
    if (toggleBtn) toggleBtn.style.display = "flex";
  }
}

// ============ TYPING INDICATOR ============

function showTypingIndicator() {
  const chatBody = document.getElementById("chatBody");
  if (!chatBody) return;

  if (document.getElementById("typingIndicator")) return;

  const indicator = document.createElement("div");
  indicator.id = "typingIndicator";
  indicator.className = "message typing-indicator";

  for (let i = 0; i < 3; i++) {
    const dot = document.createElement("div");
    dot.className = "typing-dot";
    indicator.appendChild(dot);
  }

  chatBody.appendChild(indicator);
  chatBody.scrollTop = chatBody.scrollHeight;
}

function hideTypingIndicator() {
  const indicator = document.getElementById("typingIndicator");
  if (indicator && indicator.parentNode) {
    indicator.parentNode.removeChild(indicator);
  }
}

// ============ MESSAGE HELPERS ============

function addUserMessage(text) {
  const chatBody = document.getElementById("chatBody");
  if (!chatBody) return;

  const msgDiv = document.createElement("div");
  msgDiv.className = "message user-msg";
  msgDiv.innerText = text;
  chatBody.appendChild(msgDiv);
  chatBody.scrollTop = chatBody.scrollHeight;

  chatHistory.push({ role: "user", content: text });
  saveChatHistory();
}

function addBotMessageInstant(text) {
  const chatBody = document.getElementById("chatBody");
  if (!chatBody) return;

  const msgDiv = document.createElement("div");
  msgDiv.className = "message bot-msg";
  msgDiv.innerHTML = renderMarkdown(text);
  chatBody.appendChild(msgDiv);
  chatBody.scrollTop = chatBody.scrollHeight;

  chatHistory.push({ role: "bot", content: text });
  saveChatHistory();
}

// ============ WORD-BY-WORD TYPING ============

function splitResponseIntoChunks(fullText) {
  const text = fullText.trim();
  if (!text) return [];

  let parts = text.split(/\n{2,}/).map((p) => p.trim()).filter(Boolean);

  if (parts.length === 1) {
    const sentences = text.match(/[^.!?]+[.!?]?/g) || [text];
    parts = sentences.map((s) => s.trim()).filter(Boolean);
  }

  const maxChunks = 4;
  if (parts.length <= maxChunks) return parts;

  const chunks = [];
  const chunkSize = Math.ceil(parts.length / maxChunks);
  for (let i = 0; i < parts.length; i += chunkSize) {
    chunks.push(parts.slice(i, i + chunkSize).join(" "));
  }
  return chunks;
}

function typeBotMessageChunk(text) {
  return new Promise((resolve) => {
    const chatBody = document.getElementById("chatBody");
    if (!chatBody) {
      resolve();
      return;
    }

    const msgDiv = document.createElement("div");
    msgDiv.className = "message bot-msg";
    msgDiv.innerText = "";
    chatBody.appendChild(msgDiv);
    chatBody.scrollTop = chatBody.scrollHeight;

    const words = text.split(" ");
    let index = 0;
    const speed = 120; // ms per word

    const interval = setInterval(() => {
      if (index >= words.length) {
        clearInterval(interval);
        msgDiv.innerHTML = renderMarkdown(text);
        chatBody.scrollTop = chatBody.scrollHeight;

        chatHistory.push({ role: "bot", content: text });
        saveChatHistory();

        resolve();
        return;
      }

      msgDiv.innerText += (index === 0 ? "" : " ") + words[index];
      index += 1;
      chatBody.scrollTop = chatBody.scrollHeight;
    }, speed);
  });
}

async function displayBotResponse(fullText) {
  const chunks = splitResponseIntoChunks(fullText);
  if (!chunks.length) return;

  for (let i = 0; i < chunks.length; i++) {
    if (i > 0) {
      await new Promise((res) => setTimeout(res, 350));
    }
    await typeBotMessageChunk(chunks[i]);
  }
}

// ============ FAQ CLICK HANDLER ============

function sendFAQ(question, staticAnswer) {
  addUserMessage(question);

  if (!IS_LOGGED_IN) {
    addBotMessageInstant(LOGIN_PROMPT);
    return;
  }

  setTimeout(() => {
    addBotMessageInstant(staticAnswer);
  }, 600);
}

// ============ MAIN SEND MESSAGE FLOW ============

async function sendMessage() {
  const input = document.getElementById("userInput");
  if (!input) return;

  const text = input.value.trim();
  if (!text) return;

  addUserMessage(text);
  input.value = "";

  if (!IS_LOGGED_IN) {
    addBotMessageInstant(LOGIN_PROMPT);
    return;
  }

  showTypingIndicator();

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text }),
    });

    const data = await response.json();
    hideTypingIndicator();

    if (data.error) {
      addBotMessageInstant("Sorry, something went wrong: " + data.error);
      return;
    }

    await displayBotResponse(data.response);
  } catch (err) {
    hideTypingIndicator();
    console.error(err);
    addBotMessageInstant(
      "Sorry, I couldn't reach the server. Please try again."
    );
  }
}

// ============ INITIALIZATION ============

document.addEventListener("DOMContentLoaded", () => {
  const userInput = document.getElementById("userInput");
  const faqContainer = document.getElementById("faqContainer");
  const chatBody = document.getElementById("chatBody");

  // Restore chat for logged-in users, clear for guests
  if (IS_LOGGED_IN && chatBody) {
    loadChatHistory();
  } else {
    clearChatHistory();
  }

  // Enter to send, Shift+Enter for newline
  if (userInput) {
    userInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
      }
    });

    userInput.setAttribute("enterkeyhint", "send");
  }

  // Hide FAQs when not logged in
  if (!IS_LOGGED_IN && faqContainer) {
    faqContainer.innerHTML =
      '<p class="faq-label">Please Login to view account-specific quick actions.</p>';
  }
});

// Hide FAQs when not logged in
if (!IS_LOGGED_IN && faqContainer) {
  faqContainer.innerHTML =
    '<p class="faq-label">Please Login to view account-specific quick actions.</p>';
}
