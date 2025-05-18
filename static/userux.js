const queryInput = document.getElementById('queryInput');
const submitButton = document.getElementById('submitButton');
const errorMessage = document.getElementById('errorMessage');
const loading = document.getElementById('loading');
const chatContainer = document.getElementById('chatContainer');
const chatHistory = document.getElementById('chatHistory');
const themeIcon = document.getElementById('theme-icon');

let chatSessions = JSON.parse(localStorage.getItem('chatSessions')) || [];
let currentSessionId = null;

// Theme Management
function setTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('theme', theme);
    themeIcon.innerHTML = theme === 'light' ? 
        `<path d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>` :
        `<path d="M21 12.79A9 9 0 1111.21 3 7 7 0 0021 12.79z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>`;
}

function toggleTheme() {
    const currentTheme = document.documentElement.getAttribute('data-theme') || 'dark';
    const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
    setTheme(newTheme);
}

// Initialize Theme
document.addEventListener('DOMContentLoaded', () => {
    const savedTheme = localStorage.getItem('theme') || 'dark';
    setTheme(savedTheme);
});

function scrollToBottom() {
    chatContainer.scrollTo({
        top: chatContainer.scrollHeight,
        behavior: 'smooth'
    });
}

function addMessage(content, isUser, sessionId) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${isUser ? 'user-message' : 'bot-message'} fade-in`;

    // Parse Markdown content to HTML using marked
    const markdownContent = marked.parse(content, {
        gfm: true,
        breaks: true
    });

    messageDiv.innerHTML = `<div>${markdownContent}</div>`;
    chatContainer.appendChild(messageDiv);
    scrollToBottom();

    if (sessionId) {
        const session = chatSessions.find(s => s.id === sessionId);
        if (session) {
            session.messages.push({ content, isUser });
            saveChatSessions();
        }
    }
}

function renderChatHistory() {
    chatHistory.innerHTML = '';
    chatSessions.forEach(session => {
        const historyItem = document.createElement('div');
        historyItem.className = `chat-history-item ${session.id === currentSessionId ? 'active' : ''}`;
        historyItem.innerHTML = `
            <span class="truncate flex-grow">${session.title}</span>
            <button class="delete-btn" onclick="deleteChatSession('${session.id}', event)">
                <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4" viewBox="0 0 20 20" fill="currentColor">
                    <path fill-rule="evenodd" d="M9 2a1 1 0 00-.894.553L7.382 4H4a1 1 0 000 2v10a2 2 0 002 2h8a2 2 0 002-2V6a1 1 0 100-2h-3.382l-.724-1.447A1 1 0 0011 2H9zM7 8a1 1 0 012 0v6a1 1 0 11-2 0V8zm5-1a1 1 0 00-1 1v6a1 1 0 102 0V8a1 1 0 00-1-1z" clip-rule="evenodd"/>
                </svg>
            </button>
        `;
        historyItem.addEventListener('click', (e) => {
            if (!e.target.closest('.delete-btn')) {
                loadChatSession(session.id);
                toggleHistoryDropdown(); // Close dropdown after selection
            }
        });
        chatHistory.appendChild(historyItem);
    });
}

function toggleHistoryDropdown() {
    chatHistory.classList.toggle('hidden');
}

function deleteChatSession(sessionId, event) {
    event.stopPropagation();
    chatSessions = chatSessions.filter(s => s.id !== sessionId);
    if (currentSessionId === sessionId) {
        currentSessionId = null;
        chatContainer.innerHTML = `
            <h1 class="welcome-header">Welcome to Polio AFP AI Assistant!</h1>
             <h5 class="welcome-header">What do you want to Know about Polio?</h5>
        `;
        scrollToBottom();
    }
    saveChatSessions();
    renderChatHistory();
}

function startNewChat() {
    currentSessionId = Date.now().toString();
    chatSessions.push({
        id: currentSessionId,
        title: 'New Chat',
        messages: []
    });
    chatContainer.innerHTML = `
        <h1 class="welcome-header">Welcome to Polio AFP AI Assistant!</h1>
         <h5 class="welcome-header">What do you want to know about Polio?</h5>
    `;
    queryInput.value = '';
    saveChatSessions();
    renderChatHistory();
    scrollToBottom();
}

function loadChatSession(sessionId) {
    currentSessionId = sessionId;
    const session = chatSessions.find(s => s.id === sessionId);
    chatContainer.innerHTML = `
        <h1 class="welcome-header" style="font-weight: bold;">Welcome to Polio AFP AI Assistant!</h1>
        <h5 class="welcome-header">What do you want to know about Polio?</h5>
    `;
    if (session) {
        session.messages.forEach(msg => {
            addMessage(msg.content, msg.isUser, null);
        });
    }
    renderChatHistory();
    scrollToBottom();
}

function saveChatSessions() {
    localStorage.setItem('chatSessions', JSON.stringify(chatSessions));
}

function showError(message) {
    errorMessage.textContent = message;
    errorMessage.classList.remove('hidden');
    setTimeout(() => {
        errorMessage.classList.add('hidden');
    }, 3000);
}

async function handleSubmit() {
    const query = queryInput.value.trim();
    if (!query) {
        showError('Please enter a question.');
        return;
    }

    if (!currentSessionId) {
        currentSessionId = Date.now().toString();
        chatSessions.push({
            id: currentSessionId,
            title: query.substring(0, 30) + (query.length > 30 ? '...' : ''),
            messages: []
        });
    }

    addMessage(query, true, currentSessionId);
    queryInput.value = '';
    renderChatHistory();

    loading.classList.remove('hidden');
    errorMessage.classList.add('hidden');

    try {
        const response = await fetch('/rag_query', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query })
        });

        if (!response.ok) {
            throw new Error('Failed to fetch response from server.');
        }

        const data = await response.json();
        addMessage(data.response || 'No response generated.', false, currentSessionId);

        // Update session title with the first query
        const session = chatSessions.find(s => s.id === currentSessionId);
        if (session && session.title === 'New Chat') {
            session.title = query.substring(0, 30) + (query.length > 30 ? '...' : '');
            saveChatSessions();
            renderChatHistory();
        }

    } catch (error) {
        showError('An error occurred while processing your request. Please try again.');
        console.error(error);
    } finally {
        loading.classList.add('hidden');
    }
}

submitButton.addEventListener('click', handleSubmit);
queryInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        handleSubmit();
    }
});

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    if (chatSessions.length > 0) {
        currentSessionId = chatSessions[chatSessions.length - 1].id;
        loadChatSession(currentSessionId);
    } else {
        startNewChat();
    }
    renderChatHistory();
    scrollToBottom();
});
