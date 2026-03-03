const sendBtn = document.getElementById("send-btn");
const chatMessages = document.getElementById("chat-messages");
const queryInput = document.getElementById("query-input");

let activeSession = null;

document.querySelectorAll(".session-item").forEach(item => {
    item.addEventListener("click", async (e) => {

        // Prevent session switch when clicking menu
        if (e.target.closest(".session-menu")) return;

        const sessionId = item.dataset.session;
        if (!sessionId) return;

        activeSession = sessionId;

        // Highlight active
        document.querySelectorAll(".session-item")
            .forEach(el => el.classList.remove("active"));
        item.classList.add("active");

        await loadSession(sessionId);
    });
});

async function loadSession(sessionId) {
    const response = await fetch(`/load_session/${sessionId}`);
    const data = await response.json();

    if (data.error) {
        addMessage("Failed to load session.", "bot");
        return;
    }

    chatMessages.innerHTML = "";

    const history = data.history || [];

    if (history.length === 0) {
        addMessage("New session started.", "bot");
        return;
    }

    history.forEach(msg => {
        if (msg.role === "user") {
            addMessage(msg.content, "user");
        } else if (msg.role === "assistant") {
            renderBotMessage(msg.content, msg.citations);
        }
    });
}



document.getElementById("new-chat").addEventListener("click", async () => {

    const response = await fetch("/new_session", { method: "POST" });
    const data = await response.json();

    activeSession = data.session_id;

    chatMessages.innerHTML = "";
    addMessage("New session started.", "bot");

    location.reload(); // reload sidebar
});



sendBtn.addEventListener("click", async () => {

    const query = queryInput.value.trim();
    if (!query || !activeSession) return;

    addMessage(query, "user");
    queryInput.value = "";

    const loadingMsg = document.createElement("div");
    loadingMsg.className = "message bot loading";
    loadingMsg.innerHTML = `<span class="typing-dots">Thinking...</span>`;
    chatMessages.appendChild(loadingMsg);

    const response = await fetch("/query", {
        method: "POST",
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
        body: new URLSearchParams({
            query,
            session_id: activeSession
        })
    });

    const data = await response.json();

    loadingMsg.remove();
    renderBotMessage(data.answer, data.citations);
});



function addMessage(content, type) {
    const msg = document.createElement("div");
    msg.className = `message ${type}`;
    msg.textContent = content;
    chatMessages.appendChild(msg);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function renderBotMessage(answer, citations) {
    const botMsg = document.createElement("div");
    botMsg.className = "message bot";

    const answerDiv = document.createElement("div");
    answerDiv.className = "answer-text";
    answerDiv.textContent = answer;

    botMsg.appendChild(answerDiv);

    if (citations && citations.length > 0) {
        const citationBlock = renderCitations(citations);
        botMsg.appendChild(citationBlock);
    }

    chatMessages.appendChild(botMsg);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function renderCitations(citations) {
    if (!citations || citations.length === 0) return null;

    const wrapper = document.createElement("div");
    wrapper.className = "citations";

    const title = document.createElement("div");
    title.innerHTML = "<strong>Sources:</strong>";
    wrapper.appendChild(title);

    const ul = document.createElement("ul");

    citations.forEach(c => {
        const li = document.createElement("li");
        li.textContent = `${c.source_file} (Chunk ${c.chunk_id})`;
        ul.appendChild(li);
    });

    wrapper.appendChild(ul);
    return wrapper;
}



document.addEventListener("click", function (e) {

    // Toggle dropdown
    if (e.target.classList.contains("menu-btn")) {
        const dropdown = e.target.nextElementSibling;

        document.querySelectorAll(".dropdown")
            .forEach(d => d.style.display = "none");

        dropdown.style.display =
            dropdown.style.display === "flex" ? "none" : "flex";

        return;
    }

    // Delete session
    if (e.target.classList.contains("delete-session")) {
        const li = e.target.closest(".session-item");
        const sessionId = li.dataset.session;

        fetch(`/delete_session/${sessionId}`, { method: "POST" })
            .then(() => location.reload());
    }

    // Rename session
    if (e.target.classList.contains("rename-session")) {
        const li = e.target.closest(".session-item");
        const sessionId = li.dataset.session;
        const newName = prompt("Enter new session name:");
        if (!newName) return;

        fetch(`/rename_session/${sessionId}`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ title: newName })
        }).then(() => location.reload());
    }

    // Close dropdown if clicking outside
    if (!e.target.closest(".session-menu")) {
        document.querySelectorAll(".dropdown")
            .forEach(d => d.style.display = "none");
    }
});



window.addEventListener("load", async () => {

    const existingSessions = document.querySelectorAll(".session-item");

    if (existingSessions.length > 0) {
        // Auto-select first session
        existingSessions[0].click();
    } else {
        const response = await fetch("/new_session", { method: "POST" });
        const data = await response.json();
        activeSession = data.session_id;
    }
});