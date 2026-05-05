const sendBtn = document.getElementById("send-btn");
const chatMessages = document.getElementById("chat-messages");
const queryInput = document.getElementById("query-input");

let activeSession = null;

/* ---------------- SESSION LOAD ---------------- */

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

    localStorage.setItem("activeSession", sessionId);
}

/* ---------------- SESSION CLICK ---------------- */

function attachSessionClickHandlers() {
    document.querySelectorAll(".session-item").forEach(item => {
        item.onclick = async (e) => {

            if (e.target.closest(".session-menu")) return;

            const sessionId = item.dataset.session;
            if (!sessionId) return;

            activeSession = sessionId;

            document.querySelectorAll(".session-item")
                .forEach(el => el.classList.remove("active"));

            item.classList.add("active");

            await loadSession(sessionId);
        };
    });
}

/* ---------------- NEW SESSION ---------------- */

document.getElementById("new-chat").addEventListener("click", async () => {

    const response = await fetch("/new_session", { method: "POST" });
    const data = await response.json();

    activeSession = data.session_id;

    chatMessages.innerHTML = "";
    addMessage("New session started.", "bot");

    // add to sidebar dynamically
    const sessionList = document.querySelector(".session-list");

    const li = document.createElement("li");
    li.className = "session-item active";
    li.dataset.session = activeSession;
    li.innerHTML = `<span>New Chat</span>`;

    // remove old active
    document.querySelectorAll(".session-item")
        .forEach(el => el.classList.remove("active"));

    sessionList.prepend(li);

    attachSessionClickHandlers();

    localStorage.setItem("activeSession", activeSession);
});

/* ---------------- QUERY ---------------- */

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

/* ---------------- MESSAGE HELPERS ---------------- */

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
        botMsg.appendChild(renderCitations(citations));
    }

    chatMessages.appendChild(botMsg);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function renderCitations(citations) {
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

/* ---------------- DELETE / RENAME ---------------- */

document.addEventListener("click", function (e) {

    // delete
    if (e.target.classList.contains("delete-session")) {
        const li = e.target.closest(".session-item");
        const sessionId = li.dataset.session;

        fetch(`/delete_session/${sessionId}`, { method: "POST" })
            .then(() => {
                li.remove();

                if (activeSession === sessionId) {
                    activeSession = null;
                    chatMessages.innerHTML = "";
                    addMessage("Session deleted. Start a new chat.", "bot");
                }
            });
    }

    // rename
    if (e.target.classList.contains("rename-session")) {
        const li = e.target.closest(".session-item");
        const sessionId = li.dataset.session;

        const newName = prompt("Enter new session name:");
        if (!newName) return;

        fetch(`/rename_session/${sessionId}`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ title: newName })
        }).then(() => {
            li.querySelector("span").textContent = newName;
        });
    }
});

/* ---------------- UPLOAD (NO RELOAD) ---------------- */

document.getElementById("upload-form")?.addEventListener("submit", async (e) => {
    e.preventDefault();

    const formData = new FormData(e.target);

    const response = await fetch("/upload", {
        method: "POST",
        body: formData
    });

    const data = await response.json();

    console.log("[UPLOAD]", data);

    addMessage("Document uploaded and indexed.", "bot");
});

/* ---------------- INIT ---------------- */

window.addEventListener("load", async () => {

    attachSessionClickHandlers();

    const saved = localStorage.getItem("activeSession");

    if (saved) {
        activeSession = saved;
        await loadSession(saved);
        return;
    }

    const sessions = document.querySelectorAll(".session-item");

    if (sessions.length > 0) {
        sessions[0].click();
    } else {
        const response = await fetch("/new_session", { method: "POST" });
        const data = await response.json();
        activeSession = data.session_id;
    }
});