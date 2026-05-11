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

/* ---------------- SESSION CLICK ---------------- */

document
.getElementById("session-list")
.addEventListener("click", async (e) => {

    const item = e.target.closest(".session-item");

    if (!item) return;

    // ignore menu clicks
    if (e.target.closest(".session-menu")) return;

    const sessionId = item.dataset.session;

    if (!sessionId) return;

    activeSession = sessionId;

    document.querySelectorAll(".session-item")
        .forEach(el => {
            el.classList.remove("active");
        });

    item.classList.add("active");

    await loadSession(sessionId);
});

/* ---------------- NEW SESSION ---------------- */

document.getElementById("new-chat")
.addEventListener("click", async () => {

    const response = await fetch("/new_session", {
        method: "POST"
    });

    const data = await response.json();

    activeSession = data.session_id;

    localStorage.setItem(
        "activeSession",
        activeSession
    );

    // clear chat
    chatMessages.innerHTML = "";

    addMessage(
        "New session started.",
        "bot"
    );

    // remove previous active states
    document.querySelectorAll(".session-item")
        .forEach(el => el.classList.remove("active"));

    // create sidebar item
    const li = document.createElement("li");

    li.className = "session-item active";

    li.dataset.session = activeSession;

    li.innerHTML = `
        <span class="session-title">
            New Chat
        </span>

        <div class="session-menu">

            <button class="menu-btn">
                ⋮
            </button>

            <div class="dropdown">

                <button class="rename-session">
                    Rename
                </button>

                <button class="pin-session">
                    Pin
                </button>

                <button class="delete-session">
                    Delete
                </button>

            </div>
        </div>
    `;

    document
        .getElementById("session-list")
        .prepend(li);

    // IMPORTANT
    attachSessionClickHandlers();
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

    let response = await fetch("/query", {
        method: "POST",
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
        body: new URLSearchParams({
            query,
            session_id: activeSession
        })
    });

    let data = await response.json();

    //HANDLE SESSION LOSS HERE
    if (data.error === "Session not found") {

        console.warn("[Session Lost] Creating new session...");

        const res = await fetch("/new_session", { method: "POST" });
        const newSession = await res.json();

        activeSession = newSession.session_id;
        localStorage.setItem("activeSession", activeSession);

        // 🔁 RETRY QUERY with new session
        response = await fetch("/query", {
            method: "POST",
            headers: { "Content-Type": "application/x-www-form-urlencoded" },
            body: new URLSearchParams({
                query,
                session_id: activeSession
            })
        });

        data = await response.json();
    }

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

    // MARKDOWN RENDERING
    answerDiv.innerHTML = marked.parse(answer);

    // CODE HIGHLIGHTING
    answerDiv.querySelectorAll("pre code")
        .forEach((block) => {
            hljs.highlightElement(block);
        });

    botMsg.appendChild(answerDiv);

    if (citations && citations.length > 0) {
        botMsg.appendChild(
            renderCitations(citations)
        );
    }

    chatMessages.appendChild(botMsg);

    chatMessages.scrollTop =
        chatMessages.scrollHeight;
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



function addDocumentToSidebar(filename) {

    const list = document.getElementById("document-list");

    // remove placeholder
    if (list.children.length === 1 &&
        list.children[0].textContent.includes("No documents")) {

        list.innerHTML = "";
    }

    // prevent duplicates
    const exists = [...list.children]
        .some(li => li.textContent === filename);

    if (exists) return;

    const li = document.createElement("li");

    li.className = "document-item";

    li.innerHTML = `
        <span class="document-dot"></span>
        ${filename}
    `;

    list.prepend(li);
}


function showToast(message, type="info") {

    const toast = document.createElement("div");

    toast.className = `toast ${type}`;
    toast.textContent = message;

    document.body.appendChild(toast);

    setTimeout(() => {
        toast.classList.add("show");
    }, 100);

    setTimeout(() => {
        toast.classList.remove("show");

        setTimeout(() => {
            toast.remove();
        }, 300);

    }, 3000);
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

document.addEventListener("click", (e) => {

    // open menu
    if (e.target.classList.contains("menu-btn")) {

        e.stopPropagation();

        const dropdown =
            e.target.nextElementSibling;

        // close others
        document.querySelectorAll(".dropdown")
            .forEach(d => {
                if (d !== dropdown) {
                    d.style.display = "none";
                }
            });

        dropdown.style.display =
            dropdown.style.display === "flex"
                ? "none"
                : "flex";

        return;
    }

    // close all menus
    document.querySelectorAll(".dropdown")
        .forEach(d => d.style.display = "none");
});

/* ---------------- UPLOAD ---------------- */
document.getElementById("upload-form")
?.addEventListener("submit", async (e) => {

    e.preventDefault();

    const form = e.target;
    const formData = new FormData(form);

    const uploadBtn = form.querySelector("button");

    // ---------------- LOADING STATE ----------------

    uploadBtn.disabled = true;
    uploadBtn.textContent = "Indexing...";
    uploadBtn.classList.add("loading");

    showToast("Uploading and indexing document...", "info");

    try {

        const response = await fetch("/upload", {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        if (data.error) {
            showToast(data.error, "error");
            return;
        }

        // ---------------- UPDATE DOCUMENT LIST ----------------

        addDocumentToSidebar(data.filename);

        // ---------------- SUCCESS ----------------

        showToast(
            `${data.filename} indexed successfully (${data.chunks} chunks)`,
            "success"
        );

        // reset form
        form.reset();

    } catch (err) {

        console.error(err);

        showToast(
            "Upload failed.",
            "error"
        );

    } finally {
        uploadBtn.disabled = false;
        uploadBtn.textContent = "Upload";
        uploadBtn.classList.remove("loading");
    }
});

queryInput.addEventListener("input", () => {

    queryInput.style.height = "auto";

    queryInput.style.height =
        queryInput.scrollHeight + "px";
});


queryInput.addEventListener("keydown", (e) => {

    if (e.key === "Enter" && !e.shiftKey) {

        e.preventDefault();

        sendBtn.click();
    }
});