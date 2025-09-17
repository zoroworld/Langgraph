document.addEventListener('DOMContentLoaded', () => {
    const chat = document.getElementById('chat');
    const input = document.getElementById('prompt');
    const form = document.getElementById('chat-form');
    let currentBotMsg = null;

    function appendMessage(data) {
        const row = document.createElement('div');
        row.className = 'row ' + data.sender;

        const avatar = document.createElement('div');
        avatar.className = 'avatar';
        avatar.textContent = data.avatar;

        const msg = document.createElement('div');
        msg.className = 'msg ' + data.sender;

        msg.innerHTML = marked.parse(data.message);

        row.appendChild(avatar);
        row.appendChild(msg);
        chat.appendChild(row);
        chat.scrollTop = chat.scrollHeight;
    }

    const socket = new WebSocket(
        (window.location.protocol === "https:" ? "wss://" : "ws://") +
        window.location.host +
        "/ws/chat/"
    );


    socket.onmessage = function(e) {
        const data = JSON.parse(e.data);

        if (data.sender === "bot") {
            if (data.type === "stream") {
                if (!currentBotMsg) {
                    const row = document.createElement('div');
                    row.className = 'row bot';

                    const avatar = document.createElement('div');
                    avatar.className = 'avatar';
                    avatar.textContent = data.avatar;

                    currentBotMsg = document.createElement('div');
                    currentBotMsg.className = 'msg bot';
                    currentBotMsg.innerHTML = "";

                    row.appendChild(avatar);
                    row.appendChild(currentBotMsg);
                    chat.appendChild(row);
                }

                // append with markdown rendering
				const safeMessage = data.message
    						.replace(/\n{2,}/g, "\n\n")   // double newlines = paragraph
    						.replace(/\n/g, "  \n")     // single newline = line break
							.replace(/\n(\d+)\s*\./g, '\n$1.')      // fix stray numbers
    						.replace(/\n\s*\*/g, '\n  *');          // indent sub-bullets
                currentBotMsg.innerHTML += marked.parse(safeMessage);
                chat.scrollTop = chat.scrollHeight;

            } else if (data.type === "done") {
                currentBotMsg = null;
            } else if (data.type === "init") {
                appendMessage(data);
            }
        }
    };

    socket.onopen = function() {
        console.log("✅ WebSocket connected");
    };

    socket.onclose = function() {
        console.log("❌ WebSocket closed");
    };

    form.addEventListener('submit', (e) => {
        e.preventDefault();
        const text = input.value.trim();
        if (!text) return;

        const data = {
            sender: 'user',
            avatar: '👦',
            message: text,
        };
        appendMessage(data);

        socket.send(JSON.stringify(data));
        input.value = '';
    });
});
