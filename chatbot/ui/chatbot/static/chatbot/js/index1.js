document.addEventListener('DOMContentLoaded', () => {
    const chat = document.getElementById('chat');
    const input = document.getElementById('prompt');
    const form = document.getElementById('chat-form');

    // Append a message to the chat area
    function appendMessage(text, who = 'bot') {
        const row = document.createElement('div');
        row.className = 'row ' + who;

        if (who === 'bot') {
            const avatar = document.createElement('div');
            avatar.className = 'avatar';
            avatar.textContent = '🤖';

            const msg = document.createElement('div');
            msg.className = 'msg bot';
            msg.textContent = text;

            row.appendChild(avatar);
            row.appendChild(msg);
        } else {
            const msg = document.createElement('div');
            msg.className = 'msg user';
            msg.textContent = text;
            row.appendChild(msg);
        }

        chat.appendChild(row);
        chat.scrollTop = chat.scrollHeight;
    }

    // Handle form submit
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        const text = input.value.trim();
        if (!text) return;

        appendMessage(text, 'user');
        input.value = '';

        const csrfToken = document.querySelector('[name=csrfmiddlewaretoken]').value;

        try {
            const response = await fetch('/api/chat/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/x-www-form-urlencoded',
                    'X-CSRFToken': csrfToken
                },
                body: new URLSearchParams({message: text})
            });

            if (!response.ok) throw new Error('Network error');

            const data = await response.json();
            appendMessage(data.reply, 'bot');
        } catch (err) {
            appendMessage('Error: Could not get response', 'bot');
            console.error(err);
        }
    });
});
