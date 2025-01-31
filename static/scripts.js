async function sendMessage() {
    const userInput = document.getElementById('user_input').value;
    const sendButton = document.getElementById('sendButton');
    const chatbox = document.getElementById('chatbox');

    // Disable input and button, show fetching data message
    document.getElementById('user_input').disabled = true;
    sendButton.disabled = true;
    chatbox.innerHTML += `<div class="chat-message assistant"><strong>assistant:</strong> Fetching data...</div>`;
    chatbox.scrollTop = chatbox.scrollHeight; // Scroll to the bottom

    try {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ message: userInput })
        });
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        const data = await response.json();
        chatbox.innerHTML = '';
        data.messages.forEach(msg => {
            let messageHtml = `<div class="chat-message ${msg.role}"><strong>${msg.role}:</strong> ${msg.content}`;
            if (msg.image) {
                messageHtml += `<img src="${msg.image}" class="chat-image">`;
            }
            messageHtml += `</div>`;
            chatbox.innerHTML += messageHtml;
        });
        document.getElementById('user_input').value = '';
        chatbox.scrollTop = chatbox.scrollHeight; // Scroll to the bottom
    } catch (error) {
        console.error('There was a problem with the fetch operation:', error);
        chatbox.innerHTML += `<div class="chat-message assistant"><strong>assistant:</strong> Error fetching data. Please try again.</div>`;
    } finally {
        // Re-enable input and button
        document.getElementById('user_input').disabled = false;
        sendButton.disabled = false;
    }
}
