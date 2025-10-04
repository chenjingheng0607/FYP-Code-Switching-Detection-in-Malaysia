document.addEventListener('DOMContentLoaded', () => {
    const chatLog = document.querySelector('#chat-log');
    const messageInput = document.querySelector('#chat-message-input');
    const sendButton = document.querySelector('#chat-message-submit');

    function addMessage(sender, text) {
        const messageElement = document.createElement('div');
        messageElement.classList.add('message');
        // Add a class for styling based on the sender
        messageElement.classList.add(sender.toLowerCase()); 
        
        // Correctly include the text in the message
        messageElement.innerHTML = `<b>${sender}:</b> ${text}`;
        chatLog.appendChild(messageElement);
        chatLog.scrollTop = chatLog.scrollHeight; // Scroll to the bottom
    }

    sendButton.addEventListener('click', () => {
        const messageText = messageInput.value.trim();
        if (messageText) {
            addMessage('You', messageText);
            messageInput.value = '';
        }
    });

    messageInput.addEventListener('keyup', (event) => {
        if (event.key === 'Enter') {
            sendButton.click();
        }
    });
});