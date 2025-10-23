// React Chat UI integrated with Rasa backend + audio upload
import React, { useState } from 'react';
import { ChatFeed, Message } from 'react-chat-ui';

export default function RasaChat() {
  const [messages, setMessages] = useState([
    new Message({ id: 1, message: "Hi! How can I help you today?" })
  ]);
  const [input, setInput] = useState("");
  const [uploading, setUploading] = useState(false);

  const sendMessage = async () => {
    if (!input.trim()) return;

    const userMessage = new Message({ id: 0, message: input });
    setMessages(prev => [...prev, userMessage]);

    try {
      const response = await fetch("http://localhost:5005/webhooks/rest/webhook", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({ sender: "user", message: input })
      });

      const data = await response.json();

      const botMessages = data.map(msg =>
        new Message({ id: 1, message: msg.text || "(no reply)" })
      );

      setMessages(prev => [...prev, ...botMessages]);
    } catch (err) {
      setMessages(prev => [...prev, new Message({ id: 1, message: "Error: Cannot connect to Rasa server." })]);
    }

    setInput("");
  };

  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setUploading(true);
    const reader = new FileReader();
    reader.onloadend = async () => {
      const base64Audio = reader.result;

      // Display file message from user
      setMessages(prev => [
        ...prev,
        new Message({ id: 0, message: "📤 Uploaded audio file for Parkinson's detection." })
      ]);

      try {
        const response = await fetch("http://localhost:5005/webhooks/rest/webhook", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ sender: "user", message: base64Audio })
        });

        const data = await response.json();

        const botMessages = data.map(msg =>
          new Message({ id: 1, message: msg.text || "(no reply)" })
        );

        setMessages(prev => [...prev, ...botMessages]);
      } catch (err) {
        setMessages(prev => [...prev, new Message({ id: 1, message: "Error sending file to Rasa." })]);
      }

      setUploading(false);
    };

    reader.readAsDataURL(file); // Convert file to base64
  };

  return (
    <div style={{ maxWidth: "600px", margin: "auto", paddingTop: "50px" }}>
      <ChatFeed
        messages={messages}
        isTyping={uploading}
        hasInputField={false}
        showSenderName={false}
        bubblesCentered={false}
        bubbleStyles={{
          text: {
            fontSize: 16,
            color: "white"
          },
          chatbubble: {
            backgroundColor: "#007bff",
            padding: "10px",
            borderRadius: "16px",
            maxWidth: "75%",
          }
        }}
      />
      <div style={{ display: "flex", marginTop: "10px", gap: "8px" }}>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && sendMessage()}
          placeholder="Type your message..."
          style={{ flex: 1, padding: "10px", fontSize: "16px" }}
        />
        <button onClick={sendMessage} style={{ padding: "10px 20px" }}>
          Send
        </button>
        <label style={{ padding: "10px 20px", background: "#eee", cursor: "pointer" }}>
          📎 Upload
          <input
            type="file"
            accept="audio/wav"
            onChange={handleFileUpload}
            style={{ display: "none" }}
          />
        </label>
      </div>
    </div>
  );
}
