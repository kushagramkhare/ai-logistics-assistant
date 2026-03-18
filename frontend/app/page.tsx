"use client";

import { useState, useRef, useEffect } from "react";


export default function Home() {
  const [messages, setMessages] = useState<any[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const sessionId = useRef(Math.random().toString(36).substring(7));

  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const sendMessage = async () => {
    if (!input) return;

    const userMessage = { role: "user", content: input };
    const updatedMessages = [...messages, userMessage];

    setMessages(updatedMessages);
    setInput("");
    setLoading(true);

    const response = await fetch("http://127.0.0.1:8000/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      // body: JSON.stringify({
      //   question: input,
      //   history: messages,
      // }),
      body: JSON.stringify({
        question: input,
        session_id: sessionId.current,
      }),
    });

    const data = await response.json();

    const botMessage = {
      role: "assistant",
      content: data.answer,
    };

    setMessages([...updatedMessages, botMessage]);
    setLoading(false);
  };

  return (
    <div className="flex flex-col h-screen max-w-3xl mx-auto p-4">

      <h1 className="text-2xl font-bold mb-4 text-center">
        AI Logistics Assistant
      </h1>

      <div className="flex-1 overflow-y-auto border rounded-lg p-4 space-y-3">

        {messages.map((msg, i) => (
          <div
            key={i}
            className={`p-3 rounded-lg max-w-[80%] ${
              msg.role === "user"
                ? "bg-blue-500 text-white ml-auto"
                : "bg-gray-200 text-black"
            }`}
          >
            {msg.content}
          </div>
        ))}

        {loading && (
          <div className="bg-gray-300 text-black p-3 rounded-lg w-fit animate-pulse">
            thinking...
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      <div className="flex gap-2 mt-4">
        <input
          className="border rounded p-2 flex-1"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask something..."
        />

        <button
          onClick={sendMessage}
          className="bg-blue-500 text-white px-4 py-2 rounded"
        >
          Send
        </button>
      </div>
    </div>
  );
}