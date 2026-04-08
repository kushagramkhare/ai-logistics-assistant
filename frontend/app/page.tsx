"use client";

import { useState, useRef, useEffect } from "react";
import ReactMarkdown from "react-markdown";


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

    const backendUrl = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:7860";
    const response = await fetch(`${backendUrl}/chat`, {
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
    <div className="flex flex-col h-screen bg-[#0f172a] text-white">

      <div className="p-4 text-center font-semibold text-lg border-b border-gray-700 bg-[#020617]">
        AI Logistics Assistant
      </div>

      <div className="flex-1 overflow-y-auto px-4 py-6 space-y-5 bg-[#020617]">

        {messages.map((msg, i) => {
          const isUser = msg.role === "user";

          let mainText = msg.content;
          let sources = "";

          if (!isUser && msg.content.includes("Sources:")) {
            const parts = msg.content.split("Sources:");
            mainText = parts[0];
            sources = parts[1];
          }

          return (
            <div
              key={i}
              className={`flex ${isUser ? "justify-end" : "justify-start"}`}
            >
              <div
                className={`px-4 py-3 rounded-2xl max-w-[75%] text-sm whitespace-pre-wrap ${isUser
                  ? "bg-gradient-to-r from-blue-500 to-indigo-500 text-white rounded-br-none"
                  : "bg-[#111827] text-gray-200 border border-gray-700 rounded-bl-none"
                  }`}
              >
                {/* MAIN CONTENT */}
                {isUser ? (
                  msg.content
                ) : (
                  <ReactMarkdown
                    components={{
                      h1: ({ children }) => (
                        <h1 className="text-lg font-bold mb-2">{children}</h1>
                      ),
                      h2: ({ children }) => (
                        <h2 className="text-md font-semibold mt-3 mb-1">{children}</h2>
                      ),
                      li: ({ children }) => (
                        <li className="ml-4 list-disc">{children}</li>
                      ),
                      p: ({ children }) => <p className="mb-2">{children}</p>,
                    }}
                  >
                    {mainText}
                  </ReactMarkdown>
                )}

                {/* SOURCES */}
                {!isUser && sources && (
                  <div className="mt-4 border-t border-gray-700 pt-2">
                    <p className="text-xs text-gray-400 mb-1">Sources</p>

                    {sources.split("\n").map((src, idx) =>
                      src.trim() ? (
                        <a
                          key={idx}
                          href={src.replace("- ", "").trim()}
                          target="_blank"
                          className="block text-blue-400 text-sm hover:underline"
                        >
                          {src.replace("- ", "")}
                        </a>
                      ) : null
                    )}
                  </div>
                )}
              </div>
            </div>
          );
        })}

        {loading && (
          <div className="bg-gray-300 text-black p-3 rounded-lg w-fit animate-pulse">
            <div className="flex gap-1">
              <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></span>
              <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce delay-150"></span>
              <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce delay-300"></span>
            </div>
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      <div className="sticky bottom-0 bg-[#020617] p-3 border-t border-gray-700">
        <div className="flex gap-2">
          <input
            className="bg-[#020617] border border-gray-700 rounded-full px-4 py-2 flex-1 text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask something..."
            onKeyDown={(e) => e.key === "Enter" && sendMessage()}
          />

          <button
            onClick={sendMessage}
            className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-full"
          >
            Send
          </button>
        </div>
      </div>
    </div>
  );
}
