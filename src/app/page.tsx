
"use client";

import { useState } from "react";

const suggestions = [
  {
    icon: "⚖️",
    title: "Understand a legal provision",
    description: "Explain a section of Indian law in simple terms.",
    prompt: "Explain a provision of Indian law in simple language.",
  },
  {
    icon: "📚",
    title: "Compare old and new laws",
    description: "Explore IPC to BNS and CrPC to BNSS changes.",
    prompt: "Help me compare an IPC provision with its BNS equivalent.",
  },
  {
    icon: "📄",
    title: "Simplify a legal document",
    description: "Understand complex legal language more easily.",
    prompt: "How can I simplify a legal document?",
  },
];

const initialMessages = [
  {
    role: "assistant",
    content:
      "Hello! I'm your Indian Legal AI assistant. I can help you explore legal concepts, understand statutory provisions, and make complex legal language easier to understand.",
  },
];

export default function Home() {
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState(initialMessages);
  const [language, setLanguage] = useState("English");
  const [activeNav, setActiveNav] = useState("Legal Assistant");
  const [showSources, setShowSources] = useState(true);

  function startNewChat() {
    setMessages(initialMessages);
    setQuestion("");
    setActiveNav("Legal Assistant");
  }

  function sendQuestion(value = question) {
    const trimmed = value.trim();
    if (!trimmed) return;

    setMessages((previous) => [
      ...previous,
      { role: "user", content: trimmed },
      {
        role: "assistant",
        content:
          "Your question has been added to this demo conversation. The legal RAG API is not connected yet. Once integrated, this response area will display an answer grounded in retrieved legal sources.",
      },
    ]);

    setQuestion("");
    setActiveNav("Legal Assistant");
  }

  return (
    <main className="flex min-h-screen bg-[#f7f8fc] text-slate-800">
      {/* Sidebar */}
      <aside className="hidden w-[260px] shrink-0 flex-col border-r border-slate-200 bg-white p-5 md:flex">
        <div className="mb-9 flex items-center gap-3">
          <div className="flex h-11 w-11 items-center justify-center rounded-2xl bg-indigo-600 text-xl text-white shadow-lg shadow-indigo-200">
            ⚖
          </div>
          <div>
            <h1 className="text-lg font-bold tracking-tight">
              CCpeeps
            </h1>
            <p className="text-xs text-slate-500">
              Indian Legal AI
            </p>
          </div>
        </div>

        <button
          onClick={startNewChat}
          className="flex items-center justify-center gap-2 rounded-xl bg-indigo-600 px-4 py-3 text-sm font-semibold text-white shadow-md shadow-indigo-100 transition hover:bg-indigo-700"
        >
          <span className="text-lg">＋</span>
          New conversation
        </button>

        <p className="mb-3 mt-9 text-[11px] font-bold uppercase tracking-[0.16em] text-slate-400">
          Workspace
        </p>

        <nav className="space-y-1">
          {[
            ["Legal Assistant", "✦"],
            ["Legal Research", "⌕"],
            ["My Documents", "▤"],
          ].map(([name, icon]) => (
            <button
              key={name}
              onClick={() => setActiveNav(name)}
              className={`flex w-full items-center gap-3 rounded-xl px-3 py-3 text-left text-sm transition ${
                activeNav === name
                  ? "bg-indigo-50 font-semibold text-indigo-700"
                  : "text-slate-600 hover:bg-slate-50"
              }`}
            >
              <span className="text-lg">{icon}</span>
              {name}
            </button>
          ))}
        </nav>

        <div className="mt-9">
          <div className="mb-3 flex items-center justify-between">
            <p className="text-[11px] font-bold uppercase tracking-[0.16em] text-slate-400">
              Recent chats
            </p>
            <span className="text-xs text-slate-400">Demo</span>
          </div>
          <p className="rounded-lg px-3 py-2 text-sm text-slate-400">
            Your conversations will appear here.
          </p>
        </div>

        <div className="mt-auto rounded-2xl border border-indigo-100 bg-indigo-50/70 p-4">
          <div className="mb-2 flex items-center gap-2">
            <span>🛡️</span>
            <p className="text-sm font-semibold text-slate-800">
              Legal information
            </p>
          </div>
          <p className="text-xs leading-5 text-slate-600">
            Verify legal provisions and consult a qualified
            lawyer for advice on your specific situation.
          </p>
        </div>

        <div className="mt-5 flex items-center gap-3 border-t border-slate-100 pt-5">
          <div className="flex h-10 w-10 items-center justify-center rounded-full bg-slate-100 font-bold text-slate-600">
            U
          </div>
          <div>
            <p className="text-sm font-semibold">Guest user</p>
            <p className="text-xs text-slate-500">
              Authentication coming next
            </p>
          </div>
        </div>
      </aside>

      {/* Main application */}
      <section className="flex min-h-screen min-w-0 flex-1 flex-col">
        {/* Header */}
        <header className="sticky top-0 z-10 flex items-center justify-between border-b border-slate-200/80 bg-white/90 px-4 py-4 backdrop-blur-xl sm:px-8">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-indigo-600 text-lg text-white md:hidden">
              ⚖
            </div>
            <div>
              <h2 className="text-sm font-bold sm:text-base">
                {activeNav}
              </h2>
              <p className="mt-0.5 text-xs text-slate-500">
                Indian law, made easier to understand
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2 sm:gap-4">
            <label className="hidden text-xs text-slate-500 sm:block">
              Response language
            </label>
            <select
              value={language}
              onChange={(event) => setLanguage(event.target.value)}
              className="max-w-[130px] rounded-xl border border-slate-200 bg-white px-3 py-2 text-xs font-medium outline-none focus:border-indigo-400 sm:text-sm"
            >
              <option>English</option>
              <option>Malayalam</option>
            </select>
          </div>
        </header>

        {/* Conversation */}
        <div className="mx-auto flex w-full max-w-5xl flex-1 flex-col px-4 py-8 sm:px-8 sm:py-12">
          {messages.length === initialMessages.length ? (
            <div className="my-auto">
              <div className="mx-auto max-w-3xl">
                <div className="mb-6 inline-flex items-center gap-2 rounded-full border border-indigo-100 bg-white px-3 py-2 text-xs font-medium text-indigo-700 shadow-sm">
                  <span className="h-2 w-2 rounded-full bg-emerald-500" />
                  Your legal research workspace
                </div>

                <h1 className="max-w-2xl text-4xl font-bold leading-tight tracking-tight text-slate-900 sm:text-5xl">
                  Understand the law.
                  <span className="block text-indigo-600">
                    Find clarity.
                  </span>
                </h1>

                <p className="mt-5 max-w-2xl text-sm leading-7 text-slate-500 sm:text-base">
                  Explore Indian legal concepts, understand
                  statutory provisions, and simplify legal
                  documents through one conversational interface.
                </p>

                {/* Question composer */}
                <form
                  onSubmit={(event) => {
                    event.preventDefault();
                    sendQuestion();
                  }}
                  className="mt-9 rounded-2xl border border-slate-200 bg-white p-3 shadow-xl shadow-slate-200/50 transition focus-within:border-indigo-300 focus-within:ring-4 focus-within:ring-indigo-50"
                >
                  <textarea
                    value={question}
                    onChange={(event) => setQuestion(event.target.value)}
                    onKeyDown={(event) => {
                      if (
                        event.key === "Enter" &&
                        !event.shiftKey
                      ) {
                        event.preventDefault();
                        sendQuestion();
                      }
                    }}
                    placeholder="Ask a question about Indian law..."
                    rows={3}
                    className="w-full resize-none border-0 bg-transparent px-3 py-2 text-sm leading-6 outline-none placeholder:text-slate-400"
                  />

                  <div className="flex items-center justify-between border-t border-slate-100 pt-3">
                    <p className="px-2 text-[11px] text-slate-400 sm:text-xs">
                      Enter to send · Shift + Enter for a new line
                    </p>
                    <button
                      type="submit"
                      disabled={!question.trim()}
                      className="flex items-center gap-2 rounded-xl bg-indigo-600 px-4 py-2.5 text-sm font-semibold text-white transition hover:bg-indigo-700 disabled:cursor-not-allowed disabled:opacity-40"
                    >
                      Ask <span>↑</span>
                    </button>
                  </div>
                </form>

                {/* Suggestions */}
                <div className="mt-10">
                  <p className="mb-4 text-xs font-semibold uppercase tracking-widest text-slate-400">
                    Explore what you can do
                  </p>
                  <div className="grid gap-3 sm:grid-cols-3">
                    {suggestions.map((item) => (
                      <button
                        key={item.title}
                        onClick={() => {
                          setQuestion(item.prompt);
                        }}
                        className="group rounded-2xl border border-slate-200 bg-white p-4 text-left transition hover:-translate-y-0.5 hover:border-indigo-200 hover:shadow-lg hover:shadow-indigo-100/50"
                      >
                        <span className="mb-4 flex h-10 w-10 items-center justify-center rounded-xl bg-indigo-50 text-xl transition group-hover:bg-indigo-100">
                          {item.icon}
                        </span>
                        <span className="block text-sm font-semibold text-slate-800">
                          {item.title}
                        </span>
                        <span className="mt-2 block text-xs leading-5 text-slate-500">
                          {item.description}
                        </span>
                      </button>
                    ))}
                  </div>
                </div>

                <p className="mt-8 text-center text-xs leading-5 text-slate-400">
                  Educational information only. Always verify
                  important legal information against authoritative
                  sources.
                </p>
              </div>
            </div>
          ) : (
            <div className="flex flex-1 flex-col">
              <div className="mb-6 flex items-center justify-between">
                <p className="text-xs text-slate-500">
                  Conversation · Demo mode
                </p>
                <button
                  onClick={() => setShowSources(!showSources)}
                  className="rounded-lg border border-slate-200 bg-white px-3 py-2 text-xs text-slate-600 hover:bg-slate-50"
                >
                  {showSources ? "Hide source panel" : "Show source panel"}
                </button>
              </div>

              <div className="flex-1 space-y-6">
                {messages.map((message, index) => (
                  <div
                    key={`${index}-${message.role}`}
                    className={`flex gap-3 sm:gap-4 ${
                      message.role === "user"
                        ? "flex-row-reverse"
                        : ""
                    }`}
                  >
                    <div
                      className={`flex h-9 w-9 shrink-0 items-center justify-center rounded-xl text-sm ${
                        message.role === "assistant"
                          ? "bg-indigo-600 text-white"
                          : "bg-slate-200 font-semibold text-slate-700"
                      }`}
                    >
                      {message.role === "assistant" ? "⚖" : "U"}
                    </div>

                    <div
                      className={`max-w-[85%] rounded-2xl px-4 py-4 text-sm leading-7 sm:max-w-[75%] ${
                        message.role === "assistant"
                          ? "border border-slate-200 bg-white text-slate-700"
                          : "bg-indigo-600 text-white"
                      }`}
                    >
                      {message.content}
                    </div>
                  </div>
                ))}
              </div>

              {showSources && (
                <div className="mt-6 rounded-xl border border-dashed border-slate-300 bg-white/70 p-4">
                  <p className="text-sm font-semibold text-slate-700">
                    Legal sources
                  </p>
                  <p className="mt-1 text-xs leading-5 text-slate-500">
                    Retrieved provisions and citations will appear
                    here after the RAG backend is connected.
                    No sources have been retrieved for this demo.
                  </p>
                </div>
              )}

              <form
                onSubmit={(event) => {
                  event.preventDefault();
                  sendQuestion();
                }}
                className="mt-6 rounded-2xl border border-slate-200 bg-white p-3 shadow-lg shadow-slate-200/40"
              >
                <textarea
                  value={question}
                  onChange={(event) => setQuestion(event.target.value)}
                  onKeyDown={(event) => {
                    if (
                      event.key === "Enter" &&
                      !event.shiftKey
                    ) {
                      event.preventDefault();
                      sendQuestion();
                    }
                  }}
                  placeholder="Ask another question..."
                  rows={2}
                  className="w-full resize-none bg-transparent px-3 py-2 text-sm outline-none placeholder:text-slate-400"
                />
                <div className="flex justify-end border-t border-slate-100 pt-3">
                  <button
                    type="submit"
                    disabled={!question.trim()}
                    className="rounded-xl bg-indigo-600 px-4 py-2.5 text-sm font-semibold text-white hover:bg-indigo-700 disabled:opacity-40"
                  >
                    Send question ↑
                  </button>
                </div>
              </form>
            </div>
          )}
        </div>

        {/* Footer */}
        <footer className="border-t border-slate-200/80 bg-white/60 px-4 py-4 text-center">
          <p className="text-[11px] leading-5 text-slate-400">
            CCpeeps Legal AI · Frontend prototype · Not legal advice
          </p>
        </footer>
      </section>
    </main>
  );
}
