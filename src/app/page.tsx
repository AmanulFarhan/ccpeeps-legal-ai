
"use client";

import { useState } from "react";
import ReactMarkdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";

type LegalSource = {
  title?: string | null;
  section_number?: string | null;
  section_title?: string | null;
  page_number?: number | null;
  end_page?: number | null;
  citation?: string | null;
  source_url?: string | null;
  content?: string | null;
};

type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: LegalSource[];
};

type MarkdownNode = {
  type: string;
  value?: unknown;
  url?: string;
  children?: MarkdownNode[];
};

function sourceCitationPlugin(sourceIds: Set<string>, messageId: string) {
  return () => (tree: MarkdownNode) => {
    const visit = (node: MarkdownNode) => {
      if (["code", "inlineCode", "link", "linkReference"].includes(node.type) || !node.children) return;

      const nextChildren: MarkdownNode[] = [];
      for (const child of node.children) {
        if (child.type !== "text" || typeof child.value !== "string") {
          visit(child);
          nextChildren.push(child);
          continue;
        }

        const citationPattern = /\[((?:S\d+)(?:\s*,\s*S\d+)*)\]/g;
        let cursor = 0;
        let replaced = false;
        let match: RegExpExecArray | null;
        while ((match = citationPattern.exec(child.value)) !== null) {
          replaced = true;
          if (match.index > cursor) {
            nextChildren.push({ type: "text", value: child.value.slice(cursor, match.index) });
          }

          const ids = match[1].match(/S\d+/g) ?? [];
          ids.forEach((id, index) => {
            if (index > 0) nextChildren.push({ type: "text", value: ", " });
            if (sourceIds.has(id)) {
              nextChildren.push({
                type: "link",
                url: `#${messageId}-source-${id}`,
                children: [{ type: "text", value: `[${id}]` }],
              });
            } else {
              nextChildren.push({
                type: "inlineCode",
                value: `[${id}] (source unavailable)`,
              });
            }
          });
          cursor = match.index + match[0].length;
        }

        if (cursor < child.value.length) {
          nextChildren.push({ type: "text", value: child.value.slice(cursor) });
        }
        if (!replaced) nextChildren.push(child);
      }

      node.children = nextChildren;
    };

    visit(tree);
  };
}

const markdownComponents: Components = {
  h1: ({ children }) => <h1 className="mb-4 mt-7 text-2xl font-bold tracking-tight text-slate-900 first:mt-0">{children}</h1>,
  h2: ({ children }) => <h2 className="mb-3 mt-7 text-xl font-bold tracking-tight text-slate-900 first:mt-0">{children}</h2>,
  h3: ({ children }) => <h3 className="mb-2 mt-5 text-lg font-semibold text-slate-900 first:mt-0">{children}</h3>,
  p: ({ children }) => <p className="my-3 leading-7 text-slate-700 first:mt-0 last:mb-0">{children}</p>,
  ul: ({ children }) => <ul className="my-3 list-disc space-y-1 pl-6 marker:text-indigo-500">{children}</ul>,
  ol: ({ children }) => <ol className="my-3 list-decimal space-y-1 pl-6 marker:font-semibold marker:text-indigo-600">{children}</ol>,
  li: ({ children }) => <li className="pl-1 leading-7">{children}</li>,
  blockquote: ({ children }) => <blockquote className="my-4 border-l-4 border-indigo-200 pl-4 italic text-slate-600">{children}</blockquote>,
  code: ({ children, className }) => (
    <code className={className ? "font-mono text-[0.9em]" : "rounded bg-slate-100 px-1.5 py-0.5 font-mono text-[0.9em] text-indigo-800"}>
      {children}
    </code>
  ),
  pre: ({ children }) => <pre className="my-4 overflow-x-auto rounded-xl bg-slate-900 p-4 text-sm leading-6 text-slate-100">{children}</pre>,
  table: ({ children }) => <div className="my-4 overflow-x-auto"><table className="w-full border-collapse text-left text-sm">{children}</table></div>,
  th: ({ children }) => <th className="border border-slate-200 bg-slate-50 px-3 py-2 font-semibold text-slate-800">{children}</th>,
  td: ({ children }) => <td className="border border-slate-200 px-3 py-2 align-top text-slate-700">{children}</td>,
  a: ({ children, href }) => <a href={href} target={href?.startsWith("http") ? "_blank" : undefined} rel={href?.startsWith("http") ? "noreferrer" : undefined} className="text-indigo-700 underline decoration-indigo-300 underline-offset-2 hover:text-indigo-900">{children}</a>,
  hr: () => <hr className="my-6 border-slate-200" />,
};

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

const initialMessages: ChatMessage[] = [
  {
    id: "welcome",
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
  const [isSending, setIsSending] = useState(false);
  const [expandedSources, setExpandedSources] = useState<Set<string>>(() => new Set());
  const [selectedSource, setSelectedSource] = useState<string | null>(null);

  function startNewChat() {
    setMessages(initialMessages);
    setQuestion("");
    setActiveNav("Legal Assistant");
    setExpandedSources(new Set());
    setSelectedSource(null);
  }

  function openSource(domId: string) {
    setExpandedSources((previous) => new Set(previous).add(domId));
    setSelectedSource(domId);
    window.setTimeout(() => {
      document.getElementById(domId)?.scrollIntoView({ behavior: "smooth", block: "center" });
    }, 0);
  }

  async function sendQuestion(value = question) {
    const trimmed = value.trim();
    if (!trimmed || isSending) return;

    const userMessageId = `${Date.now()}-user`;
    const answerMessageId = `${Date.now()}-assistant`;

    setMessages((previous) => [
      ...previous,
      { id: userMessageId, role: "user", content: trimmed },
      {
        id: answerMessageId,
        role: "assistant",
        content: "Searching the legal sources...",
      },
    ]);

    setQuestion("");
    setActiveNav("Legal Assistant");
    setIsSending(true);

    try {
      const response = await fetch("/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: trimmed, response_language: language }),
      });
      const result = await response.json();
      if (!response.ok) {
        throw new Error(result.error || "The legal assistant could not process this question.");
      }
      if (typeof result.answer !== "string") {
        throw new Error("The RAG backend returned an invalid answer.");
      }

      setMessages((previous) =>
        previous.map((message) =>
          message.id === answerMessageId
            ? { ...message, content: result.answer, sources: Array.isArray(result.sources) ? result.sources : [] }
            : message,
        ),
      );
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : "Something went wrong. Please try again.";
      setMessages((previous) =>
        previous.map((message) =>
          message.id === answerMessageId
            ? { ...message, content: `I couldn't get an answer: ${errorMessage}` }
            : message,
        ),
      );
    } finally {
      setIsSending(false);
    }
  }

  return (
    <main className="min-h-screen bg-[#f7f8fc] text-slate-800">
      {/* Sidebar */}
      <aside className="fixed inset-y-0 left-0 z-20 hidden h-screen w-[260px] flex-col overflow-y-auto border-r border-slate-200 bg-white p-5 md:flex">
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
      <section className="flex min-h-screen min-w-0 flex-col md:ml-[260px]">
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
                    void sendQuestion();
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
                      disabled={!question.trim() || isSending}
                      className="flex items-center gap-2 rounded-xl bg-indigo-600 px-4 py-2.5 text-sm font-semibold text-white transition hover:bg-indigo-700 disabled:cursor-not-allowed disabled:opacity-40"
                    >
                      {isSending ? "Thinking…" : <>Ask <span>↑</span></>}
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
                  Conversation
                </p>
                <button
                  onClick={() => setShowSources(!showSources)}
                  className="rounded-lg border border-slate-200 bg-white px-3 py-2 text-xs text-slate-600 hover:bg-slate-50"
                >
                  {showSources ? "Hide source panel" : "Show source panel"}
                </button>
              </div>

              <div className="flex-1 space-y-6">
                {messages.map((message) => (
                  <div
                    key={message.id}
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

                    <div className={message.role === "assistant" ? "w-full max-w-3xl min-w-0" : "max-w-[85%] sm:max-w-[75%]"}>
                      <div
                          className={`rounded-2xl px-5 py-5 text-[15px] leading-7 sm:px-6 ${
                          message.role === "assistant"
                            ? "border border-slate-200 bg-white text-slate-700"
                            : "bg-indigo-600 text-white"
                        }`}
                      >
                        {message.role === "assistant" ? (
                          <ReactMarkdown
                            remarkPlugins={[
                              remarkGfm,
                              sourceCitationPlugin(
                                new Set((message.sources ?? []).map((_, index) => `S${index + 1}`)),
                                message.id,
                              ),
                            ]}
                            components={{
                              ...markdownComponents,
                              a: ({ children, href }) => {
                                const prefix = `#${message.id}-source-`;
                                if (href?.startsWith(prefix)) {
                                  const domId = href.slice(1);
                                  const sourceId = domId.slice(prefix.length);
                                  return (
                                    <a
                                      href={href}
                                      onClick={(event) => {
                                        event.preventDefault();
                                        if ((message.sources ?? []).some((_, index) => `S${index + 1}` === sourceId)) openSource(domId);
                                      }}
                                      className="mx-0.5 inline-flex items-center rounded-md bg-indigo-50 px-1.5 py-0.5 align-baseline text-xs font-semibold text-indigo-700 no-underline ring-1 ring-inset ring-indigo-200 hover:bg-indigo-100 focus:outline-none focus:ring-2 focus:ring-indigo-500"
                                    >
                                      {children}
                                    </a>
                                  );
                                }
                                return (
                                  <a
                                    href={href}
                                    target={href?.startsWith("http") ? "_blank" : undefined}
                                    rel={href?.startsWith("http") ? "noreferrer" : undefined}
                                    className="text-indigo-700 underline decoration-indigo-300 underline-offset-2 hover:text-indigo-900"
                                  >
                                    {children}
                                  </a>
                                );
                              },
                            }}
                          >
                            {message.content}
                          </ReactMarkdown>
                        ) : (
                          message.content
                        )}
                      </div>
                      {message.role === "assistant" && showSources && message.sources && message.sources.length > 0 && (
                        <div className="mt-3 rounded-xl border border-slate-200 bg-white p-3">
                          <p className="mb-2 text-xs font-semibold text-slate-700">Retrieved legal sources</p>
                          <ul className="space-y-2">
                            {message.sources.map((source, sourceIndex) => (
                              <li
                                key={`${message.id}-source-${sourceIndex}`}
                                id={`${message.id}-source-S${sourceIndex + 1}`}
                                className={`rounded-lg border p-3 text-xs leading-5 transition ${selectedSource === `${message.id}-source-S${sourceIndex + 1}` ? "border-indigo-300 bg-indigo-50 ring-2 ring-indigo-100" : "border-slate-200 bg-white"}`}
                              >
                                <span className="font-semibold">[S{sourceIndex + 1}] </span>
                                {source.source_url ? (
                                  <a href={source.source_url} target="_blank" rel="noreferrer" className="font-medium text-indigo-700 hover:underline">
                                    {source.title || "Legal document"}
                                  </a>
                                ) : (
                                  <span className="font-medium">{source.title || "Legal document"}</span>
                                )}
                                <span>{` · Section ${source.section_number || "?"}${source.section_title ? `, ${source.section_title}` : ""} · Pages ${source.page_number ?? "?"}-${source.end_page ?? "?"}${source.citation ? ` · ${source.citation}` : ""}`}</span>
                                {source.content && (
                                  <>
                                    <p className="mt-2 line-clamp-2 text-slate-500">{source.content}</p>
                                    <details
                                      open={expandedSources.has(`${message.id}-source-S${sourceIndex + 1}`)}
                                      onToggle={(event) => {
                                        const domId = `${message.id}-source-S${sourceIndex + 1}`;
                                        const isOpen = event.currentTarget.open;
                                        setExpandedSources((previous) => {
                                          if (previous.has(domId) === isOpen) return previous;
                                          const next = new Set(previous);
                                          if (isOpen) next.add(domId);
                                          else next.delete(domId);
                                          return next;
                                        });
                                      }}
                                      className="mt-2 border-t border-slate-100 pt-2"
                                    >
                                      <summary className="cursor-pointer font-medium text-indigo-700 hover:text-indigo-900">Show full retrieved passage</summary>
                                      <p className="max-h-80 overflow-y-auto whitespace-pre-wrap pt-2 text-slate-600">{source.content}</p>
                                    </details>
                                  </>
                                )}
                              </li>
                            ))}
                          </ul>
                        </div>
                      )}
                    </div>
                  </div>
                ))}
              </div>

              <form
                onSubmit={(event) => {
                  event.preventDefault();
                  void sendQuestion();
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
                    disabled={!question.trim() || isSending}
                    className="rounded-xl bg-indigo-600 px-4 py-2.5 text-sm font-semibold text-white hover:bg-indigo-700 disabled:opacity-40"
                  >
                    {isSending ? "Thinking…" : "Send question ↑"}
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
