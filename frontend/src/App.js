import { useState, useEffect, useRef } from "react";
import ReactMarkdown from "react-markdown";
import {
  uploadDocument,
  getDocuments,
  queryDocuments,
  indexDocument,
  chatWithMemory,
  resetChat,
} from "./api";
import "./App.css";

// ── Sidebar: Document List ────────────────────────────────────────
function Sidebar({ documents, selectedDoc, onSelect, onUpload, uploading }) {
  const fileRef = useRef();

  return (
    <div className="sidebar">
      <div className="sidebar-header">
        <h2>📁 Documents</h2>
        <button
          className="upload-btn"
          onClick={() => fileRef.current.click()}
          disabled={uploading}
        >
          {uploading ? "Uploading..." : "+ Upload"}
        </button>
        <input
          ref={fileRef}
          type="file"
          accept=".pdf,.txt,.csv"
          style={{ display: "none" }}
          onChange={(e) => onUpload(e.target.files[0])}
        />
      </div>

      <div className="doc-list">
        {documents.length === 0 && (
          <p className="empty-hint">No documents yet. Upload one to start.</p>
        )}
        {documents.map((doc) => (
          <div
            key={doc.id}
            className={`doc-item ${selectedDoc?.id === doc.id ? "active" : ""}`}
            onClick={() => onSelect(doc)}
          >
            <span className="doc-icon">
              {doc.file_type === "pdf" ? "📄" : doc.file_type === "csv" ? "📊" : "📝"}
            </span>
            <div className="doc-info">
              <span className="doc-name">{doc.filename}</span>
              <span className="doc-meta">{doc.word_count} words</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ── Message Bubble ────────────────────────────────────────────────
function Message({ msg }) {
  const isUser = msg.role === "user";
  return (
    <div className={`message ${isUser ? "user" : "assistant"}`}>
      <div className="bubble">
        {isUser ? (
          <p>{msg.content}</p>
        ) : (
          <ReactMarkdown>{msg.content}</ReactMarkdown>
        )}
        {msg.sources && msg.sources.length > 0 && (
          <div className="sources">
            <span className="sources-label">Sources:</span>
            {msg.sources.map((s, i) => (
              <span key={i} className="source-tag">
                {s.filename} (chunk {s.chunk_index})
              </span>
            ))}
          </div>
        )}
        {msg.guardrail_info?.warnings?.length > 0 && (
          <div className="warnings">
            ⚠️ {msg.guardrail_info.warnings.join(", ")}
          </div>
        )}
      </div>
    </div>
  );
}

// ── Chat Panel ────────────────────────────────────────────────────
function ChatPanel({ selectedDoc, mode }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [sessionId] = useState(() => `session_${Date.now()}`);
  const bottomRef = useRef();

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const sendMessage = async () => {
    if (!input.trim() || loading) return;

    const userMsg = { role: "user", content: input };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setLoading(true);

    try {
      let response;
      const docId = selectedDoc?.id || null;

      if (mode === "query") {
        // Stage 3 — direct RAG query
        response = await queryDocuments(input, docId);
        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            content: response.answer,
            sources: response.sources,
            guardrail_info: response.guardrail_info,
          },
        ]);
      } else {
        // Stage 4 — chat with memory
        response = await chatWithMemory(input, sessionId, docId);
        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            content: response.answer,
            source: response.source,
            guardrail_info: response.guardrail_info,
          },
        ]);
      }
    } catch (err) {
      const detail = err.response?.data?.detail;
      let errorMsg = "Something went wrong.";

      if (typeof detail === "object") {
        errorMsg = `🚫 ${detail.error}\n\nReason: ${detail.reason || ""}`;
      } else if (typeof detail === "string") {
        errorMsg = `🚫 ${detail}`;
      }

      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: errorMsg },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = async () => {
    await resetChat(sessionId);
    setMessages([]);
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <div className="chat-panel">
      {/* Mode indicator */}
      <div className="chat-header">
        <span className="mode-badge">
          {mode === "query" ? "📖 RAG Query" : "💬 Chat with Memory"}
        </span>
        {selectedDoc && (
          <span className="doc-badge">🔍 {selectedDoc.filename}</span>
        )}
        {mode === "chat" && messages.length > 0 && (
          <button className="reset-btn" onClick={handleReset}>
            Clear Memory
          </button>
        )}
      </div>

      {/* Messages */}
      <div className="messages">
        {messages.length === 0 && (
          <div className="empty-chat">
            <h3>
              {mode === "query"
                ? "Ask anything about your documents"
                : "Start a conversation"}
            </h3>
            <p>
              {selectedDoc
                ? `Searching in: ${selectedDoc.filename}`
                : "Select a document from the sidebar or ask a general question"}
            </p>
          </div>
        )}
        {messages.map((msg, i) => (
          <Message key={i} msg={msg} />
        ))}
        {loading && (
          <div className="message assistant">
            <div className="bubble loading">
              <span></span><span></span><span></span>
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div className="input-area">
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={
            mode === "query"
              ? "Ask a question about the document..."
              : "Type a message... (Enter to send)"
          }
          rows={2}
          disabled={loading}
        />
        <button
          onClick={sendMessage}
          disabled={loading || !input.trim()}
          className="send-btn"
        >
          {loading ? "..." : "Send"}
        </button>
      </div>
    </div>
  );
}

// ── Index Banner ──────────────────────────────────────────────────
function IndexBanner({ doc, onIndex }) {
  const [indexing, setIndexing] = useState(false);
  const [indexed, setIndexed] = useState(false);

  const handleIndex = async () => {
    setIndexing(true);
    try {
      await indexDocument(doc.id);
      setIndexed(true);
    } catch {
      alert("Indexing failed. Check the backend logs.");
    } finally {
      setIndexing(false);
    }
  };

  if (indexed) return null;

  return (
    <div className="index-banner">
      <span>
        ⚡ <strong>{doc.filename}</strong> is not indexed yet.
        Index it to enable search.
      </span>
      <button onClick={handleIndex} disabled={indexing}>
        {indexing ? "Indexing..." : "Index Now"}
      </button>
    </div>
  );
}

// ── Main App ──────────────────────────────────────────────────────
export default function App() {
  const [documents, setDocuments] = useState([]);
  const [selectedDoc, setSelectedDoc] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [mode, setMode] = useState("chat"); // "chat" | "query"
  const [toast, setToast] = useState(null);

  // Load documents on mount
  useEffect(() => {
    loadDocuments();
  }, []);

  const loadDocuments = async () => {
    try {
      const docs = await getDocuments();
      setDocuments(docs);
    } catch {
      showToast("Failed to load documents", "error");
    }
  };

  const handleUpload = async (file) => {
    if (!file) return;
    setUploading(true);
    try {
      const result = await uploadDocument(file);
      showToast(`✅ Uploaded: ${result.filename}`, "success");
      await loadDocuments();
      // Auto-select the newly uploaded document
      setSelectedDoc({ id: result.document_id, filename: result.filename });
    } catch (err) {
      showToast("Upload failed. Check file type.", "error");
    } finally {
      setUploading(false);
    }
  };

  const showToast = (message, type = "info") => {
    setToast({ message, type });
    setTimeout(() => setToast(null), 3000);
  };

  return (
    <div className="app">
      {/* Toast notification */}
      {toast && (
        <div className={`toast ${toast.type}`}>{toast.message}</div>
      )}

      {/* Top navigation */}
      <nav className="navbar">
        <div className="nav-brand">
          <span className="brand-icon">🔬</span>
          <span className="brand-name">AI Research Assistant</span>
        </div>
        <div className="mode-toggle">
          <button
            className={mode === "chat" ? "active" : ""}
            onClick={() => setMode("chat")}
          >
            💬 Chat
          </button>
          <button
            className={mode === "query" ? "active" : ""}
            onClick={() => setMode("query")}
          >
            📖 Query
          </button>
        </div>
      </nav>

      {/* Index banner */}
      {selectedDoc && <IndexBanner doc={selectedDoc} onIndex={indexDocument} />}

      {/* Main layout */}
      <div className="main-layout">
        <Sidebar
          documents={documents}
          selectedDoc={selectedDoc}
          onSelect={setSelectedDoc}
          onUpload={handleUpload}
          uploading={uploading}
        />
        <ChatPanel
          key={`${mode}-${selectedDoc?.id}`}
          selectedDoc={selectedDoc}
          mode={mode}
        />
      </div>
    </div>
  );
}