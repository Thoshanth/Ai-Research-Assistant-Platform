import axios from "axios";

const BASE_URL = "http://127.0.0.1:8000";

const api = axios.create({
  baseURL: BASE_URL,
});

// ── Stage 1 ───────────────────────────────────────────
export const uploadDocument = async (file) => {
  const formData = new FormData();
  formData.append("file", file);
  const res = await api.post("/upload", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return res.data;
};

export const getDocuments = async () => {
  const res = await api.get("/documents");
  return res.data;
};

// ── Stage 3 ───────────────────────────────────────────
export const queryDocuments = async (question, documentId = null, topK = 3) => {
  const params = { question, top_k: topK };
  if (documentId) params.document_id = documentId;
  const res = await api.post("/query", null, { params });
  return res.data;
};

export const indexDocument = async (documentId, strategy = "recursive") => {
  const res = await api.post(`/index/${documentId}`, null, {
    params: { strategy },
  });
  return res.data;
};

// ── Stage 4 ───────────────────────────────────────────
export const chatWithMemory = async (question, sessionId = "default", documentId = null) => {
  const params = { question, session_id: sessionId };
  if (documentId) params.document_id = documentId;
  const res = await api.post("/chat", null, { params });
  return res.data;
};

export const resetChat = async (sessionId = "default") => {
  const res = await api.post("/chat/reset", null, {
    params: { session_id: sessionId },
  });
  return res.data;
};

// ── Health ────────────────────────────────────────────
export const getHealth = async () => {
  const res = await api.get("/health");
  return res.data;
};