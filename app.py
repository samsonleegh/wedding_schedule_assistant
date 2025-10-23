import os, io, glob, time, hashlib, json
import pandas as pd
import numpy as np
from typing import List, Tuple
from dataclasses import dataclass
from dotenv import load_dotenv

# Telegram
from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters

# OpenAI
from openai import OpenAI

# Files / parsing
from docx import Document as DocxDocument
from pypdf import PdfReader

# Vector store
import faiss

from datetime import datetime, timezone

load_dotenv()
import httpx
if not hasattr(httpx, "proxies"):
    httpx.proxies = None

# -----------------------------
# Environment / Config
# -----------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
STRICT_DOC_MODE = (os.getenv("STRICT_DOC_MODE", "true").lower() == "true")

DOCS_DIR = os.getenv("DOCS_DIR", "wedding_docs")
INDEX_PATH = os.getenv("INDEX_PATH", "wedding.index")
META_CSV = os.getenv("META_CSV", "wedding_chunks.csv")

# Memory settings
MEMORY_PATH = os.getenv("MEMORY_PATH", "wedding_memory.json")
MAX_MEMORY_TURNS = int(os.getenv("MAX_MEMORY_TURNS", "12"))  # per chat, rolling window
MAX_NOTES = int(os.getenv("MAX_NOTES", "20"))                 # per chat

client = OpenAI(api_key=OPENAI_API_KEY)

# -----------------------------
# Utilities to read documents
# -----------------------------

def read_txt_md(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def read_docx(path: str) -> str:
    doc = DocxDocument(path)
    return "\n".join(p.text for p in doc.paragraphs)

def read_pdf(path: str) -> str:
    reader = PdfReader(path)
    texts = []
    for page in reader.pages:
        try:
            texts.append(page.extract_text() or "")
        except Exception:
            pass
    return "\n".join(texts)

def read_csv_file(path: str, max_rows: int = 20000, max_chars: int = 400_000) -> str:
    """
    Read a CSV as text that’s friendly for RAG.
    - Coerces everything to string
    - Skips bad lines
    - Truncates very large files by rows and total chars (to avoid blowing up memory)
    """
    try:
        df = pd.read_csv(
            path,
            dtype=str,
            on_bad_lines="skip",      # pandas>=1.4
            nrows=max_rows,           # safeguard for very large CSVs
            encoding="utf-8"
        )
    except UnicodeDecodeError:
        # Fallback for odd encodings
        df = pd.read_csv(
            path,
            dtype=str,
            on_bad_lines="skip",
            nrows=max_rows,
            encoding="latin-1"
        )

    df = df.fillna("")

    # Build a compact, readable text block:
    # 1) schema
    cols = list(df.columns)
    schema_block = "COLUMNS: " + ", ".join(cols)

    # 2) first N rows as pipe-separated lines
    lines = []
    for _, row in df.iterrows():
        kv = [f"{c}={row[c]}" for c in cols]
        lines.append(" | ".join(kv))
        # Hard cap to avoid giant strings
        if sum(len(x) for x in lines) > max_chars:
            lines.append("…(truncated)")
            break

    data_block = "\n".join(lines)
    return f"{schema_block}\n{data_block}"

def load_all_docs(folder: str) -> List[Tuple[str, str]]:
    paths = []
    for ext in ("*.md", "*.txt", "*.docx", "*.pdf", "*.csv"):
        paths.extend(glob.glob(os.path.join(folder, ext)))

    docs = []
    for p in paths:
        if p.endswith((".md", ".txt")):
            text = read_txt_md(p)
        elif p.endswith(".docx"):
            text = read_docx(p)
        elif p.endswith(".pdf"):
            text = read_pdf(p)
        elif p.endswith(".csv"):
            text = read_csv_file(p)
        else:
            continue
        docs.append((p, text))
    return docs

# -----------------------------
# Chunk + Embed + Index
# -----------------------------

def chunk_text(text: str, source: str, chunk_size: int = 300, overlap: int = 80) -> List[dict]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk_words = words[i:i+chunk_size]
        chunk = " ".join(chunk_words)
        chunks.append({
            "source": source,
            "chunk": chunk,
            "hash": hashlib.md5((source + str(i) + chunk).encode("utf-8")).hexdigest()
        })
        i += (chunk_size - overlap)
    return chunks

def embed_texts(texts: List[str]) -> np.ndarray:
    # Returns an array of shape (n, d)
    # Uses OpenAI embeddings
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    vecs = [item.embedding for item in resp.data]
    return np.array(vecs).astype("float32")

@dataclass
class RAGIndex:
    index: faiss.IndexFlatIP
    df: pd.DataFrame  # columns: [source, chunk, hash]
    dim: int

def build_or_load_index(force_rebuild: bool = False) -> RAGIndex:
    docs = load_all_docs(DOCS_DIR)
    if not docs:
        raise RuntimeError(f"No docs found in {DOCS_DIR}/. Put your itinerary files there.")

    # Simple staleness check: if any file is newer than index, rebuild
    def newest_mtime():
        paths = []
        for ext in ("*.md", "*.txt", "*.docx", "*.pdf", "*.csv"):
            paths.extend(glob.glob(os.path.join(DOCS_DIR, ext)))
        return max(os.path.getmtime(p) for p in paths)

    index_exists = os.path.exists(INDEX_PATH) and os.path.exists(META_CSV)
    need_rebuild = force_rebuild
    if index_exists:
        idx_mtime = min(os.path.getmtime(INDEX_PATH), os.path.getmtime(META_CSV))
        need_rebuild = need_rebuild or (newest_mtime() > idx_mtime)

    if index_exists and not need_rebuild:
        df = pd.read_csv(META_CSV)
        vecs = np.load(INDEX_PATH)
        dim = vecs.shape[1]
        index = faiss.IndexFlatIP(dim)
        faiss.normalize_L2(vecs)
        index.add(vecs)
        return RAGIndex(index=index, df=df, dim=dim)

    # Rebuild
    all_chunks = []
    for path, text in docs:
        if not text.strip():
            continue
        all_chunks.extend(chunk_text(text, source=path))

    if not all_chunks:
        raise RuntimeError("Docs were read but produced no chunks. Check formats.")

    df = pd.DataFrame(all_chunks)
    vecs = embed_texts(df["chunk"].tolist())
    # Normalize for cosine similarity via inner product
    faiss.normalize_L2(vecs)
    np.save(INDEX_PATH, vecs)
    df.to_csv(META_CSV, index=False)

    dim = vecs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vecs)
    return RAGIndex(index=index, df=df, dim=dim)

# -----------------------------
# Retrieval + Answering
# -----------------------------

def retrieve(query: str, rag: RAGIndex, k: int = 7) -> List[dict]:
    qvec = embed_texts([query])
    faiss.normalize_L2(qvec)
    D, I = rag.index.search(qvec, k)
    results = []
    for score, idx in zip(D[0], I[0]):
        if idx == -1:
            continue
        row = rag.df.iloc[int(idx)]
        results.append({
            "score": float(score),
            "source": row["source"],
            "chunk": row["chunk"],
        })
    return results

# -----------------------------
# Simple JSON Memory Store
# -----------------------------
class MemoryStore:
    def __init__(self, path: str):
        self.path = path
        self.data = {}  # {chat_id_str: {"messages":[{role,text,ts}], "notes":[str]}}
        self._load()

    def _load(self):
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                self.data = json.load(f)
        except FileNotFoundError:
            self.data = {}
        except Exception:
            self.data = {}

    def _save(self):
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.path)

    def _ensure_chat(self, chat_id: int):
        key = str(chat_id)
        if key not in self.data:
            self.data[key] = {"messages": [], "notes": []}
        return key

    def add_message(self, chat_id: int, role: str, text: str):
        key = self._ensure_chat(chat_id)
        msgs = self.data[key]["messages"]
        msgs.append({
            "role": role,
            "text": text.strip(),
            "ts": datetime.now(timezone.utc).isoformat(),
        })
        if len(msgs) > MAX_MEMORY_TURNS:
            self.data[key]["messages"] = msgs[-MAX_MEMORY_TURNS:]
        self._save()

    def get_recent_messages(self, chat_id: int, limit: int = 6):
        key = self._ensure_chat(chat_id)
        return self.data[key]["messages"][ -limit: ]

    def add_note(self, chat_id: int, note: str):
        key = self._ensure_chat(chat_id)
        notes = self.data[key]["notes"]
        notes.append(note.strip())
        if len(notes) > MAX_NOTES:
            self.data[key]["notes"] = notes[-MAX_NOTES:]
        self._save()

    def list_notes(self, chat_id: int):
        key = self._ensure_chat(chat_id)
        return list(self.data[key]["notes"])

    def delete_note(self, chat_id: int, idx: int) -> bool:
        key = self._ensure_chat(chat_id)
        notes = self.data[key]["notes"]
        if 0 <= idx < len(notes):
            del notes[idx]
            self._save()
            return True
        return False

    def clear_notes(self, chat_id: int):
        key = self._ensure_chat(chat_id)
        self.data[key]["notes"] = []
        self._save()


MEM = MemoryStore(MEMORY_PATH)

# -----------------------------
# Prompt scaffolding
# -----------------------------
SYSTEM_PROMPT = (
    "You are a helpful, concise wedding assistant for Samuel's wedding.\n"
    "Answer ONLY using the provided context from the wedding documents.\n"
    "Use 'Conversation Memory' solely to resolve pronouns, referents, user preferences, or earlier clarifications—"
    "but DO NOT invent facts that are not in the docs.\n"
    "If the answer isn’t in the docs, say you don’t have that info and suggest who to contact (e.g., Samuel or Samson).\n"
    "Keep answers under 6 bullets or 150 words when possible. Use SGT times."
)


def build_conversation_memory_block(chat_id: int) -> str:
    msgs = MEM.get_recent_messages(chat_id, limit=6)
    notes = MEM.list_notes(chat_id)

    def clip(s: str, n: int = 160):
        s = s.replace("\n", " ").strip()
        return s if len(s) <= n else s[:n] + "…"

    lines = []
    if msgs:
        lines.append("Recent turns:")
        for m in msgs:
            role = "User" if m["role"] == "user" else "Assistant"
            lines.append(f"- {role}: {clip(m['text'])}")

    if notes:
        lines.append("Pinned notes:")
        for i, note in enumerate(notes[-5:]):
            lines.append(f"- [{i}] {clip(note)}")

    return "\n".join(lines) if lines else ""


async def answer_with_rag(question: str, rag: RAGIndex, chat_id: int | None = None) -> str:
    ctx = retrieve(question, rag, k=6)
    context_blocks = []
    for r in ctx:
        text = r["chunk"]
        if len(text) > 800:
            text = text[:800] + "…"
        context_blocks.append(f"[Source: {os.path.basename(r['source'])}]\n{text}")

    context_text = "\n\n".join(context_blocks)

    memory_text = ""
    if chat_id is not None:
        memory_text = build_conversation_memory_block(chat_id)
        if memory_text:
            memory_text = f"Conversation Memory:\n{memory_text}\n\n"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"{memory_text}"
                f"Context from docs:\n\n{context_text}\n\n"
                f"Question: {question}"
            )
        },
    ]

    completion = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        temperature=0.2,
    )
    answer = completion.choices[0].message.content.strip()

    if STRICT_DOC_MODE and (not context_blocks or "I don’t have that info" in answer):
        if len(context_blocks) == 0:
            return (
                "I couldn’t find this in the wedding docs. Please check the Family Playbook or ask the Overall IC. "
                "You can also /refresh to make sure I have the latest files."
            )
    return answer

# -----------------------------
# Telegram Handlers
# -----------------------------
RAG = None  # lazy loaded

async def ensure_rag(force: bool = False):
    global RAG
    if RAG is None or force:
        RAG = build_or_load_index(force_rebuild=force)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await ensure_rag(False)
    msg = (
        "👋 Hello! I’m the Wedding Q&A Bot. Ask me anything about roles, timings, addresses, and logistics.\n\n"
        "Examples:\n"
        "• What time is the solemnisation?\n"
        "• What’s Mum’s role during tea ceremony?\n"
        "• Where to park at the hotel?\n"
        "• Who holds the ang bao box?\n\n"
        "Admins can /refresh after updating the docs.\n\n"
        "Memory tips: /remember <note>, /notes to view/manage."
    )
    await update.message.reply_text(msg)

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Send a normal question, or use:\n"
        "/role <name> — quick role lookup\n"
        "/refresh — rebuild knowledge from latest docs (admin only, but not enforced)\n"
        "/remember <note> — pin a short note I can reference\n"
        "/notes — list notes; '/notes del <idx>' or '/notes clear'\n"
    )

# Memory commands
async def remember_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    note = " ".join(context.args).strip()
    if not note:
        await update.message.reply_text("Usage: /remember <note to pin>")
        return
    MEM.add_note(chat_id, note)
    await update.message.reply_text("🧠 Noted. Use /notes to view or manage pinned notes.")

async def notes_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    args = context.args

    if args and args[0].lower() == "del" and len(args) >= 2:
        try:
            idx = int(args[1])
            ok = MEM.delete_note(chat_id, idx)
            await update.message.reply_text("Deleted." if ok else "Index not found.")
            return
        except ValueError:
            await update.message.reply_text("Usage: /notes del <index>")
            return

    if args and args[0].lower() == "clear":
        MEM.clear_notes(chat_id)
        await update.message.reply_text("All notes cleared.")
        return

    notes = MEM.list_notes(chat_id)
    if not notes:
        await update.message.reply_text("No pinned notes. Use /remember <note> to add one.")
        return

    out = ["📌 Pinned notes:"]
    for i, n in enumerate(notes):
        out.append(f"{i}. {n}")
    out.append("\nTip: /notes del <index> or /notes clear")
    await update.message.reply_text("\n".join(out))

async def role_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await ensure_rag(False)
    chat_id = update.effective_chat.id
    name = " ".join(context.args).strip()
    if not name:
        await update.message.reply_text("Usage: /role <name>")
        return
    q = f"What is the role and responsibilities of {name}? Include timings and contact if available."

    # memory: user asked role
    MEM.add_message(chat_id, "user", f"/role {name}")

    ans = await answer_with_rag(q, RAG, chat_id=chat_id)

    # memory: assistant responded
    MEM.add_message(chat_id, "assistant", ans)

    await update.message.reply_text(ans, parse_mode=ParseMode.MARKDOWN)

async def refresh_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await ensure_rag(True)
    await update.message.reply_text("✅ Refreshed. I’m now using the latest documents in wedding_docs/.")

async def on_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await ensure_rag(False)
    chat_id = update.effective_chat.id
    text = (update.message.text or "").strip()
    if not text:
        return

    # memory: user turn
    MEM.add_message(chat_id, "user", text)

    ans = await answer_with_rag(text, RAG, chat_id=chat_id)
    if len(ans) > 3500:
        ans = ans[:3500] + "…"

    # memory: assistant turn
    MEM.add_message(chat_id, "assistant", ans)

    await update.message.reply_text(ans, parse_mode=ParseMode.MARKDOWN)

# -----------------------------
# Entrypoint
# -----------------------------

# def main():
#     if not TELEGRAM_BOT_TOKEN:
#         raise RuntimeError("TELEGRAM_BOT_TOKEN missing")
#     if not OPENAI_API_KEY:
#         raise RuntimeError("OPENAI_API_KEY missing")

#     app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

#     app.add_handler(CommandHandler("start", start))
#     app.add_handler(CommandHandler("help", help_cmd))
#     app.add_handler(CommandHandler("role", role_cmd))
#     app.add_handler(CommandHandler("refresh", refresh_cmd))
#     app.add_handler(CommandHandler("remember", remember_cmd))
#     app.add_handler(CommandHandler("notes", notes_cmd))
#     app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_message))

#     print("Bot running… Press Ctrl+C to stop.")
#     app.run_polling(drop_pending_updates=True)

# if __name__ == "__main__":
#     main()

def main():
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN missing")
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY missing")

    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    # --- handlers (unchanged) ---
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("role", role_cmd))
    app.add_handler(CommandHandler("refresh", refresh_cmd))
    app.add_handler(CommandHandler("remember", remember_cmd))
    app.add_handler(CommandHandler("notes", notes_cmd))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_message))

    # --- webhook mode for Render Web Service ---
    import os
    port = int(os.environ.get("PORT", "10000"))  # Render provides this
    base = os.environ.get("PUBLIC_BASE_URL")
    if not base:
        raise RuntimeError("PUBLIC_BASE_URL missing (e.g., https://your-app.onrender.com)")

    # Use token as a path component (keeps the endpoint obscure)
    url_path = TELEGRAM_BOT_TOKEN

    # Optional but recommended: verify Telegram via secret token header
    secret = os.environ.get("WEBHOOK_SECRET", "")

    print(f"Starting webhook server on 0.0.0.0:{port}")
    print(f"Setting webhook to: {base}/{url_path}")

    app.run_webhook(
        listen="0.0.0.0",
        port=port,
        url_path=url_path,                       # local path
        webhook_url=f"{base}/{url_path}",        # public URL
        secret_token=secret if secret else None, # enables Telegram header verification
        drop_pending_updates=True,
    )

if __name__ == "__main__":
    main()
