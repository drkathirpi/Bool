"""
BookSplit v2 - self-hosted textbook reader, splitter & AI tutor.

- Upload a PDF (or drop into DATA_DIR/books via scp/rclone; auto-indexed)
- Robust table-of-contents extraction (PyMuPDF + pypdf fallback)
- Read the whole book in the browser (page-image reader, preloading)
- Preview any chapter before downloading it as its own PDF
- AI tutor: save any page range as a persistent "topic" (text stored
  server-side once), then chat about it via NVIDIA NIM - no re-uploads.

Env:
  DATA_DIR        default /data
  APP_PASSWORD    optional basic-auth password
  NVIDIA_API_KEY  required for the AI tutor
  NIM_BASE_URL    default https://integrate.api.nvidia.com/v1
  NIM_MODEL       default meta/llama-3.3-70b-instruct
"""

import base64
import json
import os
import re
import secrets
import threading
import time
import uuid

import fitz  # PyMuPDF
import httpx
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, Response

DATA_DIR = os.environ.get("DATA_DIR", "/data")
BOOKS_DIR = os.path.join(DATA_DIR, "books")
META_DIR = os.path.join(DATA_DIR, "meta")
TEXT_DIR = os.path.join(DATA_DIR, "text")
AI_DIR = os.path.join(DATA_DIR, "ai")
for d in (BOOKS_DIR, META_DIR, TEXT_DIR, AI_DIR):
    os.makedirs(d, exist_ok=True)

META_VERSION = 2
NIM_BASE = os.environ.get("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1").rstrip("/")
NIM_MODEL = os.environ.get("NIM_MODEL", "meta/llama-3.3-70b-instruct")

app = FastAPI(title="BookSplit")


# ------------------------------------------------------------------ auth
@app.middleware("http")
async def basic_auth(request: Request, call_next):
    password = os.environ.get("APP_PASSWORD")
    if password:
        ok = False
        header = request.headers.get("authorization", "")
        if header.startswith("Basic "):
            try:
                creds = base64.b64decode(header[6:]).decode()
                supplied = creds.split(":", 1)[1] if ":" in creds else creds
                ok = secrets.compare_digest(supplied, password)
            except Exception:
                ok = False
        if not ok:
            return Response(status_code=401,
                            headers={"WWW-Authenticate": 'Basic realm="BookSplit"'})
    return await call_next(request)


# ------------------------------------------------------------------ helpers
def slugify(name: str) -> str:
    name = re.sub(r"\.pdf$", "", name, flags=re.I)
    name = re.sub(r"[^A-Za-z0-9._ -]+", "", name).strip()
    name = re.sub(r"\s+", "_", name)
    return name[:120] or f"book_{int(time.time())}"


def book_path(bid): return os.path.join(BOOKS_DIR, f"{bid}.pdf")
def meta_path(bid): return os.path.join(META_DIR, f"{bid}.json")
def text_path(bid): return os.path.join(TEXT_DIR, f"{bid}.json")
def topic_path(tid): return os.path.join(AI_DIR, f"{tid}.json")


def extract_toc(doc, path):
    """Try hard to get [[level, title, page], ...] out of any PDF."""
    toc = []
    try:
        toc = doc.get_toc(simple=True) or []
    except Exception:
        toc = []
    toc = [[t[0], str(t[1] or "").strip(), t[2]]
           for t in toc
           if len(t) >= 3 and isinstance(t[2], int) and t[2] >= 1 and str(t[1] or "").strip()]
    if toc:
        return toc

    # Fallback 1: full outline with destination resolution
    try:
        for it in (doc.get_toc(simple=False) or []):
            lvl, title, page = it[0], str(it[1] or "").strip(), it[2]
            if not isinstance(page, int) or page < 1:
                dest = it[3] if len(it) > 3 and isinstance(it[3], dict) else {}
                p = dest.get("page")
                page = (p + 1) if isinstance(p, int) and p >= 0 else 0
            if page >= 1 and title:
                toc.append([lvl, title, page])
    except Exception:
        pass
    if toc:
        return toc

    # Fallback 2: pypdf reads some outlines that PyMuPDF rejects
    try:
        from pypdf import PdfReader
        reader = PdfReader(path)
        out = []

        def walk(items, lvl):
            for it in items:
                if isinstance(it, list):
                    walk(it, lvl + 1)
                    continue
                try:
                    page = reader.get_destination_page_number(it) + 1
                except Exception:
                    continue
                title = str(getattr(it, "title", "") or "").strip()
                if title and page >= 1:
                    out.append([lvl, title, page])

        walk(reader.outline, 1)
        return out
    except Exception:
        return []


def compute_ranges(toc, page_count):
    entries = []
    for i, (level, title, page) in enumerate(toc):
        start = max(1, min(int(page), page_count))
        end = page_count
        for j in range(i + 1, len(toc)):
            if toc[j][0] <= level:
                end = max(start, min(int(toc[j][2]), page_count) - 1)
                break
        entries.append({"level": level, "title": title, "start": start, "end": end})
    return entries


def index_book(bid: str):
    path = book_path(bid)
    doc = fitz.open(path)
    meta = {
        "v": META_VERSION,
        "id": bid,
        "name": bid.replace("_", " "),
        "pages": doc.page_count,
        "size_mb": round(os.path.getsize(path) / 1048576, 1),
        "toc": compute_ranges(extract_toc(doc, path), doc.page_count),
        "indexed": int(time.time()),
    }
    doc.close()
    with open(meta_path(bid), "w") as f:
        json.dump(meta, f)
    return meta


def load_meta(bid: str):
    p = meta_path(bid)
    if not os.path.exists(p):
        if os.path.exists(book_path(bid)):
            return index_book(bid)
        raise HTTPException(404, "Book not found")
    with open(p) as f:
        meta = json.load(f)
    if meta.get("v") != META_VERSION:          # auto re-index after upgrades
        meta = index_book(bid)
    return meta


def scan_library():
    books = []
    for fn in sorted(os.listdir(BOOKS_DIR)):
        if fn.lower().endswith(".pdf"):
            try:
                books.append(load_meta(fn[:-4]))
            except Exception:
                continue
    return books


def get_page_texts(bid: str):
    tp = text_path(bid)
    if os.path.exists(tp):
        with open(tp) as f:
            return json.load(f)
    doc = fitz.open(book_path(bid))
    pages = [page.get_text("text") for page in doc]
    doc.close()
    with open(tp, "w") as f:
        json.dump(pages, f)
    return pages


# ------------------------------------------------------------------ page renderer (with small doc cache)
_docs = {}
_render_lock = threading.Lock()


def _get_doc(bid):
    d = _docs.pop(bid, None)
    if d is None:
        if len(_docs) >= 3:
            _docs.pop(next(iter(_docs))).close()
        d = fitz.open(book_path(bid))
    _docs[bid] = d
    return d


def _evict_doc(bid):
    with _render_lock:
        d = _docs.pop(bid, None)
        if d:
            d.close()


@app.get("/api/books/{bid}/page/{n}")
def api_page_image(bid: str, n: int, w: int = 1100):
    load_meta(bid)
    w = max(300, min(int(w), 1600))
    try:
        with _render_lock:
            doc = _get_doc(bid)
            if n < 1 or n > doc.page_count:
                raise HTTPException(404, "Page out of range")
            page = doc[n - 1]
            pw = max(50.0, float(page.rect.width))   # guard odd page geometry
            zoom = max(0.3, min(w / pw, 3.0))        # never explode the zoom
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
            if pix.width * pix.height > 4_500_000:   # hard memory cap ~4.5MP
                scale = (4_500_000 / (pix.width * pix.height)) ** 0.5
                pix = None
                zoom = max(0.3, zoom * scale)
                pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
            try:
                data, mt = pix.tobytes("jpeg"), "image/jpeg"
            except Exception:
                data, mt = pix.tobytes("png"), "image/png"
            pix = None
    except HTTPException:
        raise
    except Exception as e:
        print(f"[render error] {bid} p{n}: {e}", flush=True)
        raise HTTPException(500, f"Render failed: {e}")
    return Response(data, media_type=mt,
                    headers={"Cache-Control": "public, max-age=86400"})


# ------------------------------------------------------------------ library API
@app.get("/api/books")
def api_books():
    return [{k: b[k] for k in ("id", "name", "pages", "size_mb")}
            | {"chapters": len(b["toc"])} for b in scan_library()]


@app.post("/api/upload")
async def api_upload(file: UploadFile = File(...)):
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported")
    bid = slugify(file.filename)
    n = 1
    while os.path.exists(book_path(bid if n == 1 else f"{bid}_{n}")):
        n += 1
    if n > 1:
        bid = f"{bid}_{n}"
    path = book_path(bid)
    with open(path, "wb") as out:
        while chunk := await file.read(1 << 20):
            out.write(chunk)
    try:
        meta = index_book(bid)
    except Exception as e:
        os.remove(path)
        raise HTTPException(400, f"Could not read PDF: {e}")
    return {"id": bid, "pages": meta["pages"], "chapters": len(meta["toc"])}


@app.post("/api/books/{bid}/reindex")
def api_reindex(bid: str):
    if not os.path.exists(book_path(bid)):
        raise HTTPException(404, "Book not found")
    tp = text_path(bid)
    if os.path.exists(tp):
        os.remove(tp)
    meta = index_book(bid)
    return {"chapters": len(meta["toc"]), "pages": meta["pages"]}


@app.delete("/api/books/{bid}")
def api_delete(bid: str):
    _evict_doc(bid)
    removed = False
    for p in (book_path(bid), meta_path(bid), text_path(bid)):
        if os.path.exists(p):
            os.remove(p)
            removed = True
    if not removed:
        raise HTTPException(404, "Book not found")
    return {"deleted": bid}


@app.get("/api/books/{bid}/toc")
def api_toc(bid: str):
    m = load_meta(bid)
    return {"id": bid, "name": m["name"], "pages": m["pages"], "toc": m["toc"]}


@app.get("/api/books/{bid}/search")
def api_search(bid: str, q: str):
    q = q.strip()
    if len(q) < 3:
        raise HTTPException(400, "Query must be at least 3 characters")
    load_meta(bid)
    pages = get_page_texts(bid)
    ql = q.lower()
    hits = []
    for i, text in enumerate(pages):
        pos = text.lower().find(ql)
        if pos == -1:
            continue
        snippet = re.sub(r"\s+", " ", text[max(0, pos - 70): pos + len(q) + 70]).strip()
        hits.append({"page": i + 1, "snippet": snippet})
        if len(hits) >= 100:
            break
    return {"query": q, "hits": hits}


@app.get("/api/books/{bid}/extract")
def api_extract(bid: str, start: int, end: int, title: str = ""):
    meta = load_meta(bid)
    doc = fitz.open(book_path(bid))
    start, end = max(1, start), min(end, doc.page_count)
    if start > end:
        doc.close()
        raise HTTPException(400, "Invalid page range")
    out = fitz.open()
    out.insert_pdf(doc, from_page=start - 1, to_page=end - 1)
    data = out.tobytes(garbage=3, deflate=True)
    out.close()
    doc.close()
    fname = slugify(f"{meta['name']} {title}".strip()) + f"_p{start}-{end}.pdf"
    return Response(content=data, media_type="application/pdf",
                    headers={"Content-Disposition": f'attachment; filename="{fname}"'})


# ------------------------------------------------------------------ AI tutor (NVIDIA NIM)
def load_topic(tid: str):
    p = topic_path(tid)
    if not os.path.exists(p):
        raise HTTPException(404, "Topic not found")
    with open(p) as f:
        return json.load(f)


def save_topic(topic):
    with open(topic_path(topic["id"]), "w") as f:
        json.dump(topic, f)


def topic_summary(t):
    return {k: t[k] for k in ("id", "book", "book_name", "title", "start", "end")} | {
        "messages": len(t["messages"])}


@app.get("/api/topics")
def api_topics(book: str = ""):
    out = []
    for fn in sorted(os.listdir(AI_DIR)):
        if not fn.endswith(".json"):
            continue
        try:
            with open(os.path.join(AI_DIR, fn)) as f:
                t = json.load(f)
            if not book or t.get("book") == book:
                out.append(topic_summary(t))
        except Exception:
            continue
    return out


@app.post("/api/books/{bid}/topics")
def api_topic_create(bid: str, body: dict):
    meta = load_meta(bid)
    try:
        start, end = int(body.get("start")), int(body.get("end"))
    except Exception:
        raise HTTPException(400, "start and end pages required")
    start, end = max(1, start), min(end, meta["pages"])
    if start > end:
        raise HTTPException(400, "Invalid page range")
    if end - start > 400:
        raise HTTPException(400, "Topic too large - keep it under 400 pages")
    texts = get_page_texts(bid)
    pages = [{"page": i + 1, "text": texts[i]} for i in range(start - 1, end)]
    topic = {
        "id": uuid.uuid4().hex[:10],
        "book": bid,
        "book_name": meta["name"],
        "title": (body.get("title") or "").strip() or f"Pages {start}-{end}",
        "start": start, "end": end,
        "pages": pages,
        "messages": [],
        "created": int(time.time()),
    }
    save_topic(topic)
    return topic_summary(topic)


@app.get("/api/topics/{tid}/history")
def api_topic_history(tid: str):
    t = load_topic(tid)
    return {"id": tid, "title": t["title"], "messages": t["messages"]}


@app.delete("/api/topics/{tid}")
def api_topic_delete(tid: str):
    p = topic_path(tid)
    if not os.path.exists(p):
        raise HTTPException(404, "Topic not found")
    os.remove(p)
    return {"deleted": tid}


# Max characters of topic text sent per question (~4 chars per token).
# 350k chars is roughly 90k tokens - comfortably inside a 128k-context model.
MAX_CONTEXT_CHARS = int(os.environ.get("MAX_CONTEXT_CHARS", "350000"))


def _context_pages(topic):
    """Send the ENTIRE topic with every question, truncating only if it
    would overflow the model's context window."""
    pages, used, total, truncated = topic["pages"], [], 0, False
    for p in pages:
        total += len(p["text"]) + 16
        if total > MAX_CONTEXT_CHARS:
            truncated = True
            break
        used.append(p)
    return used, truncated


@app.post("/api/topics/{tid}/chat")
def api_topic_chat(tid: str, body: dict):
    message = (body.get("message") or "").strip()
    if not message:
        raise HTTPException(400, "Empty message")
    key = os.environ.get("NVIDIA_API_KEY")
    if not key:
        raise HTTPException(500, "NVIDIA_API_KEY is not set on the server")
    topic = load_topic(tid)
    ctx, truncated = _context_pages(topic)
    if not ctx:
        raise HTTPException(400, "Topic text is empty")
    excerpts = "\n\n".join(f"[Page {p['page']}]\n{p['text']}" for p in ctx)
    if truncated:
        excerpts += (f"\n\n[NOTE: text after page {ctx[-1]['page']} was omitted "
                     "because it exceeds the model's context window]")
    system = (
        "You are a precise paediatrics tutor helping a doctor revise. "
        f"Below is the COMPLETE text of the topic '{topic['title']}' from "
        f"'{topic['book_name']}' (pp. {topic['start']}-{topic['end']}). "
        "Answer using ONLY this text. Cite the page number(s) you used, "
        "e.g. (p. 412). If the text does not contain the answer, say so "
        "plainly instead of guessing. Be concise and clinically structured."
        "\n\n=== TOPIC TEXT ===\n" + excerpts
    )
    msgs = ([{"role": "system", "content": system}]
            + topic["messages"][-8:]
            + [{"role": "user", "content": message}])
    try:
        r = httpx.post(f"{NIM_BASE}/chat/completions",
                       headers={"Authorization": f"Bearer {key}"},
                       json={"model": NIM_MODEL, "messages": msgs,
                             "temperature": 0.2, "max_tokens": 1024},
                       timeout=180)
    except Exception as e:
        raise HTTPException(502, f"Could not reach NIM: {e}")
    if r.status_code != 200:
        raise HTTPException(502, f"NIM error {r.status_code}: {r.text[:300]}")
    reply = r.json()["choices"][0]["message"]["content"]
    topic["messages"] += [{"role": "user", "content": message},
                          {"role": "assistant", "content": reply}]
    topic["messages"] = topic["messages"][-40:]
    save_topic(topic)
    ctx_note = f"full topic, pp. {ctx[0]['page']}-{ctx[-1]['page']}"
    if truncated:
        ctx_note += " (truncated to fit context window)"
    return {"reply": reply, "context": ctx_note}


# ------------------------------------------------------------------ UI
PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>BookSplit</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,650&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root{--paper:#FCFCF9;--ink:#152730;--muted:#5E7079;--line:#DDE3E0;
  --teal:#0E7568;--teal-soft:#E4F0ED;--amber:#A8620A}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);
  font:15px/1.55 Inter,system-ui,sans-serif;-webkit-font-smoothing:antialiased}
header{padding:26px 20px 10px;max-width:880px;margin:0 auto}
h1{font-family:Fraunces,serif;font-weight:650;font-size:30px;margin:0;letter-spacing:-.01em}
h1 small{font-family:Inter;font-size:13px;font-weight:500;color:var(--muted);margin-left:10px}
main{max-width:880px;margin:0 auto;padding:0 20px 80px}
.card{border:1px solid var(--line);border-radius:12px;background:#fff;padding:16px;margin-top:14px}
button{font:inherit;cursor:pointer;border:none;border-radius:8px;background:none;color:inherit}
.btn{background:var(--teal);color:#fff;padding:9px 16px;font-weight:600}
.btn:disabled{opacity:.5;cursor:default}
.ghost{color:var(--teal);font-weight:600;padding:8px 10px}
.danger{color:#B3261E;font-weight:600;padding:8px 10px}
input,select{font:inherit;padding:9px 12px;border:1px solid var(--line);
  border-radius:8px;background:#fff;color:var(--ink)}
input[type=text]{width:100%}
input:focus,select:focus,button:focus-visible{outline:2px solid var(--teal);outline-offset:1px}
.row{display:flex;gap:10px;align-items:center;flex-wrap:wrap}
.muted{color:var(--muted);font-size:13px}
.tag{background:var(--teal-soft);color:var(--teal);border-radius:99px;padding:2px 10px;font-size:12px;font-weight:600}
.hidden{display:none}
progress{width:100%}
/* shelf */
.book{display:flex;justify-content:space-between;gap:12px;align-items:center;
  padding:14px 4px;border-bottom:1px dotted var(--line);cursor:pointer}
.book:last-child{border-bottom:none}
.book:hover .bname{color:var(--teal)}
.bname{font-family:Fraunces,serif;font-size:18px;font-weight:500}
/* toc ledger */
.toc-row{display:flex;align-items:baseline;gap:8px;padding:6px 0;width:100%;text-align:left}
.toc-row:hover .t-title{color:var(--teal)}
.t-title{font-weight:500;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:48%;cursor:pointer}
.lv1 .t-title{font-family:Fraunces,serif;font-size:17px;font-weight:650}
.leader{flex:1;border-bottom:1px dotted #B9C4C0;transform:translateY(-4px);min-width:14px}
.t-pages{font-variant-numeric:tabular-nums;color:var(--amber);font-size:13px;white-space:nowrap}
.mini{color:var(--teal);font-weight:600;font-size:13px;padding:2px 6px;white-space:nowrap}
.indent1{margin-left:0}.indent2{margin-left:20px}.indent3{margin-left:40px}
.indent4{margin-left:56px}.indent5{margin-left:70px}
/* search hits */
.hit{padding:10px 0;border-bottom:1px dotted var(--line)}
.hit .pg{color:var(--amber);font-variant-numeric:tabular-nums;font-weight:600;margin-right:8px}
mark{background:var(--teal-soft);color:var(--ink);border-radius:3px;padding:0 2px}
/* reader */
#reader{position:fixed;inset:0;background:#20272A;z-index:50;display:flex;flex-direction:column}
.r-bar{display:flex;gap:8px;align-items:center;padding:10px 14px;background:#152025;color:#EDF2F0;flex-wrap:wrap}
.r-bar .rt{font-family:Fraunces,serif;font-size:15px;flex:1;min-width:120px;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.r-bar button{color:#EDF2F0;font-weight:600;padding:6px 10px}
.r-bar .btn{background:var(--teal)}
.r-bar input{width:74px;padding:5px 8px;background:#0F181C;color:#fff;border-color:#33454C;text-align:center}
.r-body{flex:1;overflow:auto;text-align:center;-webkit-overflow-scrolling:touch}
.r-body img{max-width:100%;height:auto;display:block;margin:0 auto;background:#fff}
.r-nav{position:fixed;bottom:18px;left:0;right:0;display:flex;justify-content:center;gap:14px;pointer-events:none}
.r-nav button{pointer-events:auto;background:#152025E6;color:#fff;border-radius:99px;
  width:52px;height:52px;font-size:22px;font-weight:700}
/* chat */
#msgs{max-height:380px;overflow:auto;margin:10px 0;padding-right:4px}
.m{padding:9px 12px;border-radius:10px;margin:6px 0;white-space:pre-wrap}
.m.user{background:var(--teal-soft);margin-left:14%}
.m.assistant{background:#F2F2ED;margin-right:8%}
.m.sys{color:var(--muted);font-size:13px;background:none;padding:2px 0}
@media (max-width:600px){.t-title{max-width:44%}}
</style>
</head>
<body>
<header>
  <h1>BookSplit <small>read · split · ask</small></h1>
</header>
<main>

<!-- SHELF -->
<section id="shelf">
  <div class="card">
    <div class="row">
      <input type="file" id="file" accept="application/pdf" style="flex:1;min-width:200px">
      <button class="btn" id="upBtn" onclick="upload()">Upload PDF</button>
    </div>
    <progress id="upProg" class="hidden" max="100" value="0"></progress>
    <p class="muted">Files over ~100&nbsp;MB may fail through Cloudflare — drop them into <code>data/books/</code> on the server instead; they appear here automatically.</p>
  </div>
  <div class="card" id="bookList"><p class="muted">Loading library…</p></div>
</section>

<!-- BOOK -->
<section id="bookView" class="hidden">
  <div class="row" style="margin-top:14px">
    <button class="ghost" onclick="showShelf()">&larr; Library</button>
    <span class="tag" id="bMeta"></span>
    <button class="ghost" onclick="readBook()">&#128214; Read book</button>
    <button class="ghost" onclick="reindex()" id="reBtn">&#8635; Re-scan contents</button>
    <button class="danger" onclick="delBook()">Delete</button>
  </div>
  <h2 id="bTitle" style="font-family:Fraunces,serif;font-weight:650;margin:10px 0 0"></h2>

  <div class="card">
    <input type="text" id="tocFilter" placeholder="Filter contents… e.g. nephrotic, Kawasaki, surfactant" oninput="renderToc()">
    <p class="muted" style="margin:8px 0 0">Tap a title to preview it in the reader · &darr; downloads that chapter as a PDF · &#128172; saves it as an AI topic.</p>
    <div id="toc" style="margin-top:6px"></div>
  </div>

  <div class="card">
    <strong>Search inside the book</strong>
    <p class="muted" style="margin:4px 0 10px">Full-text search across every page (first search on a big book takes a moment while it indexes).</p>
    <div class="row">
      <input type="text" id="ftq" style="flex:1;min-width:180px" placeholder="e.g. hypernatraemic dehydration">
      <button class="btn" onclick="ftSearch()">Search</button>
    </div>
    <div id="ftHits"></div>
  </div>

  <div class="card">
    <strong>Custom page range</strong>
    <div class="row" style="margin-top:10px">
      <input type="number" id="rs" min="1" placeholder="From" style="width:100px">
      <input type="number" id="re" min="1" placeholder="To" style="width:100px">
      <button class="ghost" onclick="previewRange()">&#128065; Preview</button>
      <button class="btn" onclick="customExtract()">&darr; Download</button>
      <button class="ghost" onclick="rangeTopic()">&#128172; AI topic</button>
    </div>
  </div>

  <div class="card">
    <strong>AI tutor</strong> <span class="muted">(NVIDIA NIM · answers from the saved topic text, with page citations)</span>
    <div class="row" style="margin-top:10px">
      <select id="topicSel" onchange="pickTopic()" style="flex:1;min-width:180px">
        <option value="">— choose a saved topic —</option>
      </select>
      <button class="danger" onclick="delTopic()">Delete topic</button>
    </div>
    <p class="muted" style="margin:8px 0 0">Topics are stored on the server with their chapter text, so you can come back any time and keep asking — nothing is re-uploaded. The <b>full topic text</b> is sent with every question so the AI sees the whole chapter.</p>
    <div id="chat" class="hidden">
      <div id="msgs"></div>
      <div class="row">
        <input type="text" id="chatIn" style="flex:1;min-width:160px" placeholder="Ask a doubt…"
          onkeydown="if(event.key==='Enter')sendMsg()">
        <button class="btn" id="sendBtn" onclick="sendMsg()">Send</button>
      </div>
    </div>
  </div>
</section>

</main>

<!-- READER -->
<div id="reader" class="hidden">
  <div class="r-bar">
    <button onclick="closeReader()">&#10005;</button>
    <span class="rt" id="rTitle"></span>
    <span>
      <input type="number" id="rPage" onchange="rGo(+this.value)">
      <span id="rMax" class="muted" style="color:#9DB2AC"></span>
    </span>
    <button class="btn hidden" id="rDl" onclick="readerDl()">&darr; Download</button>
  </div>
  <div class="r-body" id="rBody"><img id="rImg" alt="page"></div>
  <div class="r-nav">
    <button onclick="rGo(reader.page-1)" aria-label="Previous page">&lsaquo;</button>
    <button onclick="rGo(reader.page+1)" aria-label="Next page">&rsaquo;</button>
  </div>
</div>

<script>
let cur=null,curToc=[],reader=null,topics=[],topic=null;
const $=id=>document.getElementById(id);
const esc=s=>String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));

async function loadBooks(){
  const r=await fetch('/api/books');const books=await r.json();
  $('bookList').innerHTML=books.length?books.map(b=>
    `<div class="book" onclick="openBook('${b.id}')">
       <div><div class="bname">${esc(b.name)}</div>
       <div class="muted">${b.pages} pages · ${b.chapters} contents entries · ${b.size_mb} MB</div></div>
       <div class="mini">Open &rsaquo;</div></div>`).join('')
    :'<p class="muted">No books yet — upload a PDF above to start.</p>';
}

function upload(){
  const f=$('file').files[0];if(!f)return alert('Choose a PDF first');
  $('upBtn').disabled=true;$('upProg').classList.remove('hidden');
  const fd=new FormData();fd.append('file',f);
  const xhr=new XMLHttpRequest();xhr.open('POST','/api/upload');
  xhr.upload.onprogress=e=>{if(e.lengthComputable)$('upProg').value=100*e.loaded/e.total};
  xhr.onload=()=>{$('upBtn').disabled=false;$('upProg').classList.add('hidden');
    if(xhr.status===200){$('file').value='';loadBooks();}
    else alert('Upload failed: '+xhr.responseText)};
  xhr.onerror=()=>{$('upBtn').disabled=false;$('upProg').classList.add('hidden');alert('Upload failed (network)')};
  xhr.send(fd);
}

function showShelf(){$('bookView').classList.add('hidden');$('shelf').classList.remove('hidden');loadBooks();}

async function openBook(id){
  const r=await fetch(`/api/books/${id}/toc`);const d=await r.json();
  cur=d;curToc=d.toc;
  $('bTitle').textContent=d.name;$('bMeta').textContent=`${d.pages} pages`;
  $('tocFilter').value='';$('ftq').value='';$('ftHits').innerHTML='';
  $('rs').max=$('re').max=d.pages;
  renderToc();
  topic=null;$('chat').classList.add('hidden');
  loadTopics();
  $('shelf').classList.add('hidden');$('bookView').classList.remove('hidden');
  window.scrollTo(0,0);
}

function renderToc(){
  const q=$('tocFilter').value.trim().toLowerCase();
  const rows=curToc.map((e,i)=>({...e,i})).filter(e=>!q||e.title.toLowerCase().includes(q));
  $('toc').innerHTML=rows.length?rows.map(e=>{
    const lv=Math.min(e.level,5);
    return `<div class="toc-row lv${lv} indent${lv}">
      <span class="t-title" onclick="previewToc(${e.i})">${esc(e.title)}</span>
      <span class="leader"></span>
      <span class="t-pages">pp. ${e.start}\u2013${e.end}</span>
      <button class="mini" onclick="dl(${e.i})" title="Download PDF">\u2193</button>
      <button class="mini" onclick="tocTopic(${e.i})" title="Save as AI topic">\uD83D\uDCAC</button>
    </div>`;}).join('')
  :(curToc.length?'<p class="muted">No matching entries.</p>'
    :'<p class="muted">No table of contents could be read from this PDF — try <b>&#8635; Re-scan contents</b> above, or use full-text search / a custom page range.</p>');
}

function previewToc(i){const e=curToc[i];openReader(e.start,e.start,e.end,e.title,true);}
function openAt(p){openReader(p,1,cur.pages,cur.name,false);}
function dl(i){const e=curToc[i];
  location.href=`/api/books/${cur.id}/extract?start=${e.start}&end=${e.end}&title=${encodeURIComponent(e.title)}`;}
function customExtract(){
  const s=+$('rs').value,e=+$('re').value;
  if(!s||!e||s>e)return alert('Enter a valid page range');
  location.href=`/api/books/${cur.id}/extract?start=${s}&end=${e}`;
}
function previewRange(){
  const s=+$('rs').value,e=+$('re').value;
  if(!s||!e||s>e)return alert('Enter a valid page range');
  openReader(s,s,e,`Pages ${s}\u2013${e}`,true);
}
function readBook(){openReader(1,1,cur.pages,cur.name,false);}

async function reindex(){
  $('reBtn').disabled=true;
  const r=await fetch(`/api/books/${cur.id}/reindex`,{method:'POST'});
  $('reBtn').disabled=false;
  if(r.ok){const d=await r.json();await openBook(cur.id);
    alert(`Re-scan complete: ${d.chapters} contents entries found.`);}
  else alert('Re-scan failed: '+await r.text());
}

async function ftSearch(){
  const q=$('ftq').value.trim();if(q.length<3)return alert('Type at least 3 characters');
  $('ftHits').innerHTML='<p class="muted">Searching\u2026 (indexing on first run)</p>';
  const r=await fetch(`/api/books/${cur.id}/search?q=`+encodeURIComponent(q));
  if(!r.ok){$('ftHits').innerHTML='<p class="muted">Search failed.</p>';return}
  const d=await r.json();
  const rx=new RegExp(q.replace(/[.*+?^${}()|[\]\\]/g,'\\$&'),'ig');
  $('ftHits').innerHTML=d.hits.length?d.hits.map(h=>
    `<div class="hit"><span class="pg">p. ${h.page}</span>${esc(h.snippet).replace(rx,m=>'<mark>'+m+'</mark>')}
      <button class="mini" onclick="openAt(${h.page})">\uD83D\uDC41 open</button></div>`).join('')
    :'<p class="muted">No matches found.</p>';
}

async function delBook(){
  if(!confirm(`Delete "${cur.name}" and its index?`))return;
  await fetch(`/api/books/${cur.id}`,{method:'DELETE'});showShelf();
}

/* ---------------- reader ---------------- */
function imgW(){return Math.min(1600,Math.round(Math.min(window.innerWidth,980)*(window.devicePixelRatio||1)));}
function openReader(p,min,max,title,showDl){
  reader={page:p,min:min,max:max,title:title,dl:showDl};
  $('rTitle').textContent=title;
  $('rDl').classList.toggle('hidden',!showDl);
  $('reader').classList.remove('hidden');
  document.body.style.overflow='hidden';
  rGo(p);
}
function rGo(p){
  if(!reader)return;
  p=Math.max(reader.min,Math.min(reader.max,p||reader.min));
  reader.page=p;reader.retry=0;
  $('rPage').value=p;$('rMax').textContent=' / '+reader.max;
  const w=imgW();
  $('rImg').src=`/api/books/${cur.id}/page/${p}?w=${w}`;
  $('rBody').scrollTop=0;
  if(p<reader.max){const pre=new Image();pre.src=`/api/books/${cur.id}/page/${p+1}?w=${w}`;}
}
$('rImg').addEventListener('error',()=>{
  if(!reader)return;
  if(reader.retry<2){reader.retry++;
    setTimeout(()=>{$('rImg').src=
      `/api/books/${cur.id}/page/${reader.page}?w=${reader.retry===1?800:500}&r=${Date.now()}`;},600);}
  else $('rImg').alt='Could not load page '+reader.page+' — check server logs';
});
function closeReader(){reader=null;$('reader').classList.add('hidden');document.body.style.overflow='';}
function readerDl(){if(reader)location.href=
  `/api/books/${cur.id}/extract?start=${reader.min}&end=${reader.max}&title=${encodeURIComponent(reader.title)}`;}
document.addEventListener('keydown',e=>{
  if(!reader)return;
  if(e.key==='ArrowRight')rGo(reader.page+1);
  else if(e.key==='ArrowLeft')rGo(reader.page-1);
  else if(e.key==='Escape')closeReader();
});
let tX=null;
$('rBody').addEventListener('touchstart',e=>{tX=e.changedTouches[0].clientX},{passive:true});
$('rBody').addEventListener('touchend',e=>{
  if(tX===null||!reader)return;
  const dx=e.changedTouches[0].clientX-tX;tX=null;
  if(Math.abs(dx)>60)rGo(reader.page+(dx<0?1:-1));
},{passive:true});

/* ---------------- AI tutor ---------------- */
async function loadTopics(selId){
  const r=await fetch('/api/topics?book='+cur.id);topics=await r.json();
  $('topicSel').innerHTML='<option value="">\u2014 choose a saved topic \u2014</option>'+
    topics.map(t=>`<option value="${t.id}">${esc(t.title)} (pp. ${t.start}\u2013${t.end})</option>`).join('');
  if(selId){$('topicSel').value=selId;pickTopic();}
}
function tocTopic(i){const e=curToc[i];makeTopic(e.start,e.end,e.title);}
function rangeTopic(){
  const s=+$('rs').value,e=+$('re').value;
  if(!s||!e||s>e)return alert('Enter a valid page range');
  makeTopic(s,e,`Pages ${s}-${e}`);
}
async function makeTopic(start,end,title){
  title=prompt('Name this topic:',title);if(title===null)return;
  const r=await fetch(`/api/books/${cur.id}/topics`,{method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({start,end,title})});
  if(!r.ok)return alert('Could not create topic: '+await r.text());
  const t=await r.json();await loadTopics(t.id);
  $('chat').scrollIntoView({behavior:'smooth'});
}
async function pickTopic(){
  const id=$('topicSel').value;
  topic=id?topics.find(t=>t.id===id):null;
  $('chat').classList.toggle('hidden',!topic);
  if(!topic)return;
  const r=await fetch(`/api/topics/${topic.id}/history`);const d=await r.json();
  renderMsgs(d.messages);
}
function renderMsgs(ms){
  $('msgs').innerHTML=ms.length?ms.map(m=>`<div class="m ${m.role}">${esc(m.content)}</div>`).join('')
    :'<div class="m sys">Ask your first question about this topic.</div>';
  $('msgs').scrollTop=1e9;
}
async function sendMsg(){
  const v=$('chatIn').value.trim();if(!v||!topic)return;
  $('chatIn').value='';$('sendBtn').disabled=true;
  $('msgs').insertAdjacentHTML('beforeend',`<div class="m user">${esc(v)}</div><div class="m sys" id="think">Thinking\u2026</div>`);
  $('msgs').scrollTop=1e9;
  const r=await fetch(`/api/topics/${topic.id}/chat`,{method:'POST',
    headers:{'Content-Type':'application/json'},body:JSON.stringify({message:v})});
  $('sendBtn').disabled=false;
  const think=$('think');
  if(!r.ok){think.textContent='Error: '+await r.text();return}
  const d=await r.json();
  think.outerHTML=`<div class="m assistant">${esc(d.reply)}</div>
    <div class="m sys">context: ${esc(d.context)}</div>`;
  $('msgs').scrollTop=1e9;
}
async function delTopic(){
  if(!topic)return alert('Choose a topic first');
  if(!confirm(`Delete topic "${topic.title}" and its chat history?`))return;
  await fetch(`/api/topics/${topic.id}`,{method:'DELETE'});
  topic=null;$('chat').classList.add('hidden');loadTopics();
}

loadBooks();
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def home():
    return PAGE
