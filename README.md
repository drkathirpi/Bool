# BookSplit

Self-hosted textbook reader, chapter splitter and AI tutor.

- Upload a PDF (or scp/rclone big files into `data/books/` — auto-indexed)
- Table of contents parsed from PDF bookmarks (PyMuPDF, with pypdf fallback)
- Read the whole book in the browser; preview any chapter before downloading
- Download any chapter/page range as its own PDF
- AI tutor via NVIDIA NIM: save a chapter as a "topic" once, chat about it forever
  (topic text + chat history persist on the server; only the most relevant
  pages are sent per question to save tokens)

## Deploy on Coolify (GitHub)

1. Push this repo to GitHub.
2. Coolify -> + New -> Public/Private Repository -> pick this repo.
   Build Pack: **Docker Compose** (it will use docker-compose.yml + Dockerfile).
3. Environment variables:
   - `NVIDIA_API_KEY` = your NIM key (required for AI tutor)
   - `NIM_MODEL` = optional, default `meta/llama-3.3-70b-instruct`
   - `APP_PASSWORD` = optional basic-auth password (uncomment in compose)
4. On the host: `mkdir -p /opt/booksplit/data/books`
5. Set the domain (e.g. books.drkathiravan.uk) pointing at port 8000, deploy.

## Notes

- Cloudflare caps uploads at ~100 MB; drop bigger PDFs into
  `/opt/booksplit/data/books/` via scp/rclone — they appear automatically.
- After upgrading versions, books re-index themselves automatically; the
  "Re-scan contents" button forces it manually.
- AI topics live in `/opt/booksplit/data/ai/` and survive book deletion.
