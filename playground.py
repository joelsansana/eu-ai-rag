#!/usr/bin/env python
"""Dev playground GUI for the eu-ai-rag API.

Serves a single-page console and proxies /ask and /health to the running
API (default http://localhost:8000). Same-origin proxying means the API
needs no CORS middleware and nothing in the repo changes. Stdlib only.

Run:    python playground.py        # then open http://localhost:8877
Stop:   Ctrl-C

If the API has API_TOKEN set, the proxy reads the same variable (or
PLAYGROUND_TOKEN) and forwards it as `Authorization: Bearer ...`, so the
console keeps working with auth enabled. Override the endpoints with
PLAYGROUND_API_HOST / PLAYGROUND_API_PORT / PLAYGROUND_PORT.
"""

from __future__ import annotations

import http.client
import http.server
import json
import os
import sys

API_HOST = os.environ.get("PLAYGROUND_API_HOST", "localhost")
API_PORT = int(os.environ.get("PLAYGROUND_API_PORT", "8000"))
PORT = int(os.environ.get("PLAYGROUND_PORT", "8877"))
TOKEN = os.environ.get("API_TOKEN") or os.environ.get("PLAYGROUND_TOKEN")

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>eu-ai-rag · playground</title>
<style>
  :root {
    --paper: #faf9f6;
    --panel: #ffffff;
    --ink: #1c1b18;
    --muted: #726c61;
    --faint: #a39c8f;
    --border: #e5e1d8;
    --accent: oklch(0.52 0.14 252);
    --accent-ink: oklch(0.42 0.14 252);
    --good: oklch(0.55 0.12 150);
    --serif: "Iowan Old Style", Georgia, "Times New Roman", serif;
    --sans: -apple-system, "Segoe UI", "Helvetica Neue", sans-serif;
    --mono: ui-monospace, "SF Mono", "Cascadia Code", Menlo, monospace;
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; padding: 0; }
  body {
    background: var(--paper);
    color: var(--ink);
    font-family: var(--sans);
    font-size: 15px;
    line-height: 1.55;
    -webkit-font-smoothing: antialiased;
  }
  .wrap { max-width: 860px; margin: 0 auto; padding: 28px 24px 80px; }

  /* ---- top bar ---- */
  header {
    display: flex; align-items: baseline; gap: 14px;
    padding-bottom: 18px; border-bottom: 1px solid var(--border);
    margin-bottom: 26px;
  }
  .wordmark { font-family: var(--mono); font-size: 14px; font-weight: 600; letter-spacing: 0.02em; }
  .wordmark em { font-style: normal; color: var(--accent-ink); }
  .status { margin-left: auto; font-family: var(--mono); font-size: 12px; color: var(--muted); display: flex; align-items: center; gap: 6px; }
  .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--faint); }
  .dot.up { background: var(--good); }

  /* ---- query console ---- */
  .console { background: var(--panel); border: 1px solid var(--border); border-radius: 10px; padding: 14px 14px 12px; }
  textarea {
    width: 100%; min-height: 64px; resize: vertical;
    border: none; outline: none; background: transparent;
    font-family: var(--sans); font-size: 15px; color: var(--ink);
  }
  textarea::placeholder { color: var(--faint); }
  .controls { display: flex; align-items: center; gap: 10px; margin-top: 8px; flex-wrap: wrap; }
  .field { display: flex; align-items: center; gap: 6px; font-family: var(--mono); font-size: 12px; color: var(--muted); }
  select, input[type=number] {
    font-family: var(--mono); font-size: 12px; color: var(--ink);
    background: var(--paper); border: 1px solid var(--border); border-radius: 6px;
    padding: 4px 6px;
  }
  input[type=number] { width: 58px; }
  button.ask {
    margin-left: auto; font-family: var(--sans); font-size: 14px; font-weight: 600;
    color: white; background: var(--accent); border: none; border-radius: 7px;
    padding: 8px 22px; cursor: pointer;
  }
  button.ask:hover { background: var(--accent-ink); }
  button.ask:disabled { opacity: 0.45; cursor: default; }
  button.ask.cancel { background: transparent; color: var(--accent-ink); border: 1px solid var(--accent); }
  .hint { font-family: var(--mono); font-size: 11px; color: var(--faint); margin-top: 8px; }

  /* ---- presets ---- */
  .presets { display: flex; gap: 8px; margin: 14px 0 30px; flex-wrap: wrap; }
  .preset {
    font-family: var(--mono); font-size: 12px; color: var(--muted);
    background: transparent; border: 1px solid var(--border); border-radius: 999px;
    padding: 5px 12px; cursor: pointer;
  }
  .preset:hover { border-color: var(--accent); color: var(--accent-ink); }
  .preset .tag { color: var(--faint); margin-left: 6px; }

  /* ---- answer ---- */
  .section-label {
    font-family: var(--mono); font-size: 11px; letter-spacing: 0.08em;
    text-transform: uppercase; color: var(--faint); margin: 26px 0 10px;
  }
  .answer {
    font-family: var(--serif); font-size: 17.5px; line-height: 1.65;
    white-space: pre-wrap;
  }
  .answer.pending { color: var(--faint); font-family: var(--mono); font-size: 13px; }
  .answer.error { color: #8c3a2e; font-family: var(--mono); font-size: 13px; white-space: pre-wrap; }
  .meta-row { display: flex; gap: 16px; margin-top: 10px; font-family: var(--mono); font-size: 12px; color: var(--muted); }
  .meta-row b { color: var(--ink); font-weight: 600; }

  /* ---- sources ---- */
  .source { border-top: 1px solid var(--border); padding: 10px 0 12px; }
  .source:last-child { border-bottom: 1px solid var(--border); }
  .source-head { display: flex; align-items: baseline; gap: 10px; cursor: pointer; }
  .source-num { font-family: var(--mono); font-size: 11px; color: var(--faint); }
  .source-header-text { font-family: var(--mono); font-size: 12.5px; color: var(--ink); }
  .score { margin-left: auto; font-family: var(--mono); font-size: 11px; color: var(--muted); white-space: nowrap; }
  .source-body {
    font-family: var(--serif); font-size: 14.5px; color: var(--muted);
    margin: 8px 0 0 22px; white-space: pre-wrap;
    display: none;
  }
  .source.open .source-body { display: block; }

  .empty { color: var(--faint); font-family: var(--mono); font-size: 12.5px; }

  @media (prefers-reduced-motion: no-preference) {
    .answer, .source { animation: rise 0.25s ease-out; }
    @keyframes rise { from { opacity: 0; transform: translateY(4px); } to { opacity: 1; transform: none; } }
  }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div class="wordmark">eu-ai-rag <em>/ playground</em></div>
    <div class="status"><span class="dot" id="dot"></span><span id="status">checking…</span></div>
  </header>

  <div class="console">
    <textarea id="q" placeholder="Ask the corpus — AI Act or NIS2…" autofocus></textarea>
    <div class="controls">
      <label class="field">k <input type="number" id="k" value="5" min="1" max="20"></label>
      <label class="field">regulation
        <select id="reg">
          <option value="">any</option>
          <option value="ai_act">AI Act</option>
          <option value="nis2">NIS2</option>
        </select>
      </label>
      <button class="ask" id="ask">Ask</button>
    </div>
    <div class="hint">Ctrl/⌘ + Enter to ask · click a source to expand its text</div>
  </div>

  <div class="presets" id="presets"></div>

  <div id="answer-block" hidden>
    <div class="section-label">Answer</div>
    <div class="answer" id="answer"></div>
    <div class="meta-row" id="meta"></div>
  </div>

  <div id="sources-block" hidden>
    <div class="section-label" id="sources-label">Sources</div>
    <div id="sources"></div>
  </div>
</div>

<script>
const $ = id => document.getElementById(id);
const API_ORIGIN = "__API_ORIGIN__";

let controller = null;   // in-flight request, for Cancel
let busy = false;        // is the Ask button currently a Cancel button?
let ticker = null;       // elapsed-ms interval while waiting
let requestSeq = 0;      // ignores responses from superseded requests

const PRESETS = [
  {label: "deployer → workers", tag: "answerable",
   q: "What must deployers of high-risk AI systems tell workers before using the system at the workplace?"},
  {label: "labour law recital", tag: "hard · Hit@5 miss",
   q: "To what extent does the AI Act affect national labour law, workers' fundamental rights and platform work?"},
  {label: "NIS2 fines in NL", tag: "trap · should abstain",
   q: "What is the procedure for NIS2 fines in the Netherlands?"},
];

PRESETS.forEach(p => {
  const b = document.createElement("button");
  b.className = "preset";
  b.innerHTML = p.label + '<span class="tag">' + p.tag + "</span>";
  b.onclick = () => { $("q").value = p.q; ask(); };
  $("presets").appendChild(b);
});

checkHealth();

async function checkHealth() {
  try {
    const resp = await fetch("/health");
    if (!resp.ok) throw new Error("HTTP " + resp.status);
    $("dot").classList.add("up");
    $("status").textContent = "api up · " + API_ORIGIN;
  } catch {
    $("status").textContent = "api unreachable · " + API_ORIGIN;
  }
}

$("q").addEventListener("keydown", e => {
  if ((e.ctrlKey || e.metaKey) && e.key === "Enter") ask();
});

// One handler for both states. A second handler here would fire on every
// click, so "Cancel" would start a fresh request and abort it again.
$("ask").addEventListener("click", () => {
  if (busy) { if (controller) controller.abort(); }
  else ask();
});

function setBusy(next) {
  busy = next;
  const b = $("ask");
  b.textContent = next ? "Cancel" : "Ask";
  b.classList.toggle("cancel", next);
}

function startTicker() {
  stopTicker();
  const t0 = performance.now();
  ticker = setInterval(() => {
    $("lat").textContent = Math.round(performance.now() - t0) + " ms…";
  }, 100);
}

function stopTicker() {
  if (ticker) { clearInterval(ticker); ticker = null; }
}

// `detail` is how FastAPI reports errors; fall back to the raw payload.
function detailOf(text) {
  try {
    const parsed = JSON.parse(text);
    return parsed && parsed.detail ? parsed.detail : text;
  } catch {
    return text;
  }
}

function ask() {
  const question = $("q").value.trim();
  if (!question) return;

  if (controller) controller.abort();   // never let two answers race
  const seq = ++requestSeq;
  const mine = () => seq === requestSeq;

  const body = {question, k: Math.max(1, Math.min(20, +$("k").value || 5))};
  const reg = $("reg").value;
  if (reg) body.regulation = reg;

  controller = new AbortController();
  setBusy(true);
  $("answer-block").hidden = false;
  $("sources-block").hidden = true;
  $("sources").replaceChildren();
  const a = $("answer");
  a.className = "answer pending";
  a.textContent = "retrieving · generating…";
  $("meta").innerHTML = "<span>latency <b id='lat'>…</b></span>";
  startTicker();

  fetch("/ask", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(body),
    signal: controller.signal,
  }).then(async resp => {
    if (!resp.ok) {
      const detail = resp.status === 401
        ? "401 — the API requires a token. Restart the playground with API_TOKEN set."
        : resp.status + " — " + detailOf(await resp.text());
      throw new Error(detail);
    }
    const reader = resp.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    while (true) {
      const {done, value} = await reader.read();
      if (done) break;
      // sse_starlette frames events with CRLF (DEFAULT_SEPARATOR in
      // sse_starlette/event.py), so normalise before splitting on "\n\n".
      buf += dec.decode(value, {stream: true}).replace(/\r\n?/g, "\n");
      let idx;
      while ((idx = buf.indexOf("\n\n")) !== -1) {
        handleEvent(buf.slice(0, idx), mine);
        buf = buf.slice(idx + 2);
      }
    }
    if (buf.trim()) handleEvent(buf, mine);   // trailing partial event
  }).catch(err => {
    if (err.name === "AbortError") {
      a.className = "answer pending";
      a.textContent = "cancelled.";
    } else if (mine()) {
      a.className = "answer error";
      a.textContent = String(err.message || err);
    }
  }).finally(() => {
    if (mine()) { stopTicker(); setBusy(false); }
  });

  function handleEvent(block, isMine) {
    if (!isMine()) return;
    let ev = "message";
    const dataLines = [];
    for (const line of block.split("\n")) {
      if (line.startsWith(":")) continue;            // sse keep-alive
      if (line.startsWith("event:")) ev = line.slice(6).trim();
      else if (line.startsWith("data:")) {
        let d = line.slice(5);
        if (d.startsWith(" ")) d = d.slice(1);      // spec: drop one space
        dataLines.push(d);
      }
    }
    // Repeated data: lines are one payload, joined with newlines.
    const data = dataLines.join("\n");
    if (ev === "answer") {
      a.className = "answer";
      a.textContent = data;
    } else if (ev === "sources") {
      renderSources(JSON.parse(data));
    } else if (ev === "done") {
      stopTicker();
      $("lat").textContent = JSON.parse(data).latency_ms + " ms";
    } else if (ev === "error") {
      a.className = "answer error";
      a.textContent = "api error: " + detailOf(data);
    }
  }
}

function renderSources(sources) {
  $("sources-block").hidden = false;
  $("sources-label").textContent = "Sources · " + sources.length;
  const box = $("sources");
  box.replaceChildren();
  sources.forEach((s, i) => {
    const div = document.createElement("div");
    div.className = "source";

    const head = document.createElement("div");
    head.className = "source-head";

    const num = document.createElement("span");
    num.className = "source-num";
    num.textContent = "[" + (i + 1) + "]";

    const title = document.createElement("span");
    title.className = "source-header-text";
    title.textContent = s.header || "(no header)";

    const score = document.createElement("span");
    score.className = "score";
    score.textContent = typeof s.score === "number" ? s.score.toFixed(3) : "";

    head.append(num, title, score);

    const bodyEl = document.createElement("div");
    bodyEl.className = "source-body";
    bodyEl.textContent = s.text || "";

    head.onclick = () => div.classList.toggle("open");
    div.append(head, bodyEl);
    box.appendChild(div);
  });
}
</script>
</body>
</html>
"""


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        sys.stderr.write("playground: %s\n" % (fmt % args))

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            page = HTML.replace("__API_ORIGIN__", f"{API_HOST}:{API_PORT}")
            self._send(200, page.encode(), "text/html; charset=utf-8")
        elif path == "/health":
            self._proxy("GET", "/health")
        else:
            self._send_json(404, {"detail": "not found"})

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        if path == "/ask":
            self._proxy("POST", "/ask", stream=True)
        else:
            self._send_json(404, {"detail": "not found"})

    def _send(self, status, body, content_type):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, status, payload):
        self._send(status, json.dumps(payload).encode(), "application/json")

    def _proxy(self, method, path, stream=False):
        length = int(self.headers.get("Content-Length") or 0)
        data = self.rfile.read(length) if length else None
        headers = {"Content-Type": "application/json"} if data else {}
        if TOKEN:
            headers["Authorization"] = f"Bearer {TOKEN}"

        conn = http.client.HTTPConnection(API_HOST, API_PORT, timeout=300)
        started = False
        try:
            conn.request(method, path, body=data, headers=headers)
            resp = conn.getresponse()
            if stream and resp.status < 400:
                self.send_response(resp.status)
                self.send_header(
                    "Content-Type",
                    resp.getheader("Content-Type", "text/plain"),
                )
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()
                started = True
                while True:
                    chunk = resp.read(256)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    self.wfile.flush()
            else:
                # Errors are buffered whole so the browser sees FastAPI's
                # {"detail": ...} body and the real status code.
                body = resp.read()
                self._send(
                    resp.status,
                    body,
                    resp.getheader("Content-Type", "application/json"),
                )
                started = True
        except OSError as e:
            # Headers already flushed? Then the stream is truncated and the
            # only thing left to do is say so on stderr.
            if started:
                sys.stderr.write(f"playground: upstream stream aborted: {e}\n")
            else:
                self._send_json(
                    502,
                    {
                        "detail": (
                            f"cannot reach the API at {API_HOST}:{API_PORT}: {e}"
                        )
                    },
                )
        finally:
            conn.close()


if __name__ == "__main__":
    try:
        server = http.server.ThreadingHTTPServer(("localhost", PORT), Handler)
    except OSError as e:
        sys.exit(
            f"playground: cannot bind localhost:{PORT} ({e}).\n"
            f"playground: another playground is probably still running — "
            f"stop it, or pick another port with PLAYGROUND_PORT."
        )
    auth = "token from env" if TOKEN else "no token (API_TOKEN unset)"
    print(f"playground: http://localhost:{PORT}", flush=True)
    print(f"playground: proxying to {API_HOST}:{API_PORT} · {auth}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nplayground: down")
