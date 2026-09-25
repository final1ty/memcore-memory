#!/usr/bin/env python3
"""Import past Claude Code sessions (~/.claude/projects/*/*.jsonl) into the memcore container over REST.

One episodic memory per session: its last compact summary (the one that covers the whole session so far),
or, for a session never compacted, its title and first prompts. Obvious secrets are redacted. A session
already imported (its id appears in a stored memory) is skipped, so re-running only adds new sessions.
Dry run by default; pass "go" to write. Key: ~/.config/memcore/api-key."""
import json, re, sys, glob, os, time, urllib.request
K = open(os.path.expanduser("~/.config/memcore/api-key")).read().strip()
B = "http://localhost:8000"
MIN_CHARS = 400  # uncompacted sessions with less than this much prompt text are not worth a memory


def req(path, body=None):
    r = urllib.request.Request(B + path, json.dumps(body).encode() if body else None,
                               {"Authorization": "Bearer " + K, "Content-Type": "application/json"})
    for _ in range(10):
        try:
            return json.load(urllib.request.urlopen(r))
        except urllib.error.HTTPError as e:
            if e.code != 429:
                raise
            time.sleep(15)
    raise RuntimeError("still rate limited")


SECRETS = [
    (re.compile(r"\b(sk-[A-Za-z0-9_-]{16,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|xox[abp]-[A-Za-z0-9-]{10,}|AKIA[0-9A-Z]{16})\b"), "[REDACTED]"),
    (re.compile(r"(?i)\b(password|passwd|jelszó|api[_-]?key|secret|token|MNEM_API_KEY|MNEM_MASTER_PASSWORD)(\s*[=:]\s*)[\"']?[^\s\"'`,;]{6,}"), r"\1\2[REDACTED]"),
    (re.compile(r"(?<![A-Za-z0-9+/])[A-Za-z0-9+/]{42,43}=(?![A-Za-z0-9+/=])"), "[REDACTED KEY]"),  # WireGuard-style keys
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S), "[REDACTED KEY]"),
]


def redact(s):
    for rx, rep in SECRETS:
        s = rx.sub(rep, s)
    return s


def text_of(msg):
    c = (msg or {}).get("content")
    if isinstance(c, str):
        return c
    return "\n".join(p.get("text", "") for p in c or [] if isinstance(p, dict) and p.get("type") == "text")


def project_name(f):
    p = f.split("/projects/")[1].split("/")[0]
    p = p.replace("-mnt-nas7-SkyNas-", "").replace("-home-skynas-", "~/")
    return {"-": "global", "-home-skynas": "~", "-mnt-nas7-SkyNas": "SkyNas"}.get(p, p)


def session(f):
    title = first_ts = last_ts = summary = None
    prompts = []
    for line in open(f, errors="replace"):
        try:
            o = json.loads(line)
        except ValueError:
            continue
        ts = o.get("timestamp")
        if ts:
            first_ts = first_ts or ts
            last_ts = ts
        t = o.get("type")
        if t == "custom-title" and o.get("customTitle"):
            title = o["customTitle"]
        elif t == "ai-title" and o.get("aiTitle") and not title:
            title = o["aiTitle"]
        elif t == "user" and o.get("isCompactSummary"):
            summary = text_of(o.get("message"))
        elif t == "user" and not o.get("isMeta") and not o.get("toolUseResult") and len(prompts) < 6:
            s = text_of(o.get("message")).strip()
            if s and not s.startswith("<") and "tool_result" not in s:
                prompts.append(s[:1500])
    return title, first_ts, last_ts, summary, prompts


norm = lambda s: re.sub(r"\W+", " ", s.lower()).strip()
existing = " ".join(norm(req(f"/memory/{m['id']}")["content"][:300]) for m in req("/memories?limit=500"))
plan, seen_bodies = [], set()
for f in sorted(glob.glob(os.path.expanduser("~/.claude/projects/*/*.jsonl"))):
    sid = os.path.basename(f)[:-6]
    if time.time() - os.path.getmtime(f) < 600:  # still running: its summary is not final yet
        plan.append(("LIVE", sid, project_name(f), None, 0, None, None)); continue
    title, t0, t1, summary, prompts = session(f)
    proj = project_name(f)
    if summary:
        summary = summary.split("If you need specific details from before compaction")[0]
        summary = re.sub(r"^This session is being continued.*?\n\n", "", summary, flags=re.S).strip()
        body, kind, imp = summary, "compact-summary", 0.7
    elif sum(map(len, prompts)) >= MIN_CHARS:
        body, kind, imp = "Felhasználói kérések:\n" + "\n---\n".join(prompts), "prompts", 0.5
    else:
        plan.append(("SKIP", sid, proj, title, 0, None, None)); continue
    key = norm(body)[:2000]
    if key in seen_bodies:  # forks and repeated automated runs carry the same text
        plan.append(("SAME", sid, proj, title, 0, None, None)); continue
    seen_bodies.add(key)
    head = f"CLAUDE-CODE-MUNKAMENET [{proj}] {(t0 or '')[:10]}..{(t1 or '')[:10]} {title or '(cím nélkül)'} (session {sid})"
    content = redact(head + "\n\n" + body)
    state = "DUP" if norm(sid) in existing else "NEW"
    plan.append((state, sid, proj, title, len(content), content,
                 {"importance": imp, "source": "claude-code-session", "kind": kind, "project": proj,
                  "session_id": sid, "started": t0, "ended": t1}))

if sys.argv[1:] == ["go"]:
    n = 0
    for state, sid, proj, title, _, content, meta in plan:
        if state != "NEW":
            continue
        req("/memory", {"content": content, "tier": "episodic", "importance": meta["importance"],
                        "entities": [proj] + ([title] if title else []), "metadata": meta})
        n += 1; time.sleep(3.2)
    print("imported", n)
else:
    for state, sid, proj, title, size, _, meta in plan:
        print(state.ljust(4), (meta or {}).get("kind", "-").ljust(15), str(size).rjust(6), proj[:28].ljust(28), (title or "")[:50])
    import collections
    print(dict(collections.Counter(p[0] for p in plan)), "of", len(plan))
    print("redactions:", sum(c.count("[REDACTED") for *_, c, _m in [(p[5],p[6]) for p in plan] if c))
