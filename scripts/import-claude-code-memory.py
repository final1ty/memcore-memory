#!/usr/bin/env python3
"""Import ~/.claude/projects/*/memory/*.md into the memcore container over REST (dedups on content; honours the 20 writes/min limit). Dry run by default; pass "go" to write. Key: ~/.config/memcore/api-key."""
import json, re, sys, glob, os, time, urllib.request
K=open(os.path.expanduser("~/.config/memcore/api-key")).read().strip()
B="http://localhost:8000"
def req(path, body=None):
    r=urllib.request.Request(B+path, json.dumps(body).encode() if body else None,
        {"Authorization":"Bearer "+K,"Content-Type":"application/json"})
    for _ in range(10):
        try: return json.load(urllib.request.urlopen(r))
        except urllib.error.HTTPError as e:
            if e.code!=429: raise
            time.sleep(15)
    raise RuntimeError("still rate limited")
norm=lambda s: re.sub(r"\W+"," ",s.lower()).strip()
existing=[norm(req(f"/memory/{m['id']}")["content"]) for m in req("/memories?limit=500")]
IMP={"feedback":0.9,"user":0.9,"project":0.75,"reference":0.7}
TIER={"feedback":"semantic","user":"semantic","project":"episodic","reference":"episodic"}
plan=[]
for f in sorted(glob.glob(os.path.expanduser("~/.claude/projects/*/memory/*.md"))):
    if f.endswith("MEMORY.md"): continue
    t=open(f).read()
    m=re.match(r"---\n(.*?)\n---\n(.*)", t, re.S)
    if not m: continue
    fm, body = m.group(1), m.group(2).strip()
    g=lambda k: (re.search(rf"^\s*{k}:\s*\"?(.*?)\"?\s*$", fm, re.M) or [None,""])[1]
    name, desc, typ = g("name"), g("description"), g("type") or "project"
    proj=f.split("/projects/")[1].split("/memory")[0].replace("-mnt-nas7-SkyNas-","").replace("-home-skynas-","~/") or "-"
    proj = {"-":"global","-mnt-nas7-SkyNas":"SkyNas"}.get(proj, proj)
    content=f"CLAUDE-CODE-MEMÓRIA [{proj}] {name} — {desc}\n\n{body}"
    nb=norm(body)
    dup=any(nb[:200] in e or norm(desc)[:120] in e for e in existing) if nb else True
    plan.append((dup,name,proj,typ,content))
if sys.argv[1:]==["go"]:
    n=0
    for dup,name,proj,typ,content in plan:
        if dup: continue
        req("/memory",{"content":content,"tier":TIER.get(typ,"episodic"),"importance":IMP.get(typ,0.7),
            "entities":[proj,name],"metadata":{"importance":IMP.get(typ,0.7),"source":"claude-code-memory","memory_type":typ,"project":proj}}); n+=1; time.sleep(3.2)
    print("imported",n)
else:
    for dup,name,proj,typ,_ in plan: print("DUP " if dup else "NEW ",typ.ljust(9),proj[:40].ljust(40),name)
    print(sum(not p[0] for p in plan),"new of",len(plan))
