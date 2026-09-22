#!/usr/bin/env bash
#
# Deploy the fixed image to SkyNAS without losing the container's memories.
#
# The container writes into its own writable layer, not the named volume, so
# recreating it destroys everything it holds. Worse, its on-disk master.key no
# longer decrypts its own database (CLAUDE.md, master-key bug) - so the store
# cannot be copied across either. It has to be re-encrypted under a new key while
# the running process is still alive to decrypt it.
#
# This script therefore rescues the CURRENT contents every time it runs, rather
# than trusting a directory prepared earlier. The first version of this file
# staged a fixed path holding a single memory; by the time it was ready to run,
# the container held 120, and running it would have destroyed 119 of them.
#
# Safe to re-run. It refuses to continue if the staged store holds fewer records
# than the live API reports.

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VOLUME="${MEMCORE_VOLUME:-memcore-memory-100_mnem_data}"
ENDPOINT="${MEMCORE_ENDPOINT:-http://192.168.1.183:8000}"
BACKUP_ROOT="${MEMCORE_BACKUP_ROOT:-/mnt/nas7/SkyNas/memcore-memory/backups}"
PYTHON="${MEMCORE_PYTHON:-$PROJECT_DIR/.venv/bin/python}"
SOURCE="${1:-}"

say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
die() { printf '\n\033[31mABORT: %s\033[0m\n' "$*" >&2; exit 1; }

rows_in() { "$PYTHON" -c "
import sqlite3,sys
print(sqlite3.connect(sys.argv[1]).execute('select count(*) from memories').fetchone()[0])
" "$1/memory.db"; }

live_total() { curl -sf --max-time 10 "$ENDPOINT/health" | "$PYTHON" -c "
import json,sys; print(sum(json.load(sys.stdin).get('tier_counts',{}).values()))
"; }

say "Preflight"
command -v docker >/dev/null || die "docker not on PATH"
[ -x "$PYTHON" ] || die "no interpreter at $PYTHON"
docker volume inspect "$VOLUME" >/dev/null 2>&1 || die "volume $VOLUME does not exist"
LIVE=$(live_total) || die "$ENDPOINT is not answering - it must be up to decrypt its own store"
echo "live memories: $LIVE"

if [ -n "$SOURCE" ]; then
    say "Using the store given on the command line"
    [ -f "$SOURCE/memory.db" ] && [ -f "$SOURCE/master.key" ] || die "$SOURCE is not a data dir"
else
    say "Step 1/4 - rescue the container's current contents"
    SOURCE="$BACKUP_ROOT/$(date +%Y%m%d-%H%M%S)-recovered-store"
    "$PYTHON" "$PROJECT_DIR/scripts/rescue-container-store.py" "$SOURCE" \
        --endpoint "$ENDPOINT" || die "rescue failed - nothing has been changed"
fi

STAGED=$(rows_in "$SOURCE")
echo "staged records: $STAGED   live: $LIVE"
[ "$STAGED" -ge "$LIVE" ] || die \
    "the store to stage holds $STAGED records but the container serves $LIVE. \
Deploying it would destroy $((LIVE - STAGED)) memories."

say "Step 2/4 - stage into $VOLUME"
docker run --rm -v "$VOLUME":/dst -v "$SOURCE":/src:ro alpine \
    sh -c 'cp -a /src/. /dst/ && cd /dst && ls -la'

say "Step 3/4 - verify the volume against the source"
for f in master.key memory.db memory.kg.db; do
    [ -f "$SOURCE/$f" ] || continue
    want=$(sha256sum "$SOURCE/$f" | cut -d' ' -f1)
    got=$(docker run --rm -v "$VOLUME":/v alpine sha256sum "/v/$f" | cut -d' ' -f1)
    [ "$want" = "$got" ] || die "$f differs between source and volume"
    echo "  $f OK"
done

say "Step 4/4 - rebuild and recreate"
cd "$PROJECT_DIR"
docker compose up -d --build

say "Wait for the API"
for _ in $(seq 1 90); do
    curl -sf --max-time 2 "$ENDPOINT/health" >/dev/null 2>&1 && break
    sleep 1
done

say "Verify the deployment"
AFTER=$(live_total) || die "the API did not come back up - the staged store is intact at $SOURCE"
echo "memories after deploy: $AFTER   (expected >= $STAGED)"
[ "$AFTER" -ge "$STAGED" ] || die "record count dropped from $STAGED to $AFTER"
echo "mcp bridge : HTTP $(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "$ENDPOINT/mcp/tools")  (expect 200)"
echo "effective config:"
docker exec mnemosyne python -c "
from memcore_memory.config import settings
print('  data_dir :', settings.data_dir)
print('  db_path  :', settings.db_path)
print('  provider :', settings.embedding_provider)
print('  dim      :', settings.embedding_dim)
"

say "Confirm the master key survived startup"
got=$(docker run --rm -v "$VOLUME":/v alpine sha256sum /v/master.key | cut -d' ' -f1)
want=$(sha256sum "$SOURCE/master.key" | cut -d' ' -f1)
[ "$want" = "$got" ] && echo "master.key unchanged - the key-rotation bug is fixed" \
                     || die "master.key was rewritten at startup - do not trust this deployment"

say "Done. Data now lives in $VOLUME, so backups and recreation are safe."
echo "Rescue copy kept at: $SOURCE"
