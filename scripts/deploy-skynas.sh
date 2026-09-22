#!/usr/bin/env bash
#
# Stage the recovered store into the named volume, then rebuild and recreate the
# container so the config, forgetting, BM25, key and MCP-bridge fixes go live.
#
# The two steps have to happen in this order. The volume is currently empty and the
# container writes into its own writable layer, so recreating first would throw the
# data away. The source is the *recovered* store, not /root/.memcore: the container's
# on-disk key no longer decrypts its own database (see CLAUDE.md, master-key bug), so
# copying the writable layer across would move an unreadable store into the volume.
#
# Safe to re-run. Every step verifies before moving on.

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VOLUME="memcore-memory-100_mnem_data"
SOURCE="${MEMCORE_RECOVERED_STORE:-/mnt/nas7/SkyNas/memcore-memory/backups/20260922-recovered-store}"
ENDPOINT="${MEMCORE_ENDPOINT:-http://192.168.1.183:8000}"

say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
die() { printf '\n\033[31mABORT: %s\033[0m\n' "$*" >&2; exit 1; }

say "Preflight"
[ -f "$SOURCE/master.key" ] && [ -f "$SOURCE/memory.db" ] || die "no recovered store at $SOURCE"
docker volume inspect "$VOLUME" >/dev/null 2>&1 || die "volume $VOLUME does not exist"
command -v docker >/dev/null || die "docker not on PATH"
echo "source : $SOURCE"
ls -la "$SOURCE"

# One last read of whatever the running instance can still serve, so there is a
# plaintext copy from *this* moment even if everything below goes wrong.
RESCUE="$SOURCE/../predeploy-$(date +%Y%m%d-%H%M%S).json"
if curl -sf --max-time 5 "$ENDPOINT/memories" -o "$RESCUE" 2>/dev/null; then
    echo "pre-deploy snapshot: $RESCUE ($(wc -c <"$RESCUE") bytes)"
else
    echo "pre-deploy snapshot: endpoint not answering, skipping"
fi

say "Step 1/2 - stage the recovered store into $VOLUME"
docker run --rm -v "$VOLUME":/dst -v "$SOURCE":/src:ro alpine \
    sh -c 'cp -a /src/. /dst/ && cd /dst && ls -la && sha256sum *'

say "Verify the volume against the source"
for f in master.key memory.db memory.kg.db; do
    [ -f "$SOURCE/$f" ] || continue
    want=$(sha256sum "$SOURCE/$f" | cut -d' ' -f1)
    got=$(docker run --rm -v "$VOLUME":/v alpine sha256sum "/v/$f" | cut -d' ' -f1)
    [ "$want" = "$got" ] || die "$f differs between source and volume"
    echo "  $f OK"
done

say "Step 2/2 - rebuild and recreate"
cd "$PROJECT_DIR"
docker compose up -d --build

say "Wait for the API"
for _ in $(seq 1 60); do
    curl -sf --max-time 2 "$ENDPOINT/health" >/dev/null 2>&1 && break
    sleep 1
done

say "Verify the deployment"
echo "health     : $(curl -s --max-time 5 "$ENDPOINT/health")"
echo "memories   : $(curl -s --max-time 5 "$ENDPOINT/memories")"
echo "mcp bridge : HTTP $(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "$ENDPOINT/mcp/tools")  (expect 200)"
echo "effective config:"
docker exec mnemosyne python -c "
from memcore_memory.config import settings
print('  data_dir :', settings.data_dir)
print('  db_path  :', settings.db_path)
print('  provider :', settings.embedding_provider)
print('  dim      :', settings.embedding_dim)
"
echo "volume now holds:"
docker run --rm -v "$VOLUME":/v alpine ls -la /v

say "Confirm the master key survived startup"
got=$(docker run --rm -v "$VOLUME":/v alpine sha256sum /v/master.key | cut -d' ' -f1)
want=$(sha256sum "$SOURCE/master.key" | cut -d' ' -f1)
[ "$want" = "$got" ] && echo "master.key unchanged - the key-rotation bug is fixed" \
                     || die "master.key was rewritten at startup - do not trust this deployment"

say "Done. Restart Claude Code so it picks up the MCP bridge from .mcp.json."
