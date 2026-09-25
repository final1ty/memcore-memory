#!/usr/bin/env bash
#
# Deploy a new image to SkyNAS without losing the container's memories.
#
#   deploy-skynas.sh [--keep-plaintext] [--drop-kg-hand-added] [SOURCE]
#
# Since the 16:39 deploy on 2026-09-22 the container serves /data, the named volume,
# and its on-disk master.key is valid. So the volume IS the live store, and staging
# into it while the container runs would swap the key and database underneath a
# process that keeps writing under the old key. The container is therefore stopped
# before anything is staged, and only restarted on the staged files.
#
# The order, and why:
#   1. build the new image while the old container still serves
#   2. rescue the current contents through the API (rescue-container-store.py) - on
#      every run, with or without SOURCE. The first version of this file staged a
#      fixed path holding one memory while the container held 120; a later one
#      skipped the rescue whenever SOURCE was given. The rescue copy is the rollback.
#   3. refuse unless every live id is in the store to stage - ids, not counts: a
#      store with more records can still lack the ones that matter
#   4. stop the container, tar the volume, and compare every row of the stopped
#      volume with the rescue's snapshot (store_fingerprint.py). Any write that
#      landed in between - a new memory, a delete, an edit, a rehearsal, a tier
#      move, a hand-added relation - would be undone by staging, so it aborts and
#      restarts the old container; re-running rescues those writes too
#   5. stage only the known store files into the volume, verify them, recreate
#
# SOURCE, when given, is a data dir to deploy instead of the rescue copy. It must
# hold every live id, and its master.key must be a raw 32-byte key that decrypts
# its memory.db, checked before anything is stopped: the compose file sets no
# MNEM_MASTER_PASSWORD, so a password-wrapped key would stop the new container
# from starting, and a mismatched one would leave it refusing its own store.
# Only master.key, memory.db, memory.kg.db and the vector
# sidecar are staged - never a whole directory, which once would have copied a
# .master-password file into the volume and every backup of it.
#
# The rescue leaves a plaintext JSON copy of every record next to its output. It
# is deleted once the deploy has verified, or when the deploy aborts, unless
# --keep-plaintext is given.
#
# The rescue carries hand-added graph entities and relations across by reading the
# graph with the container's master.key. When that key cannot read it, the rescue
# refuses rather than drop them; --drop-kg-hand-added is passed through to accept
# the loss. Nothing else needs it.

set -euo pipefail
umask 077

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVICE=mnemosyne
CONTAINER="${MEMCORE_CONTAINER:-mnemosyne}"
ENDPOINT="${MEMCORE_ENDPOINT:-http://192.168.1.183:8000}"
BACKUP_ROOT="${MEMCORE_BACKUP_ROOT:-/mnt/nas7/SkyNas/memcore-memory/backups}"
PYTHON="${MEMCORE_PYTHON:-$PROJECT_DIR/.venv/bin/python}"
STORE_FILES="master.key memory.db memory.kg.db vectors.vectors.json"

KEEP_PLAINTEXT=0
RESCUE_FLAGS=()
SOURCE=""
while [ $# -gt 0 ]; do
    case "$1" in
        --keep-plaintext) KEEP_PLAINTEXT=1 ;;
        --drop-kg-hand-added|--drop-kg-relations) RESCUE_FLAGS+=(--drop-kg-hand-added) ;;
        -h|--help) awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "${BASH_SOURCE[0]}"; exit 0 ;;
        -*) printf 'unknown option %s\n' "$1" >&2; exit 2 ;;
        *) [ -z "$SOURCE" ] || { printf 'only one SOURCE\n' >&2; exit 2; }; SOURCE="$1" ;;
    esac
    shift
done

say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
die() { printf '\n\033[31mABORT: %s\033[0m\n' "$*" >&2; exit 1; }
compose() { (cd "$PROJECT_DIR" && docker compose "$@"); }

# Where the run stands, for the exit trap: what it has to undo depends on it.
RESCUE=""
RESCUED=0
STOPPED=0
STAGING=0
STARTED=0
VOLUME_TGZ=""
on_exit() {
    local rc=$?
    [ "$rc" -ne 0 ] || return 0
    # Also when the rescue itself failed after writing it: the live store has not
    # been touched at that point, so the plaintext protects nothing, and once the
    # rescue verified, its copy is the rollback point instead.
    if [ -n "$RESCUE" ] && [ -e "$RESCUE.plaintext.json" ]; then
        if [ "$KEEP_PLAINTEXT" = 0 ]; then
            rm -f "$RESCUE.plaintext.json"
            printf '\nRemoved the plaintext snapshot %s.\n' "$RESCUE.plaintext.json" >&2
        else
            printf '\nPlaintext snapshot kept: %s - delete it when done.\n' "$RESCUE.plaintext.json" >&2
        fi
    fi
    if [ "$STOPPED" = 1 ] && [ "$STAGING" = 0 ]; then
        printf '\nNothing was staged; starting the container again on its own store.\n' >&2
        compose start "$SERVICE" >&2 || printf 'could not start it: run  docker compose start %s\n' "$SERVICE" >&2
    elif [ "$STAGING" = 1 ] && [ "$STARTED" = 0 ]; then
        # Starting now would run on a half-staged volume. Left stopped on purpose.
        printf '\nThe volume may be half-staged, so the container is left STOPPED.\n' >&2
        printf 'Its previous contents: %s\n' "$VOLUME_TGZ" >&2
        printf 'Restore:  docker run --rm -v %s:/v -v %s:/b alpine sh -c '"'"'cd /v && rm -f %s && tar xzf /b/%s'"'"'\n' \
            "$VOLUME" "$BACKUP_ROOT" "$STORE_FILES" "$(basename "$VOLUME_TGZ")" >&2
        printf 'then:     docker compose start %s\n' "$SERVICE" >&2
    fi
}
trap on_exit EXIT

# The ids in an SQLite store on this host, one per line, sorted. Read-only, and it
# refuses a store with a non-empty -wal beside it: staging copies the database
# files alone, which would drop whatever the -wal still holds.
ids_in() {
    local f
    for f in memory.db memory.kg.db; do
        [ ! -s "$1/$f-wal" ] || { echo "$1/$f-wal is not empty - the store is open or was not closed cleanly" >&2; return 1; }
    done
    "$PYTHON" - "$1/memory.db" <<'PY'
import sqlite3, sys
con = sqlite3.connect(f"file:{sys.argv[1]}?mode=ro", uri=True)
for (i,) in con.execute("select id from memories order by id"):
    print(i)
PY
}

# A row-level fingerprint of the store in the volume, taken with the running
# image's own python since the volume is root-owned. store_fingerprint.py copies
# the database and any leftover -wal to /tmp first, so the -wal is replayed.
volume_fingerprint() { docker run --rm -v "$VOLUME":/v:ro -v "$PROJECT_DIR/scripts":/s:ro \
    --entrypoint python "$IMAGE" /s/store_fingerprint.py /v; }

# Fails closed: a /health without tier_counts used to count as 0 live memories,
# which let any store through the check below.
live_total() { curl -sf --max-time 10 "$ENDPOINT/health" | "$PYTHON" -c "
import json,sys
tc = json.load(sys.stdin).get('tier_counts')
if not isinstance(tc, dict): sys.exit('health response has no tier_counts')
print(sum(tc.values()))
"; }

# Every id in $1 (a file of ids) must be in $2. Prints what is missing.
require_ids() {
    local missing
    missing=$(LC_ALL=C comm -23 "$1" "$2")
    [ -z "$missing" ] || die "$(printf '%s\n' "$missing" | grep -c .) live id(s) are not in the store to stage, e.g.
$(printf '%s\n' "$missing" | head -5)
$3"
}

check_key() {
    local size
    size=$(stat -c %s "$1/master.key")
    [ "$size" = 32 ] || die "$1/master.key is $size bytes, not a raw 32-byte key. The compose file sets \
no MNEM_MASTER_PASSWORD, so the recreated container could not unlock it."
}

key_matches() {
    # Size alone let a mismatched key through every check before the stop; the new
    # container then refused its store (MasterKeyMismatch) after the old one was gone.
    # Read from a copy, so the check writes nothing into $1. Call after ids_in, which
    # refuses a non-empty -wal.
    "$PYTHON" - "$1" "$PROJECT_DIR/src" <<'PY' || die "$1/master.key is not the key $1/memory.db was written under"
import asyncio, shutil, sqlite3, sys, tempfile
from pathlib import Path
src = Path(sys.argv[1])
sys.path.insert(0, sys.argv[2])
from memcore_memory.crypto.aes_gcm import AES256GCM
from memcore_memory.storage.encrypted_sqlite import EncryptedStore

async def main():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "memory.db"
        shutil.copyfile(src / "memory.db", db)
        store = EncryptedStore(db, AES256GCM((src / "master.key").read_bytes()), blind_index=False)
        con = sqlite3.connect(db)
        stored = None
        if con.execute("SELECT 1 FROM sqlite_master WHERE name='store_meta'").fetchone():
            row = con.execute("SELECT v FROM store_meta WHERE k='key_id'").fetchone()
            stored = row[0] if row else None
        con.close()
        if stored is not None and stored != store.key_id:
            sys.exit(f"the store records key id {stored}, master.key is {store.key_id}")
        items, bad = await store.scan()
        if bad:
            sys.exit(f"master.key decrypts {len(items)} of {len(items) + len(bad)} rows")

asyncio.run(main())
PY
}

say "Preflight"
command -v docker >/dev/null || die "docker not on PATH"
[ -x "$PYTHON" ] || die "no interpreter at $PYTHON"
# From compose itself: the name follows the project name, and a hardcoded one went
# stale the moment the checkout was renamed.
VOLUME="${MEMCORE_VOLUME:-$(compose config --format json | "$PYTHON" -c \
    'import json,sys; print(json.load(sys.stdin)["volumes"]["mnem_data"]["name"])')}" \
    || die "could not read the volume name from docker compose config"
docker volume inspect "$VOLUME" >/dev/null 2>&1 || die "volume $VOLUME does not exist"
MOUNTED=$(docker inspect "$CONTAINER" --format \
    '{{range .Mounts}}{{if eq .Destination "/data"}}{{.Name}}{{end}}{{end}}') \
    || die "no container $CONTAINER - the rescue needs it running"
[ "$MOUNTED" = "$VOLUME" ] || die "$CONTAINER serves /data from '$MOUNTED', but compose would mount \
'$VOLUME'. Deploying would start the new container on a different store."
IMAGE=$(docker inspect "$CONTAINER" --format '{{.Image}}')
LIVE=$(live_total) || die "$ENDPOINT is not answering with tier_counts - it must be up for the rescue"
echo "volume: $VOLUME   live memories: $LIVE"

if [ -n "$SOURCE" ]; then
    # Absolute, always: docker reads a bare name as a named volume and would mount
    # an empty one.
    SOURCE="$(realpath -e -- "$SOURCE")" || die "cannot resolve SOURCE"
    [ -f "$SOURCE/memory.db" ] && [ -f "$SOURCE/master.key" ] || die "$SOURCE is not a data dir"
    check_key "$SOURCE"
    ids_in "$SOURCE" >/dev/null || die "cannot read the ids in $SOURCE"
    key_matches "$SOURCE"
    echo "store to deploy: $SOURCE"
fi

say "Step 1/5 - build the new image (the old container keeps serving)"
compose build || die "build failed - nothing has been changed"

say "Step 2/5 - rescue the container's current contents"
mkdir -p "$BACKUP_ROOT"
STAMP=$(date +%Y%m%d-%H%M%S)
RESCUE="$BACKUP_ROOT/$STAMP-recovered-store"
"$PYTHON" "$PROJECT_DIR/scripts/rescue-container-store.py" "$RESCUE" \
    --endpoint "$ENDPOINT" --container "$CONTAINER" --keep-plaintext ${RESCUE_FLAGS[@]+"${RESCUE_FLAGS[@]}"} \
    || die "rescue failed - the container and its store have not been changed"
[ -f "$RESCUE/source-fingerprint.json" ] || die "the rescue left no source-fingerprint.json - is \
rescue-container-store.py older than this script?"
RESCUED=1
[ -n "$SOURCE" ] || SOURCE="$RESCUE"
check_key "$SOURCE"
key_matches "$SOURCE"

say "Step 3/5 - compare the store to stage with the live one"
WORK=$(mktemp -d)
ids_in "$RESCUE" >"$WORK/live.ids" || die "cannot read the rescue copy"
ids_in "$SOURCE" >"$WORK/source.ids" || die "cannot read $SOURCE"
STAGED=$(wc -l <"$WORK/source.ids" | tr -d ' ')
echo "records - live: $LIVE   rescued: $(wc -l <"$WORK/live.ids" | tr -d ' ')   to stage: $STAGED"
require_ids "$WORK/live.ids" "$WORK/source.ids" "Deploying $SOURCE would destroy them."
[ "$STAGED" -ge "$LIVE" ] || die "the store to stage holds $STAGED records but the container serves $LIVE"

say "Step 4/5 - stop the container, back up and re-check the volume"
compose stop "$SERVICE" || die "could not stop $SERVICE - nothing staged"
STOPPED=1
VOLUME_TGZ="$BACKUP_ROOT/$STAMP-volume-before-deploy.tgz"
docker run --rm -v "$VOLUME":/v:ro -v "$BACKUP_ROOT":/b alpine \
    sh -c "umask 077 && tar czf /b/$(basename "$VOLUME_TGZ") -C /v ." \
    || die "could not archive the volume - nothing staged"
echo "volume archived: $VOLUME_TGZ"
volume_fingerprint >"$WORK/volume.fp.json" || die "cannot fingerprint the stopped volume"
CHANGED=$("$PYTHON" "$PROJECT_DIR/scripts/store_fingerprint.py" --compare \
    "$RESCUE/source-fingerprint.json" "$WORK/volume.fp.json") \
    || die "the store changed between the rescue snapshot and the stop:
$CHANGED
Staging would undo those writes. Re-run the deploy to rescue them too."
echo "stopped volume matches the rescue snapshot row for row"

say "Step 5/5 - stage into $VOLUME and recreate"
STAGING=1
# The old store files go first: a stale vector sidecar or -wal left beside the new
# database would be read as if it belonged to it.
docker run --rm -v "$VOLUME":/dst -v "$SOURCE":/src:ro alpine sh -c '
    set -e
    cd /dst
    for f in '"$STORE_FILES"'; do rm -f "$f" "$f-wal" "$f-shm" "$f-journal" "$f.lock"; done
    for f in '"$STORE_FILES"'; do [ ! -f "/src/$f" ] || cp -p "/src/$f" "/dst/$f"; done
    ls -la /dst'
for f in $STORE_FILES; do
    if [ -f "$SOURCE/$f" ]; then
        want=$(sha256sum "$SOURCE/$f" | cut -d' ' -f1)
        got=$(docker run --rm -v "$VOLUME":/v:ro alpine sha256sum "/v/$f" | cut -d' ' -f1)
        [ "$want" = "$got" ] || die "$f differs between source and volume"
        echo "  $f OK"
    fi
done
# --force-recreate: with an unchanged image compose would otherwise leave the old
# container as it is, and it would never read the staged files.
compose up -d --no-build --force-recreate "$SERVICE"
STARTED=1

say "Wait for the API"
for _ in $(seq 1 90); do
    curl -sf --max-time 2 "$ENDPOINT/livez" >/dev/null 2>&1 && break
    curl -sf --max-time 2 "$ENDPOINT/health" >/dev/null 2>&1 && break
    sleep 1
done

say "Verify the deployment"
AFTER=$(live_total) || die "the API did not come back up. The staged store is at $SOURCE, the \
previous volume contents in $VOLUME_TGZ"
echo "memories after deploy: $AFTER   (expected >= $STAGED)"
[ "$AFTER" -ge "$STAGED" ] || die "record count dropped from $STAGED to $AFTER"
echo "mcp bridge : HTTP $(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "$ENDPOINT/mcp/tools")  (expect 200, or 401 with MNEM_API_KEY set)"
echo "effective config:"
docker exec "$CONTAINER" python -c "
from memcore_memory.config import settings
print('  data_dir :', settings.data_dir)
print('  db_path  :', settings.db_path)
print('  provider :', settings.embedding_provider)
print('  dim      :', settings.embedding_dim)
"

say "Confirm the master key survived startup"
got=$(docker run --rm -v "$VOLUME":/v:ro alpine sha256sum /v/master.key | cut -d' ' -f1)
want=$(sha256sum "$SOURCE/master.key" | cut -d' ' -f1)
[ "$want" = "$got" ] || die "master.key was rewritten at startup - do not trust this deployment"
echo "master.key unchanged"

rm -rf "$WORK"
if [ "$KEEP_PLAINTEXT" = 1 ]; then
    echo "plaintext snapshot kept: $RESCUE.plaintext.json - delete it when done"
else
    rm -f "$RESCUE.plaintext.json"
fi

say "Done. Data lives in $VOLUME."
echo "Deployed from   : $SOURCE"
echo "Rescue copy     : $RESCUE  (raw key inside - the rollback point)"
echo "Previous volume : $VOLUME_TGZ"
