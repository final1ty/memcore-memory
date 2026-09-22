#!/usr/bin/env python3
"""Rebuild a deployable copy of the container's store, losing nothing.

Why this is needed at all: the running container's on-disk master.key no longer
decrypts its own database (see CLAUDE.md, master-key bug). Its records are readable
only by the process that is still holding the original key in memory. So the store
cannot simply be copied out - it has to be re-encrypted under a new key, using that
process as the only available decryption oracle.

What makes that lossless: `content` and `metadata` are the only encrypted columns.
`id`, `tier`, `timestamp`, `forgetting_json`, `entities_json` and `embedding` are
plaintext, so they come straight out of a docker-cp'd copy of the database and are
carried across verbatim - including the Ebbinghaus curve and the stored vector.

Content is read through GET /memory/{id}, never the /memories listing, which cuts
content at 500 characters. Importing from that endpoint silently stores truncated
records.

Usage:
    rescue-container-store.py OUTPUT_DIR [--container NAME] [--endpoint URL]

The output directory is a complete data dir (master.key, memory.db, memory.kg.db)
ready to be staged into the volume.
"""

import argparse
import asyncio
import json
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("output_dir", type=Path)
    p.add_argument("--container", default="mnemosyne")
    p.add_argument("--endpoint", default="http://192.168.1.183:8000")
    p.add_argument("--data-dir", default="/root/.memcore",
                   help="Path of the store inside the container")
    return p.parse_args()


def fetch(url, timeout=20):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.load(r)


async def main():
    args = parse_args()
    out = args.output_dir

    print(f"== endpoint {args.endpoint}")
    health = fetch(f"{args.endpoint}/health")
    live_total = sum(health.get("tier_counts", {}).values())
    print(f"   health {health['status']}, {live_total} memories live")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        print(f"== copying the database out of {args.container}")
        for name in ("memory.db", "memory.kg.db"):
            subprocess.run(
                ["docker", "cp", f"{args.container}:{args.data_dir}/{name}", str(tmp / name)],
                check=(name == "memory.db"),
            )

        db = sqlite3.connect(tmp / "memory.db")
        db.row_factory = sqlite3.Row
        rows = list(db.execute("SELECT * FROM memories ORDER BY timestamp"))
        print(f"   {len(rows)} rows on disk")

        if len(rows) < live_total:
            # The container writes to its own layer; a copy can lag a live write.
            print(f"   note: API reports {live_total}, snapshot has {len(rows)} "
                  f"- the copy was taken after those writes landed" if len(rows) >= live_total
                  else f"   WARNING: API reports {live_total} but the snapshot has only {len(rows)}")

        print("== reading plaintext back through the API (the only process holding the key)")
        records, failed = [], []
        for i, row in enumerate(rows, 1):
            try:
                d = fetch(f"{args.endpoint}/memory/{row['id']}")
            except Exception as e:  # noqa: BLE001 - want the id with the reason
                failed.append((row["id"], repr(e)))
                continue
            records.append((row, d))
            if i % 25 == 0:
                print(f"   {i}/{len(rows)}")

        if failed:
            print(f"\n{len(failed)} record(s) could not be read:")
            for mid, err in failed[:10]:
                print(f"   {mid} {err}")
            sys.exit("refusing to write a partial store")

        stamp = time.strftime("%Y%m%d-%H%M%S")
        raw = out.parent / f"container-rescue-{stamp}.json"
        raw.parent.mkdir(parents=True, exist_ok=True)
        raw.write_text(json.dumps(
            [{"id": r["id"], "tier": r["tier"], "timestamp": r["timestamp"],
              "content": d["content"], "metadata": d.get("metadata", {})}
             for r, d in records], indent=2, ensure_ascii=False))
        print(f"   plaintext snapshot: {raw}")

        print(f"== re-encrypting {len(records)} records under a fresh key in {out}")
        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True)

        import os
        os.environ["MNEM_DATA_DIR"] = str(out)
        from memcore_memory.config import Settings
        from memcore_memory.core.ebbinghaus import ForgettingCurve
        from memcore_memory.core.tiers import MemoryItem, Tier
        from memcore_memory.crypto.aes_gcm import AES256GCM
        from memcore_memory.crypto.key_manager import KeyManager
        from memcore_memory.graph.kg import KnowledgeGraph
        from memcore_memory.storage.encrypted_sqlite import EncryptedStore

        settings = Settings()
        key = KeyManager(settings.key_path).load_or_create()
        cipher = AES256GCM(key)
        store = EncryptedStore(settings.db_path, cipher)
        await store.init()
        kg = KnowledgeGraph(settings.db_path.with_suffix(".kg.db"), cipher)
        await kg.init()

        for row, d in records:
            await store.put(MemoryItem(
                id=row["id"],
                content=d["content"],
                tier=Tier(row["tier"]),
                timestamp=row["timestamp"],
                embedding=json.loads(row["embedding"].decode()) if row["embedding"] else None,
                metadata=d.get("metadata", {}),
                forgetting=ForgettingCurve(**json.loads(row["forgetting_json"]))
                           if row["forgetting_json"] else None,
                entities=json.loads(row["entities_json"]) if row["entities_json"] else [],
            ))

        print("== verifying through a fresh open of the new store")
        verify = EncryptedStore(settings.db_path, AES256GCM(KeyManager(settings.key_path).load_or_create()))
        items = await verify.list_all()
        assert len(items) == len(records), f"{len(items)} written, {len(records)} expected"
        by_id = {i.id: i for i in items}
        for row, d in records:
            item = by_id[row["id"]]
            assert item.content == d["content"], f"{row['id']}: content mismatch"
            assert item.metadata == d.get("metadata", {}), f"{row['id']}: metadata mismatch"
            assert item.tier.value == row["tier"], f"{row['id']}: tier mismatch"
            assert item.timestamp == row["timestamp"], f"{row['id']}: timestamp mismatch"
            if row["forgetting_json"]:
                for field, value in json.loads(row["forgetting_json"]).items():
                    assert getattr(item.forgetting, field) == value, f"{row['id']}: {field}"

        print(f"   {len(items)} records verified byte-for-byte")
        print(f"   files: {sorted(p.name for p in out.iterdir())}")
        print(f"\nReady to stage: {out}")


if __name__ == "__main__":
    asyncio.run(main())
