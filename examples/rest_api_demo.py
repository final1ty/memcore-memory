# REST API + Python SDK demo.
#
# There is deliberately no default URL. localhost:8000 on the SkyNAS host is the live
# production container, and the REST API writes without a password, so a default
# would put a test memory into the real store. Start a throwaway server instead:
#
#   MNEM_DATA_DIR=$(mktemp -d) memcore server start --host 127.0.0.1 --port 8765
#   python examples/rest_api_demo.py http://127.0.0.1:8765
#
# MEMCORE_DEMO_API_KEY is sent as the API key if the server has MNEM_API_KEY set.
import os
import sys

from memcore_memory.sdk import MnemosyneClient


def main(base_url: str):
    with MnemosyneClient(base_url, api_key=os.environ.get("MEMCORE_DEMO_API_KEY")) as client:
        # No entities: the demo removes its memory at the end, and a KG entity is
        # the kind of side effect that can outlive the row.
        res = client.add("Python SDK test memory", tier="episodic", importance=0.8, metadata={"demo": True})
        try:
            print(res)
            print(client.recall("SDK test", k=5))
            print(client.health())
        finally:
            client.delete(res["id"])


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(f"usage: {sys.argv[0]} <base-url of a throwaway server>\n"
                 "e.g.   MNEM_DATA_DIR=$(mktemp -d) memcore server start --host 127.0.0.1 --port 8765\n"
                 f"       {sys.argv[0]} http://127.0.0.1:8765")
    main(sys.argv[1])
