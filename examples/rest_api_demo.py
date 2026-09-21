
# REST API + Python SDK demo
from mnemosyne.sdk import MnemosyneClient

client = MnemosyneClient("http://localhost:8000")

# add
res = client.add("Python SDK test memory", tier="episodic", importance=0.8, entities=["Python"])
print(res)

# recall
results = client.recall("SDK test", k=5)
print(results)

# health
print(client.health())
