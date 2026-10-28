import time

print("Testing importlib.metadata.packages_distributions()...")
t0 = time.time()
import importlib.metadata
result = importlib.metadata.packages_distributions()
elapsed = time.time() - t0
print(f"Took {elapsed:.2f} seconds. Found {len(result)} packages.")

print("\nChecking which packages are missing top_level.txt...")
missing = []
for dist in importlib.metadata.distributions():
    if not any(f.name == "top_level.txt" for f in (dist.files or [])):
        missing.append(dist.metadata["Name"])
print(f"Packages missing top_level.txt: {missing}")