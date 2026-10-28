import importlib.metadata
from pathlib import Path

# These are the packages your diagnostic identified as missing top_level.txt
missing_packages = [
    'aiohappyeyeballs', 'annotated-doc', 'attrs', 'click', 'colorama',
    'filelock', 'fsspec', 'hf-xet', 'httpcore2', 'httpx2', 'idna',
    'Jinja2', 'markdown-it-py', 'mdurl', 'numpy', 'packaging', 'pandas',
    'portalocker', 'pyarrow', 'Pygments', 'rich', 'safetensors',
    'tokenizers', 'torch', 'truststore', 'typer', 'typing_extensions',
    'urllib3'
]

for dist in importlib.metadata.distributions():
    dist_name = dist.metadata['Name']
    if dist_name not in missing_packages:
        continue

    dist_info_path = Path(dist._path)  # The .dist-info directory
    top_level_file = dist_info_path / 'top_level.txt'

    if top_level_file.exists():
        print(f"✅ {dist_name} already has top_level.txt")
        continue

    # Infer the top-level import name from the distribution name.
    # This is a good heuristic: normalize hyphens to underscores and
    # take the first part before any dot.
    import_name = dist_name.lower().replace('-', '_').split('.')[0]

    top_level_file.write_text(f"{import_name}\n", encoding='utf-8')
    print(f"📝 Created {top_level_file} with: {import_name}")