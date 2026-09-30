"""Download official UCI members with fixed checksums; no dataset in Git."""
import hashlib
from io import BytesIO
from pathlib import Path
import sys
from urllib.request import urlopen
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.credit_benchmarks import SOURCES, load_benchmark

def main() -> None:
    for name, spec in SOURCES.items():
        directory = ROOT / "data/raw" / name
        path = directory / spec.filename
        if not path.exists():
            with urlopen(spec.url, timeout=60) as response:
                archive = ZipFile(BytesIO(response.read()))
            content = archive.read(spec.filename)  # never extract arbitrary paths
            if hashlib.sha256(content).hexdigest() != spec.sha256:
                raise ValueError(f"Source checksum mismatch: {name}")
            directory.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        _, _, metadata = load_benchmark(name)
        print(name, metadata)

if __name__ == "__main__":
    main()
