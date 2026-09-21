"""Fetch pinned public build dependencies; no changes to installed packages."""
import hashlib
import io
import json
from pathlib import Path
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
# Digest recorded for the release archive used to build the paper environments.
RAYLIB_SHA256 = '3d95ef03d5b38dfa55c0a16ca122d382134b078f0e5b270b52fe7eae0549c000'


def main():
    target = ROOT / '.build_deps'
    target.mkdir(exist_ok=True)
    url = 'https://github.com/raysan5/raylib/releases/download/5.5/raylib-5.5_linux_amd64.tar.gz'
    if not (target / 'raylib-5.5_linux_amd64/lib/libraylib.a').exists():
        payload = urllib.request.urlopen(url, timeout=120).read()
        digest = hashlib.sha256(payload).hexdigest()
        if digest != RAYLIB_SHA256:
            raise ValueError(f'Raylib archive checksum mismatch: {digest}')
        with tarfile.open(fileobj=io.BytesIO(payload), mode='r:gz') as archive:
            archive.extractall(target, filter='data')
        (target / 'raylib-download.json').write_text(json.dumps({
            'url': url, 'sha256': hashlib.sha256(payload).hexdigest()
        }, indent=2))
    print('Raylib 5.5 ready:', target)


if __name__ == '__main__':
    main()
