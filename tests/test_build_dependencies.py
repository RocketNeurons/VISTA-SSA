import importlib.util
from unittest.mock import Mock

import pytest

from vista.config import ROOT


def test_corrupt_raylib_download_is_rejected_before_extraction(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "fetch_build_dependencies", ROOT / "scripts/fetch_build_dependencies.py"
    )
    fetch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fetch)
    monkeypatch.setattr(fetch, "ROOT", tmp_path)
    monkeypatch.setattr(fetch.urllib.request, "urlopen", Mock(return_value=Mock(read=lambda: b"corrupt")))
    unpack = Mock()
    monkeypatch.setattr(fetch.tarfile, "open", unpack)
    with pytest.raises(ValueError, match="checksum mismatch"):
        fetch.main()
    unpack.assert_not_called()
