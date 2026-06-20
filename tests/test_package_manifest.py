from __future__ import annotations

import json
import zipfile

import pytest

import app_services


def _write_package(zip_path, manifest: dict) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(app_services.PACKAGE_MANIFEST_NAME, json.dumps(manifest, ensure_ascii=False))


def test_read_package_manifest_missing_file(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        app_services._read_package_manifest(tmp_path / "missing.zip")


def test_read_package_manifest_rejects_missing_manifest(tmp_path) -> None:
    package_path = tmp_path / "empty.zip"
    with zipfile.ZipFile(package_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("other.json", "{}")

    with pytest.raises(ValueError, match="missing package manifest"):
        app_services._read_package_manifest(package_path)


def test_read_package_manifest_rejects_wrong_package_type(tmp_path) -> None:
    package_path = tmp_path / "wrong-type.zip"
    _write_package(package_path, {"package_type": "other"})

    with pytest.raises(ValueError, match="not a StoryRAG knowledge package"):
        app_services._read_package_manifest(package_path)


def test_import_knowledge_package_rejects_embedding_model_mismatch(tmp_path, monkeypatch) -> None:
    package_path = tmp_path / "mismatch.zip"
    monkeypatch.setitem(app_services.COLLECTION_METADATA, "embedding_model", "local-model")
    local_model = str(app_services.COLLECTION_METADATA.get("embedding_model"))
    package_model = f"{local_model}-different"
    _write_package(
        package_path,
        {
            "package_type": app_services.PACKAGE_TYPE,
            "package_version": app_services.PACKAGE_VERSION,
            "embedding_model": package_model,
            "corpora": [],
        },
    )

    with pytest.raises(ValueError, match="Embedding model mismatch"):
        app_services.import_knowledge_package(package_path)
