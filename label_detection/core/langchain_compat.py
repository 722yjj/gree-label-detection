"""Compatibility aliases for PaddleX imports against LangChain 1.x.

PaddleX 3.4.0 still imports a couple of pre-1.0 LangChain module paths:

- ``langchain.docstore.document.Document``
- ``langchain.text_splitter.RecursiveCharacterTextSplitter``

The project pins LangChain 1.x, where those objects live in split packages.
This module installs narrow aliases before PaddleX/PaddleOCR is imported.
"""

from __future__ import annotations

import sys
import types


def ensure_langchain_legacy_imports() -> None:
    """Expose the old LangChain module paths required by PaddleX 3.4.0."""

    try:
        import langchain
    except ImportError:
        langchain = types.ModuleType("langchain")
        sys.modules["langchain"] = langchain

    if "langchain.docstore.document" not in sys.modules:
        try:
            from langchain_classic.docstore.document import Document
        except ImportError:
            from langchain_core.documents import Document

        docstore_mod = sys.modules.setdefault(
            "langchain.docstore",
            types.ModuleType("langchain.docstore"),
        )
        document_mod = types.ModuleType("langchain.docstore.document")
        document_mod.Document = Document
        sys.modules["langchain.docstore.document"] = document_mod
        setattr(docstore_mod, "document", document_mod)
        setattr(langchain, "docstore", docstore_mod)

    if "langchain.text_splitter" not in sys.modules:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        text_splitter_mod = types.ModuleType("langchain.text_splitter")
        text_splitter_mod.RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter
        sys.modules["langchain.text_splitter"] = text_splitter_mod
        setattr(langchain, "text_splitter", text_splitter_mod)


ensure_langchain_legacy_imports()
