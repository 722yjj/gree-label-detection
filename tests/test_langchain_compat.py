def test_langchain_legacy_paths_are_available_for_paddlex():
    from label_detection.core.langchain_compat import ensure_langchain_legacy_imports

    ensure_langchain_legacy_imports()

    from langchain.docstore.document import Document
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    assert Document is not None
    assert RecursiveCharacterTextSplitter is not None
