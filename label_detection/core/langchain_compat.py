# Compatibility shim to fix langchain import issues
# This redirects old langchain imports to new langchain_core locations

# Create the missing langchain module structure
import sys
import types

# Create langchain package if it doesn't exist
try:
    import langchain
except ImportError:
    sys.modules["langchain"] = types.ModuleType("langchain")

# 1. Fix langchain.docstore import
# Create langchain.docstore package
sys.modules["langchain.docstore"] = types.ModuleType("langchain.docstore")

# Import the Document class from the new location and expose it in the old location
from langchain_core.documents.base import Document

# Expose Document in the old location
sys.modules["langchain.docstore.document"] = types.ModuleType(
    "langchain.docstore.document"
)
sys.modules["langchain.docstore.document"].Document = Document

# 2. Fix langchain.text_splitter import
# Import RecursiveCharacterTextSplitter from the new location
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Create langchain.text_splitter module
sys.modules["langchain.text_splitter"] = types.ModuleType("langchain.text_splitter")
sys.modules["langchain.text_splitter"].RecursiveCharacterTextSplitter = (
    RecursiveCharacterTextSplitter
)

# Make sure the old import paths work
import langchain.docstore.document
import langchain.text_splitter

print("LangChain compatibility shim loaded successfully!")
