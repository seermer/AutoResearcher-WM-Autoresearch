"""Additional tools. Starts empty on purpose — the meta agent populates it.

Add a tool by defining it here (or in a sibling module) and appending it to TOOLS:

    from langchain_core.tools import tool

    @tool
    def zip_manifest(path: str) -> str:
        \"\"\"List the entries of a data shard zip without extracting it.\"\"\"
        ...

    TOOLS = [zip_manifest]
"""
TOOLS: list = []
