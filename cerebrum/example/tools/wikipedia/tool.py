from cerebrum.tool.base import BaseTool
import wikipedia
from typing import Optional, Any

class Wikipedia(BaseTool):
    """Wikipedia tool, refactored from langchain.

    To use, you should have the ``wikipedia`` python package installed.
    This wrapper will use the Wikipedia API to conduct searches and
    fetch page summaries. By default, it will return the page summaries
    of the top-k results.
    It limits the Document content by doc_content_chars_max.
    """
    def __init__(self):
        super().__init__()
        self.WIKIPEDIA_MAX_QUERY_LENGTH = 300
        self.top_k_results = 3
        self.lang = "en"
        self.load_all_available_meta: bool = False
        self.doc_content_chars_max: int = 4000
        self.wiki_client = self.build_client()

    def build_client(self):
        try:
            wikipedia.set_lang(self.lang)
            return wikipedia
        except ImportError:
            raise ImportError(
                "Could not import wikipedia python package. "
                "Please install it with `pip install wikipedia`."
            )

    def _fetch_page(self, page_title: str) -> Optional[Any]:
        """Fetch a Wikipedia page."""
        try:
            return self.wiki_client.page(page_title)
        except Exception:
            return None

    def run(self, params) -> str:
        """Run Wikipedia search and get page summaries."""
        query = params["query"]
        page_titles = self.wiki_client.search(query, results=self.top_k_results)
        summaries = []
        for page_title in page_titles[: self.top_k_results]:
            if wiki_page := self._fetch_page(page_title):
                if summary := self._formatted_page_summary(page_title, wiki_page):
                    summaries.append(summary)
        if not summaries:
            return "No good Wikipedia Search Result was found"
        return "\n\n".join(summaries)[: self.doc_content_chars_max]

    @staticmethod
    def _formatted_page_summary(page_title: str, wiki_page: Any) -> Optional[str]:
        return f"Page: {page_title}\nSummary: {wiki_page.summary}"

    def get_tool_call_format(self):
        tool_call_format = {
            "type": "function",
            "function": {
                "name": "wikipedia",
                "description": "Search information on the Wikipedia",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Query used for searching information on the Wikipedia"
                        }
                    },
                    "required": [
                        "query"
                    ]
                }
            }
        }
        return tool_call_format
