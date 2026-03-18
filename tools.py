from langchain_tavily import TavilySearch
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np
from dotenv import load_dotenv

load_dotenv()

search_tool = TavilySearch(max_results=10)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)


def fetch_full_page(url: str, query: str = "") -> str:
    """Fetch page content and return the most relevant chunks."""
    try:
        loader = WebBaseLoader(url)
        docs = loader.load()
        if not docs:
            return "Could not load page content."

        content = docs[0].page_content
        chunks = text_splitter.split_text(content)

        if not query or len(chunks) <= 10:
            return content[:10000]

        query_embedding = embeddings.embed_query(query)
        chunk_embeddings = embeddings.embed_documents(chunks)

        similarities = []
        for chunk_emb in chunk_embeddings:
            score = np.dot(query_embedding, chunk_emb) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(chunk_emb)
            )
            similarities.append(score)

        ranked_indices = np.argsort(similarities)[::-1]

        selected_chunks = []
        total_chars = 0
        for idx in ranked_indices:
            if total_chars + len(chunks[idx]) > 10000:
                break
            selected_chunks.append((idx, chunks[idx]))
            total_chars += len(chunks[idx])

        selected_chunks.sort(key=lambda x: x[0])

        return "\n\n".join([chunk for _, chunk in selected_chunks])

    except Exception as e:
        return f"Error fetching {url}: {str(e)}"
