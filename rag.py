# rag.py
import os
import re
import json
import numpy as np
from bs4 import BeautifulSoup
import requests
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# try to import faiss; fallback to cosine
try:
    import faiss
    _HAS_FAISS = True
except Exception:
    _HAS_FAISS = False

# model for embeddings
EMBED_MODEL = "all-MiniLM-L6-v2"
INDEX_META = "index_meta.npz"
FAISS_INDEX_FILE = "faiss.index"


def clean_html_to_text(html):
    soup = BeautifulSoup(html, "html.parser")
    for s in soup(["script", "style", "nav", "footer", "header", "aside"]):
        s.decompose()
    text = soup.get_text(separator="\n")
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return "\n".join(lines)


def fetch_and_save(url, outdir="kb"):
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    txt = clean_html_to_text(resp.text)
    safe = re.sub(r"[^0-9a-zA-Z]+", "_", url)[:200]
    os.makedirs(outdir, exist_ok=True)
    fn = os.path.join(outdir, f"{safe}.txt")
    with open(fn, "w", encoding="utf-8") as f:
        f.write(url + "\n\n")
        f.write(txt)
    return fn


def load_kb_texts(kb_dir="kb"):
    docs = []
    if not os.path.exists(kb_dir):
        return docs
    for fn in os.listdir(kb_dir):
        if fn.endswith(".txt"):
            path = os.path.join(kb_dir, fn)
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            source = content.splitlines()[0] if content else ""
            body = "\n".join(content.splitlines()[2:]) if len(content.splitlines()) > 2 else ""
            docs.append({"id": fn, "source": source, "text": body})
    return docs


def chunk_text(text, chunk_size=800):
    paragraphs = text.split("\n\n")
    chunks = []
    buf = ""
    for p in paragraphs:
        if len(buf) + len(p) < chunk_size:
            buf = (buf + "\n\n" + p) if buf else p
        else:
            if buf:
                chunks.append(buf)
            buf = p
    if buf:
        chunks.append(buf)
    return chunks


def build_index(kb_dir="kb", index_meta=INDEX_META):
    docs = load_kb_texts(kb_dir)
    if not docs:
        raise Exception("No docs found in kb/. Add some .txt files or use fetch_and_save to scrape pages.")
    model = SentenceTransformer(EMBED_MODEL)
    embeddings = []
    metadata = []
    for d in tqdm(docs, desc="Embedding docs"):
        chunks = chunk_text(d["text"])
        for i, ch in enumerate(chunks):
            emb = model.encode(ch)
            embeddings.append(emb)
            metadata.append({"doc_id": d["id"], "source": d["source"], "chunk_id": i, "text": ch})
    X = np.array(embeddings).astype("float32")
    if _HAS_FAISS:
        dim = X.shape[1]
        faiss_index = faiss.IndexFlatIP(dim)
        faiss.normalize_L2(X)
        faiss_index.add(X)
        faiss.write_index(faiss_index, FAISS_INDEX_FILE)
        np.savez(index_meta, metadata=metadata)
    else:
        np.savez(index_meta, embeddings=X, metadata=metadata)
    return True


def _load_index(index_meta=INDEX_META):
    if _HAS_FAISS and os.path.exists(FAISS_INDEX_FILE):
        idx = faiss.read_index(FAISS_INDEX_FILE)
        meta = np.load(index_meta, allow_pickle=True)["metadata"]
        return ("faiss", idx, meta)
    else:
        arr = np.load(index_meta, allow_pickle=True)
        return ("fallback", arr["embeddings"], arr["metadata"])


def retrieve(query, top_k=4, index_meta=INDEX_META):
    model = SentenceTransformer(EMBED_MODEL)
    qvec = model.encode(query).astype("float32").reshape(1, -1)
    mode, idx_obj, meta = _load_index(index_meta)
    results = []
    if mode == "faiss":
        faiss.normalize_L2(qvec)
        D, I = idx_obj.search(qvec, top_k)
        for score, ii in zip(D[0], I[0]):
            m = meta[int(ii)]
            results.append({"score": float(score), "metadata": m})
    else:
        X = idx_obj
        from sklearn.metrics.pairwise import cosine_similarity
        sims = cosine_similarity(qvec, X)[0]
        arg = sims.argsort()[::-1][:top_k]
        for i in arg:
            results.append({"score": float(sims[i]), "metadata": meta[int(i)]})
    return results


# 🔹 Answer with OpenAI if possible, else fallback to HuggingFace
from transformers import pipeline
from openai import OpenAI

def answer_with_rag(question, retrieved, openai_api_key=None):
    if openai_api_key is None:
        openai_api_key = os.environ.get("OPENAI_API_KEY")

    docs_text = ""
    sources = []
    for i, r in enumerate(retrieved):
        meta = r["metadata"].item() if isinstance(r["metadata"], np.ndarray) else r["metadata"]
        txt = meta.get("text") if isinstance(meta, dict) else meta["text"]
        src = meta.get("source") or meta.get("doc_id") or ""
        docs_text += f"\n---DOC {i+1} SOURCE: {src}---\n{txt[:1000]}\n"
        if src:
            sources.append(src)

    system_prompt = (
        "You are a helpful customer-support assistant. "
        "Use the provided documents to answer the customer's question. "
        "Give a concise direct answer. At the end, list the Sources (URLs)."
    )
    user_prompt = f"{docs_text}\n\nQuestion: {question}\n\nAnswer:"

    # ✅ OpenAI GPT-3.5 if key available
    if openai_api_key:
        try:
            client = OpenAI(api_key=openai_api_key)
            resp = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=512,
                temperature=0.0,
            )
            answer = resp.choices[0].message.content.strip()
            unique_sources = list(dict.fromkeys([s for s in sources if s]))
            return {"answer": answer, "sources": unique_sources, "model": "OpenAI GPT-3.5"}
        except Exception as e:
            print(f"⚠️ OpenAI failed: {e}. Falling back to local model.")

    # ✅ Local HuggingFace fallback
    generator = pipeline("text2text-generation", model="google/flan-t5-base")
    local_prompt = f"{system_prompt}\n\n{docs_text}\n\nQuestion: {question}\nAnswer:"
    result = generator(local_prompt, max_new_tokens=200, do_sample=False)
    answer = result[0]["generated_text"]

    unique_sources = list(dict.fromkeys([s for s in sources if s]))
    return {"answer": answer, "sources": unique_sources, "model": "Local flan-t5-base"}
