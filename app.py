import os, json, numpy as np, faiss, streamlit as st
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
import google.generativeai as genai

# Use one of the supported models from list_models()
GEMINI_MODEL = "gemini-2.5-flash"

def load_index():
    with open("index/meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    docs = meta["docs"]
    info = meta["meta"]

    X = np.load("index/embeddings.npy")
    index = faiss.read_index("index/faiss.index")
    return index, X, docs, info

def embed_query(vectorizer, text):
    vec = vectorizer.transform([text]).toarray().astype("float32")
    faiss.normalize_L2(vec)
    return vec

def build_context(docs, info, indices):
    lines = []
    for idx in indices:
        src = info[idx]["source"]
        cid = info[idx]["chunk_id"]
        content = docs[idx].strip().replace("\n", " ")
        lines.append(f"[{src} #{cid}] {content}")
    return "\n\n".join(lines)

def answer_with_gemini(context, question):
    prompt = (
        "You are AvenirEdge's assistant. Answer only using the provided context.\n"
        "If the answer is not in the context, say you don't know.\n"
        "Cite sources as [filename #chunk]. Be concise.\n\n"
        f"Context:\n{context}\n\nQuestion:\n{question}"
    )
    model = genai.GenerativeModel(GEMINI_MODEL)
    response = model.generate_content(prompt)
    return response.text

def main():
    load_dotenv()
    gemini_key = os.getenv("GEMINI_API_KEY")
    if not gemini_key:
        st.error("GEMINI_API_KEY missing in .env")
        st.stop()

    genai.configure(api_key=gemini_key)

    st.set_page_config(page_title="AvenirEdge AI Assistant", page_icon="🤖")
    st.title("🤖 AvenirEdge AI Assistant")
    st.write("Ask anything about AvenirEdge courses and learning paths. Answers use only AvenirEdge content.")

    @st.cache_resource
    def cached_load():
        return load_index()
    index, X, docs, info = cached_load()

    # Load vectorizer with same vocab as build_index
    vectorizer = TfidfVectorizer(max_features=768)
    vectorizer.fit(docs)

    query = st.text_input("Your question")
    k = st.slider("Number of context chunks", 2, 8, 4)

    if st.button("Get answer") and query.strip():
        try:
            qvec = embed_query(vectorizer, query)
            scores, idxs = index.search(qvec.reshape(1, -1), k)
            top_indices = idxs[0].tolist()
            context = build_context(docs, info, top_indices)
            answer = answer_with_gemini(context, query)
            st.markdown("### Answer")
            st.write(answer)
            with st.expander("Show context"):
                st.code(context)
        except Exception as e:
            st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
