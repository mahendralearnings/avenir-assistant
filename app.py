import os, json, numpy as np, streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai

# Use one of the supported models from list_models()
GEMINI_MODEL = "gemini-2.5-flash"

def load_index():
    with open("index/meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    docs = meta["docs"]
    info = meta["meta"]
    X = np.load("index/embeddings.npy")
    return X, docs, info

def embed_query(vectorizer, text):
    vec = vectorizer.transform([text]).toarray().astype("float32")
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
    api_key = st.secrets.get("GEMINI_API_KEY", "")
    if not api_key:
        st.error("GEMINI_API_KEY missing in Streamlit Secrets")
        st.stop()

    genai.configure(api_key=api_key)

    st.set_page_config(page_title="AvenirEdge AI Assistant", page_icon="🤖")
    st.title("🤖 AvenirEdge AI Assistant")
    st.write("Ask anything about AvenirEdge courses and learning paths. Answers use only AvenirEdge content.")

    @st.cache_resource
    def cached_load():
        return load_index()
    X, docs, info = cached_load()

    vectorizer = TfidfVectorizer(max_features=768)
    vectorizer.fit(docs)

    query = st.text_input("Your question")
    k = st.slider("Number of context chunks", 2, 8, 4)

    if st.button("Get answer") and query.strip():
        try:
            qvec = embed_query(vectorizer, query)
            sims = cosine_similarity(qvec, X)[0]
            top_indices = np.argsort(sims)[-k:][::-1]
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
