"""
RAG Chatbot v5 - UPLOADED DATA + INTERNET TRAINING
Upload -> Bot finds TOPICS in your file -> Searches INTERNET for those topics
-> Trains on BOTH (Uploaded + Internet) -> Answers ANY type of question

Flow:
  1. UPLOAD your PDF/TXT/CSV
  2. TRAIN: Chunk + Embed uploaded data
  3. INTERNET ENRICH (optional, toggle ON):
     - Extract keywords/topics from uploaded text (TF + LLM or frequency)
     - Search Wikipedia + DuckDuckGo for those topics
     - Fetch top 3-5 pages, clean, chunk, embed as "internet" source
  4. CHAT: Hybrid Retrieval (Uploaded 70% weight + Internet) + Intent-aware LLM
     Answers: Summary, Important Topics, Explain, MCQ, Compare, Viva Q&A, Application, etc.
     EVERY answer cites: [Uploaded] or [Internet] source.

No internet? Works 100% on uploaded data only.
No Groq key? Internet + retrieval still works, LLM summary needs key (else offline chunks).
"""
import os, io, re, time, hashlib, collections
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

from langchain_text_splitters import RecursiveCharacterTextSplitter
# BULLETPROOF EMBEDDINGS: tries 3 ways, never crashes with ModuleNotFoundError
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
    except ImportError:
        HuggingFaceEmbeddings = None  # fallback to direct SentenceTransformer below
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import pandas as pd
try:
    import PyPDF2
except ImportError: PyPDF2=None
try:
    import docx
except ImportError: docx=None

# Internet helpers - in internet_enrich.py
try:
    from internet_enrich import extract_topics, search_internet, fetch_url_text
except ImportError:
    extract_topics = search_internet = fetch_url_text = None

st.set_page_config(page_title="RAG Chatbot v5 - Upload + Internet Learning", page_icon="🌐", layout="wide")
st.markdown("""<style>
.main-title{font-size:36px;font-weight:800;background:linear-gradient(90deg,#6c5ce7,#00b894,#f39c12);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.card{background:#f1f2ff;padding:12px;border-radius:12px}
.source-u{background:#d4edda;padding:8px;border-radius:8px;border-left:4px solid #28a745;font-size:13px;margin:5px 0}
.source-i{background:#cce5ff;padding:8px;border-radius:8px;border-left:4px solid #007bff;font-size:13px;margin:5px 0}
.chunk{background:#eef2ff;padding:10px;border-radius:10px;margin:6px 0;font-size:13px;border-left:4px solid #6c5ce7}
</style>""", unsafe_allow_html=True)

for k,v in {"vectorstore":None,"messages":[],"trained_files":[],"internet_files":[],"doc_count":0,"chunk_count":0,"uploaded_text":"","topics":[]}.items():
    if k not in st.session_state: st.session_state[k]=v

# ---------- SIDEBAR ----------
with st.sidebar:
    st.header("⚙️ Settings")
    groq_key = st.text_input("Groq API Key (for smart answers)", value=os.getenv("GROQ_API_KEY",""), type="password")
    if not groq_key: st.warning("No key = Offline Mode (chunk paste only)")
    else: st.success("✅ Groq ON")
    model_name = st.selectbox("LLM Model", ["llama-3.3-70b-versatile","llama-3.1-8b-instant"])
    st.divider()
    st.subheader("🌐 Internet Learning")
    use_internet = st.toggle("Enable Internet Training", value=True, help="Bot searches internet for topics found in YOUR upload and learns from it too.")
    max_internet = st.slider("Internet sources per upload", 1, 5, 3)
    st.caption("Sources: Wikipedia + DuckDuckGo. Only topics FROM your upload are searched - bot does NOT go off-topic.")
    st.divider()
    st.subheader("📊 Knowledge Base")
    if st.session_state.vectorstore is None: st.error("❌ NOT TRAINED")
    else:
        st.success(f"✅ TRAINED\n📄 Uploaded: {st.session_state.doc_count} files\n🌐 Internet: {len(st.session_state.internet_files)} sources\n🧩 Total chunks: {st.session_state.chunk_count}")
    if st.session_state.topics: st.write("**Topics found:**", ", ".join(st.session_state.topics[:8]))
    if st.button("🗑️ Clear All"):
        for k in ["vectorstore","messages","trained_files","internet_files","doc_count","chunk_count","uploaded_text","topics"]:
            st.session_state[k]= [] if k in ("messages","trained_files","internet_files","topics") else (0 if k in ("doc_count","chunk_count") else "" if k=="uploaded_text" else None)
        st.rerun()

@st.cache_resource(show_spinner="Loading ML embedding model...")
def get_embeddings():
    if HuggingFaceEmbeddings is not None:
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    # FALLBACK: direct sentence-transformers, FAISS-compatible wrapper
    from sentence_transformers import SentenceTransformer
    class DirectEmbeddings:
        def __init__(self):
            self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        def embed_documents(self, texts):
            return self.model.encode(texts, show_progress_bar=False).tolist()
        def embed_query(self, text):
            return self.model.encode([text], show_progress_bar=False)[0].tolist()
    return DirectEmbeddings()

def read_file(name, data):
    n=name.lower()
    try:
        if n.endswith(".pdf"):
            r=PyPDF2.PdfReader(io.BytesIO(data)); return "\n".join([(p.extract_text() or "") for p in r.pages])
        if n.endswith(".docx"):
            d=docx.Document(io.BytesIO(data)); return "\n".join([p.text for p in d.paragraphs])
        if n.endswith(".csv"):
            df=pd.read_csv(io.BytesIO(data)); return f"CSV {name} columns {list(df.columns)}\n\n"+df.to_string(index=False)
        if n.endswith(".xlsx"):
            df=pd.read_excel(io.BytesIO(data)); return df.to_string(index=False)
        return data.decode("utf-8", errors="ignore")
    except Exception as e: return f"[ERROR {name}: {e}]"

def clean_text(t):
    t=re.sub(r'\s+',' ',t); t=re.sub(r'Page \d+','',t,flags=re.I); return t.strip()

def keyword_score(q, chunk):
    qq=set(re.findall(r'\w+', q.lower())); stop={'what','is','the','a','an','of','in','to','for','give','me','summarize','summarise','important','topics','list','explain','tell','define'}
    qq={w for w in qq if w not in stop and len(w)>2}
    if not qq: return 0
    c=set(re.findall(r'\w+', chunk.lower())); return len(qq & c)/len(qq)

def hybrid_retrieve(question, top_k=6):
    # Retrieve more, then re-rank, prefer Uploaded for factual questions, Internet for explain/extra
    docs_scores = st.session_state.vectorstore.similarity_search_with_score(question, k=top_k*2)
    ranked=[]
    for doc, dist in docs_scores:
        vec=max(0,1-dist/2); kw=keyword_score(question, doc.page_content)
        # Boost uploaded sources slightly (they are ground truth)
        boost=0.05 if doc.metadata.get("kind")=="uploaded" else 0
        final=0.65*vec+0.30*kw+boost
        ranked.append((doc,final,vec,kw))
    ranked.sort(key=lambda x: x[1], reverse=True)
    seen=set(); out=[]
    for doc,f,v,k in ranked:
        key=doc.page_content[:80]
        if key in seen: continue
        seen.add(key); out.append((doc,f,v,k))
        if len(out)>=top_k: break
    return out

def detect_intent(q):
    q=q.lower()
    if any(w in q for w in ["summar","summary","gist","overall"]): return "SUMMARY"
    if any(w in q for w in ["important","key topics","main topics","must know"]): return "TOPICS"
    if any(w in q for w in ["mcq","multiple choice","quiz","test me","practice question"]): return "MCQ"
    if any(w in q for w in ["compare","difference","vs ","versus","distinguish"]): return "COMPARE"
    if any(w in q for w in ["explain","elaborate","in detail","in simple","easy language"]): return "EXPLAIN"
    if any(w in q for w in ["viva","interview","likely question","exam question"]): return "VIVA"
    if any(w in q for w in ["application","real life","example","use case","where is it used"]): return "APPLICATION"
    if any(w in q for w in ["formula","define","definition","what is","what are"]): return "DEFINE"
    if any(w in q for w in ["list","all topics","chapters","enumerate"]): return "LIST"
    return "QA"

PROMPTS={
"SUMMARY":"Summarize ENTIRE uploaded data + internet enrichment. Structure: Overview, Main Points (cover every section), Conclusion. Use ONLY context. Mark [Uploaded] vs [Internet] facts.",
"TOPICS":"Extract TOP 5-8 IMPORTANT TOPICS from context. For each: 1-2 line explanation + WHY important + 1 exam point. Rank by importance. Mark [Uploaded]/[Internet].",
"MCQ":"Create 5 MCQs from context. Each: Question, Options A-D, Correct Answer + 1-line Explanation. Mix easy + hard. Mark [Uploaded]/[Internet] per Q.",
"COMPARE":"Compare the two things asked, in a TABLE: Feature | First | Second. Then 3-line conclusion. Use ONLY context.",
"EXPLAIN":"Explain in SIMPLE, easy language like teaching a beginner. Definition -> Simple Example -> Real-life analogy -> Key points. Use ONLY context.",
"VIVA":"Act as viva examiner. Give 5 Likely Viva Questions + Model Answers (2-3 lines each) from context. Then 1 tricky question + answer.",
"APPLICATION":"Give Real-life Applications + Examples from context + internet. For each: Where used, How it works, Example. Mark sources.",
"DEFINE":"Give precise Definition (as in uploaded data) + Formula if any + 1 Example + 1 line in simple words. If not in context, say so.",
"LIST":"List ALL topics/headings/sections in context as numbered list, exhaustive.",
"QA":"Answer question precisely from context. Short answer if asked short, detailed if asked detail. If not in context say: 'I could not find this in uploaded + internet data.' Do NOT invent. Cite [Uploaded]/[Internet].",
}

def answer(question, groq_key, model_name):
    ranked=hybrid_retrieve(question, 6)
    docs=[r[0] for r in ranked]
    # Label context by kind so LLM can cite
    parts=[]
    for d in docs:
        kind="Uploaded" if d.metadata.get("kind")=="uploaded" else "Internet"
        parts.append(f"[{kind} | {d.metadata.get('source')}] {d.page_content}")
    context="\n\n---\n\n".join(parts)
    intent=detect_intent(question)
    history="\n".join([f"{m['role']}: {m['content'][:250]}" for m in st.session_state.messages[-4:]])
    debug=[{"source":d.metadata.get("source"),"kind":d.metadata.get("kind"),"chunk":d.metadata.get("chunk"),"final":round(float(f),3),"preview":d.page_content[:180]} for d,f,v,k in ranked]
    if not groq_key:
        ans=f"📌 **Offline Mode — Top chunks for '{question}' (Intent: {intent})**\n\n"
        for d in docs: ans+=f"**[{'Uploaded' if d.metadata.get('kind')=='uploaded' else 'Internet'}] {d.metadata.get('source')}**\n{d.page_content[:700]}\n\n---\n\n"
        ans+="*Add Groq key for Summary/MCQ/Viva answers.*"
        return ans, docs, debug, intent
    try:
        llm=ChatGroq(groq_api_key=groq_key, model_name=model_name, temperature=0.1)
        prompt=ChatPromptTemplate.from_template("{full}")
        chain=prompt|llm|StrOutputParser()
        full=f"""{PROMPTS[intent]}

CHAT HISTORY:
{history}

CONTEXT (Uploaded + Internet):
{context}

QUESTION: {question}

ANSWER (structured, cite [Uploaded] or [Internet] for each point):"""
        return chain.invoke({"full":full}), docs, debug, intent
    except Exception as e:
        return f"Groq Error: {e}\n\nFallback:\n{context[:1500]}", docs, debug, intent

# ---------- HEADER ----------
st.markdown('<div class="main-title">🌐 RAG Chatbot v5 — Upload + Internet Learning</div>', unsafe_allow_html=True)
st.write("**Upload your file → Bot learns YOUR data + searches INTERNET for same topics → Answers ANY question type** — Summary, Topics, MCQ, Explain, Compare, Viva, Applications.")
st.info("💡 **Example:** Upload Doppler notes → Bot finds topics `Doppler Effect, Radar, Frequency` → Searches Wikipedia/DuckDuckGo for them → Trains on both → Ask `Explain Doppler with real-life examples + MCQs`")

left, right = st.columns([1, 1.6])
with left:
    st.subheader("📤 Step 1: Upload")
    uploaded=st.file_uploader("Upload PDF / TXT / CSV / Excel / DOCX", type=["pdf","txt","csv","xlsx","docx","md","py","java"], accept_multiple_files=True)
    if uploaded:
        for f in uploaded: st.write(f"- {f.name} ({len(f.getvalue())/1024:.1f} KB)")
    st.subheader("🧠 Step 2: Train + Internet Learn")
    if st.button("🚀 TRAIN — Upload + Internet", type="primary", use_container_width=True):
        if not uploaded: st.warning("Upload at least 1 file!")
        else:
            with st.spinner("Step 1/3: Reading & chunking YOUR upload..."):
                texts, metas=[], []
                full_text=""
                for f in uploaded:
                    if any(t["name"]==f.name for t in st.session_state.trained_files): st.write(f"⏭️ Already trained: {f.name}"); continue
                    raw=read_file(f.name, f.getvalue()); text=clean_text(raw); full_text+=text+"\n"
                    splitter=RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=150, separators=["\n\n","\n",". "," ",""])
                    chunks=splitter.split_text(text)
                    for i,ch in enumerate(chunks): texts.append(ch); metas.append({"source":f.name,"chunk":i,"kind":"uploaded"})
                    st.session_state.trained_files.append({"name":f.name,"chunks":len(chunks)}); st.session_state.doc_count+=1
                    st.write(f"✅ Uploaded: {f.name} → {len(chunks)} chunks")
                st.session_state.uploaded_text=full_text
                if texts:
                    emb=get_embeddings()
                    if st.session_state.vectorstore is None: st.session_state.vectorstore=FAISS.from_texts(texts, emb, metadatas=metas)
                    else: st.session_state.vectorstore.add_texts(texts, metadatas=metas)
                    st.session_state.chunk_count+=len(texts)
            # INTERNET STEP
            if use_internet and full_text and extract_topics:
                with st.spinner("Step 2/3: Finding topics in YOUR upload → Searching internet..."):
                    topics=extract_topics(full_text, groq_key, model_name)
                    st.session_state.topics=topics
                    st.write(f"🔍 Topics found in YOUR file: **{', '.join(topics)}**")
                    results=search_internet(topics, max_per_topic=1, max_total=max_internet)
                    st.write(f"🌐 Found {len(results)} internet sources")
                    for r in results: st.write(f"- 🌐 {r['title']} — {r['url'][:70]}")
                with st.spinner("Step 3/3: Learning from internet sources (chunk + embed)..."):
                    itexts, imetas=[], []
                    for r in results:
                        if any(t["name"]==r["title"] for t in st.session_state.internet_files): continue
                        txt=fetch_url_text(r["url"])
                        if len(txt)<300: continue
                        splitter=RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=150)
                        chunks=splitter.split_text(clean_text(txt))
                        for i,ch in enumerate(chunks): itexts.append(ch); imetas.append({"source":r["title"],"chunk":i,"kind":"internet","url":r["url"]})
                        st.session_state.internet_files.append({"name":r["title"],"chunks":len(chunks),"url":r["url"]})
                        st.write(f"✅ Internet: {r['title']} → {len(chunks)} chunks")
                    if itexts:
                        emb=get_embeddings(); st.session_state.vectorstore.add_texts(itexts, metadatas=imetas)
                        st.session_state.chunk_count+=len(itexts)
            st.success(f"🎉 TRAINED! Total {st.session_state.chunk_count} chunks (Uploaded + Internet). Now ask ANY question type!")
            st.rerun()
    if st.button("📄 Load Sample Data (Doppler + Internet)"):
        sample=open("sample_data.txt",encoding="utf-8").read() if os.path.exists("sample_data.txt") else ""
        st.session_state.uploaded_text=sample
        splitter=RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=150)
        chunks=splitter.split_text(clean_text(sample)); emb=get_embeddings()
        metas=[{"source":"sample_data.txt","chunk":i,"kind":"uploaded"} for i in range(len(chunks))]
        if st.session_state.vectorstore is None: st.session_state.vectorstore=FAISS.from_texts(chunks, emb, metadatas=metas)
        else: st.session_state.vectorstore.add_texts(chunks, metadatas=metas)
        st.session_state.chunk_count+=len(chunks); st.session_state.doc_count+=1
        st.session_state.trained_files.append({"name":"sample_data.txt","chunks":len(chunks)})
        st.session_state.topics=["Doppler Effect","Speed Radar","Frequency","Java"]
        st.success(f"✅ Sample trained: {len(chunks)} chunks. Now use Internet toggle + Ask below."); st.rerun()
    if st.session_state.trained_files:
        st.subheader("📚 Uploaded Files")
        for t in st.session_state.trained_files: st.write(f"✅ {t['name']} — {t['chunks']} chunks")
    if st.session_state.internet_files:
        st.subheader("🌐 Internet Sources Learned")
        for t in st.session_state.internet_files: st.write(f"🌐 {t['name']} — {t['chunks']} chunks")

with right:
    st.subheader("💬 Step 3: Ask ANY Type of Question")
    st.caption("Try the buttons — each gives a DIFFERENT answer type from SAME knowledge:")
    b1,b2,b3,b4=st.columns(4)
    if b1.button("📝 Summarize"): st.session_state.test_q="summarize the uploaded data"
    if b2.button("⭐ Important Topics"): st.session_state.test_q="important topics in the uploaded data"
    if b3.button("❓ Make 5 MCQs"): st.session_state.test_q="make 5 MCQs from the uploaded data with answers"
    if b4.button("🎤 Viva Questions"): st.session_state.test_q="give me viva questions and answers from this data"
    b5,b6,b7,b8=st.columns(4)
    if b5.button("💡 Explain Simply"): st.session_state.test_q="explain the main concept in simple easy language with real-life example"
    if b6.button("⚖️ Compare"): st.session_state.test_q="compare the main concepts in this data in a table"
    if b7.button("🌍 Real Applications"): st.session_state.test_q="what are real-life applications of this data?"
    if b8.button("📋 List All Topics"): st.session_state.test_q="list all topics in this data"
    if st.session_state.get("test_q"):
        prompt=st.session_state.test_q; st.session_state.test_q=None
    else:
        prompt=st.chat_input("Ask anything... e.g. Explain Doppler + MCQs + Viva")
    if st.session_state.vectorstore is None: st.warning("👈 Upload → TRAIN first, or Load Sample Data.")
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("debug"):
                with st.expander("🔍 Sources & Retrieved Chunks"):
                    for d in msg["debug"]:
                        cls="source-u" if d["kind"]=="uploaded" else "source-i"
                        st.markdown(f"<div class='{cls}'>{'📄 [Uploaded]' if d['kind']=='uploaded' else '🌐 [Internet]'} {d['source']} | Score {d['final']}<br>{d['preview']}...</div>", unsafe_allow_html=True)
    if prompt:
        if st.session_state.vectorstore is None: st.error("Train first!")
        else:
            st.session_state.messages.append({"role":"user","content":prompt})
            with st.chat_message("user"): st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner("🔍 Searching Uploaded + Internet knowledge..."):
                    ans, docs, debug, intent = answer(prompt, groq_key, model_name)
                    st.markdown(f"**Detected Type: `{intent}`**")
                    st.markdown(ans)
                    with st.expander("🔍 Sources — Uploaded vs Internet"):
                        for d in debug:
                            cls="source-u" if d["kind"]=="uploaded" else "source-i"
                            st.markdown(f"<div class='{cls}'>{'📄 [Uploaded]' if d['kind']=='uploaded' else '🌐 [Internet]'} {d['source']} | Score {d['final']}<br>{d['preview']}...</div>", unsafe_allow_html=True)
                    st.session_state.messages.append({"role":"assistant","content":ans,"debug":debug})
st.divider()
st.caption("v5: Uploaded (ground truth) + Internet (enrichment) | Hybrid retrieval | Intent-aware prompts | Every answer cites source")
