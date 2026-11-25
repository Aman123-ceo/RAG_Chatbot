# 🌐 RAG Chatbot v5 — Upload + Internet Learning

## NEW IN v5
Upload → Bot extracts TOPICS from YOUR file → Searches INTERNET (Wikipedia + DuckDuckGo) for those topics → Trains on BOTH → Answers ANY question type.

## FLOW
```
YOUR PDF/TXT
   ↓
TOPIC EXTRACTION (LLM or frequency)
   e.g. Doppler Effect, Speed Radar, Frequency
   ↓
INTERNET SEARCH (Wikipedia + DuckDuckGo)
   ↓  Fetch top 3 pages, clean, chunk
UPLOADED CHUNKS (ground truth, boosted) + INTERNET CHUNKS (enrichment)
   ↓  Embed → FAISS
QUESTION → Hybrid Retrieval → LLM answers with [Uploaded]/[Internet] citations
```

## QUESTION TYPES IT CAN ANSWER (auto-detected)
- Summarize → Summary prompt
- Important Topics → Topics prompt
- Make MCQs / Quiz → 5 MCQs + Answers + Explanation
- Viva Questions → Viva Q + Model Answers
- Explain Simply → Simple language + analogy
- Compare → Table comparison
- Real Applications → Real-life uses
- List All Topics → Exhaustive list
- Define / Formula → Definition + Formula + Example
- Any normal question → Precise Q&A

Try the 8 buttons above chat - same knowledge, 8 different answer types!

## RUN (venv)
```
python -m venv venv
.\venv\Scripts\Activate.ps1
.\venv\Scripts\python.exe -m pip install -r requirements.txt
streamlit run app.py
```
Or double-click setup.bat

## GROQ KEY (free, console.groq.com)
Needed for Summary/MCQ/Viva. Without it = Offline chunk paste only.

## FILES
- app.py — main app
- internet_enrich.py — topic extraction + Wikipedia/DDG search + fetch
- sample_data.txt, requirements.txt, setup.bat, README.md

## VIVA LINE
"My bot does not just answer from upload. It first understands what the upload is ABOUT (topics), then it self-learns from internet sources on those SAME topics, merges both into one FAISS knowledge base, and then answers any question type with source citations - so uploaded data stays ground truth, internet adds depth."
