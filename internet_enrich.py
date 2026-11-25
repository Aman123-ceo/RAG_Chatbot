"""
internet_enrich.py - Bot learns from INTERNET using topics from YOUR upload

1. extract_topics(uploaded_text): finds important keywords/topics in YOUR file
   - If Groq key available: LLM extracts 3-5 topics (best)
   - Else: frequency-based keyword extraction (no key needed)

2. search_internet(topics): Wikipedia + DuckDuckGo
   - Wikipedia: free, no key, reliable
   - DuckDuckGo: free, no key, via ddgs library or wikipedia fallback

3. fetch_url_text(url): downloads page, strips HTML, returns clean text

All free, no paid API needed.
"""
import re, collections, urllib.parse
try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    requests = BeautifulSoup = None

STOP = set("what is the and for with from this that have been will would about into over under after before between through during each other more most other some such only own same than too very can just should now also using used use used based between including".split())

def extract_topics(text, groq_key=None, model_name="llama-3.3-70b-versatile"):
    """Find 3-5 main topics in uploaded text."""
    # Try LLM first (most accurate)
    if groq_key:
        try:
            from langchain_groq import ChatGroq
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_core.output_parsers import StrOutputParser
            llm = ChatGroq(groq_api_key=groq_key, model_name=model_name, temperature=0)
            prompt = ChatPromptTemplate.from_template(
                "Extract EXACTLY 3 to 5 main topics from this document as a comma-separated list. "
                "Only topics, no explanation. Example: Doppler Effect, Speed Radar, Frequency Shift\n\nDocument (first 4000 chars):\n{text}\n\nTopics:")
            chain = prompt | llm | StrOutputParser()
            out = chain.invoke({"text": text[:4000]})
            topics = [t.strip() for t in out.replace("\n", ",").split(",") if len(t.strip())>2]
            topics = topics[:5]
            if topics: return topics
        except Exception as e:
            print("LLM topic extraction failed, using frequency method:", e)
    # Fallback: frequency - most repeated meaningful words/phrases
    words = re.findall(r"[A-Za-z][A-Za-z\- ]{2,30}", text)
    # Bigrams (2-word phrases) are better topics than single words
    bigrams = []
    tokens = re.findall(r"[a-z]+", text.lower())
    for i in range(len(tokens)-1):
        a,b = tokens[i], tokens[i+1]
        if a not in STOP and b not in STOP and len(a)>2 and len(b)>2:
            bigrams.append(f"{a} {b}")
    freq = collections.Counter(bigrams + [w.lower() for w in words if w.lower() not in STOP and len(w)>3])
    topics=[]
    for phrase, _ in freq.most_common(20):
        phrase=phrase.strip().title()
        if phrase and phrase not in topics and len(phrase)>3:
            topics.append(phrase)
        if len(topics)>=5: break
    return topics or ["General Knowledge"]

def search_internet(topics, max_per_topic=1, max_total=3):
    """Search Wikipedia + DuckDuckGo for each topic. Returns [{title, url, snippet}]"""
    results=[]
    # 1. Wikipedia first - most reliable, no blocking
    for topic in topics:
        if len(results)>=max_total: break
        try:
            # Wikipedia API search
            if requests:
                r = requests.get("https://en.wikipedia.org/w/api.php",
                    params={"action":"query","list":"search","srsearch":topic,"format":"json","srlimit":1},
                    timeout=10, headers={"User-Agent":"RAG-Chatbot/1.0"})
                data=r.json()
                if data.get("query",{}).get("search"):
                    item=data["query"]["search"][0]
                    title=item["title"]
                    url="https://en.wikipedia.org/wiki/"+urllib.parse.quote(title.replace(" ","_"))
                    results.append({"title":title,"url":url,"snippet":re.sub(r"<.*?>","",item.get("snippet",""))})
                    continue
        except Exception as e:
            print("Wikipedia search failed:", e)
    # 2. DuckDuckGo via ddgs if still need more
    if len(results)<max_total:
        try:
            from ddgs import DDGS
            for topic in topics:
                if len(results)>=max_total: break
                try:
                    with DDGS() as ddgs:
                        for hit in ddgs.text(topic, max_results=max_per_topic):
                            if any(hit["href"] in r["url"] for r in results): continue
                            results.append({"title":hit["title"],"url":hit["href"],"snippet":hit.get("body","")})
                except Exception as e:
                    print("DDGS failed for", topic, e)
        except ImportError:
            print("ddgs not installed - using Wikipedia only. pip install ddgs")
    return results[:max_total]

def fetch_url_text(url, max_chars=8000):
    """Download page, strip HTML, return clean text."""
    if not requests: return ""
    try:
        r=requests.get(url, timeout=15, headers={"User-Agent":"Mozilla/5.0 RAG-Chatbot"})
        if r.status_code!=200: return ""
        soup=BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script","style","nav","header","footer","aside","form"]): tag.decompose()
        text=soup.get_text(separator=" ")
        text=re.sub(r"\s+"," ",text).strip()
        return text[:max_chars]
    except Exception as e:
        print("Fetch failed", url, e); return ""
