# Install required packages if not installed:
# pip install streamlit requests pandas transformers beautifulsoup4 lxml scikit-learn

import requests
import pandas as pd
from bs4 import BeautifulSoup
from transformers import pipeline
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer

# ------------------------------
# 1. Fetch papers from arXiv
# ------------------------------
def fetch_arxiv_papers(query, max_results=10):
    url = f'http://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results={max_results}'
    response = requests.get(url)

    # Use XML parser (lxml recommended)
    try:
        soup = BeautifulSoup(response.text, 'lxml-xml')
    except:
        soup = BeautifulSoup(response.text, 'html.parser')

    entries = soup.find_all('entry')
    papers = []
    for entry in entries:
        papers.append({
            'title': entry.title.text.strip().replace('\n', ' '),
            'authors': ', '.join([author.find('name').text for author in entry.find_all('author')]),
            'summary': entry.summary.text.strip().replace('\n', ' '),
            'link': entry.id.text
        })
    return papers

# ------------------------------
# 2. Lightweight Summarizer
# ------------------------------
def get_summarizer():
    try:
        # DistilBART is much lighter than facebook/bart-large-cnn
        return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=-1)
    except Exception as e:
        print("Falling back to T5-small due to error:", e)
        return pipeline("summarization", model="t5-small", device=-1)

summarizer = get_summarizer()

def summarize_text(text):
    try:
        summary = summarizer(text, max_length=80, min_length=25, do_sample=False)
        return summary[0]['summary_text']
    except:
        return text[:200] + "..."

# ------------------------------
# 3. Keyword Extraction (Simple TF-IDF)
# ------------------------------
def extract_keywords(text, top_n=5):
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform([text])
    indices = X.toarray()[0].argsort()[-top_n:][::-1]
    keywords = [vectorizer.get_feature_names_out()[i] for i in indices]
    return ', '.join(keywords)

# ------------------------------
# 4. Streamlit App
# ------------------------------
st.title("📚 Autonomous Research Assistant")
query = st.text_input("Enter Research Topic", "Artificial Intelligence")
max_results = st.slider("Number of Papers to Fetch", 5, 20, 10)

if st.button("Fetch & Summarize"):
    papers = fetch_arxiv_papers(query, max_results)
    
    data = []
    for paper in papers:
        summary = summarize_text(paper['summary'])
        keywords = extract_keywords(paper['summary'])
        data.append({
            'Title': paper['title'],
            'Authors': paper['authors'],
            'Summary': summary,
            'Keywords': keywords,
            'Link': paper['link']
        })
    
    df = pd.DataFrame(data)
    st.dataframe(df)

    # Pretty printing with links
    for i, row in df.iterrows():
        st.markdown(f"### [{row['Title']}]({row['Link']})")
        st.write(f"**Authors:** {row['Authors']}")
        st.write(f"**Summary:** {row['Summary']}")
        st.write(f"**Keywords:** {row['Keywords']}")
        st.markdown("---")
