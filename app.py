import streamlit as st
import re
import nltk
import requests
import pickle
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from streamlit_autorefresh import st_autorefresh

nltk.download('stopwords')

# Load model and vectorizer
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Text preprocessing
port_stem = PorterStemmer()
def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if word not in stopwords.words('english')]
    return ' '.join(stemmed_content)

# NewsAPI
API_KEY = "ac985774b8bb448fbd720422fd53b8fc"

def fetch_news():
    url = f"https://newsapi.org/v2/everything?q=india&language=en&pageSize=5&sortBy=publishedAt&apiKey={API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data.get("articles", [])
    else:
        return []

# Streamlit UI
st.set_page_config(page_title="TruthVerify", layout="wide")

# Custom CSS with updated design
st.markdown("""
    <style>
    html, body {
        font-family: 'Roboto', sans-serif;
        background-color: #f0f2f6;
    }
    .main {
        background-color: #f0f2f6;
    }
    h1, h2, h3 {
        color: #1e3a8a;
        font-family: 'Poppins', sans-serif;
    }
    .stTextArea textarea {
        background-color: #ffffff !important;
        border: none;
        border-radius: 0.75rem;
        padding: 1rem;
        font-size: 1rem;
        color: #333;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
    }
    .stTextArea textarea:focus {
        box-shadow: 0 4px 12px rgba(30,58,138,0.15);
    }
    .stButton>button {
        background-color: #1e3a8a;
        color: white;
        border-radius: 0.5rem;
        padding: 0.5rem 2rem;
        font-weight: 500;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #1e40af;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(30,58,138,0.2);
    }
    .news-card {
        background-color: #ffffff;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        border-radius: 1rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
        border-left: 5px solid #cbd5e1;
    }
    .news-card:hover {
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        transform: translateY(-3px);
    }
    .news-card.fake {
        border-left: 5px solid #ef4444;
    }
    .news-card.real {
        border-left: 5px solid #10b981;
    }
    .headline {
        font-size: 1.25rem;
        font-weight: 600;
        color: #111827;
        margin-bottom: 0.75rem;
    }
    .result-card {
        padding: 1.5rem;
        border-radius: 1rem;
        margin: 1.5rem 0;
        animation: fadeIn 0.5s ease-out;
    }
    .result-card.fake {
        background-color: #fee2e2;
        border: 1px solid #f87171;
    }
    .result-card.real {
        background-color: #d1fae5;
        border: 1px solid #34d399;
    }
    .confidence-meter {
        height: 10px;
        background-color: #e5e7eb;
        border-radius: 5px;
        margin-top: 0.5rem;
        overflow: hidden;
    }
    .confidence-fill {
        height: 100%;
        border-radius: 5px;
    }
    .confidence-fill.fake {
        background-color: #ef4444;
    }
    .confidence-fill.real {
        background-color: #10b981;
    }
    .sidebar-content {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 1rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        padding: 1.5rem;
        color: #6b7280;
        font-size: 0.875rem;
        border-top: 1px solid #e5e7eb;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .stSlider>div>div {
        background-color: #1e3a8a !important;
    }
    </style>
""", unsafe_allow_html=True)

# App header with logo emoji
col1, col2 = st.columns([1, 5])
with col1:
    st.markdown('<div style="font-size:3.5rem;text-align:center">🔍</div>', unsafe_allow_html=True)
with col2:
    st.title("TruthVerify: Advanced Fake News Detection")
    
st.markdown("<p style='color:#6b7280;margin-top:-10px;'>Powered by AI to identify misinformation</p>", unsafe_allow_html=True)

# Sidebar with improved design
with st.sidebar:
    st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
    st.markdown("### 🧠 About TruthVerify")
    st.info("This application uses advanced machine learning to detect fake news and misinformation in real time.")
    
    st.markdown("### 👨‍💻 Developers")
    st.markdown("""
    - **Neeraj Chandel**
    - **Khushi**
    
    Under the guidance of:  
    **Dr. Sunita Soni** (HOD CSE)
    """)
    
    st.markdown("### 🔧 How it works")
    st.write("1. Text preprocessing and cleaning")
    st.write("2. Feature extraction with NLP")
    st.write("3. ML classification algorithms")
    st.write("4. Real-time NewsAPI integration")
    st.markdown('</div>', unsafe_allow_html=True)

# Two tabs for different functionalities
tab1, tab2 = st.tabs(["✍️ Check News Article", "🌐 Live News Analysis"])

with tab1:
    st.markdown("### Submit a news article or headline to verify")
    st.markdown("<p style='color:#6b7280;font-size:0.9rem;'>Our AI will analyze the content and determine if it's likely to be fake or legitimate.</p>", 
                unsafe_allow_html=True)
    
    # User input with improved UI
    user_input = st.text_area("", placeholder="Paste news article or headline here...", height=150)
    
    col1, col2 = st.columns([1, 4])
    with col1:
        check_button = st.button("Analyze")
    
    # Analysis results with improved visual feedback
    if check_button:
        if user_input.strip() == "":
            st.warning("⚠️ Please enter some text to analyze.")
        else:
            with st.spinner("Analyzing content..."):
                # Process the text
                preprocessed_input = stemming(user_input)
                vectorized_input = vectorizer.transform([preprocessed_input])
                prediction = model.predict(vectorized_input)
                prob = model.predict_proba(vectorized_input).max()
                
                # Show results with better visualization
                if prediction[0] == 1:
                    st.markdown(f"""
                        <div class="result-card fake">
                            <h3 style="color:#b91c1c;margin:0">🚨 Fake News Detected</h3>
                            <p>Our analysis indicates this content is likely to be false or misleading.</p>
                            <p><strong>Confidence:</strong> {prob:.2%}</p>
                            <div class="confidence-meter">
                                <div class="confidence-fill fake" style="width:{prob*100}%"></div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                        <div class="result-card real">
                            <h3 style="color:#047857;margin:0">✅ Likely Authentic</h3>
                            <p>Our analysis indicates this content appears to be legitimate.</p>
                            <p><strong>Confidence:</strong> {prob:.2%}</p>
                            <div class="confidence-meter">
                                <div class="confidence-fill real" style="width:{prob*100}%"></div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)

with tab2:
    st.markdown("### Real-Time News Analysis")
    st.markdown("<p style='color:#6b7280;font-size:0.9rem;'>Live news headlines with automatic fact-checking.</p>", 
                unsafe_allow_html=True)
    
    # Refresh settings with better UI
    col1, col2 = st.columns([3, 1])
    with col1:
        refresh_interval = st.slider("Auto-refresh interval (seconds)", 0, 60, 30)
    with col2:
        if st.button("Refresh Now"):
            st.experimental_rerun()
    
    if refresh_interval > 0:
        st_autorefresh(interval=refresh_interval * 1000, key="auto-refresh")
    
    # News feed with improved card design
    with st.spinner("Fetching latest headlines..."):
        articles = fetch_news()
        
    if not articles:
        st.error("⚠️ Could not fetch news. Please check API key or try again.")
    else:
        for article in articles:
            title = article.get("title", "No title")
            description = article.get("description", "No description available.")
            source = article.get("source", {}).get("name", "Unknown source")
            url = article.get("url", "#")
            
            # Process the headline
            preprocessed = stemming(title)
            vectorized = vectorizer.transform([preprocessed])
            pred = model.predict(vectorized)
            prob = model.predict_proba(vectorized).max()
            
            card_class = "fake" if pred[0] == 1 else "real"
            emoji = "🚨" if pred[0] == 1 else "✅"
            verdict = "Potentially Fake" if pred[0] == 1 else "Likely Authentic"
            
            st.markdown(f"""
                <div class="news-card {card_class}">
                    <div class="headline">{title}</div>
                    <p style="color:#4b5563;margin-bottom:1rem;font-size:0.9rem;">{description[:150]}...</p>
                    <div style="display:flex;justify-content:space-between;align-items:center">
                        <div>
                            <span style="background-color:{'#fee2e2' if pred[0] == 1 else '#d1fae5'};
                                  padding:0.25rem 0.75rem;border-radius:9999px;font-size:0.8rem;
                                  color:{'#b91c1c' if pred[0] == 1 else '#047857'};font-weight:500;">
                                {emoji} {verdict} ({prob:.0%})
                            </span>
                        </div>
                        <div style="color:#6b7280;font-size:0.8rem;">Source: {source}</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

# Footer
st.markdown("""
    <div class="footer">
        <p>© 2025 TruthVerify - Fighting misinformation with AI</p>
        <p style="font-size:0.8rem;margin-top:0.5rem;">Powered by NewsAPI and Streamlit</p>
    </div>
""", unsafe_allow_html=True)
