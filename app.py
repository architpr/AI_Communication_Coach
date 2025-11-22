import streamlit as st
import spacy
import language_tool_python
from sentence_transformers import SentenceTransformer, util
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import re
from datetime import datetime

# --- 1. CONFIGURATION & STYLE ---
st.set_page_config(
    page_title="Nirmaan AI Coach", 
    page_icon="🎙️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a "Product-Like" feel
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    h1 { color: #2c3e50; font-family: 'Helvetica Neue', sans-serif; }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    .chip-found {
        background-color: #d4edda;
        color: #155724;
        padding: 5px 10px;
        border-radius: 15px;
        margin: 2px;
        display: inline-block;
        font-size: 0.9em;
        border: 1px solid #c3e6cb;
    }
    .chip-missing {
        background-color: #f8d7da;
        color: #721c24;
        padding: 5px 10px;
        border-radius: 15px;
        margin: 2px;
        display: inline-block;
        font-size: 0.9em;
        border: 1px solid #f5c6cb;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# --- 2. LOAD AI MODELS (CACHED) ---
@st.cache_resource
def load_nlp():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        from spacy.cli import download
        download("en_core_web_sm")
        return spacy.load("en_core_web_sm")

@st.cache_resource
def load_similarity_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def load_grammar_tool():
    try:
        return language_tool_python.LanguageTool('en-US')
    except Exception as e:
        return None

# --- 3. STRICT RUBRIC LOGIC ---

def analyze_content_strict(text):
    """
    Weightage: 40% Total
    - Salutation: 5%
    - Keywords: 30%
    - Flow: 5%
    """
    text_lower = text.lower()
    
    # 1. Salutation (5 pts)
    salutations = ['hello', 'hi', 'good morning', 'good afternoon', 'good evening', 'namaste']
    has_salutation = any(s in text_lower for s in salutations)
    score_salutation = 5 if has_salutation else 0

    # 2. Flow (5 pts) - Logic: Salutation -> Name -> Closing
    # Find indices of key components
    try:
        idx_salutation = min([text_lower.find(s) for s in salutations if s in text_lower]) if has_salutation else -1
        
        name_indicators = ['my name', 'myself', 'i am']
        idx_name = -1
        for n in name_indicators:
            if n in text_lower:
                idx_name = text_lower.find(n)
                break
        
        closing_indicators = ['thank', 'that\'s all', 'listening']
        idx_closing = -1
        for c in closing_indicators:
            if c in text_lower:
                idx_closing = text_lower.find(c)
                break
        
        # Logic: Salutation must come before Name, Name must come before Closing
        is_flow_correct = False
        if idx_salutation != -1 and idx_name != -1 and idx_closing != -1:
            if idx_salutation < idx_name < idx_closing:
                is_flow_correct = True
        elif idx_name != -1 and idx_closing != -1:
             if idx_name < idx_closing:
                 is_flow_correct = True # Partial credit logic if salutation missing but flow ok
        
        score_flow = 5 if is_flow_correct else 0
    except:
        score_flow = 0 # Fallback

    # 3. Keywords (30 pts) - Derived from Excel List
    required_keywords = {
        "Name": ['name', 'myself', 'i am'],
        "Age": ['years old', 'age'],
        "Class/School": ['class', 'grade', 'studying', 'school', 'college'],
        "Family": ['family', 'mother', 'father', 'parents', 'siblings', 'brother', 'sister'],
        "Hobbies": ['hobby', 'hobbies', 'playing', 'reading', 'dancing', 'singing', 'drawing', 'interest', 'enjoy'],
        "Goal": ['goal', 'aim', 'ambition', 'become a', 'want to be'],
        "Unique Point": ['fact', 'special', 'unique', 'don\'t know about me']
    }
    
    found_sections = []
    missing_sections = []
    
    for category, keywords in required_keywords.items():
        if any(k in text_lower for k in keywords):
            found_sections.append(category)
        else:
            missing_sections.append(category)
    
    # Calculate Score: (Found / Total Categories) * 30
    total_categories = len(required_keywords)
    score_keywords = (len(found_sections) / total_categories) * 30

    total_content_score = score_salutation + score_flow + score_keywords

    return {
        "total_score": total_content_score,
        "breakdown": {
            "Salutation (5)": score_salutation,
            "Flow (5)": score_flow,
            "Keywords (30)": round(score_keywords, 1)
        },
        "found_sections": found_sections,
        "missing_sections": missing_sections,
        "is_flow_correct": is_flow_correct
    }

def analyze_speech_rate_strict(text, duration):
    """
    Weightage: 10%
    - Ideal (111-140 WPM): 10
    - Slow (81-110 WPM): 6
    - Too Slow (<80 WPM): 2
    - Fast (>140 WPM): 6 (Assumed based on Slow penalty)
    """
    if duration <= 0: return {"score": 0, "wpm": 0, "feedback": "Invalid"}
    
    words = len(text.split())
    minutes = duration / 60
    wpm = words / minutes
    
    if 111 <= wpm <= 140:
        score = 10
        feedback = "Ideal Pace (111-140 WPM)"
    elif 81 <= wpm <= 110:
        score = 6
        feedback = "Slow (81-110 WPM)"
    elif wpm < 80:
        score = 2
        feedback = "Too Slow (<80 WPM)"
    else:
        # > 140 WPM
        score = 6
        feedback = "Fast (>140 WPM)"
        
    return {"score": score, "wpm": int(wpm), "feedback": feedback}

def analyze_grammar_strict(text):
    """
    Weightage: 20% Total
    - Grammar Errors: 10%
      Formula: 1 - min(errors per 100 words / 10, 1) -> Scaled to 10
    - Vocabulary (TTR): 10%
      >0.9: 10 | 0.7-0.89: 8 | 0.5-0.69: 6 | 0.3-0.49: 4 | <0.3: 2
    """
    # 1. Grammar Logic
    tool = load_grammar_tool()
    words = text.split()
    word_count = len(words)
    
    if not tool or word_count == 0:
        score_grammar = 10
        matches = []
        error_count = 0
    else:
        matches = tool.check(text)
        error_count = len(matches)
        # Formula from Excel: 1 - min(errors_per_100/10, 1)
        errors_per_100 = (error_count / word_count) * 100
        calc_val = min(errors_per_100 / 10, 1)
        score_grammar = (1 - calc_val) * 10 # Scale 0-1 to 0-10

    # 2. Vocabulary Logic (TTR)
    text_lower = text.lower()
    words_lower = text_lower.split()
    if not words_lower:
        ttr = 0
    else:
        ttr = len(set(words_lower)) / len(words_lower)
        
    if ttr >= 0.9: score_vocab = 10
    elif ttr >= 0.7: score_vocab = 8
    elif ttr >= 0.5: score_vocab = 6
    elif ttr >= 0.3: score_vocab = 4
    else: score_vocab = 2
    
    return {
        "total_score": score_grammar + score_vocab,
        "grammar_score": round(score_grammar, 1),
        "vocab_score": score_vocab,
        "error_count": error_count,
        "matches": matches,
        "ttr": round(ttr, 2)
    }

def analyze_clarity_strict(text):
    """
    Weightage: 30% (Assumed Remaining Weight)
    Metric: Filler Word Rate
    Formula: (Filler Count / Total Words) * 100
    List from Excel: um, uh, like, you know, so, actually, basically, right, i mean, well, kinda, sort of, okay, hmm, ah
    """
    fillers = ['um', 'uh', 'like', 'you know', 'so', 'actually', 'basically', 'right', 'i mean', 'well', 'kinda', 'sort of', 'okay', 'hmm', 'ah']
    words = text.lower().split()
    total_words = len(words)
    
    if total_words == 0:
        return {"score": 30, "filler_percentage": 0, "filler_count": 0}
    
    # Exact match count
    filler_count = 0
    for w in words:
        # Strip punctuation for better matching
        clean_w = re.sub(r'[^\w\s]', '', w)
        if clean_w in fillers:
            filler_count += 1
            
    filler_rate = (filler_count / total_words) * 100
    
    # Bins (Assumed standard as Excel snippet cut off)
    # Scaling 10 point scale to 30 points (x3)
    if filler_rate < 2: 
        raw_score = 10
    elif filler_rate < 5: 
        raw_score = 8
    elif filler_rate < 9: 
        raw_score = 6
    elif filler_rate < 12: 
        raw_score = 4
    else: 
        raw_score = 2
        
    final_score = raw_score * 3 # Scale to 30%
    
    return {
        "score": final_score,
        "filler_rate": round(filler_rate, 2),
        "filler_count": filler_count
    }

def create_gauge_chart(score, title):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = score,
        title = {'text': title},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar': {'color': "#4CAF50" if score > 80 else "#FFC107" if score > 50 else "#FF5252"},
            'steps' : [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 80], 'color': "gray"}],
        }
    ))
    fig.update_layout(height=250, margin=dict(l=10, r=10, t=40, b=10))
    return fig

# --- 4. APP LAYOUT ---

st.title("🎙️ AI Communication Coach")
st.markdown("##### Nirmaan Organization - Intern Case Study")
st.caption("Strict Rubric Implementation v1.0")

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712035.png", width=80)
    st.header("Configuration")
    
    if st.button("📄 Load Case Study"):
        st.session_state['transcript_text'] = """Hello everyone, myself Muskan, studying in class 8th B section from Christ Public School. 
I am 13 years old. I live with my family. There are 3 people in my family, me, my mother and my father.
One special thing about my family is that they are very kind hearted to everyone and soft spoken. One thing I really enjoy is play, playing cricket and taking wickets.
A fun fact about me is that I see in mirror and talk by myself. One thing people don't know about me is that I once stole a toy from one of my cousin.
My favorite subject is science because it is very interesting. Through science I can explore the whole world and make the discoveries and improve the lives of others. 
Thank you for listening."""
        st.session_state['duration'] = 52
    
    transcript = st.text_area("Paste Transcript:", value=st.session_state.get('transcript_text', ""), height=300)
    duration = st.number_input("Speech Duration (seconds):", value=st.session_state.get('duration', 60), min_value=10)
    
    st.markdown("---")
    analyze_btn = st.button("🚀 Analyze Now", type="primary")

# Main Logic
if analyze_btn and transcript:
    with st.spinner("🤖 Applying Nirmaan Rubric Logic..."):
        
        # 1. Run Analysis
        content = analyze_content_strict(transcript)
        grammar = analyze_grammar_strict(transcript)
        speech = analyze_speech_rate_strict(transcript, duration)
        clarity = analyze_clarity_strict(transcript)
        
        # 2. Calculate Final Score (Simple Sum because we scaled sections to weights)
        final_score = (
            content['total_score'] + 
            grammar['total_score'] + 
            speech['score'] + 
            clarity['score']
        )
        
        # 3. Dashboard
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.plotly_chart(create_gauge_chart(int(final_score), "Overall Score"), use_container_width=True)
            
            if final_score > 80:
                st.success("🌟 Excellent")
            elif final_score > 50:
                st.warning("⚠️ Average")
            else:
                st.error("🛑 Poor")
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            # Parameter Breakdown Chart
            df_scores = pd.DataFrame({
                'Category': ['Content (40%)', 'Grammar & Vocab (20%)', 'Speech Rate (10%)', 'Clarity (30%)'],
                'Score Achieved': [content['total_score'], grammar['total_score'], speech['score'], clarity['score']],
                'Max Score': [40, 20, 10, 30]
            })
            
            fig = px.bar(df_scores, x='Category', y=['Score Achieved', 'Max Score'], barmode='group', title="Score Breakdown vs Max Weightage")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # 4. Detailed Rubric Breakdown & Report
        t1, t2, t3, t4, t5 = st.tabs(["📝 Content", "✍️ Grammar", "⏱️ Speed", "🗣️ Clarity", "📥 Download Report"])
        
        with t1:
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Total Content Score", f"{content['total_score']} / 40")
                st.write("**Sub-Scores:**")
                st.write(f"- Salutation: {content['breakdown']['Salutation (5)']} / 5")
                st.write(f"- Flow of Thoughts: {content['breakdown']['Flow (5)']} / 5")
                st.write(f"- Keywords: {content['breakdown']['Keywords (30)']} / 30")
                
                if content['is_flow_correct']:
                    st.success("✅ Flow detected: Salutation -> Name -> Closing")
                else:
                    st.warning("⚠️ Flow issue: Ensure you start with a greeting, say your name, and end with a thank you.")

            with c2:
                st.write("**Keyword Analysis:**")
                for s in content['found_sections']:
                    st.markdown(f"<span class='chip-found'>✅ {s}</span>", unsafe_allow_html=True)
                for s in content['missing_sections']:
                    st.markdown(f"<span class='chip-missing'>❌ {s}</span>", unsafe_allow_html=True)

        with t2:
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Grammar Score", f"{grammar['grammar_score']} / 10")
                st.caption("Formula: 1 - min(errors/10 words, 1)")
                if grammar['matches']:
                    for m in grammar['matches']:
                        # SAFE MODE: No slicing
                        st.error(f"Issue: {m.message}")
                        st.caption(f"Context: {m.context}")
                else:
                    st.success("No Grammar Errors")
            with c2:
                st.metric("Vocabulary (TTR)", f"{grammar['vocab_score']} / 10")
                st.write(f"**Diversity Ratio:** {grammar['ttr']}")
                st.caption("Range: >0.9=10pts, 0.7=8pts, 0.5=6pts...")

        with t3:
            st.metric("Score", f"{speech['score']} / 10")
            st.metric("Speed", f"{speech['wpm']} WPM")
            st.info(f"**Verdict:** {speech['feedback']}")
            st.caption("Rubric: Ideal=111-140, Slow=81-110, Too Slow=<80")

        with t4:
            st.metric("Score", f"{clarity['score']} / 30")
            st.metric("Filler Rate", f"{clarity['filler_rate']}%")
            st.write(f"**Filler Count:** {clarity['filler_count']}")
            st.caption("Penalty applied for words like: um, uh, like, actually, basically...")

        with t5:
            st.header("📄 Download Assessment Report")
            st.write("Click the button below to save a detailed text file of this analysis.")
            
            # Generate Report Text
            report_content = f"""
NIRMAAN AI COMMUNICATION COACH - ASSESSMENT REPORT
Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
--------------------------------------------------

OVERALL SCORE: {int(final_score)} / 100

1. CONTENT & STRUCTURE (Score: {content['total_score']}/40)
   - Salutation: {'Present' if content['breakdown']['Salutation (5)'] > 0 else 'Missing'}
   - Flow: {'Correct' if content['is_flow_correct'] else 'Needs Improvement'}
   - Keywords Found: {', '.join(content['found_sections'])}
   - Keywords Missing: {', '.join(content['missing_sections'])}

2. LANGUAGE & GRAMMAR (Score: {grammar['total_score']}/20)
   - Grammar Errors: {grammar['error_count']}
   - Vocabulary Diversity (TTR): {grammar['ttr']}

3. SPEECH RATE (Score: {speech['score']}/10)
   - Speed: {speech['wpm']} WPM
   - Verdict: {speech['feedback']}

4. CLARITY & FLOW (Score: {clarity['score']}/30)
   - Filler Word Rate: {clarity['filler_rate']}%
   - Filler Words Found: {clarity['filler_count']}

--------------------------------------------------
Generated by Nirmaan AI Coach
            """
            
            st.download_button(
                label="📥 Download Report (.txt)",
                data=report_content,
                file_name=f"Assessment_Report_{datetime.now().strftime('%Y%m%d')}.txt",
                mime="text/plain",
                type="primary"
            )

elif analyze_btn and not transcript:
    st.warning("⚠️ Please paste a transcript in the sidebar first.")