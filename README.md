# 🎙️ Nirmaan AI Communication Coach

### 🚀 AI-Powered Student Assessment Tool
**Built for the Nirmaan Organization AI Internship Case Study**

---

## 📖 Overview
This application is an AI-driven tool designed to evaluate student self-introductions based on a **strict grading rubric**. It accepts a text transcript, analyzes it using Natural Language Processing (NLP) and rule-based logic, and provides a detailed score out of 100 along with actionable feedback.

## 🔗 Quick Links
- **🔴 Live Demo:** [PASTE_YOUR_STREAMLIT_LINK_HERE]
- **🎥 Video Walkthrough:** [PASTE_YOUR_VIDEO_LINK_HERE]

---

## 🛠️ Key Features

### 1. 📝 Content & Structure Analysis (40%)
- **Flow Detection:** Checks if the introduction follows the logical order: `Salutation -> Name -> Body -> Closing`.
- **Keyword Extraction:** Scans for 7 mandatory categories: *Name, Age, Family, School, Hobbies, Goal, Unique Fact*.
- **Semantic Analysis:** Uses `sentence-transformers` to ensure the context matches a standard introduction.

### 2. ✍️ Grammar & Vocabulary (20%)
- **Grammar Check:** Integrated with **LanguageTool** to catch grammatical errors.
  - *Formula:* `Score = 1 - (Errors / Word Count)`
- **Vocabulary Diversity:** Calculates **Type-Token Ratio (TTR)** to reward rich vocabulary usage.

### 3. ⏱️ Speech Rate Analysis (10%)
- Calculates **Words Per Minute (WPM)** based on transcript length and duration.
- **Strict Scoring:**
  - ✅ **Ideal:** 111-140 WPM (10 points)
  - 🐢 **Slow:** 81-110 WPM (6 points)
  - 🐇 **Fast:** >140 WPM (6 points)
  - 🐌 **Too Slow:** <80 WPM (2 points)

### 4. 🗣️ Clarity & Filler Words (30%)
- Detects filler words (*um, uh, like, actually, basically, you know*).
- Penalizes the score based on the **Filler Word Rate (%)**.

### 5. 📊 Visual Dashboard & Reports
- **Gauge Charts:** Instant visual feedback for the overall score.
- **Radar Charts:** To visualize strengths and weaknesses across 4 categories.
- **Downloadable Reports:** Generates a `.txt` summary of the assessment.

---

## 💻 Tech Stack
- **Frontend:** [Streamlit](https://streamlit.io/) (Python)
- **NLP Engine:** [Spacy](https://spacy.io/)
- **Grammar Engine:** [LanguageTool](https://pypi.org/project/language-tool-python/)
- **AI Models:** `all-MiniLM-L6-v2` (Sentence Transformers)
- **Visualization:** Plotly Express

---

## ⚙️ How to Run Locally

If you want to run this project on your own machine, follow these steps:

1. **Clone the Repository**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/nirmaan-ai-coach.git](https://github.com/YOUR_USERNAME/nirmaan-ai-coach.git)
   cd nirmaan-ai-coach

   ## ⚙️ How to Run Locally

If you want to run this project on your own machine, follow these steps:

1.  **Clone the Repository**

    ```bash
    git clone [https://github.com/YOUR_USERNAME/nirmaan-ai-coach.git](https://github.com/YOUR_USERNAME/nirmaan-ai-coach.git)
    cd nirmaan-ai-coach
    ```

2.  **Install Dependencies**

    ```bash
    pip install -r requirements.txt
    ```

3.  **Download Language Models**

    ```bash
    python -m spacy download en_core_web_sm
    ```

4.  **Run the App**

    ```bash
    streamlit run app.py
    ```

-----

## 📂 Project Structure
```text
├── app.py               # The main application file (Logic + UI)
├── requirements.txt     # List of python dependencies
└── README.md            # Project documentation
