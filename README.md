# 🤖 AI Resume Matcher

A smart and scalable Resume-to-Job Matching application that leverages **NLP**, **BERT embeddings**, and **FAISS** to semantically match PDF resumes with the most relevant job descriptions.

Built with ❤️ using **Streamlit**, this app allows users to upload their resume and receive instant job recommendations.

---

![Initial UI](assets/initial%20ui.png)

![Upload Resume](assets/after%20uploading%20the%20CV.png)

## 🔍 Demo

👉 [Click here to try it live on Streamlit Cloud](https://your-streamlit-link.streamlit.app)

---

## 🚀 Key Features

| Feature                            | Description                                                                |
| ---------------------------------- | -------------------------------------------------------------------------- |
| 🧠 Semantic Matching               | Uses **BERT (MiniLM)** embeddings for deep understanding of resume content |
| 📂 PDF Upload                      | Accepts **PDF resumes**, extracts and cleans content                       |
| ⚡ Fast Similarity Search          | Powered by **FAISS** for real-time matching                                |
| 📊 Similarity Score Display        | Shows score between your resume and each job                               |
| 🧩 Duplicate Job Filtering         | Avoids repeated job titles to improve diversity                            |
| 💬 Interactive Streamlit Interface | User-friendly and responsive web app                                       |

---

## 🧠 Tech Stack

- **Frontend/UI**: [Streamlit](https://streamlit.io)
- **Language Model**: [SentenceTransformers - MiniLM](https://www.sbert.net/)
- **Similarity Engine**: [FAISS](https://github.com/facebookresearch/faiss)
- **Preprocessing**: [NLTK](https://www.nltk.org/), Regex, Lemmatization

---

---

## 📦 Setup Instructions

### ✅ Prerequisites

- Python 3.11
- pip

> ⚠️ Note: **Streamlit Cloud doesn't support Python 3.12**, so we pin to 3.11.

---

### 🛠️ Installation

```bash
git clone https://github.com/your-username/ai-resume-matcher.git
cd ai-resume-matcher
pip install -r requirements.txt
```
