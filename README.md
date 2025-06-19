# 🎬 Video Summarizer & Highlights Extractor with Hugging Face Whisper & Gemini

An AI-powered Streamlit app that **transcribes**, **summarizes**, and **highlights** video content from either a **YouTube link** or a **local video file**.

**Try it live:** [Vedio Summarizer](https://vedio-summarizer.streamlit.app/)


## Features

-  Upload local videos or paste YouTube links
-  Transcribe using Hugging Face Whisper (`openai/whisper-large-v3`)
-  Summarize with Google Gemini (`gemini-1.5-flash`)
-  Extract key highlights using spaCy NLP
-  Download summary and highlights as a `.txt` file


## 🖼️ UI Preview

![App Preview](https://github.com/NamineniBrahmani/VedioSummarizer/blob/main/image.jpg)

---

## Installation

### Prerequisites

- Python 3.10+
- Hugging Face API Key
- Gemini API Key (Google Generative AI)

### Set Up

1. **Clone the repository**

```bash
git clone https://github.com/NamineniBrahmani/VideoSummarizer.git
cd VideoSummarizer
```

2. **Create `.env` file**

```bash
GEMINI_API_KEY=your_gemini_api_key
HUGGINGFACE_API_KEY=your_huggingface_api_key
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

4. **Run the app**

```bash
streamlit run app.py
```

### Supported Inputs
- Local upload: `.mp4`, `.mov`, `.mkv`, `.webm`, `.wav`, `.m4a`
- YouTube links: Automatically downloads and processes audio

## Tech Stack

| Tool          | Purpose                                 |
| ------------- | --------------------------------------- |
| Streamlit     | Web interface                           |
| Hugging Face  | Whisper transcription (`large-v3`)      |
| Google Gemini | Text summarization (`gemini-1.5-flash`) |
| spaCy         | Highlight extraction (`en_core_web_sm`) |
| yt\_dlp       | YouTube audio download                  |
| dotenv        | API key handling                        |

## Output Example
== SUMMARY ==
- This video covers the core principles of ...

== HIGHLIGHTS ==
1. The speaker emphasizes the importance of ...
2. A key takeaway from the discussion is ...
...

## Models Used
- `openai/whisper-large-v3` – for accurate audio transcription
- `gemini-1.5-flash` – for high-quality summarization
- `en_core_web_sm` – for extracting important sentences (spaCy)

## Download
- At the end of processing, you can download the combined summary and key highlights as a text file.

# Try it out now
👉 [Launch the App](https://vedio-summarizer.streamlit.app/)

## License
- This project is licensed under the MIT License.

## Acknowledgements
- [Hugging Face](https://huggingface.co/)
- [Google Generative AI](https://ai.google.dev/)
- [Streamlit](https://streamlit.io/)
- [spaCy](https://spacy.io/)
- [yt_dlp](https://github.com/yt-dlp/yt-dlp)

## Author
- Mail: naminenibrahmani@gmail.com
- [Linkedin](https://www.linkedin.com/in/brahmani-namineni)
