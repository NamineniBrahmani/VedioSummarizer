import os
import glob
import streamlit as st
import tempfile
import requests
import google.generativeai as genai
import spacy
from collections import Counter
from dotenv import load_dotenv
import yt_dlp
import mimetypes
import atexit
import uuid  # To avoid filename collisions

# Load secrets
load_dotenv()
GEMINI_KEY = os.getenv("GEMINI_API_KEY") or st.secrets["GEMINI_API_KEY"]
HF_TOKEN = os.getenv("HUGGINGFACE_API_KEY") or st.secrets["HUGGINGFACE_API_KEY"]
genai.configure(api_key=GEMINI_KEY)

st.title("🎬 Video Summarizer & Highlights Extractor with Hugging Face + Gemini")

# Step 1: Input type
input_mode = st.radio("Select input type:", ["🔗 YouTube Link","📎 Upload Local Video"])
video_path = ""
transcript_text = ""

# Step 2: Load video
if input_mode == "📎 Upload Local Video":
    uploaded = st.file_uploader("Upload a video file", type=["mp4", "mov", "mkv", "webm", "wav", "m4a"])
    if uploaded:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded.name)[1]) as temp_file:
            temp_file.write(uploaded.read())
            video_path = temp_file.name
            atexit.register(lambda: os.remove(video_path) if os.path.exists(video_path) else None)

elif input_mode == "🔗 YouTube Link":
    url = st.text_input("Enter YouTube video URL")
    if url:
        try:
            st.info("Downloading audio from YouTube...")
            temp_dir = tempfile.gettempdir()
            unique_id = str(uuid.uuid4())[:8]
            base_path = os.path.join(temp_dir, f"ytvideo_{unique_id}")
            ydl_opts = {
                "format": "bestaudio[ext=mp3]/bestaudio[ext=m4a]",
                "outtmpl": base_path + ".%(ext)s",
                "quiet": True,
                "postprocessors": [
        {
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }]  # skip ffmpeg postprocessing
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])

            matching_files = glob.glob(base_path + ".*")
            if matching_files:
                video_path = matching_files[0]
                atexit.register(lambda: os.remove(video_path) if os.path.exists(video_path) else None)
                st.success(f"Downloaded: {os.path.basename(video_path)}")
            else:
                st.error("Video download failed.")
        except Exception as e:
            st.error(f"Download error: {e}")

# Step 3: Transcribe with Hugging Face Whisper
def transcribe_with_huggingface(audio_file):
    mime_type, _ = mimetypes.guess_type(audio_file)
    if not mime_type:
        mime_type = "application/octet-stream"
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": mime_type  # auto-detect correct content type
    }
    api_url = "https://api-inference.huggingface.co/models/openai/whisper-large-v3"
    with open(audio_file, "rb") as f:
        audio_data = f.read()
    response = requests.post(api_url, headers=headers, data=audio_data)
    if response.status_code == 200:
        return response.json().get("text", "")
    elif response.status_code == 503:
        raise RuntimeError("Model is loading. Please retry in a few seconds.")
    elif response.status_code == 401:
        raise RuntimeError("Unauthorized: Check Hugging Face API key.")
    elif response.status_code == 400:
        raise RuntimeError(f"Bad request: possibly wrong audio format.\n{response.text}")
    else:
        raise RuntimeError(f"API error: {response.status_code}\n{response.text}")



if video_path:
    with st.spinner("🔊 Transcribing with Whisper (Hugging Face API)..."):
        try:
            transcript_text = transcribe_with_huggingface(video_path)
            st.success("Transcript ready!")
            st.text_area("Transcript Preview", transcript_text[:1500], height=250)
        except Exception as e:
            st.error(str(e))

# Step 4: Gemini Summary
summary = ""
if transcript_text.strip():
    st.subheader("🧠 Gemini Summary")
    with st.spinner("Summarizing transcript..."):
        try:
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(f"Summarize this transcript:\n{transcript_text}")
            summary = response.text
            st.text_area("Summary", summary, height=300)
        except Exception as e:
            st.error(f"Gemini error: {e}")

# Step 5: spaCy Highlights
highlights = []
if transcript_text.strip():
    st.subheader("✨ Key Highlights")
    try:
        try:
            nlp = spacy.load("en_core_web_sm")
        except:
            from spacy.cli import download
            download("en_core_web_sm")
            nlp = spacy.load("en_core_web_sm")

        doc = nlp(transcript_text)
        word_freq = Counter([t.text.lower() for t in doc if t.is_alpha and not t.is_stop])
        sent_scores = {sent: sum(word_freq.get(w.text.lower(), 0) for w in sent) for sent in doc.sents}
        top_sents = sorted(sent_scores.items(), key=lambda x: x[1], reverse=True)[:7]
        highlights = [s.text.strip() for s, _ in top_sents]

        for i, h in enumerate(highlights, 1):
            st.markdown(f"{i}. **{h}**")
    except Exception as e:
        st.error(f"Highlight extraction failed: {e}")

# Step 6: Download summary + highlights
if summary and highlights:
    output = f"== SUMMARY ==\n{summary}\n\n== HIGHLIGHTS ==\n" + "\n".join(highlights)
    st.download_button("📥 Download Summary + Highlights", output, file_name="video_summary.txt")
