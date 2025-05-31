import streamlit as st
import asyncio
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig
from dotenv import load_dotenv
import os

# Load environment variable
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set in .env file.")

# External OpenAI Client setup
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# Model setup
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

# Run configuration
config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

# Translator agent
translator = Agent(
    name='Translator Agent',
    instructions="""
You are a translator agent. Your job is to translate the given text from one language to another.

Always detect the source language automatically and translate it into the target language as requested.

Keep the meaning, tone, and context accurate while translating.

Do not change names, dates, or important information.

Respond only with the translated text, nothing else.
"""
)

# Async run function
async def run_translator_agent(user_input):
    return await Runner.run(translator, input=user_input, run_config=config)

# Page config
st.set_page_config(page_title="Translator Agent by Alisha Kafeel",
                   page_icon="🌍", layout="centered")

# Styling
st.markdown("""
    <style>
    .title {
        font-size: 2.8em;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.2em;
        color: #4B7BEC;
    }
    .subtitle {
        font-size: 1.1em;
        text-align: center;
        color: #6c757d;
        margin-bottom: 2em;
    }
    .footer {
        margin-top: 3em;
        text-align: center;
        font-size: 0.9em;
        color: #adb5bd;
    }
    .translate-button div.stButton > button {
        background-color: #4B7BEC;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.5em 1.5em;
        transition: 0.3s;
    }
    .translate-button div.stButton > button:hover {
        background-color: #3a5fcd;
    }
    </style>
    <div class="title">🌐 Translator Agent</div>
    <div class="subtitle">Powered by Gemini API<br>Created by <strong>Alisha Kafeel</strong></div>
""", unsafe_allow_html=True)

# Main translation UI
with st.container():
    st.markdown("### 🌐 Select Languages")

    col1, col2 = st.columns(2)
    with col1:
        source_lang = st.selectbox("Source Language", [
            "Auto Detect", "English", "Urdu", "Arabic", "French", "German", "Hindi", "Spanish", "Chinese", "Japanese"
        ])
    with col2:
        target_lang = st.selectbox("Target Language", [
            "Urdu", "English", "Arabic", "French", "German", "Hindi", "Spanish", "Chinese", "Japanese"
        ])

    st.markdown("### 📝 Enter your text below")
    user_input = st.text_area(
        label="",
        placeholder="Example: I am learning Python programming.",
        height=150
    )

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.container():
            st.markdown('<div class="translate-button">', unsafe_allow_html=True)
            translate_clicked = st.button("🌍 Translate", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    if translate_clicked:
        if user_input.strip() == "":
            st.warning("⚠️ Please enter some text to translate.")
        else:
            with st.spinner("Translating..."):
                # Construct instruction for the agent
                if source_lang == "Auto Detect":
                    lang_instruction = f"Translate this into {target_lang}: {user_input}"
                else:
                    lang_instruction = f"Translate from {source_lang} to {target_lang}: {user_input}"

                response = asyncio.run(run_translator_agent(lang_instruction))
                st.success("✅ Translation Complete:")
                st.markdown(
                    f"<div style='padding: 15px; background-color: #f1f3f5; border-radius: 10px; font-size: 1.1em; color: #333;'>{response.final_output}</div>",
                    unsafe_allow_html=True
                )

# Footer
st.markdown("""
    <div class="footer">
        &copy; 2025 Alisha Kafeel. All rights reserved.
    </div>
""", unsafe_allow_html=True)
