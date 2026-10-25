# install the required libraries
# pip install -q -r ./requirements.txt
# openai
# langchain
# langchain-openai
# langchain-core
# streamlit
# tiktoken

# run the app: streamlit run ./chatgpt_interactive.py

import html
import json
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import streamlit as st
from openai import OpenAI, AuthenticationError, APIConnectionError
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage


st.set_page_config(
    page_title='Custom ChatGPT',
    page_icon=':material/auto_awesome:',
    layout='wide',
    initial_sidebar_state='expanded',
)

# ---------------------------------------------------------------- constants
HISTORY_FILE = Path(__file__).with_name('chat_history.json')
MODELS = ['gpt-5.2', 'gpt-4.1', 'gpt-4o', 'gpt-4o-mini']
DEFAULT_PERSONA = 'You are a helpful assistant.'
USER_AVATAR = ':material/person:'
AI_AVATAR = ':material/auto_awesome:'
SUGGESTIONS = [
    ('💡', 'Explain a concept', 'Explain quantum computing like I\'m five years old.'),
    ('✍️', 'Write a poem', 'Write a short, uplifting poem about Monday mornings.'),
    ('🧳', 'Plan a trip', 'Plan a 3-day itinerary for a first-time visit to Istanbul.'),
    ('🐍', 'Improve my Python', 'What are the most common Python mistakes beginners make, and how do I avoid them?'),
]
TYPING_HTML = (
    '<div class="typing"><span></span><span></span><span></span>'
    '<em>Thinking…</em></div>'
)

# ---------------------------------------------------------------- styles
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');

:root {
  --bg: #F4F6FB;
  --surface: #FFFFFF;
  --ink: #0F172A;
  --muted: #64748B;
  --line: rgba(15, 23, 42, .08);
  --navy: #0B1220;
  --navy-2: #111A2E;
  --indigo: #4F46E5;
  --blue: #2563EB;
  --cyan: #06B6D4;
  --grad: linear-gradient(120deg, #4F46E5 0%, #2563EB 50%, #06B6D4 100%);
  --grad-soft: linear-gradient(135deg, #EEF2FF 0%, #E0F2FE 100%);
  --shadow: 0 10px 30px -12px rgba(37, 56, 140, .25);
  --glow: 0 8px 24px -6px rgba(79, 70, 229, .55);
}

/* ---------- base (force a light main area even if the OS is in dark mode) */
html, body, .stApp, [data-testid="stAppViewContainer"] {
  background: var(--bg) !important; color: var(--ink) !important;
  font-family: 'Manrope', system-ui, sans-serif !important;
}
[data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] *,
.stApp label, .stApp label * { color: var(--ink); font-family: 'Manrope', system-ui, sans-serif; }
[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li { line-height: 1.7; }
[data-testid="stMarkdownContainer"] code { background: #EEF2FF; color: #4338CA; border-radius: 6px; }
[data-testid="stMarkdownContainer"] pre { background: #0F172A !important; border-radius: 14px; border: 1px solid rgba(255,255,255,.06); }
[data-testid="stMarkdownContainer"] pre code { background: transparent !important; color: #E2E8F0 !important; }
header[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stAppDeployButton"], [data-testid="stMainMenu"], [data-testid="stDecoration"], footer { display: none !important; }
[data-testid="stMainBlockContainer"] { max-width: 880px; padding-top: 1.6rem; padding-bottom: 7rem; position: relative; z-index: 1; }

/* ---------- animated aurora background */
.blob { position: fixed; border-radius: 50%; filter: blur(90px); opacity: .55; z-index: 0;
        pointer-events: none; animation: drift 26s ease-in-out infinite alternate; }
.b1 { width: 520px; height: 520px; background: #C7D2FE; top: -180px; left: 28%; }
.b2 { width: 460px; height: 460px; background: #A5F3FC; bottom: -200px; right: -100px; animation-delay: -9s; }
.b3 { width: 380px; height: 380px; background: #DDD6FE; top: 40%; left: -140px; animation-delay: -17s; }
@keyframes drift {
  0%   { transform: translate(0, 0) scale(1); }
  50%  { transform: translate(60px, -40px) scale(1.12); }
  100% { transform: translate(-40px, 50px) scale(.94); }
}
@keyframes rise  { from { opacity: 0; transform: translateY(12px); } to { opacity: 1; transform: none; } }
@keyframes pop   { from { opacity: 0; transform: scale(.94) translateY(16px); } to { opacity: 1; transform: none; } }
@keyframes float { 0%, 100% { transform: translateY(0); } 50% { transform: translateY(-8px); } }
@keyframes shine { to { background-position: 200% center; } }
@keyframes blink { 0%, 80%, 100% { transform: scale(.55); opacity: .35; } 40% { transform: scale(1); opacity: 1; } }
@keyframes pulse { 0%, 100% { box-shadow: 0 0 0 0 rgba(34, 197, 94, .5); } 50% { box-shadow: 0 0 0 6px rgba(34, 197, 94, 0); } }
@keyframes ring  { 0% { transform: scale(.9); opacity: .6; } 100% { transform: scale(1.7); opacity: 0; } }

.grad-text { background: var(--grad); background-size: 200% auto; -webkit-background-clip: text;
             background-clip: text; -webkit-text-fill-color: transparent; animation: shine 7s linear infinite; }

/* ---------- inputs */
[data-baseweb="input"], [data-baseweb="textarea"], [data-baseweb="select"] > div {
  background: #FFFFFF !important; border: 1px solid var(--line) !important; border-radius: 12px !important;
  transition: border-color .2s ease, box-shadow .2s ease;
}
[data-baseweb="input"]:focus-within, [data-baseweb="textarea"]:focus-within {
  border-color: var(--indigo) !important; box-shadow: 0 0 0 4px rgba(79, 70, 229, .15);
}
.stApp input, .stApp textarea, [data-baseweb="select"] * { color: var(--ink) !important; background: transparent !important; }

/* ---------- buttons */
[data-testid^="stBaseButton-secondary"] {
  background: #FFFFFF; color: var(--ink); border: 1px solid var(--line); border-radius: 12px; font-weight: 600;
  transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease;
}
[data-testid^="stBaseButton-secondary"]:hover { transform: translateY(-1px); border-color: rgba(79, 70, 229, .45);
  color: var(--indigo); box-shadow: var(--shadow); }
[data-testid^="stBaseButton-secondary"]:active, [data-testid^="stBaseButton-primary"]:active { transform: scale(.97); }
[data-testid^="stBaseButton-secondary"]:focus-visible, [data-testid^="stBaseButton-primary"]:focus-visible {
  outline: 2px solid var(--cyan); outline-offset: 2px;
}
[data-testid^="stBaseButton-primary"] {
  background: var(--grad) !important; background-size: 200% auto !important; border: none !important;
  border-radius: 12px !important; color: #fff !important; font-weight: 700; box-shadow: var(--glow);
  transition: background-position .4s ease, transform .18s ease, box-shadow .18s ease;
}
[data-testid^="stBaseButton-primary"] * { color: #fff !important; }
[data-testid^="stBaseButton-primary"]:hover { background-position: right center !important; transform: translateY(-2px);
  box-shadow: 0 12px 28px -6px rgba(79, 70, 229, .65); }

/* ---------- sidebar (dark navy for contrast) */
[data-testid="stSidebar"] { background: linear-gradient(180deg, var(--navy) 0%, var(--navy-2) 100%) !important;
                            border-right: 1px solid rgba(255, 255, 255, .06); }
[data-testid="stSidebar"] *, [data-testid="stSidebar"] label * { color: #E2E8F0; }
[data-testid="stSidebar"] [data-baseweb="input"], [data-testid="stSidebar"] [data-baseweb="textarea"],
[data-testid="stSidebar"] [data-baseweb="select"] > div {
  background: rgba(255, 255, 255, .06) !important; border: 1px solid rgba(255, 255, 255, .1) !important;
}
[data-testid="stSidebar"] [data-baseweb="input"]:focus-within, [data-testid="stSidebar"] [data-baseweb="textarea"]:focus-within {
  border-color: var(--cyan) !important; box-shadow: 0 0 0 3px rgba(6, 182, 212, .22);
}
[data-testid="stSidebar"] input, [data-testid="stSidebar"] textarea,
[data-testid="stSidebar"] [data-baseweb="select"] * { color: #F1F5F9 !important; background: transparent !important; }
[data-testid="stSidebar"] input::placeholder { color: #8391A7 !important; }
[data-testid="stSidebar"] [data-testid^="stBaseButton-secondary"] {
  background: rgba(255, 255, 255, .06); border: 1px solid rgba(255, 255, 255, .1); color: #E2E8F0; box-shadow: none;
}
[data-testid="stSidebar"] [data-testid^="stBaseButton-secondary"]:hover {
  background: rgba(255, 255, 255, .12); border-color: rgba(6, 182, 212, .6); color: #fff; box-shadow: none;
}
[data-testid="stSidebar"] [data-testid="stExpander"] details { background: rgba(255, 255, 255, .04);
  border: 1px solid rgba(255, 255, 255, .08); border-radius: 14px; }

.brand { display: flex; align-items: center; gap: .75rem; margin: .2rem 0 1.1rem; animation: rise .6s both; }
.brand .logo { width: 42px; height: 42px; border-radius: 13px; background: var(--grad); display: grid;
               place-items: center; font-size: 1.35rem; font-weight: 800; color: #fff !important; box-shadow: var(--glow);
               animation: float 5s ease-in-out infinite; }
.brand b { font-size: 1.05rem; font-weight: 800; letter-spacing: -.01em; display: block; color: #fff !important; }
.brand small { color: #8FA0BA !important; font-size: .75rem; }
.group-label { font-size: .7rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase;
               color: #7C8CA6 !important; margin: 1.1rem 0 .25rem .3rem; }
.empty-hist { text-align: center; color: #8FA0BA !important; font-size: .85rem; padding: 1.2rem .5rem;
              border: 1px dashed rgba(255, 255, 255, .18); border-radius: 14px; margin-top: .8rem; }

[class*="st-key-chatrow_"], .st-key-chat_active { border-radius: 12px; transition: background .2s ease, transform .2s ease; animation: rise .4s both; }
[class*="st-key-chatrow_"]:hover { background: rgba(255, 255, 255, .07); transform: translateX(2px); }
.st-key-chat_active { background: rgba(79, 70, 229, .28); position: relative; box-shadow: inset 0 0 0 1px rgba(129, 140, 248, .35); }
.st-key-chat_active::before { content: ''; position: absolute; left: 0; top: 22%; bottom: 22%; width: 3px;
                              border-radius: 3px; background: var(--cyan); }
[class*="st-key-chatrow_"] button, .st-key-chat_active button {
  background: transparent !important; border: none !important; box-shadow: none !important;
  justify-content: flex-start; padding: .38rem .6rem; min-height: 0; transform: none !important;
}
[class*="st-key-chatrow_"] button p, .st-key-chat_active button p {
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis; font-size: .87rem; text-align: left; color: #CBD5E1 !important;
}
.st-key-chat_active button p { font-weight: 700; color: #fff !important; }
[class*="st-key-chatrow_"] [data-testid="stColumn"]:last-child,
.st-key-chat_active [data-testid="stColumn"]:last-child { opacity: 0; transition: opacity .2s ease; }
[class*="st-key-chatrow_"]:hover [data-testid="stColumn"]:last-child,
.st-key-chat_active:hover [data-testid="stColumn"]:last-child,
[class*="st-key-chatrow_"]:focus-within [data-testid="stColumn"]:last-child,
.st-key-chat_active:focus-within [data-testid="stColumn"]:last-child { opacity: 1; }
[class*="st-key-del_"] button:hover p, [class*="st-key-del_"] button:hover span { color: #F87171 !important; }

.keycard { display: flex; align-items: center; gap: .65rem; background: rgba(255, 255, 255, .05);
           border: 1px solid rgba(255, 255, 255, .1); border-radius: 14px; padding: .7rem .85rem; margin: .9rem 0 .6rem; }
.keycard .dot { width: 9px; height: 9px; border-radius: 50%; background: #22C55E; animation: pulse 2s infinite; flex: none; }
.keycard b { font-size: .82rem; display: block; color: #fff !important; }
.keycard code { font-size: .72rem; color: #8FA0BA !important; background: none !important; }

/* ---------- top bar */
.topbar { display: flex; justify-content: space-between; align-items: center; gap: 1rem; padding: .75rem 1.1rem;
          background: rgba(255, 255, 255, .8); backdrop-filter: blur(14px); border: 1px solid rgba(255, 255, 255, .95);
          border-radius: 16px; box-shadow: var(--shadow); margin-bottom: 1.2rem; animation: rise .5s both; }
.topbar .t { font-weight: 700; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.pill { display: inline-flex; align-items: center; gap: .45rem; font-size: .75rem; font-weight: 700;
        padding: .32rem .8rem; border-radius: 999px; white-space: nowrap;
        background: var(--grad-soft); color: #3730A3 !important; border: 1px solid rgba(79, 70, 229, .2); }
.pill::before { content: ''; width: 7px; height: 7px; border-radius: 50%; background: #22C55E; animation: pulse 2s infinite; }

/* ---------- hero / empty state */
.hero { text-align: center; margin: 6vh 0 2.2rem; animation: rise .7s both; }
.hero .mark { position: relative; width: 72px; height: 72px; border-radius: 22px; background: var(--grad);
              display: inline-grid; place-items: center; font-size: 2.3rem; font-weight: 800; color: #fff !important;
              box-shadow: var(--glow); animation: float 4.5s ease-in-out infinite; }
.hero .mark::after { content: ''; position: absolute; inset: 0; border-radius: 22px; border: 2px solid var(--indigo);
                     animation: ring 2.8s ease-out infinite; }
.hero .hello { color: var(--muted) !important; font-weight: 600; margin-top: 1.1rem; }
.hero h1 { font-size: clamp(2rem, 4.4vw, 3.2rem) !important; font-weight: 800 !important; letter-spacing: -.035em;
           line-height: 1.1 !important; margin: .3rem 0 .4rem !important; padding: 0 !important; }
.hero .sub { color: var(--muted) !important; font-size: 1rem; }

[class*="st-key-sugg_"] { animation: rise .6s both; }
.st-key-sugg_1 { animation-delay: .07s; } .st-key-sugg_2 { animation-delay: .14s; } .st-key-sugg_3 { animation-delay: .21s; }
[class*="st-key-sugg_"] button {
  min-height: 118px; width: 100%; background: rgba(255, 255, 255, .85) !important; backdrop-filter: blur(10px);
  border: 1px solid rgba(255, 255, 255, .95) !important; border-radius: 20px !important;
  box-shadow: 0 8px 22px -14px rgba(37, 56, 140, .35); align-items: flex-start; justify-content: flex-start;
  padding: 1.05rem 1.15rem !important; position: relative; overflow: hidden;
  transition: transform .25s cubic-bezier(.2, .8, .2, 1), box-shadow .25s ease, border-color .25s ease !important;
}
[class*="st-key-sugg_"] button::after { content: ''; position: absolute; left: 0; top: 0; height: 3px; width: 100%;
  background: var(--grad); transform: scaleX(0); transform-origin: left; transition: transform .35s ease; }
[class*="st-key-sugg_"] button p { text-align: left; font-size: .86rem; line-height: 1.55; color: var(--muted) !important; }
[class*="st-key-sugg_"] button p strong { font-size: 1rem; font-weight: 800; color: var(--ink) !important; }
[class*="st-key-sugg_"] button span[data-testid="stIconMaterial"] { color: var(--indigo) !important; font-size: 1.4rem; }
[class*="st-key-sugg_"] button:hover { transform: translateY(-5px) !important;
  box-shadow: 0 22px 36px -16px rgba(79, 70, 229, .5) !important; border-color: rgba(79, 70, 229, .4) !important; }
[class*="st-key-sugg_"] button:hover::after { transform: scaleX(1); }

/* ---------- chat messages */
[data-testid="stChatMessage"] { background: transparent !important; padding: .4rem 0; gap: .8rem;
                                animation: rise .45s cubic-bezier(.2, .8, .2, 1) both; }
[data-testid="stChatMessage"] [data-testid^="stChatMessageAvatar"] {
  background: #FFFFFF; border-radius: 12px; width: 38px; height: 38px;
  box-shadow: 0 6px 16px -8px rgba(37, 56, 140, .45); border: 1px solid var(--line); color: var(--indigo);
}
[data-testid="stChatMessage"]:has([class*="st-key-assistant_msg_"]) [data-testid^="stChatMessageAvatar"] {
  background: var(--grad); border: none; box-shadow: var(--glow);
}
[data-testid="stChatMessage"]:has([class*="st-key-assistant_msg_"]) [data-testid^="stChatMessageAvatar"] * { color: #fff !important; }
[data-testid="stChatMessage"]:has([class*="st-key-assistant_msg_"]) [data-testid="stChatMessageContent"] {
  background: #FFFFFF; border: 1px solid var(--line); border-radius: 6px 20px 20px 20px;
  padding: .9rem 1.2rem; box-shadow: var(--shadow); position: relative;
}
[data-testid="stChatMessage"]:has([class*="st-key-assistant_msg_"]) [data-testid="stChatMessageContent"]::before {
  content: ''; position: absolute; left: 0; top: 14px; bottom: 14px; width: 3px; border-radius: 3px; background: var(--grad);
}
[data-testid="stChatMessage"]:has([class*="st-key-user_msg_"]) { flex-direction: row-reverse; }
[data-testid="stChatMessage"]:has([class*="st-key-user_msg_"]) [data-testid="stChatMessageContent"] {
  background: var(--grad); border-radius: 20px 6px 20px 20px; box-shadow: var(--glow);
  padding: .75rem 1.15rem; flex: 0 1 auto; max-width: 78%; margin-left: auto;
}
[data-testid="stChatMessage"]:has([class*="st-key-user_msg_"]) [data-testid="stChatMessageContent"] *  { color: #fff !important; }
[data-testid="stChatMessage"]:has([class*="st-key-user_msg_"]) [data-testid="stChatMessageContent"] code {
  background: rgba(255, 255, 255, .2) !important;
}

.typing { display: flex; align-items: center; gap: 6px; padding: .2rem 0; }
.typing span { width: 8px; height: 8px; border-radius: 50%; background: var(--grad); animation: blink 1.2s infinite both; }
.typing span:nth-child(2) { animation-delay: .15s; } .typing span:nth-child(3) { animation-delay: .3s; }
.typing em { margin-left: .4rem; color: var(--muted) !important; font-size: .85rem; font-style: normal; font-weight: 600; }

/* ---------- chat input */
[data-testid="stBottom"] > div { background: linear-gradient(to top, var(--bg) 65%, rgba(244, 246, 251, 0)) !important; }
[data-testid="stBottomBlockContainer"] { max-width: 880px; background: transparent !important; }
[data-testid="stChatInput"] { background: #FFFFFF !important; border: 1px solid var(--line) !important;
  border-radius: 22px !important; box-shadow: var(--shadow); transition: box-shadow .25s ease, border-color .25s ease; }
[data-testid="stChatInput"]:focus-within { border-color: var(--indigo) !important;
  box-shadow: 0 0 0 4px rgba(79, 70, 229, .16), var(--shadow); }
[data-testid="stChatInput"] > div { background: transparent !important; border: none !important; }
[data-testid="stChatInputSubmitButton"] { background: var(--grad) !important; border-radius: 14px !important;
  box-shadow: var(--glow); transition: transform .18s ease; }
[data-testid="stChatInputSubmitButton"]:hover { transform: scale(1.08); }
[data-testid="stChatInputSubmitButton"] svg { fill: #fff !important; color: #fff !important; }
.fineprint { text-align: center; font-size: .74rem; color: var(--muted) !important; margin-top: .5rem; }

/* ---------- API key gate */
.st-key-gate_card { background: rgba(255, 255, 255, .85); backdrop-filter: blur(22px);
  border: 1px solid rgba(255, 255, 255, .95); border-radius: 28px; padding: 2.3rem 2.1rem 1.8rem;
  box-shadow: 0 30px 80px -30px rgba(37, 56, 140, .45); margin-top: 8vh;
  animation: pop .8s cubic-bezier(.2, .9, .3, 1.2) both; }
.st-key-gate_card [data-testid="stForm"] { border: none; padding: 0; }
.gate-head { text-align: center; }
.gate-head .mark { width: 64px; height: 64px; border-radius: 20px; background: var(--grad); color: #fff !important;
                   display: inline-grid; place-items: center; font-size: 2rem; font-weight: 800; box-shadow: var(--glow);
                   animation: float 4.5s ease-in-out infinite; }
.gate-head h2 { font-size: 1.5rem !important; text-wrap: balance; font-weight: 800 !important; letter-spacing: -.03em; padding: .8rem 0 .2rem !important; }
.gate-head p { color: var(--muted) !important; }
.perks { display: flex; justify-content: center; flex-wrap: wrap; gap: .45rem; margin: .9rem 0 1.3rem; }
.perks span { font-size: .75rem; font-weight: 700; padding: .3rem .75rem; border-radius: 999px;
              background: var(--grad-soft); border: 1px solid rgba(79, 70, 229, .18); color: #3730A3 !important; }
.gate-note { text-align: center; font-size: .78rem; color: var(--muted) !important; margin-top: .9rem; line-height: 1.6; }
.gate-note a { color: var(--indigo) !important; font-weight: 700; text-decoration: none; }
.gate-note a:hover { text-decoration: underline; }

@media (prefers-reduced-motion: reduce) { *, *::before, *::after { animation: none !important; transition: none !important; } }
</style>
<div class="blob b1"></div><div class="blob b2"></div><div class="blob b3"></div>
"""

GATE_CSS = """
<style>
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"], [data-testid="stExpandSidebarButton"] { display: none !important; }
</style>
"""


# ---------------------------------------------------------------- helpers
def load_chats():
    if HISTORY_FILE.exists():
        try:
            return json.loads(HISTORY_FILE.read_text(encoding='utf-8'))
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def save_chats():
    HISTORY_FILE.write_text(
        json.dumps(st.session_state.chats, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )


def make_title(prompt):
    text = ' '.join(prompt.split())
    return f'{text[:38]}{"…" if len(text) > 38 else ""}'


def time_group(ts):
    day = datetime.fromtimestamp(ts).date()
    today = datetime.now().date()
    if day == today:
        return '☀️ Today'
    if day == today - timedelta(days=1):
        return '🌙 Yesterday'
    if day > today - timedelta(days=7):
        return '🗓️ Previous 7 days'
    return '📦 Older'


def greeting():
    hour = datetime.now().hour
    if hour < 12:
        return 'Good morning ☀️'
    if hour < 18:
        return 'Good afternoon 🌤️'
    return 'Good evening 🌙'


def validate_key(key):
    try:
        OpenAI(api_key=key).models.list()
    except AuthenticationError:
        return 'OpenAI rejected that key. Check it and try again.'
    except APIConnectionError:
        return 'Could not reach OpenAI. Check your internet connection.'
    except Exception as e:
        return f'Something went wrong: {e}'
    return None


def to_langchain(messages):
    # the persona becomes the SystemMessage, followed by the whole conversation
    history = [SystemMessage(content=st.session_state.persona.strip() or DEFAULT_PERSONA)]
    for msg in messages:
        cls = HumanMessage if msg['role'] == 'user' else AIMessage
        history.append(cls(content=msg['content']))
    return history


def chunk_text(content):
    # newer models may stream a list of content blocks instead of a plain string
    if isinstance(content, str):
        return content
    return ''.join(b.get('text', '') if isinstance(b, dict) else str(b) for b in content)


# ---------------------------------------------------------------- callbacks
def new_chat():
    st.session_state.current_id = None


def open_chat(chat_id):
    st.session_state.current_id = chat_id


def delete_chat(chat_id):
    st.session_state.chats.pop(chat_id, None)
    if st.session_state.current_id == chat_id:
        st.session_state.current_id = None
    save_chats()


def clear_all():
    st.session_state.chats = {}
    st.session_state.current_id = None
    save_chats()


def change_key():
    st.session_state.api_key = None


def use_suggestion(text):
    st.session_state.pending_prompt = text


# ---------------------------------------------------------------- state
st.session_state.setdefault('api_key', None)
st.session_state.setdefault('chats', load_chats())
st.session_state.setdefault('current_id', None)
st.session_state.setdefault('persona', DEFAULT_PERSONA)

st.markdown(CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------- API key gate
if not st.session_state.api_key:
    st.markdown(GATE_CSS, unsafe_allow_html=True)
    _, middle, _ = st.columns([1, 1.6, 1])
    with middle, st.container(key='gate_card'):
        st.markdown(
            '<div class="gate-head"><div class="mark">🔮</div>'
            '<h2>Welcome to <span class="grad-text">Custom ChatGPT</span> 🎉</h2>'
            '<p>Enter your OpenAI API key to start chatting.</p></div>'
            '<div class="perks"><span>⚡ Streaming replies</span><span>🗂️ Saved chats</span>'
            '<span>🎭 Custom persona</span></div>',
            unsafe_allow_html=True,
        )
        with st.form('gate_form', border=False):
            key = st.text_input('OpenAI API key', type='password', placeholder='sk-...')
            submitted = st.form_submit_button('Get started', icon=':material/arrow_forward:', type='primary', width='stretch')
        if submitted:
            key = key.strip()
            if not key:
                st.warning('Enter your API key to continue.')
            else:
                with st.spinner('Verifying your key...'):
                    error = validate_key(key)
                if error:
                    st.error(error)
                else:
                    st.session_state.api_key = key
                    st.toast('API key verified. Let\'s chat!', icon=':material/check_circle:')
                    time.sleep(0.6)
                    st.rerun()
        st.markdown(
            '<div class="gate-note">🔒 Your key stays in this browser session and is never written to disk.<br>'
            'Need one? <a href="https://platform.openai.com/api-keys" target="_blank">Get an API key</a></div>',
            unsafe_allow_html=True,
        )
    st.stop()


# ---------------------------------------------------------------- sidebar
chats = st.session_state.chats

with st.sidebar:
    st.markdown(
        '<div class="brand"><div class="logo">🔮</div>'
        '<div><b>Custom ChatGPT</b><small>Your AI workspace</small></div></div>',
        unsafe_allow_html=True,
    )
    st.button('New chat', icon=':material/add:', on_click=new_chat, type='primary', width='stretch')
    query = st.text_input('Search', placeholder='Search chats', label_visibility='collapsed')

    ordered = sorted(chats.items(), key=lambda kv: kv[1]['updated'], reverse=True)
    if query:
        q = query.lower()
        ordered = [
            (cid, c) for cid, c in ordered
            if q in c['title'].lower() or any(q in m['content'].lower() for m in c['messages'])
        ]

    if not ordered:
        st.markdown(
            f'<div class="empty-hist">{"🤷 No matching chats" if query else "🌱 No chats yet. Send a message to start."}</div>',
            unsafe_allow_html=True,
        )

    last_group = None
    for cid, c in ordered:
        group = time_group(c['updated'])
        if group != last_group:
            st.markdown(f'<div class="group-label">{group}</div>', unsafe_allow_html=True)
            last_group = group
        row_key = 'chat_active' if cid == st.session_state.current_id else f'chatrow_{cid}'
        with st.container(key=row_key):
            left, right = st.columns([6, 1], gap='small', vertical_alignment='center')
            left.button(c['title'], key=f'open_{cid}', on_click=open_chat, args=(cid,), width='stretch')
            right.button('', key=f'del_{cid}', icon=':material/close:', on_click=delete_chat, args=(cid,), help='Delete chat')

    st.write('')
    with st.expander('Settings', icon=':material/tune:'):
        model = st.selectbox('Model', MODELS)
        temperature = st.slider('Creativity', 0.0, 1.5, 0.5, 0.1)
        st.text_area('Persona (system role)', key='persona', height=90)

    k = st.session_state.api_key
    st.markdown(
        f'<div class="keycard"><div class="dot"></div><div><b>Connected to OpenAI</b>'
        f'<code>{html.escape(k[:5])}…{html.escape(k[-4:])}</code></div></div>',
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns(2)
    c1.button('Key', icon=':material/key:', on_click=change_key, width='stretch', help='Use a different API key')
    c2.button('Clear', icon=':material/delete:', on_click=clear_all, width='stretch', help='Delete all saved chats')


# ---------------------------------------------------------------- main area
prompt = st.chat_input('Message Custom ChatGPT')
prompt = st.session_state.pop('pending_prompt', None) or prompt

current = chats.get(st.session_state.current_id)

if prompt:
    if current is None:
        chat_id = uuid.uuid4().hex[:10]
        current = {'title': make_title(prompt), 'created': time.time(), 'updated': time.time(), 'messages': []}
        chats[chat_id] = current
        st.session_state.current_id = chat_id
    current['messages'].append({'role': 'user', 'content': prompt})
    current['updated'] = time.time()
    save_chats()

if current is None:
    # empty state, ChatGPT style
    st.markdown(
        f'<div class="hero"><div class="mark">🪄</div><div class="hello">{greeting()}</div>'
        '<h1>How can I <span class="grad-text">help</span> you today?</h1>'
        '<div class="sub">✨ Pick a starter below or type your own message.</div></div>',
        unsafe_allow_html=True,
    )
    cols = st.columns(2)
    for i, (icon, title, text) in enumerate(SUGGESTIONS):
        with cols[i % 2], st.container(key=f'sugg_{i}'):
            st.button(f'{icon}  **{title}**\n\n{text}', on_click=use_suggestion, args=(text,), width='stretch')
else:
    st.markdown(
        f'<div class="topbar"><div class="t">{html.escape(current["title"])}</div>'
        f'<div class="pill">{html.escape(model)}</div></div>',
        unsafe_allow_html=True,
    )
    for i, msg in enumerate(current['messages']):
        avatar = USER_AVATAR if msg['role'] == 'user' else AI_AVATAR
        with st.chat_message(msg['role'], avatar=avatar):
            with st.container(key=f'{msg["role"]}_msg_{i}'):
                st.markdown(msg['content'])

    # the last message is from the user -> stream the assistant's answer
    if current['messages'] and current['messages'][-1]['role'] == 'user':
        with st.chat_message('assistant', avatar=AI_AVATAR):
            with st.container(key=f'assistant_msg_{len(current["messages"])}'):
                box = st.empty()
                box.markdown(TYPING_HTML, unsafe_allow_html=True)
                reply = ''
                try:
                    chat = ChatOpenAI(model=model, temperature=temperature, api_key=st.session_state.api_key)
                    for chunk in chat.stream(to_langchain(current['messages'])):
                        reply += chunk_text(chunk.content)
                        box.markdown(reply + ' ▌')
                    box.markdown(reply)
                except AuthenticationError:
                    reply = ''
                    box.error('Your API key is no longer valid. Use the Key button in the sidebar to enter a new one.')
                except Exception as e:
                    reply = ''
                    box.error(f'Something went wrong: {e}')

        if reply:
            current['messages'].append({'role': 'assistant', 'content': reply})
            current['updated'] = time.time()
            save_chats()
            st.rerun()
        else:
            # drop the unanswered message so it isn't re-sent on every rerun
            current['messages'].pop()
            if not current['messages']:
                delete_chat(st.session_state.current_id)
            save_chats()

st.markdown(
    '<div class="fineprint">✨ Custom ChatGPT can make mistakes. Check important information.</div>',
    unsafe_allow_html=True,
)