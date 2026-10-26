# install the required libraries
# pip install -q -r ./requirements.txt
# openai
# langchain
# langchain-openai
# langchain-core
# streamlit
# tiktoken

# run the app: streamlit run ./chatgpt.py

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
    page_title='Your Custom ChatGPT',
    page_icon='✨',
    layout='wide',
    initial_sidebar_state='expanded',
)

# ---------------------------------------------------------------- constants
HISTORY_FILE = Path(__file__).with_name('chat_history.json')
MODELS = ['gpt-5.2', 'gpt-4.1', 'gpt-4o', 'gpt-4o-mini']
DEFAULT_PERSONA = 'You are a helpful assistant.'
USER_AVATAR = '🧑‍💻'
AI_AVATAR = '✨'
TITLE_EMOJIS = ['💡', '🚀', '🎨', '📚', '🧩', '🌟', '🔮', '🍀', '🎯', '🪄', '🌈', '🧠']
SUGGESTIONS = [
    ('🧠', 'Explain like I\'m five', 'Explain quantum computing like I\'m five years old.'),
    ('✍️', 'Write a poem', 'Write a short, uplifting poem about Monday mornings.'),
    ('🧳', 'Plan a trip', 'Plan a 3-day itinerary for a first-time visit to Istanbul.'),
    ('🐍', 'Level up my Python', 'What are the most common Python mistakes beginners make, and how do I avoid them?'),
]
TYPING_HTML = (
    '<div class="typing"><span></span><span></span><span></span>'
    '<em>Thinking…</em></div>'
)

# ---------------------------------------------------------------- styles
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Instrument+Serif:ital@0;1&display=swap');

:root {
  --bg: #FBF8F3;
  --ink: #1F1B2E;
  --muted: #6E6884;
  --line: rgba(31, 27, 46, .08);
  --card: rgba(255, 255, 255, .78);
  --violet: #7C5CFF;
  --pink: #FF6FB5;
  --peach: #FFB86B;
  --grad: linear-gradient(120deg, #7C5CFF 0%, #FF6FB5 55%, #FFB86B 100%);
  --grad-soft: linear-gradient(135deg, #EEE8FF 0%, #FFE6F2 100%);
  --shadow: 0 12px 32px -14px rgba(76, 52, 170, .28);
}

/* ---------- base (force a light look even if the OS is in dark mode) */
html, body, .stApp, [data-testid="stAppViewContainer"] {
  background: var(--bg) !important;
  color: var(--ink) !important;
  font-family: 'Plus Jakarta Sans', sans-serif !important;
}
[data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] *,
.stApp label, .stApp label * { color: var(--ink); font-family: 'Plus Jakarta Sans', sans-serif; }
[data-testid="stMarkdownContainer"] code { background: #F2EEFF; color: #5B3FE0; border-radius: 6px; }
[data-testid="stMarkdownContainer"] pre { background: #FFFFFF !important; border: 1px solid var(--line); border-radius: 14px; }
header[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stAppDeployButton"], [data-testid="stMainMenu"], [data-testid="stDecoration"], footer { display: none !important; }
[data-testid="stMainBlockContainer"] { max-width: 880px; padding-top: 2.2rem; padding-bottom: 7rem; position: relative; z-index: 1; }

/* ---------- animated background blobs */
.blob { position: fixed; border-radius: 50%; filter: blur(70px); opacity: .6; z-index: 0;
        pointer-events: none; animation: drift 24s ease-in-out infinite alternate; }
.b1 { width: 460px; height: 460px; background: #DCCFFF; top: -140px; left: 25%; }
.b2 { width: 420px; height: 420px; background: #FFD3E8; bottom: -160px; right: -80px; animation-delay: -8s; }
.b3 { width: 360px; height: 360px; background: #FFE7C9; top: 38%; left: -120px; animation-delay: -16s; }
@keyframes drift {
  0%   { transform: translate(0, 0) scale(1); }
  50%  { transform: translate(70px, -50px) scale(1.12); }
  100% { transform: translate(-50px, 60px) scale(.94); }
}
@keyframes rise  { from { opacity: 0; transform: translateY(14px); } to { opacity: 1; transform: none; } }
@keyframes pop   { from { opacity: 0; transform: scale(.92) translateY(20px); } to { opacity: 1; transform: none; } }
@keyframes float { 0%, 100% { transform: translateY(0) rotate(-4deg); } 50% { transform: translateY(-12px) rotate(4deg); } }
@keyframes shine { to { background-position: 200% center; } }
@keyframes blink { 0%, 80%, 100% { transform: scale(.5); opacity: .35; } 40% { transform: scale(1); opacity: 1; } }
@keyframes pulse { 0%, 100% { box-shadow: 0 0 0 0 rgba(52, 199, 89, .45); } 50% { box-shadow: 0 0 0 6px rgba(52, 199, 89, 0); } }

.grad-text { background: var(--grad); background-size: 200% auto; -webkit-background-clip: text;
             background-clip: text; -webkit-text-fill-color: transparent; animation: shine 6s linear infinite; }
.serif { font-family: 'Instrument Serif', serif !important; font-style: italic; font-weight: 400; }

/* ---------- inputs */
[data-baseweb="input"], [data-baseweb="textarea"], [data-baseweb="select"] > div {
  background: #FFFFFF !important; border: 1px solid var(--line) !important; border-radius: 14px !important;
}
[data-baseweb="input"]:focus-within, [data-baseweb="textarea"]:focus-within {
  border-color: var(--violet) !important; box-shadow: 0 0 0 3px rgba(124, 92, 255, .15);
}
.stApp input, .stApp textarea, [data-baseweb="select"] * { color: var(--ink) !important; background: transparent !important; }

/* ---------- buttons */
[data-testid^="stBaseButton-secondary"] {
  background: #FFFFFF; color: var(--ink); border: 1px solid var(--line); border-radius: 14px;
  transition: transform .2s ease, box-shadow .2s ease, border-color .2s ease;
}
[data-testid^="stBaseButton-secondary"]:hover { transform: translateY(-1px); border-color: rgba(124, 92, 255, .4);
  color: var(--violet); box-shadow: var(--shadow); }
[data-testid^="stBaseButton-primary"] {
  background: var(--grad) !important; background-size: 200% auto !important; border: none !important;
  border-radius: 14px !important; color: #fff !important; font-weight: 700;
  box-shadow: 0 10px 24px -10px rgba(255, 111, 181, .7); transition: all .35s ease;
}
[data-testid^="stBaseButton-primary"] * { color: #fff !important; }
[data-testid^="stBaseButton-primary"]:hover { background-position: right center !important; transform: translateY(-2px); }

/* ---------- sidebar */
[data-testid="stSidebar"] { background: rgba(255, 255, 255, .62) !important; backdrop-filter: blur(20px);
                            border-right: 1px solid var(--line); }
[data-testid="stSidebar"] * { color: var(--ink); }
.brand { display: flex; align-items: center; gap: .7rem; margin: .2rem 0 1.1rem; animation: rise .6s both; }
.brand .logo { width: 42px; height: 42px; border-radius: 14px; background: var(--grad); display: grid;
               place-items: center; font-size: 22px; box-shadow: 0 8px 20px -8px rgba(124, 92, 255, .7);
               animation: float 5s ease-in-out infinite; }
.brand b { font-size: 1.08rem; font-weight: 800; letter-spacing: -.02em; display: block; }
.brand small { color: var(--muted) !important; font-size: .75rem; }
.group-label { font-size: .7rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase;
               color: var(--muted) !important; margin: 1rem 0 .25rem .3rem; }
.empty-hist { text-align: center; color: var(--muted) !important; font-size: .85rem; padding: 1.2rem .5rem;
              border: 1px dashed rgba(124, 92, 255, .3); border-radius: 16px; margin-top: .8rem; }

[class*="st-key-chatrow_"], .st-key-chat_active { border-radius: 12px; transition: background .2s ease; animation: rise .4s both; }
[class*="st-key-chatrow_"]:hover { background: rgba(124, 92, 255, .07); }
.st-key-chat_active { background: #FFFFFF; box-shadow: var(--shadow); position: relative; }
.st-key-chat_active::before { content: ''; position: absolute; left: 0; top: 22%; bottom: 22%; width: 3px;
                              border-radius: 3px; background: var(--grad); }
[class*="st-key-chatrow_"] button, .st-key-chat_active button {
  background: transparent !important; border: none !important; box-shadow: none !important;
  justify-content: flex-start; padding: .35rem .55rem; min-height: 0; transform: none !important;
}
[class*="st-key-chatrow_"] button p, .st-key-chat_active button p {
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis; font-size: .87rem; text-align: left;
}
.st-key-chat_active button p { font-weight: 700; }
[class*="st-key-chatrow_"] [data-testid="stColumn"]:last-child,
.st-key-chat_active [data-testid="stColumn"]:last-child { opacity: 0; transition: opacity .2s ease; }
[class*="st-key-chatrow_"]:hover [data-testid="stColumn"]:last-child,
.st-key-chat_active:hover [data-testid="stColumn"]:last-child { opacity: 1; }
[class*="st-key-del_"] button:hover p { color: #E5484D !important; }

[data-testid="stSidebar"] [data-testid="stExpander"] details { background: rgba(255, 255, 255, .75);
  border: 1px solid var(--line); border-radius: 16px; }
.keycard { display: flex; align-items: center; gap: .6rem; background: #FFFFFF; border: 1px solid var(--line);
           border-radius: 16px; padding: .7rem .85rem; margin: .9rem 0 .6rem; box-shadow: var(--shadow); }
.keycard .dot { width: 9px; height: 9px; border-radius: 50%; background: #34C759; animation: pulse 2s infinite; }
.keycard b { font-size: .82rem; display: block; }
.keycard code { font-size: .72rem; color: var(--muted) !important; background: none !important; }

/* ---------- top bar */
.topbar { display: flex; justify-content: space-between; align-items: center; gap: 1rem; padding: .7rem 1rem;
          background: var(--card); backdrop-filter: blur(14px); border: 1px solid rgba(255, 255, 255, .9);
          border-radius: 18px; box-shadow: var(--shadow); margin-bottom: 1.2rem; animation: rise .5s both; }
.topbar .t { font-weight: 700; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.pill { font-size: .75rem; font-weight: 700; padding: .3rem .75rem; border-radius: 999px; white-space: nowrap;
        background: var(--grad-soft); color: #5B3FE0 !important; border: 1px solid rgba(124, 92, 255, .2); }

/* ---------- hero / empty state */
.hero { text-align: center; margin: 7vh 0 2.2rem; animation: rise .7s both; }
.hero .orb { font-size: 68px; display: inline-block; animation: float 4s ease-in-out infinite;
             filter: drop-shadow(0 14px 18px rgba(124, 92, 255, .35)); }
.hero .hello { color: var(--muted) !important; font-weight: 600; margin-top: .6rem; }
.hero h1 { font-size: clamp(2rem, 4.4vw, 3.3rem) !important; font-weight: 800 !important; letter-spacing: -.035em;
           line-height: 1.1 !important; margin: .3rem 0 0 !important; padding: 0 !important; }
.hero h1 .serif { font-size: 1.1em; }

[class*="st-key-sugg_"] { animation: rise .6s both; }
.st-key-sugg_1 { animation-delay: .08s; } .st-key-sugg_2 { animation-delay: .16s; } .st-key-sugg_3 { animation-delay: .24s; }
[class*="st-key-sugg_"] button {
  height: 108px; width: 100%; background: var(--card) !important; backdrop-filter: blur(10px);
  border: 1px solid rgba(255, 255, 255, .95) !important; border-radius: 20px !important;
  box-shadow: 0 8px 22px -14px rgba(76, 52, 170, .35); align-items: flex-start; justify-content: flex-start;
  padding: 1rem 1.1rem !important; transition: transform .25s cubic-bezier(.2, .8, .2, 1), box-shadow .25s ease !important;
}
[class*="st-key-sugg_"] button p { font-weight: 700; text-align: left; font-size: .92rem; line-height: 1.6; }
[class*="st-key-sugg_"] button:hover { transform: translateY(-5px) rotate(-.6deg) !important;
  box-shadow: 0 20px 34px -16px rgba(124, 92, 255, .5) !important; border-color: rgba(124, 92, 255, .35) !important; }

/* ---------- chat messages */
[data-testid="stChatMessage"] { background: transparent !important; padding: .35rem 0; gap: .75rem;
                                animation: rise .45s cubic-bezier(.2, .8, .2, 1) both; }
[data-testid="stChatMessage"] [data-testid^="stChatMessageAvatar"] {
  background: #FFFFFF; border-radius: 14px; width: 38px; height: 38px; font-size: 20px;
  box-shadow: 0 6px 16px -8px rgba(76, 52, 170, .4); border: 1px solid var(--line);
}
[data-testid="stChatMessage"]:has([class*="st-key-assistant_msg_"]) [data-testid="stChatMessageContent"] {
  background: #FFFFFF; border: 1px solid var(--line); border-radius: 6px 20px 20px 20px;
  padding: .85rem 1.15rem; box-shadow: var(--shadow);
}
[data-testid="stChatMessage"]:has([class*="st-key-user_msg_"]) { flex-direction: row-reverse; }
[data-testid="stChatMessage"]:has([class*="st-key-user_msg_"]) [data-testid="stChatMessageContent"] {
  background: var(--grad-soft); border: 1px solid rgba(124, 92, 255, .15); border-radius: 20px 6px 20px 20px;
  padding: .75rem 1.1rem; flex: 0 1 auto; max-width: 78%; margin-left: auto;
}

.typing { display: flex; align-items: center; gap: 6px; padding: .2rem 0; }
.typing span { width: 8px; height: 8px; border-radius: 50%; background: var(--grad); animation: blink 1.2s infinite both; }
.typing span:nth-child(2) { animation-delay: .15s; } .typing span:nth-child(3) { animation-delay: .3s; }
.typing em { margin-left: .4rem; color: var(--muted) !important; font-size: .85rem; }

/* ---------- chat input */
[data-testid="stBottom"] > div { background: linear-gradient(to top, var(--bg) 65%, rgba(251, 248, 243, 0)) !important; }
[data-testid="stBottomBlockContainer"] { max-width: 880px; background: transparent !important; }
[data-testid="stChatInput"] { background: #FFFFFF !important; border: 1px solid var(--line) !important;
  border-radius: 24px !important; box-shadow: var(--shadow); transition: box-shadow .25s ease; }
[data-testid="stChatInput"]:focus-within { box-shadow: 0 0 0 3px rgba(124, 92, 255, .18), var(--shadow); }
[data-testid="stChatInput"] > div { background: transparent !important; border: none !important; }
[data-testid="stChatInputSubmitButton"] { background: var(--grad) !important; border-radius: 14px !important; }
[data-testid="stChatInputSubmitButton"] svg { fill: #fff !important; color: #fff !important; }
.fineprint { text-align: center; font-size: .72rem; color: var(--muted) !important; margin-top: .4rem; }

/* ---------- API key gate */
.st-key-gate_card { background: rgba(255, 255, 255, .78); backdrop-filter: blur(22px);
  border: 1px solid rgba(255, 255, 255, .95); border-radius: 30px; padding: 2.4rem 2.2rem 1.8rem;
  box-shadow: 0 30px 80px -30px rgba(76, 52, 170, .4); margin-top: 9vh;
  animation: pop .8s cubic-bezier(.2, .9, .3, 1.25) both; }
.st-key-gate_card [data-testid="stForm"] { border: none; padding: 0; }
.gate-head { text-align: center; }
.gate-head .orb { font-size: 60px; display: inline-block; animation: float 4s ease-in-out infinite; }
.gate-head h2 { font-size: 1.9rem !important; font-weight: 800 !important; letter-spacing: -.03em; padding: .4rem 0 .2rem !important; }
.gate-head p { color: var(--muted) !important; }
.perks { display: flex; justify-content: center; flex-wrap: wrap; gap: .45rem; margin: .9rem 0 1.3rem; }
.perks span { font-size: .75rem; font-weight: 600; padding: .3rem .7rem; border-radius: 999px;
              background: var(--grad-soft); border: 1px solid rgba(124, 92, 255, .15); }
.gate-note { text-align: center; font-size: .78rem; color: var(--muted) !important; margin-top: .9rem; }
.gate-note a { color: var(--violet) !important; font-weight: 700; text-decoration: none; }
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
    emoji = TITLE_EMOJIS[sum(map(ord, text)) % len(TITLE_EMOJIS)]
    return f'{emoji} {text[:38]}{"…" if len(text) > 38 else ""}'


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
        return '🙅 OpenAI rejected that key. Double-check it and try again.'
    except APIConnectionError:
        return '📡 Could not reach OpenAI. Please check your internet connection.'
    except Exception as e:
        return f'😕 Something went wrong: {e}'
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
            '<div class="gate-head"><div class="orb">🤖</div>'
            '<h2>Welcome to your <span class="grad-text">Custom ChatGPT</span></h2>'
            '<p>Plug in your OpenAI API key to unlock your personal AI studio.</p></div>'
            '<div class="perks"><span>⚡ Streaming replies</span><span>🗂️ Saved chats</span>'
            '<span>🎭 Custom persona</span></div>',
            unsafe_allow_html=True,
        )
        with st.form('gate_form', border=False):
            key = st.text_input('🔑 OpenAI API key', type='password', placeholder='sk-...')
            submitted = st.form_submit_button('🚀 Unlock the magic', type='primary', width='stretch')
        if submitted:
            key = key.strip()
            if not key:
                st.warning('✋ Please paste your API key first.')
            else:
                with st.spinner('🔍 Verifying your key...'):
                    error = validate_key(key)
                if error:
                    st.error(error)
                else:
                    st.session_state.api_key = key
                    st.toast('Key verified — let\'s chat!', icon='🎉')
                    time.sleep(0.6)
                    st.rerun()
        st.markdown(
            '<div class="gate-note">🔒 Your key lives only in this browser session and is never written to disk.<br>'
            'Need one? <a href="https://platform.openai.com/api-keys" target="_blank">Get an API key ↗</a></div>',
            unsafe_allow_html=True,
        )
    st.stop()


# ---------------------------------------------------------------- sidebar
chats = st.session_state.chats

with st.sidebar:
    st.markdown(
        '<div class="brand"><div class="logo">✨</div>'
        '<div><b>Custom ChatGPT</b><small>your personal AI studio</small></div></div>',
        unsafe_allow_html=True,
    )
    st.button('✏️  New chat', on_click=new_chat, type='primary', width='stretch')
    query = st.text_input('Search', placeholder='🔎  Search chats...', label_visibility='collapsed')

    ordered = sorted(chats.items(), key=lambda kv: kv[1]['updated'], reverse=True)
    if query:
        q = query.lower()
        ordered = [
            (cid, c) for cid, c in ordered
            if q in c['title'].lower() or any(q in m['content'].lower() for m in c['messages'])
        ]

    if not ordered:
        st.markdown(
            f'<div class="empty-hist">{"🤷 No matching chats" if query else "🌱 No chats yet — say hi!"}</div>',
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
            right.button('✕', key=f'del_{cid}', on_click=delete_chat, args=(cid,), help='Delete chat')

    st.write('')
    with st.expander('⚙️  Settings'):
        model = st.selectbox('🧬 Model', MODELS)
        temperature = st.slider('🎨 Creativity', 0.0, 1.5, 0.5, 0.1)
        st.text_area('🎭 Persona (system role)', key='persona', height=90)

    k = st.session_state.api_key
    st.markdown(
        f'<div class="keycard"><div class="dot"></div><div><b>Connected to OpenAI</b>'
        f'<code>{html.escape(k[:5])}…{html.escape(k[-4:])}</code></div></div>',
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns(2)
    c1.button('🔁 Key', on_click=change_key, width='stretch', help='Use a different API key')
    c2.button('🧹 Clear', on_click=clear_all, width='stretch', help='Delete all saved chats')


# ---------------------------------------------------------------- main area
prompt = st.chat_input('Message Custom ChatGPT... ✨')
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
        f'<div class="hero"><div class="orb">🪄</div><div class="hello">{greeting()}</div>'
        '<h1>How can I <span class="serif grad-text">help</span> you today?</h1></div>',
        unsafe_allow_html=True,
    )
    cols = st.columns(2)
    for i, (emoji, title, text) in enumerate(SUGGESTIONS):
        with cols[i % 2], st.container(key=f'sugg_{i}'):
            st.button(f'{emoji}\n\n{title}', on_click=use_suggestion, args=(text,), width='stretch')
else:
    st.markdown(
        f'<div class="topbar"><div class="t">{html.escape(current["title"])}</div>'
        f'<div class="pill">⚡ {html.escape(model)}</div></div>',
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
                    box.error('🔑 Your API key stopped working. Use the 🔁 Key button in the sidebar to enter a new one.')
                except Exception as e:
                    reply = ''
                    box.error(f'😵 Oops, something went wrong: {e}')

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
    '<div class="fineprint">✨ Custom ChatGPT can make mistakes. Check important info.</div>',
    unsafe_allow_html=True,
)
