import os, re
import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="AIDOK 💬 (Instruct)", page_icon="💬")
st.title("AIDOK 💬 (Instruct)")

# --- Secrets/ENV ---
BASE_URL = st.secrets.get("LLAMA_BASE_URL", os.getenv("LLAMA_BASE_URL", "https://api.ai-dok.com/v1"))
API_KEY  = st.secrets.get("LLAMA_API_KEY",  os.getenv("LLAMA_API_KEY"))
# llama.cpp는 현재 로드된 모델을 'local'로 쓰는 게 가장 안전함.
# KoEn 모델을 명시하고 싶다면 /v1/models로 확인한 이름을 넣어도 되지만, 보통 'local' 권장.
MODEL    = st.secrets.get("LLAMA_MODEL", "local")

if not API_KEY:
    st.error("API 키가 없습니다. Streamlit Secrets에 LLAMA_API_KEY를 설정하세요.")
    st.stop()

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

if "history" not in st.session_state:
    st.session_state.history = []  # list[(role, content)]

# 간단 중복 제거(연속 문장 중복 방지)
def dedupe_sentences(text: str) -> str:
    sents = re.split(r'(?<=[.!?。…])\s+', text.strip())
    out = []
    for s in sents:
        s = s.strip()
        if s and (not out or s != out[-1]):
            out.append(s)
    return " ".join(out).strip()

# Instruction 래퍼 (instruct 전용)
def wrap_instruction(user_query: str, context: str = "") -> str:
    # 과거 히스토리의 길이를 줄이기 위해 간단 요약/첫 줄만 포함(반복 유발 방지)
    ctx = context.strip()
    if ctx:
        instr = f"{ctx}\n\nUser: {user_query}"
    else:
        instr = user_query
    return (
        "Answer in one short sentence without repetition or invented facts.\n\n"
        "Below is an instruction.\n\n"
        f"### Instruction:\n{instr}\n\n"
        "### Response:"
    )

# --- Sidebar ---
with st.sidebar:
    clear = st.button("대화내용 초기화")
    instruction_wrapper = st.text_area(
        "Instruction Wrapper",
        value=(
            "Answer in one short sentence without repetition or invented facts.\n\n"
            "Below is an instruction.\n\n"
            "### Instruction:\n{instruction}\n\n"
            "### Response:"
        ),
        height=160,
        help="필요하면 커스터마이즈하세요. (KoEn instruct 모델용)"
    )
    max_turns = st.number_input("최근 전송 턴 수(요약)", 1, 10, 3)
    max_tokens = st.number_input("max_tokens", 32, 1024, 80, 16)
    temperature = st.slider("temperature", 0.0, 1.0, 0.2, 0.05)
    freq_pen = st.slider("frequency_penalty", 0.0, 2.0, 1.0, 0.1)
    pres_pen = st.slider("presence_penalty", 0.0, 2.0, 0.3, 0.1)

if clear:
    st.session_state.history.clear()
    st.rerun()

# 히스토리 렌더
for role, content in st.session_state.history:
    st.chat_message(role).write(content)

# 최근 컨텍스트 가볍게 구성(반복 줄이기 위해 한두 줄만)
def build_context():
    recent = st.session_state.history[-(max_turns*2):]
    lines = []
    for r, c in recent:
        if r == "user":
            lines.append(f"User: {c.splitlines()[0]}")
        elif r == "assistant":
            lines.append(f"Assistant: {c.splitlines()[0]}")
    return "\n".join(lines)

if user_in := st.chat_input("질문을 입력하세요"):
    st.session_state.history.append(("user", user_in))
    st.chat_message("user").write(user_in)

    ctx = build_context()
    # 사용자가 사이드바에서 래퍼를 수정했을 수도 있으니 반영
    prompt = instruction_wrapper.format(instruction=(ctx + f"\n\nUser: {user_in}" if ctx else user_in))

    msg = st.chat_message("assistant")
    placeholder = msg.empty()
    collected = []

    try:
        # Instruct 모델 → completions API
        stream = client.completions.create(
            model=MODEL,
            prompt=prompt,
            stream=True,
            temperature=float(temperature),
            max_tokens=int(max_tokens),
            top_p=0.85,
            frequency_penalty=float(freq_pen),
            presence_penalty=float(pres_pen),
            stop=["### Instruction:", "###", "\n\n###"]
        )

        for chunk in stream:
            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue
            piece = getattr(choices[0], "text", "") or ""
            if not piece:
                continue
            collected.append(piece)
            partial = dedupe_sentences("".join(collected))
            placeholder.markdown(partial)

        final = dedupe_sentences("".join(collected)).strip() or "_(빈 응답)_"
        placeholder.markdown(final)
        st.session_state.history.append(("assistant", final))

    except Exception as e:
        placeholder.error(f"요청 중 오류가 발생했습니다: {e}")
