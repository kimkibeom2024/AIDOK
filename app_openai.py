import os, re
import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="AIDOK 💬", page_icon="💬")
st.title("AIDOK 💬")

# --- 1) 배포/로컬 공통: Secrets/ENV ---
BASE_URL = st.secrets.get("LLAMA_BASE_URL", os.getenv("LLAMA_BASE_URL", "https://api.ai-dok.com/v1"))
API_KEY  = st.secrets.get("LLAMA_API_KEY",  os.getenv("LLAMA_API_KEY"))
MODEL    = st.secrets.get("LLAMA_MODEL",    os.getenv("LLAMA_MODEL", "local"))

if not API_KEY:
    st.error("API 키가 없습니다. Streamlit Secrets에 LLAMA_API_KEY를 설정하세요.")
    st.stop()

# --- 2) 클라이언트 1회 생성 ---
if "client" not in st.session_state:
    st.session_state.client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

if "history" not in st.session_state:
    st.session_state.history = []  # list[(role, content)]

# ---------- 반복 억제 유틸 ----------
def dedupe_sentences(text: str) -> str:
    """연속 중복 문장을 제거"""
    # 문장 단위로 쪼개고, 인접 중복만 제거(순서 유지)
    sents = re.split(r'(?<=[.!?。…]|[.!?]\))\s+', text.strip())
    out = []
    for s in sents:
        s = s.strip()
        if not s:
            continue
        if not out or s != out[-1]:
            out.append(s)
    return " ".join(out).strip()

def dedupe_soft(text: str) -> str:
    """토큰 스트림 중간에 자주 호출하는 가벼운 중복 억제(짧은 꼬리 반복 제거)"""
    # 끝부분에서 같은 구절이 두 번 이상 반복되면 하나만 남김
    # 예: "…합니다. 합니다. 합니다." -> "…합니다."
    text = re.sub(r'(\b[^，,。.?!\s]{2,}[，,。.?!]?)\s+\1(\s+\1)+', r'\1', text)
    text = re.sub(r'(합니다[.]?\s*)(\1)+', r'\1', text)
    return text

# --- 3) 사이드바: 프롬프트/초기화 ---
with st.sidebar:
    clear = st.button("대화내용 초기화")
    default_sys = (
        "Answer clearly in one or two sentences. "
        "Do not repeat words or ideas. If unsure, say you are not sure."
    )
    system_prompt = st.text_area("시스템 프롬프트", value=default_sys, height=120)
    max_turns = st.number_input("최근 전송 턴 수", min_value=1, max_value=10, value=3, step=1)
    max_tokens = st.number_input("max_tokens", min_value=32, max_value=1024, value=160, step=16)

if clear:
    st.session_state.history.clear()
    st.rerun()

# --- 4) 기록 렌더링 ---
for role, content in st.session_state.history:
    st.chat_message(role).write(content)

# --- 5) 입력 & 스트리밍 출력 ---
if user_in := st.chat_input("메시지를 입력하세요"):
    st.session_state.history.append(("user", user_in))
    st.chat_message("user").write(user_in)

    # 최근 N턴만 전송 (system은 항상 포함)
    recent = st.session_state.history[-(max_turns * 2):]
    messages = [{"role": "system", "content": system_prompt}] + [
        {"role": r, "content": c} for r, c in recent
    ]

    msg = st.chat_message("assistant")
    placeholder = msg.empty()
    collected = []

    try:
        stream = st.session_state.client.chat.completions.create(
            model=MODEL,
            messages=messages,
            stream=True,
            # ---- 반복·장황 억제 파라미터 ----
            temperature=0.2,
            max_tokens=int(max_tokens),
            top_p=0.85,
            frequency_penalty=0.9,   # 강하게
            presence_penalty=0.3,
            # stop=["\n\n\n"],       # 필요시 강제 중단 트리거 추가(권장되진 않음)
        )

        for chunk in stream:
            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue
            c0 = choices[0]

            # 최종 청크면 종료
            if getattr(c0, "finish_reason", None) is not None:
                break

            delta = getattr(c0, "delta", None)
            text  = getattr(delta, "content", None) if delta is not None else None
            if not text:
                continue

            collected.append(text)
            partial = "".join(collected)
            # 스트리밍 중간에도 가벼운 중복 정리
            partial = dedupe_soft(partial)
            placeholder.markdown(partial)

        full = "".join(collected)
        full = dedupe_sentences(dedupe_soft(full)).strip()
        if not full:
            full = "_(빈 응답)_"
        placeholder.markdown(full)
        st.session_state.history.append(("assistant", full))

    except Exception as e:
        placeholder.error(f"요청 중 오류가 발생했습니다: {e}")
