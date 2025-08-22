import os
import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="AIDOK 💬", page_icon="💬")
st.title("AIDOK 💬")

# --- 1) 배포/로컬 공통: Secrets/ENV에서 읽기 ---
BASE_URL = st.secrets.get("LLAMA_BASE_URL", os.getenv("LLAMA_BASE_URL", "https://api.ai-dok.com/v1"))
API_KEY  = st.secrets.get("LLAMA_API_KEY",  os.getenv("LLAMA_API_KEY"))
MODEL    = st.secrets.get("LLAMA_MODEL",    os.getenv("LLAMA_MODEL", "local"))

if not API_KEY:
    st.error("API 키가 없습니다. Streamlit Secrets에 LLAMA_API_KEY를 설정하세요.")
    st.stop()

# --- 2) 클라이언트 1회 생성 (리런에도 유지) ---
if "client" not in st.session_state:
    st.session_state.client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

if "history" not in st.session_state:
    st.session_state.history = []  # list[(role, content)]

# --- 3) 사이드바: 프롬프트/초기화 ---
with st.sidebar:
    clear = st.button("대화내용 초기화")
    default_sys = "You are a helpful assistant. Answer concisely and directly. Avoid repetition."
    system_prompt = st.text_area("시스템 프롬프트", value=default_sys, height=120)

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

    msg = st.chat_message("assistant")
    placeholder = msg.empty()
    collected = []

    # 메시지 배열 구성 (system 포함)
    messages = [{"role": "system", "content": system_prompt}] + [
        {"role": r, "content": c} for r, c in st.session_state.history
    ]

    try:
        stream = st.session_state.client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.2,     # 장황·반복 방지
            max_tokens=512,
            top_p=0.9,
            frequency_penalty=0.6,
            presence_penalty=0.2,
            stream=True,
            # timeout=60,        # 필요 시 주석 해제
        )

        for chunk in stream:
            # --- 방탄 처리 ---
            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue
            c0 = choices[0]

            # 최종 청크(finish_reason 존재)면 종료
            if getattr(c0, "finish_reason", None) is not None:
                break

            delta = getattr(c0, "delta", None)
            text  = getattr(delta, "content", None) if delta is not None else None
            if not text:
                continue

            collected.append(text)
            placeholder.markdown("".join(collected))

        full = "".join(collected).strip() or "_(빈 응답)_"
        st.session_state.history.append(("assistant", full))
        if full:
            placeholder.markdown(full)

    except Exception as e:
        placeholder.error(f"요청 중 오류가 발생했습니다: {e}")
