import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import ChatMessage
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.prompts import load_prompt


st.set_page_config(page_title="AIDOK 💬", page_icon="💬")
st.title("AIDOK 💬")

if "messages" not in st.session_state:
    st.session_state["messages"] = []


def print_history():
    for msg in st.session_state["messages"]:
        st.chat_message(msg.role).write(msg.content)


def add_history(role, content):
    st.session_state["messages"].append(ChatMessage(role=role, content=content))


# 체인을 생성합니다.
def create_chain(prompt, model):
    llm = ChatOpenAI(
        base_url="http://172.24.128.1:5555/v1", 
        api_key="lm-studio",
        model=model,
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()],
    )       
    
    chain = prompt | llm | StrOutputParser()
    return chain


with st.sidebar:
    clear_btn = st.button("대화내용 초기화")
    prompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request."""
    system_prompt = st.text_area("프롬프트", value=prompt)
    prompt_template = system_prompt + "\n\n###Instruction:\n{instruction}\n\n###Response:"
    prompt = PromptTemplate.from_template(prompt_template)
    st.session_state["chain"] = create_chain(prompt, "KB8407/Llama-3-KoEn-8B-Instruct-AIDOK-gguf")


if clear_btn:
    retriever = st.session_state["messages"].clear()

print_history()


if "chain" not in st.session_state:
    # user_prompt
    prompt_template = system_prompt + "\n\n###Instruction:\n{instruction}\n\n###Response:"
    prompt = PromptTemplate.from_template(prompt_template)
    st.session_state["chain"] = create_chain(prompt, "KB8407/Llama-3-KoEn-8B-Instruct-AIDOK-gguf")

if user_input := st.chat_input():
    add_history("user", user_input)
    st.chat_message("user").write(user_input)
    with st.chat_message("assistant"):
        chat_container = st.empty()

        stream_response = st.session_state["chain"].stream(
            {"instruction": user_input}
        )  # 문서에 대한 질의
        ai_answer = ""
        for chunk in stream_response:
            ai_answer += chunk
            chat_container.markdown(ai_answer)
        add_history("ai", ai_answer)