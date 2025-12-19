"""
Naive Chatbot v0.1
Gemini 2.5 Flash Lite 기반 간단한 챗봇
"""
import streamlit as st
import google.generativeai as genai

# 페이지 설정
st.set_page_config(
    page_title="AI 챗봇",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 AI 챗봇 v0.1")
st.caption("Gemini 2.5 Flash Lite 기반")

# API 키 설정
try:
    api_key = st.secrets["GEMINI_API_KEY"]
except KeyError:
    api_key = st.sidebar.text_input("Gemini API Key", type="password")
    if not api_key:
        st.info("👈 사이드바에 API 키를 입력해주세요.")
        st.stop()

# Gemini 모델 설정
genai.configure(api_key=api_key)

# 시스템 프롬프트
SYSTEM_PROMPT = """당신은 친절하고 도움이 되는 AI 비서입니다.

역할:
- 사용자의 질문에 정확하고 친절하게 답변합니다
- 어려운 개념은 쉽게 풀어서 설명합니다
- 한국어로 답변합니다

규칙:
- 모르는 것은 솔직히 "모르겠습니다"라고 말합니다
- 불확실한 정보는 "~일 수 있습니다"로 표현합니다
- 유해하거나 부적절한 요청은 정중히 거절합니다
"""

model = genai.GenerativeModel(
    "gemini-2.5-flash-lite",  # Rate Limit 회피용 모델
    system_instruction=SYSTEM_PROMPT
)

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat" not in st.session_state:
    st.session_state.chat = model.start_chat(history=[])

# 저장된 메시지 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 사용자 입력 처리
if prompt := st.chat_input("무엇이든 물어보세요!"):
    # 사용자 메시지 표시
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    # AI 응답 생성
    with st.chat_message("assistant"):
        with st.spinner("생각 중..."):
            try:
                response = st.session_state.chat.send_message(prompt)
                st.markdown(response.text)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response.text
                })
            except Exception as e:
                st.error(f"오류가 발생했습니다: {e}")

# 사이드바
with st.sidebar:
    st.title("⚙️ 설정")

    # 대화 초기화 버튼
    if st.button("🗑️ 대화 초기화", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chat = model.start_chat(history=[])
        st.rerun()

    st.markdown("---")

    # 대화 통계
    st.subheader("📊 대화 통계")
    st.write(f"총 메시지: {len(st.session_state.messages)}개")
    user_msgs = len([m for m in st.session_state.messages if m["role"] == "user"])
    st.write(f"사용자: {user_msgs}개")
    st.write(f"AI: {len(st.session_state.messages) - user_msgs}개")

    st.markdown("---")
    st.caption("Made with ❤️ using Streamlit & Gemini")
