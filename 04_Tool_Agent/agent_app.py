"""
검색 Agent 챗봇 v0.3
Gemini 2.5 Flash Lite + DuckDuckGo 검색 기반 Agent
"""
import streamlit as st
import os
from datetime import datetime

# LangChain 관련 임포트
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool

# 페이지 설정
st.set_page_config(
    page_title="검색 Agent",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 검색 에이전트 챗봇 v0.3")
st.caption("Gemini 2.5 Flash Lite + DuckDuckGo 검색")


# ============ Tool 정의 ============

@tool
def web_search(query: str) -> str:
    """웹에서 최신 정보를 검색합니다.

    Args:
        query: 검색할 키워드나 질문
    """
    try:
        from ddgs import DDGS
        results = DDGS().text(query, region='kr-kr', max_results=5)
        if not results:
            return "검색 결과가 없습니다."

        output = []
        for r in results:
            output.append(f"- {r['title']}: {r['body']}")
        return "\n".join(output)
    except Exception as e:
        return f"검색 중 오류 발생: {str(e)}"


@tool
def get_current_time() -> str:
    """현재 날짜와 시간을 반환합니다."""
    return datetime.now().strftime("%Y년 %m월 %d일 %H시 %M분 %S초")


@tool
def calculate(expression: str) -> str:
    """수학 표현식을 계산합니다.

    Args:
        expression: 계산할 수식 (예: "2 + 2 * 3", "100 / 4")
    """
    try:
        # 안전하지 않은 함수 제거
        allowed_chars = set("0123456789+-*/.() ")
        if not all(c in allowed_chars for c in expression):
            return "허용되지 않는 문자가 포함되어 있습니다."
        result = eval(expression)
        return f"{expression} = {result}"
    except Exception as e:
        return f"계산 오류: {str(e)}"


# ============ API 키 설정 ============

try:
    api_key = st.secrets["GEMINI_API_KEY"]
except KeyError:
    api_key = st.sidebar.text_input("Gemini API Key", type="password")
    if not api_key:
        st.info("👈 사이드바에 API 키를 입력해주세요.")
        st.stop()

os.environ["GOOGLE_API_KEY"] = api_key


# ============ Agent 설정 ============

@st.cache_resource
def create_agent(_api_key):
    """Agent 생성 (캐싱)"""
    # LLM 설정
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",  # Rate Limit 회피
        google_api_key=_api_key,
        temperature=0
    )

    # 도구 목록
    tools = [web_search, get_current_time, calculate]

    # 프롬프트 설정
    prompt = ChatPromptTemplate.from_messages([
        ("system", """당신은 웹 검색과 계산이 가능한 AI 비서입니다.

역할:
- 최신 정보가 필요하면 web_search를 사용합니다
- 계산이 필요하면 calculate를 사용합니다
- 현재 시간이 필요하면 get_current_time을 사용합니다

규칙:
- 항상 한국어로 답변합니다
- 검색 결과를 바탕으로 정확한 정보를 제공합니다
- 불확실한 정보는 솔직히 말합니다
- 검색 결과가 없으면 그렇다고 알려줍니다"""),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    # Agent 생성
    agent = create_tool_calling_agent(llm, tools, prompt)

    # 실행기 생성
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=5,
        max_execution_time=60,
        handle_parsing_errors=True
    )

    return agent_executor


# ============ 세션 상태 초기화 ============

if "messages" not in st.session_state:
    st.session_state.messages = []


# ============ 사이드바 ============

with st.sidebar:
    st.header("⚙️ 설정")

    if st.button("🗑️ 대화 초기화", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")

    st.subheader("📊 대화 통계")
    st.write(f"총 메시지: {len(st.session_state.messages)}개")

    st.markdown("---")

    st.subheader("🛠️ 사용 가능한 도구")
    st.markdown("""
    - 🔍 **웹 검색**: 최신 정보 검색
    - 🕐 **현재 시간**: 날짜/시간 확인
    - 🧮 **계산기**: 수학 계산
    """)

    st.markdown("---")

    st.subheader("💡 질문 예시")
    st.markdown("""
    - "오늘 날씨 어때?"
    - "최신 AI 뉴스 알려줘"
    - "100 * 25 + 300 계산해줘"
    - "지금 몇 시야?"
    - "비트코인 현재 시세는?"
    """)


# ============ 메인 채팅 영역 ============

# 저장된 메시지 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 입력 처리
if prompt_input := st.chat_input("무엇이든 물어보세요..."):
    # 사용자 메시지 표시
    with st.chat_message("user"):
        st.markdown(prompt_input)
    st.session_state.messages.append({"role": "user", "content": prompt_input})

    # Agent 응답 생성
    with st.chat_message("assistant"):
        with st.spinner("검색 및 분석 중..."):
            try:
                agent_executor = create_agent(api_key)
                response = agent_executor.invoke({
                    "input": prompt_input
                })
                answer = response["output"]
                st.markdown(answer)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })
            except Exception as e:
                error_msg = f"오류가 발생했습니다: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })
