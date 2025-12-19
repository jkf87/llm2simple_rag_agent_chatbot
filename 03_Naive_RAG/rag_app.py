"""
RAG 챗봇 v0.2
Gemini 2.5 Flash Lite + ChromaDB 기반 문서 QA 챗봇
"""
import streamlit as st
import google.generativeai as genai
import chromadb
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 페이지 설정
st.set_page_config(
    page_title="RAG 챗봇",
    page_icon="📚",
    layout="wide"
)

st.title("📚 RAG 챗봇 v0.2")
st.caption("문서 기반 질의응답 시스템 (Gemini + ChromaDB)")

# API 키 설정
try:
    api_key = st.secrets["GEMINI_API_KEY"]
except KeyError:
    api_key = st.sidebar.text_input("Gemini API Key", type="password")
    if not api_key:
        st.info("👈 사이드바에 API 키를 입력해주세요.")
        st.stop()

genai.configure(api_key=api_key)


# ============ 헬퍼 함수들 ============

def get_pdf_text(pdf_docs):
    """PDF 파일들에서 텍스트 추출"""
    text = ""
    for pdf in pdf_docs:
        try:
            reader = PdfReader(pdf)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        except Exception as e:
            st.warning(f"PDF 읽기 오류: {e}")
    return text


def get_text_chunks(text):
    """텍스트를 청크로 분할"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )
    chunks = splitter.split_text(text)
    return chunks


def get_embedding(text):
    """단일 텍스트를 임베딩 벡터로 변환"""
    try:
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=text
        )
        return result['embedding']
    except Exception as e:
        st.error(f"임베딩 오류: {e}")
        return None


def get_embeddings(texts):
    """여러 텍스트를 임베딩 벡터로 변환"""
    embeddings = []
    progress_bar = st.progress(0)
    for i, text in enumerate(texts):
        emb = get_embedding(text)
        if emb:
            embeddings.append(emb)
        progress_bar.progress((i + 1) / len(texts))
    progress_bar.empty()
    return embeddings


def create_vector_store(chunks):
    """청크들을 ChromaDB에 저장"""
    client = chromadb.Client()

    # 기존 컬렉션 삭제 후 재생성
    try:
        client.delete_collection("documents")
    except:
        pass

    collection = client.create_collection(
        name="documents",
        metadata={"hnsw:space": "cosine"}
    )

    # 임베딩 생성
    st.info("임베딩 생성 중...")
    embeddings = get_embeddings(chunks)

    if len(embeddings) != len(chunks):
        st.error("일부 청크의 임베딩 생성에 실패했습니다.")
        return None

    # ChromaDB에 저장
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[f"chunk_{i}" for i in range(len(chunks))],
        metadatas=[{"chunk_id": i} for i in range(len(chunks))]
    )

    return collection


def answer_question(question, collection):
    """질문에 대한 RAG 기반 답변 생성"""
    # 1. 질문 임베딩
    q_embedding = get_embedding(question)
    if q_embedding is None:
        return "임베딩 생성에 실패했습니다.", None

    # 2. 유사 문서 검색
    results = collection.query(
        query_embeddings=[q_embedding],
        n_results=4
    )

    if not results['documents'][0]:
        return "관련 문서를 찾을 수 없습니다.", None

    # 3. 컨텍스트 구성
    context = "\n\n---\n\n".join(results['documents'][0])

    # 디버깅: 검색된 문서와 유사도 점수 저장
    retrieved_docs = []
    for i, (doc, distance) in enumerate(zip(results['documents'][0], results['distances'][0])):
        similarity = 1 - distance  # cosine distance를 similarity로 변환
        retrieved_docs.append({
            "rank": i + 1,
            "similarity": f"{similarity:.3f}",
            "preview": doc[:200] + "..." if len(doc) > 200 else doc
        })

    # 4. RAG 프롬프트 생성
    prompt = f"""다음 문서들을 참고하여 질문에 정확하게 답변하세요.
문서에 없는 내용은 "문서에서 해당 정보를 찾을 수 없습니다"라고 답변하세요.

[참고 문서]
{context}

[질문]
{question}

[답변]"""

    # 5. Gemini로 답변 생성
    try:
        model = genai.GenerativeModel("gemini-2.5-flash-lite")
        response = model.generate_content(prompt)
        return response.text, retrieved_docs
    except Exception as e:
        return f"답변 생성 오류: {e}", retrieved_docs


# ============ 세션 상태 초기화 ============

if "collection" not in st.session_state:
    st.session_state.collection = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "processed" not in st.session_state:
    st.session_state.processed = False


# ============ 사이드바 ============

with st.sidebar:
    st.header("📁 문서 업로드")

    pdf_docs = st.file_uploader(
        "PDF 파일을 업로드하세요",
        accept_multiple_files=True,
        type=['pdf']
    )

    if st.button("📥 문서 처리", use_container_width=True, type="primary"):
        if pdf_docs:
            with st.spinner("문서 처리 중..."):
                # 1. 텍스트 추출
                st.info("1/3: 텍스트 추출 중...")
                raw_text = get_pdf_text(pdf_docs)

                if not raw_text.strip():
                    st.error("PDF에서 텍스트를 추출할 수 없습니다.")
                else:
                    # 2. 청킹
                    st.info("2/3: 문서 분할 중...")
                    chunks = get_text_chunks(raw_text)

                    # 3. Vector Store 생성
                    st.info("3/3: 벡터 저장소 생성 중...")
                    collection = create_vector_store(chunks)

                    if collection:
                        st.session_state.collection = collection
                        st.session_state.processed = True
                        st.success(f"✅ {len(chunks)}개 청크 처리 완료!")
        else:
            st.warning("PDF 파일을 먼저 업로드하세요.")

    st.markdown("---")

    # 대화 초기화
    if st.button("🗑️ 대화 초기화", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    # 문서 초기화
    if st.button("📄 문서 초기화", use_container_width=True):
        st.session_state.collection = None
        st.session_state.processed = False
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")

    # 상태 표시
    st.subheader("📊 상태")
    if st.session_state.processed:
        st.success("✅ 문서 로드됨")
    else:
        st.info("⏳ 문서 대기 중")

    st.write(f"대화 수: {len(st.session_state.messages)}개")


# ============ 메인 채팅 영역 ============

# 저장된 메시지 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 입력 처리
if prompt := st.chat_input("문서에 대해 질문하세요"):
    # 사용자 메시지 표시
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 응답 생성
    with st.chat_message("assistant"):
        if st.session_state.collection is None:
            response = "⚠️ 먼저 PDF 문서를 업로드하고 처리해주세요."
            st.warning(response)
        else:
            with st.spinner("검색 및 답변 생성 중..."):
                response, retrieved_docs = answer_question(prompt, st.session_state.collection)
                st.markdown(response)

                # 검색된 문서 표시 (디버깅용)
                if retrieved_docs:
                    with st.expander("🔍 검색된 관련 문서 (디버깅)"):
                        for doc in retrieved_docs:
                            st.markdown(f"**#{doc['rank']}** (유사도: {doc['similarity']})")
                            st.caption(doc['preview'])
                            st.markdown("---")

        st.session_state.messages.append({"role": "assistant", "content": response})
