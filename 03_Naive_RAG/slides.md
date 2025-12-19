---
marp: true
theme: default
paginate: true
backgroundColor: #fff
style: |
  section {
    font-family: 'Noto Sans KR', sans-serif;
  }
  h1 { color: #7c3aed; }
  h2 { color: #8b5cf6; }
  code { background-color: #f3f4f6; }
---

# Naive RAG 이해 및 간단한 RAG Agent

## 섹션 3 (75분)

---

# 학습 목표

1. RAG의 개념과 작동 원리를 이해한다
2. Vector Database (ChromaDB)를 사용한다
3. 문서 청킹과 임베딩을 구현한다
4. RAG 기반 챗봇을 완성한다

---

# LLM의 한계 복습

| 한계 | 설명 |
|------|------|
| **지식 컷오프** | 학습 데이터 이후 정보 모름 |
| **할루시네이션** | 그럴듯하지만 틀린 정보 생성 |
| **도메인 지식** | 특정 분야 전문 지식 부족 |

### 해결책: RAG!

---

# RAG란?

## Retrieval-Augmented Generation
## 검색 증강 생성

> LLM의 지식을 **외부 데이터**로 확장하는 기술

```
사용자 질문 → [검색] → 관련 문서 → [LLM] → 정확한 답변
```

---

# RAG 3단계 작동 원리

```
┌─────────────────────────────────────────────┐
│ 1. Retrieval (검색)                          │
│    사용자 질문 → Vector DB에서 유사 문서 검색  │
├─────────────────────────────────────────────┤
│ 2. Augmentation (증강)                       │
│    검색된 문서를 프롬프트에 추가               │
├─────────────────────────────────────────────┤
│ 3. Generation (생성)                         │
│    LLM이 보강된 컨텍스트로 답변 생성          │
└─────────────────────────────────────────────┘
```

---

# RAG가 해결하는 문제

| LLM 한계 | RAG 해결 방법 |
|---------|--------------|
| 지식 컷오프 | 최신 데이터 **실시간** 검색 |
| 할루시네이션 | **검증된 소스** 기반 답변 |
| 도메인 지식 | **커스텀 지식** 베이스 연결 |

---

# RAG 파이프라인 전체 흐름

```
[문서 준비 단계]
PDF/텍스트 → 청킹 → 임베딩 → Vector DB 저장

[질문 응답 단계]
질문 → 임베딩 → Vector DB 검색 → 상위 K개 문서
                                    ↓
                            프롬프트에 추가
                                    ↓
                            LLM 답변 생성
```

---

# 임베딩이란?

## 텍스트를 숫자 벡터로 변환

```python
"고양이"  →  [0.12, -0.34, 0.56, ..., 0.89]
"강아지"  →  [0.11, -0.32, 0.58, ..., 0.87]
"자동차"  →  [0.78, 0.23, -0.45, ..., 0.12]
```

### 핵심
- 의미가 비슷하면 → 벡터도 비슷
- **코사인 유사도**로 비교

---

# Vector Database 비교

| 항목 | ChromaDB | FAISS | Pinecone |
|------|----------|-------|----------|
| **유형** | 완전한 DB | 검색 라이브러리 | 클라우드 |
| **설치** | 매우 쉬움 | 중간 | API |
| **로컬** | O | O | X |
| **메타데이터** | O 내장 | X | O |
| **비용** | 무료 | 무료 | 유료 |

---

# 초보자 추천: ChromaDB

### 선택 이유
- **즉시 사용 가능**: 설정 없이 바로 시작
- **메타데이터 필터링**: 문서 관리 내장
- **영속성 자동화**: 저장/로드 자동

### 설치
```bash
pip install chromadb
```

---

# ChromaDB 기본 사용법

```python
import chromadb

# 클라이언트 생성
client = chromadb.Client()

# 컬렉션 생성
collection = client.create_collection("my_docs")

# 문서 추가
collection.add(
    documents=["AI는 인공지능입니다", "RAG는 검색 증강 생성입니다"],
    ids=["doc1", "doc2"]
)

# 검색
results = collection.query(
    query_texts=["인공지능이란?"],
    n_results=2
)
```

---

# 문서 청킹이란?

## 긴 문서를 작은 조각으로 분할

```
┌──────────────────────────────────────┐
│          원본 문서 (10페이지)          │
└──────────────────────────────────────┘
                  ↓
┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐
│청크1│ │청크2│ │청크3│ │청크4│ │청크5│
└─────┘ └─────┘ └─────┘ └─────┘ └─────┘
  500자   500자   500자   500자   500자
        └──100자 오버랩──┘
```

---

# 청킹 파라미터

| 파라미터 | 권장값 | 설명 |
|---------|-------|------|
| **chunk_size** | 500-1000 | 청크 크기 (토큰/문자) |
| **chunk_overlap** | 10-20% | 오버랩 (문장 연결성) |

### 청킹이 중요한 이유
- 너무 크면 → 검색 정밀도 낮음
- 너무 작으면 → 문맥 손실

---

# 청킹 구현 코드

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 텍스트 분할기 설정
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # 청크 크기
    chunk_overlap=200,    # 오버랩 (20%)
    separators=["\n\n", "\n", ".", "!", "?", ",", " "]
)

# 텍스트 분할
text = "긴 문서 내용..."
chunks = splitter.split_text(text)

print(f"생성된 청크 수: {len(chunks)}")
```

---

# PDF에서 텍스트 추출

```python
from PyPDF2 import PdfReader

def get_pdf_text(pdf_files):
    """PDF 파일들에서 텍스트 추출"""
    text = ""
    for pdf in pdf_files:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text += page.extract_text()
    return text

# 사용
pdf_text = get_pdf_text(uploaded_pdfs)
chunks = splitter.split_text(pdf_text)
```

---

# RAG 챗봇 v0.2 - 전체 구조

```
┌─────────────────────────────────────────┐
│            Streamlit UI                  │
├─────────────────────────────────────────┤
│  사이드바: PDF 업로드 → 처리 → 저장      │
├─────────────────────────────────────────┤
│  메인: 질문 → ChromaDB 검색 → LLM 응답   │
└─────────────────────────────────────────┘
```

---

# RAG 챗봇 - Gemini 임베딩 설정

```python
import google.generativeai as genai

# Gemini 임베딩 함수
def get_embedding(text):
    """텍스트를 임베딩 벡터로 변환"""
    result = genai.embed_content(
        model="models/text-embedding-004",
        content=text
    )
    return result['embedding']

# 여러 텍스트 임베딩
def get_embeddings(texts):
    return [get_embedding(t) for t in texts]
```

---

# RAG 챗봇 - Vector Store 생성

```python
import chromadb

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

    # 임베딩 생성 및 저장
    embeddings = get_embeddings(chunks)
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[f"chunk_{i}" for i in range(len(chunks))]
    )

    return collection
```

---

# RAG 챗봇 - 검색 및 응답

```python
def answer_question(question, collection):
    """질문에 대한 RAG 기반 답변 생성"""
    # 1. 질문 임베딩
    q_embedding = get_embedding(question)

    # 2. 유사 문서 검색
    results = collection.query(
        query_embeddings=[q_embedding],
        n_results=4
    )

    # 3. 컨텍스트 구성
    context = "\n\n".join(results['documents'][0])

    # 4. LLM 프롬프트 생성
    prompt = f"""다음 문서를 참고하여 질문에 답변하세요.

문서:
{context}

질문: {question}

답변:"""

    # 5. Gemini로 답변 생성
    model = genai.GenerativeModel("gemini-2.5-flash-lite")
    response = model.generate_content(prompt)
    return response.text
```

---

# RAG 챗봇 - Streamlit UI

```python
import streamlit as st

st.title("📚 RAG 챗봇 v0.2")

# 사이드바 - 문서 업로드
with st.sidebar:
    st.title("📁 문서 업로드")
    pdf_docs = st.file_uploader(
        "PDF 파일 업로드",
        accept_multiple_files=True,
        type=['pdf']
    )

    if st.button("📥 문서 처리"):
        with st.spinner("처리 중..."):
            # 1. 텍스트 추출
            raw_text = get_pdf_text(pdf_docs)
            # 2. 청킹
            chunks = get_text_chunks(raw_text)
            # 3. Vector Store 생성
            st.session_state.collection = create_vector_store(chunks)
            st.success(f"✅ {len(chunks)}개 청크 처리 완료!")
```

---

# RAG 챗봇 - 대화 처리

```python
# 질문 입력
if prompt := st.chat_input("문서에 대해 질문하세요"):
    # 사용자 메시지 표시
    with st.chat_message("user"):
        st.write(prompt)

    # RAG 응답 생성
    with st.chat_message("assistant"):
        if "collection" not in st.session_state:
            st.error("먼저 문서를 업로드해주세요!")
        else:
            with st.spinner("검색 중..."):
                answer = answer_question(
                    prompt,
                    st.session_state.collection
                )
                st.write(answer)
```

---

# 실습: RAG 챗봇 만들기

### 준비물
- Python 3.9+
- 테스트용 PDF 파일

### 패키지 설치
```bash
pip install streamlit google-generativeai chromadb PyPDF2 langchain
```

### 실행
```bash
streamlit run rag_app.py
```

---

# 초보자 흔한 오류

| 문제 | 증상 | 해결책 |
|------|------|--------|
| 청크 너무 큼 | 검색 정밀도 낮음 | 500-1000으로 축소 |
| 오버랩 없음 | 문장 중간 잘림 | 최소 10% 오버랩 |
| 임베딩 불일치 | 검색 실패 | 동일 모델 사용 |
| 메모리 부족 | 대용량 PDF | 배치 처리 |

---

# RAG 개선 방향

### Advanced RAG 기법
- **Hybrid Search**: 키워드 + 벡터 검색 결합
- **Re-ranking**: 검색 결과 재정렬
- **Query Expansion**: 질문 확장
- **Multi-hop RAG**: 다단계 추론

### 다음 섹션에서
- Tool Calling으로 웹 검색 추가!

---

# 섹션 3 정리

## 배운 내용

1. **RAG 개념**: Retrieval → Augmentation → Generation
2. **임베딩**: 텍스트를 벡터로 변환
3. **ChromaDB**: 간단한 Vector Database
4. **청킹**: 문서를 작은 조각으로 분할
5. **RAG 챗봇**: 문서 기반 질의응답

---

# 참고 자료

| 자료 | URL |
|------|-----|
| LangChain RAG 튜토리얼 | python.langchain.com/docs/tutorials/rag |
| ChromaDB 문서 | docs.trychroma.com |
| RAG from Scratch | github.com/langchain-ai/rag-from-scratch |
| Weaviate RAG 소개 | weaviate.io/blog/introduction-to-rag |

---

# 다음 섹션 예고

## 섹션 4: Tool을 이용한 검색 Agent

- Tool Calling 개념
- DuckDuckGo 검색 연동
- 검색 Agent 완성
- 최종 배포
