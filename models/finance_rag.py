import os
import json
import asyncio
from typing import List, TypedDict
from tqdm import tqdm
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# LangGraph 관련 임포트
from langgraph.graph import StateGraph, END

# 1. 상태(State) 정의: 노드 간에 전달될 데이터 구조
class AgentState(TypedDict):
    question: str
    context: List[Document]
    answer: str
    retry_count: int
    relevance: str  # <--- 이 줄이 반드시 있어야 합니다!

class FinanceRAG:
    def __init__(self, db_dir="./finance_local_db"):
        load_dotenv()
        self.db_dir = db_dir
        self.embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask", model_kwargs={'device': 'cpu'})
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)
        
        # 벡터 DB 로드
        self.vector_db = Chroma(persist_directory=self.db_dir, embedding_function=self.embeddings)
        
        # 2. 그래프 구축
        self.app = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(AgentState)

        # 1. 노드 정의: 각 단계의 역할 지정
        workflow.add_node("retrieve", self.node_retrieve)               # RAG: 질문 관련 문서 검색
        workflow.add_node("grade_documents", self.node_grade_documents) # QC: 검색된 문서의 적합성 평가
        workflow.add_node("generate", self.node_generate)               # 최종 답변 생성

        # 2. 기본 엣지: 검색이 끝나면 무조건 평가 단계로 이동
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "grade_documents")
        
        # 3. 조건부 엣지: 평가 결과(relevance)에 따른 분기 처리
        workflow.add_conditional_edges(
            "grade_documents",
            self.decide_to_generate,
            {
                "generate": "generate", # 적합: 답변 생성으로 이동
                "rewrite": "retrieve",  # 부적합: 검색 쿼리 재조정 후 다시 검색
                "end": END              # 실패: 재시도 횟수 초과 시 종료
            }
        )
        
        # 4. 종료 엣지: 답변 생성이 완료되면 끝
        workflow.add_edge("generate", END)

        return workflow.compile()

    # --- [노드 함수들] ---

    def node_retrieve(self, state: AgentState):
        print("🔍 [Node: Retrieve] 관련 데이터를 찾는 중...")
        question = state["question"]
        # k=5로 검색
        docs = self.vector_db.as_retriever(search_kwargs={"k": 5}).invoke(question)
        return {"context": docs, "retry_count": state.get("retry_count", 0) + 1}
    
    # === [ langgraph 통과 함수 ]
    # def node_grade_documents(self, state: AgentState):
    #     print("⚖️ [Node: Grade] (임시) API 호출 없이 통과 모드...")
        
    #     # LLM 호출 코드를 주석 처리하거나 무시하고
    #     # 무조건 'yes'를 반환하도록 만듭니다.
    #     return {"relevance": "yes"}

    def node_grade_documents(self, state: AgentState):
        print("⚖️ [Node: Grade] 데이터 품질 검사 시작...")
        question = state["question"]
        docs = state["context"]

        if not docs:
            print("❌ [Grade] 검색된 문서가 아예 없음")
            return {"relevance": "no"}

        # 1. LLM에게 판단 요청 (더 직관적인 프롬프트)
        prompt = ChatPromptTemplate.from_template("""
        너는 데이터 분석가야. 아래 [문서]에 [질문]에 대한 답을 할 수 있는 숫자가 하나라도 들어있니?

        예를 들어 질문이 '삼성전자 2025년 영업이익'이고, 
        문서에 '삼성전자', '2025', '영업이익'이라는 글자와 숫자가 있다면 무조건 'yes'라고 해.

        답변은 딱 한 단어 'yes' 또는 'no'로만 해.

        [질문]: {question}
        [문서]: {docs}

        결정:
        """)
    
        chain = prompt | self.llm | StrOutputParser()
        # LLM의 실제 답변을 raw_result에 담아 출력해봅니다.
        raw_result = chain.invoke({"question": question, "docs": docs}).lower().strip()

        print(f"🤖 [Grade] LLM의 실제 판단: '{raw_result}'")

        # 2. 결과 판정 (안전장치 추가: yes가 포함되어 있거나, 특정 키워드 매칭 시 통과)
        if "yes" in raw_result:
            print("✅ [Grade] 결과: YES")
            return {"relevance": "yes"} # <--- 키 이름이 AgentState와 같아야 함
        else:
            print("❌ [Grade] 결과: NO")
            return {"relevance": "no"}

    def decide_to_generate(self, state: AgentState):
        # state["relevance"]를 직접 접근해서 값이 있는지 확인
        relevance = state.get("relevance")
        retry_count = state.get("retry_count", 0)
        
        # 디버깅 로그 추가
        print(f"🧐 [Decision Debug] 현재 상태의 relevance: '{relevance}'")
    
        if relevance == "yes":
            print("✨ [Decision] 통과! 생성 노드로 이동")
            return "generate"
        
        if retry_count > 2:
            return "end"
        
        return "rewrite"

    def node_generate(self, state: AgentState):
        print("✍️ [Node: Generate] 답변 생성 중...")
        docs = state["context"]
        question = state["question"]
        
        if not docs: return {"relevance": "no"}
    
        # 1차 필터링: 질문의 핵심 단어가 문서에 포함되어 있다면 LLM 호출 없이 통과!
        keywords = [question[:4], "삼성", "전자"] # 예시 키워드
        if any(k in docs[0].page_content for k in keywords):
            print("⚡ [Grade] 키워드 매칭으로 API 호출 없이 통과!")
            return {"relevance": "yes"}
        context = "\n\n".join([d.page_content for d in state["context"]])
        
        prompt = ChatPromptTemplate.from_template("""
        당신은 금융 분석 전문가입니다. 아래 제공된 재무 데이터를 바탕으로 질문에 답하세요.
        데이터에 없는 내용은 지어내지 마세요.

        [데이터]
        {context}

        질문: {question}
        """)
        
        chain = prompt | self.llm | StrOutputParser()
        answer = chain.invoke({"context": context, "question": question})
        return {"answer": answer}

    # --- [외부 호출 메서드] ---

    async def query_stream(self, question: str):
        inputs = {"question": question, "retry_count": 0}
        
        # 1. 그래프 실행
        final_state = await asyncio.to_thread(self.app.invoke, inputs)
        
        # 2. 판단 결과 확인
        if final_state.get("relevance") != "yes":
            yield "❌ 질문과 관련된 정확한 데이터를 찾지 못했습니다. (데이터 부족)"
            return

        # 3. 'yes'일 때만 Gemini 스트리밍 시작
        context = "\n\n".join([d.page_content for d in final_state["context"]])
        prompt = f"아래 데이터를 바탕으로 답하세요.\n\n{context}\n\n질문: {question}"
        
        async for chunk in self.llm.astream(prompt):
            yield chunk.content