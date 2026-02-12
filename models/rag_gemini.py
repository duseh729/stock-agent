# rag를 적용한 llm
import os
import json
import time
from tqdm import tqdm
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.documents import Document

# api 메서드
import asyncio

class FinanceRAG:
    def __init__(self, db_dir="./finance_local_db"):
        load_dotenv()
        self.db_dir = db_dir
        
        # 1. 로컬 임베딩 모델 (4500U CPU에서 작동)
        print("💡 로컬 모델을 로드합니다...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="jhgan/ko-sroberta-multitask",
            model_kwargs={'device': 'cpu'}
        )
        
        # 2. 답변용 LLM
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
        self.vector_db = None

        # 기존 DB 로드
        if os.path.exists(self.db_dir) and os.path.isdir(self.db_dir) and os.listdir(self.db_dir):
            print(f"📦 기존 DB 로드 완료: {self.db_dir}")
            self.vector_db = Chroma(
                persist_directory=self.db_dir, 
                embedding_function=self.embeddings
            )
        else:
            print("ℹ️ 기존 DB가 없습니다. 새로 구축이 필요합니다.")

    def _parse_jsonl(self, file_path):
        """파일을 읽어 Document 객체 리스트로 변환하는 공통 로직"""
        documents = []
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in tqdm(lines, desc="데이터 파싱 중"):
                try:
                    item = json.loads(line)
                    content_json = json.loads(item['output'])
                    meta = content_json['metadata']
                    
                    text_content = f"기업명: {meta['company']}, 연도: {meta['fiscal_year']}\n"
                    text_content += f"재무: {content_json['financial_metrics']}\n"
                    text_content += f"비율: {content_json['analysis_ratios']}"
                    
                    documents.append(Document(
                        page_content=text_content, 
                        metadata={"company": str(meta['company']), "year": str(meta['fiscal_year'])}
                    ))
                except: continue
        return documents

    def ingest_local_json(self, file_path):
        """처음부터 DB를 생성 (전체 6,000줄용)"""
        docs = self._parse_jsonl(file_path)
        print(f"🚀 {len(docs)}건 로컬 임베딩 시작... (4500U 기준 약 5-10분)")
        self.vector_db = Chroma.from_documents(
            documents=docs,
            embedding=self.embeddings,
            persist_directory=self.db_dir
        )
        print("✅ DB 구축 완료!")

    def update_data(self, file_path):
        """기존 DB에 새로운 데이터를 추가"""
        if self.vector_db is None:
            self.ingest_local_json(file_path)
            return
            
        docs = self._parse_jsonl(file_path)
        print(f"🚀 {len(docs)}건의 데이터를 추가 중...")
        self.vector_db.add_documents(docs)
        print("✅ 데이터 업데이트 완료!")

    def query(self, question):
        if not self.vector_db:
            return "❌ DB가 없습니다. 학습을 먼저 진행하세요."
        
        retriever = self.vector_db.as_retriever(search_kwargs={"k": 5})
        docs = retriever.invoke(question)
        context = "\n\n".join([d.page_content for d in docs])
        
        prompt = f"아래 재무 데이터를 바탕으로 답하세요.\n\n[데이터]\n{context}\n\n질문: {question}"
        return self.llm.invoke(prompt).content

    async def query_stream(self, question: str):
        if not self.vector_db:
            yield "❌ DB가 없습니다."
            return

        # 1. 검색 (이 단계는 스트리밍이 아니므로 빠르게 수행)
        retriever = self.vector_db.as_retriever(search_kwargs={"k": 5})
        docs = await asyncio.to_thread(retriever.invoke, question) # 비동기 처리
        context = "\n\n".join([d.page_content for d in docs])

        prompt = f"아래 재무 데이터를 바탕으로 답하세요.\n\n[데이터]\n{context}\n\n질문: {question}"

        # 2. 생성 및 스트리밍 (Gemini가 한 글자씩 보냄)
        async for chunk in self.llm.astream(prompt):
            yield chunk.content # 한 토큰씩 클라이언트로 전달