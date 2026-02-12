from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from finance_rag import FinanceRAG

app = FastAPI()
rag = FinanceRAG() # 서버 시작 시 DB 로드

# CORS 설정 추가
from fastapi.middleware.cors import CORSMiddleware # 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # 모든 도메인에서의 접속을 허용합니다.
    allow_credentials=True,
    allow_methods=["*"],      # OPTIONS를 포함한 모든 메서드를 허용합니다.
    allow_headers=["*"],      # 모든 헤더를 허용합니다.
)

class ChatRequest(BaseModel):
    question: str

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    # StreamingResponse를 사용하여 한 토큰씩 응답
    return StreamingResponse(
        rag.query_stream(request.question), 
        media_type="text/event-stream"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)