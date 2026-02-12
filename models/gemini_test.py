# rag_gemini 테스트용
from rag_gemini import FinanceRAG

# 1. 기존 DB 로드 (학습된 6,000건이 이미 들어있는 상태)
rag = FinanceRAG()

# 2. 새로운 데이터만 추가하기 (기존 데이터는 그대로 유지됨)
# new_data = "./dart_financial_analysis_dataset.jsonl"
# rag.update_data(new_data)

# 3. 질문하기 (기존 데이터 + 새 데이터 모두 검색 가능)
print(rag.query("종근당홀딩스 재무 상태는 어때?"))