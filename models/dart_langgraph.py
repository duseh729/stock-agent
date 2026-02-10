import json
import re
from typing import TypedDict, Optional
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END

# 1. 상태 정의 (GraphState 확장)
class GraphState(TypedDict):
    user_query: str          # 사용자의 질문 (예: "삼성전자 투자할만해?")
    company_name: Optional[str] # 추출된 기업명
    raw_text: Optional[str]  # 가져온 공시 텍스트
    financial_data: Optional[dict]
    error_msg: Optional[str]
    retry_count: int

# 2. 모델 설정
llm = ChatOllama(model="dart_model_v1", temperature=0)

# 3. [신규] 기업명 추출 노드 (company_extractor_node)
def company_extractor_node(state: GraphState):
    user_query = state["user_query"]
    
    prompt = f"""### Instruction:
다음 [사용자 질문]에서 분석 대상인 '기업 이름'만 추출하라. 
조사(은/는/이/가)를 제외하고 기업 이름만 딱 하나 출력해. 
기업명이 없으면 'None'이라고 답해.

[사용자 질문]: {user_query}

### Response:
"""
    response = llm.invoke(prompt)
    company_name = response.content.strip().replace("'", "").replace('"', "")
    
    # 간단한 정제 (마침표 등 제거)
    company_name = re.sub(r'[^\w\s]', '', company_name)
    
    print(f"🔍 단계 1 [기업명 추출]: {company_name}")
    
    # 실제 환경에서는 여기서 DART API 등을 호출해 raw_text를 가져와야 합니다.
    # 일단은 테스트를 위해 더미 데이터를 넣어줍니다.
    dummy_text = f"({company_name})는 2025년 매출액 2조 5,000억원, 영업이익 3,000억원을 기록하였다."
    
    return {
        "company_name": company_name,
        "raw_text": dummy_text
    }

# 4. 추출 노드 (extractor_node) - 기존 유지 및 프롬프트 보강
def extractor_node(state: GraphState):
    raw_text = state["raw_text"]
    error_msg = state["error_msg"]
    retry_count = state.get("retry_count", 0)
    
    correction_prompt = ""
    if error_msg:
        correction_prompt = f"\n\n[이전 시도 에러]: {error_msg}\n주의: 자릿수를 다시 확인하세요 (1조=0 12개, 1000억=0 11개)."

    instruction = f"""다음 기업 공시 텍스트에서 주요 재무 지표를 JSON 형식으로 추출해줘.
    
[자릿수 규칙]
- 1조: 1,000,000,000,000 (0이 12개)
- 1,000억: 100,000,000,000 (0이 11개)

텍스트: {raw_text}{correction_prompt}"""

    prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
    response = llm.invoke(prompt)
    content = response.content.strip()
    
    try:
        json_match = re.search(r"\{.*\}", content, re.DOTALL)
        data = json.loads(json_match.group().replace("'", '"')) if json_match else None
    except:
        data = None
        
    print(f"📊 단계 2 [지표 추출 시도]: {retry_count + 1}회차")
    return {"financial_data": data, "retry_count": retry_count + 1}

# 5. 검증 노드 (validator_node) - 기존 유지
def validator_node(state: GraphState):
    data = state["financial_data"]
    raw_text = state["raw_text"]
    
    if not data:
        return {"error_msg": "JSON 형식을 파싱할 수 없습니다."}
    
    revenue = data.get("revenue", 0) or 0
    # '조' 단위 검증 추가
    if "조" in raw_text and revenue < 10**12:
        return {"error_msg": "텍스트에 '조'가 있는데 결과는 '억' 단위입니다. 0의 개수를 12개로 맞추세요."}
    
    if revenue < data.get("profit", 0):
        return {"error_msg": "매출액이 영업이익보다 작을 수 없습니다."}
    
    return {"error_msg": None}

# 6. 조건부 에지 및 그래프 빌드
def should_continue(state: GraphState):
    if state["error_msg"] is None or state["retry_count"] >= 3:
        return "end"
    return "continue"

workflow = StateGraph(GraphState)

# 노드 추가
workflow.add_node("company_extractor", company_extractor_node)
workflow.add_node("extractor", extractor_node)
workflow.add_node("validator", validator_node)

# 연결 (company_extractor -> extractor -> validator)
workflow.set_entry_point("company_extractor")
workflow.add_edge("company_extractor", "extractor")
workflow.add_edge("extractor", "validator")

workflow.add_conditional_edges(
    "validator",
    should_continue,
    {"continue": "extractor", "end": END}
)

app = workflow.compile()

# 7. 실행부
if __name__ == "__main__":
    query = "삼성전자 이번에 투자할만 하냐? 실적 좀 봐줘"
    
    initial_state = {
        "user_query": query,
        "company_name": None,
        "raw_text": None,
        "financial_data": None,
        "error_msg": None,
        "retry_count": 0
    }
    
    result = app.invoke(initial_state)
    
    print("\n" + "="*50)
    print(f"[최종 분석 대상 기업]: {result['company_name']}")
    print("[추출된 재무 데이터]")
    print(json.dumps(result["financial_data"], indent=4, ensure_ascii=False))
    print("="*50)