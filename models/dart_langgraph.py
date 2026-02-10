import json
import re
from typing import TypedDict, Optional
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END

# 1. 상태 정의 (GraphState)
class GraphState(TypedDict):
    raw_text: str
    financial_data: Optional[dict]
    error_msg: Optional[str]
    retry_count: int

# 2. 모델 설정
# 모델 이름은 사용자의 환경에 맞춰 'dart_model_v1'으로 설정
llm = ChatOllama(
    model="dart_model_v1",
    temperature=0,
)
# 3. 추출 노드 (extractor_node)
def extractor_node(state: GraphState):
    raw_text = state["raw_text"]
    error_msg = state["error_msg"]
    retry_count = state.get("retry_count", 0)
    
    # 에러 메시지가 있을 경우 보정 지시사항 추가
    correction_prompt = ""
    if error_msg:
        correction_prompt = f"\n\n[이전 시도 에러]: {error_msg}\n주의: 지난번 결과는 논리적으로 틀렸습니다. 특히 '억' 단위의 0 개수(8개)를 정확히 확인하고 다시 추출하세요."

    instruction = f"""다음 기업 공시 텍스트에서 주요 재무 지표를 JSON 형식으로 추출해줘. 
결과는 반드시 {{'company': '...', 'year': 2025, 'revenue': 150000000000, 'profit': 25000000000}} 형태의 순수 JSON만 출력해.

텍스트: {raw_text}{correction_prompt}"""

    prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
    
    response = llm.invoke(prompt)
    content = response.content.strip()
    
    # JSON 문자열만 추출하는 간단한 전처리
    try:
        json_match = re.search(r"\{.*\}", content, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group().replace("'", '"'))
        else:
            data = None
    except:
        data = None
        
    return {
        "financial_data": data,
        "retry_count": retry_count + 1
    }

# 4. 검증 노드 (validator_node)
def validator_node(state: GraphState):
    data = state["financial_data"]
    raw_text = state["raw_text"]
    
    if not data:
        return {"error_msg": "JSON 형식을 파싱할 수 없습니다."}
    
    revenue = data.get("revenue", 0)
    profit = data.get("profit", 0)
    
    # 체크 1: "억"이 있는데 1억 미만인 경우 (0 개수 실수)
    if "억" in raw_text and revenue < 100000000:
        return {"error_msg": f"매출액({revenue})이 텍스트의 '억' 단위에 비해 너무 작습니다. 0의 개수를 다시 확인하세요."}
    
    # 체크 2: 매출액 < 영업이익 논리 오류
    if revenue < profit:
        return {"error_msg": f"매출액({revenue})이 영업이익({profit})보다 작을 수 없습니다."}
    
    return {"error_msg": None}

# 5. 조건부 에지 (conditional_edge) logic
def should_continue(state: GraphState):
    if state["error_msg"] is None:
        return "end"
    if state["retry_count"] >= 3:
        print("!!! 최대 재시도 횟수 초과 (검증 실패) !!!")
        return "end"
    return "continue"

# 6. 그래프 빌드
workflow = StateGraph(GraphState)

workflow.add_node("extractor", extractor_node)
workflow.add_node("validator", validator_node)

workflow.set_entry_point("extractor")

workflow.add_edge("extractor", "validator")

workflow.add_conditional_edges(
    "validator",
    should_continue,
    {
        "continue": "extractor",
        "end": END
    }
)

app = workflow.compile()

# 7. 실행부
if __name__ == "__main__":
    test_text = "(주)에이아이컴퍼니는 2025년 매출액 1,500억원, 영업이익 250억원을 기록하였다."
    
    initial_state = {
        "raw_text": test_text,
        "financial_data": None,
        "error_msg": None,
        "retry_count": 0
    }
    
    result = app.invoke(initial_state)
    
    print("\n" + "="*50)
    print("[최종 추출 결과]")
    print(json.dumps(result["financial_data"], indent=4, ensure_ascii=False))
    print(f"재시도 횟수: {result['retry_count']}")
    print("="*50)