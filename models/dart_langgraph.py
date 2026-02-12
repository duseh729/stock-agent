"""
dart_langgraph.py — LangGraph 기반 재무 분석 파이프라인 (실험용, 현재 미사용)

[역할]
  사용자 질문에서 기업명을 추출(Gemini Flash) → DART에서 재무 데이터 수집
  → 로컬 파인튜닝 모델(Ollama)로 분석 → 결과 검증의 멀티스텝 워크플로우.

[파이프라인 구조]
  company_extractor (Gemini Flash)
    → 기업명 추출 + DART 재무 데이터 수집 (fetch_financials.get_refined_financials)
  extractor (Ollama dart_model_v1)
    → 파인튜닝 모델로 8대 재무 지표 추출 및 JSON 생성
  validator
    → 매출액 누락, 논리 오류 검증 (최대 3회 재시도)

[의존]
  - fetch_financials.py (backend/src/tools/) → get_refined_financials()
  - langchain_ollama, langchain_google_genai, langgraph
  - Ollama에 dart_model_v1 모델이 로컬에 등록되어 있어야 실행 가능

[비고]
  서비스 메인 흐름에서는 호출되지 않으며, 단독 실행(__main__)으로 테스트하는 용도.
"""
import json
import re
import os
from typing import TypedDict, Optional
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from fetch_financials import get_refined_financials
from dotenv import load_dotenv

load_dotenv()

# ==========================================
# 1. 모델 인스턴스 분리
# ==========================================
# [비서] Gemini 1.5 Flash - 기업명 추출용 (API 사용)
llm_general = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0
)

# [분석가] 로컬 파인튜닝 모델 - 재무 분석용 (Ollama 사용)
llm_analyser = ChatOllama(model="dart_model_v1", temperature=0)

class GraphState(TypedDict):
    user_query: str
    company_name: Optional[str]
    raw_text: Optional[str]
    financial_data: Optional[dict]
    error_msg: Optional[str]
    retry_count: int

# ==========================================
# 2. 노드 정의
# ==========================================

def company_extractor_node(state: GraphState):
    print("--- [NODE] 기업명 추출 (Gemini-Flash) ---")
    user_query = state["user_query"]
    
    # Gemini는 지시를 매우 잘 따릅니다.
    prompt = f"다음 질문에서 기업 이름만 한 단어로 추출해줘. 다른 말은 절대 하지 마. 없으면 'None'.\n질문: {user_query}"
    
    response = llm_general.invoke(prompt)
    # Gemini 응답에서 기업명만 정제
    company_name = response.content.strip().split('\n')[0].replace('*', '')
    company_name = re.sub(r'[^\w\s]', '', company_name).split(' ')[0]
    
    print(f"🔍 추출된 기업명: {company_name}")

    if company_name == "None" or not company_name:
        return {"error_msg": "기업명을 찾지 못했습니다.", "company_name": "None"}

    # 정제 툴 호출
    refined_dict = get_refined_financials(company_name, 2025)
    if not refined_dict:
        return {"company_name": company_name, "error_msg": "DART 데이터 로드 실패"}

    return {
        "company_name": company_name,
        "raw_text": json.dumps(refined_dict, ensure_ascii=False, indent=2),
        "error_msg": None
    }

def extractor_node(state: GraphState):
    print(f"--- [NODE] 재무 지표 추출 (dart_model_v1) ---")
    raw_text = state["raw_text"]
    error_msg = state.get("error_msg")
    
    correction = f"\n\n[보정 요청]: {error_msg}" if error_msg else ""
    instruction = "제시된 재무 데이터를 바탕으로 핵심 지표 8종을 추출하고 주요 재무 비율을 분석하여 JSON으로 응답하세요."
    input_data = f"{state['company_name']}의 재무 데이터: {raw_text}"

    # 파인튜닝 시 사용했던 포맷 그대로 유지
    prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_data}{correction}\n\n### Response:\n"
    
    response = llm_analyser.invoke(prompt)
    
    try:
        json_match = re.search(r"\{.*\}", response.content, re.DOTALL)
        data = json.loads(json_match.group().replace("'", '"'))
        return {"financial_data": data, "retry_count": state["retry_count"] + 1}
    except:
        return {"error_msg": "JSON 생성 실패", "retry_count": state["retry_count"] + 1}

def validator_node(state: GraphState):
    data = state["financial_data"]
    if not data: return {"error_msg": "데이터 파싱 에러"}
    
    # v1 모델이 뱉은 키값(financial_metrics)에 맞춰 체크
    metrics = data.get("financial_metrics", {})
    rev = metrics.get("매출액") or data.get("revenue") or 0
    pro = metrics.get("영업이익") or data.get("profit") or 0
    
    if rev == 0: return {"error_msg": "매출액 누락"}
    if rev < pro: return {"error_msg": "매출액이 영업이익보다 작음"}
    return {"error_msg": None}

# ==========================================
# 3. 그래프 구성
# ==========================================

def route_after_extraction(state: GraphState):
    return "end" if state.get("error_msg") else "continue"

def should_continue(state: GraphState):
    return "end" if state["error_msg"] is None or state["retry_count"] >= 3 else "continue"

workflow = StateGraph(GraphState)
workflow.add_node("company_extractor", company_extractor_node)
workflow.add_node("extractor", extractor_node)
workflow.add_node("validator", validator_node)

workflow.set_entry_point("company_extractor")
workflow.add_conditional_edges("company_extractor", route_after_extraction, {"continue": "extractor", "end": END})
workflow.add_edge("extractor", "validator")
workflow.add_conditional_edges("validator", should_continue, {"continue": "extractor", "end": END})

app = workflow.compile()

if __name__ == "__main__":
    result = app.invoke({"user_query": "삼성전자 이번 실적 분석해줘", "retry_count": 0})
    print(f"\n✅ 분석 결과:\n{json.dumps(result['financial_data'], indent=4, ensure_ascii=False)}")