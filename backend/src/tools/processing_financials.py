"""
processing_financials.py — DART 재무제표 원시 데이터 정제·분석 모듈 (현재 미사용)

[역할]
  fetch_financials.py가 가져온 DART 원시 DataFrame을 받아
  8대 핵심 재무 지표를 추출하고, 파생 분석 비율을 계산하여
  파인튜닝 학습 데이터셋(JSONL) 형식으로 변환하는 유틸리티.

[주요 함수]
  - refine_dart_res(df, corp_name):
      DART DataFrame에서 매출액·영업이익·당기순이익·자산/부채/자본총계·
      영업활동현금흐름·자본금 등 8대 지표를 account_id + 명칭으로 추출.
  - analyze_and_format(data):
      정제된 dict를 부채비율, ROE 등 파생 지표와 함께
      instruction/input/output 형태의 학습 데이터 JSON으로 포맷팅.
  - clean_amount(val):
      문자열 숫자를 정수로 변환하는 헬퍼.

[의존]
  - pandas, json, re (표준/서드파티 라이브러리만 사용)

[참조하는 곳]
  - fetch_financials.py → refine_dart_res()
"""
import pandas as pd
import json
import re

# ==========================================
# 1. 설정 및 8대 핵심 지표 매핑 (ID + 명칭)
# ==========================================
# ID는 정확한 매칭을 위해, 명칭은 ID가 누락된 경우를 대비한 백업입니다.
TARGET_MAPPING = {
    '매출액': {'id': 'ifrs-full_Revenue', 'synonyms': ['매출액', '수익(매출액)', '영업수익']},
    '영업이익': {'id': 'dart_OperatingIncomeLoss', 'synonyms': ['영업이익', '영업이익(손실)']},
    '당기순이익': {'id': 'ifrs-full_ProfitLoss', 'synonyms': ['당기순이익', '당기순이익(손실)']},
    '자산총계': {'id': 'ifrs-full_Assets', 'synonyms': ['자산총계']},
    '부채총계': {'id': 'ifrs-full_Liabilities', 'synonyms': ['부채총계']},
    '자본총계': {'id': 'ifrs-full_Equity', 'synonyms': ['자본총계']},
    '영업활동현금흐름': {'id': 'ifrs-full_CashFlowsFromUsedInOperatingActivities', 'synonyms': ['영업활동으로 인한 현금흐름', '영업활동현금흐름']},
    '자본금': {'id': 'ifrs-full_IssuedCapital', 'synonyms': ['자본금']}
}

def clean_amount(val):
    """문자열 숫자를 정수로 변환"""
    if pd.isna(val) or val == '': return 0
    if isinstance(val, (int, float)): return int(val)
    # 숫자, 마이너스 부호, 소수점 제외 제거
    cleaned = re.sub(r'[^0-9.\-]', '', str(val))
    try:
        return int(float(cleaned))
    except:
        return 0

def refine_dart_res(df, corp_name):
    """ID와 명칭을 모두 활용하여 8대 지표 추출"""
    if df is None or df.empty: return None
    
    refined_data = {'corp_name': corp_name.strip()}
    refined_data['year'] = str(df.iloc[0].get('bsns_year', '2025'))
    
    # 금액 컬럼 자동 선택
    amount_col = 'thstrm_amount' if 'thstrm_amount' in df.columns else 'thstrm_add_amount'

    for key, target in TARGET_MAPPING.items():
        val = 0
        # 1순위: account_id로 매칭
        if 'account_id' in df.columns:
            match = df[df['account_id'].str.strip() == target['id']]
            print(match)
            if not match.empty:
                val = clean_amount(match.iloc[0][amount_col])
        
        # 2순위: ID 매칭 실패 시 이름(synonyms)으로 매칭
        if val == 0:
            match = df[df['account_nm'].str.strip().isin(target['synonyms'])]
            if not match.empty:
                val = clean_amount(match.iloc[0][amount_col])
        
        refined_data[key] = val
            
    return refined_data

def analyze_and_format(data):
    """정제된 데이터를 바탕으로 분석 및 JSON 포맷팅 (학습 데이터셋 형식)"""
    if not data: return None

    # 계산을 위한 변수 할당 (0으로 나누기 방지)
    a, l, e = data.get('자산총계', 0), data.get('부채총계', 0), data.get('자본총계', 0)
    sales, op_income, net_income = data.get('매출액', 0), data.get('영업이익', 0), data.get('당기순이익', 0)

    # 파생 분석 지표 계산
    ratios = {}
    if e != 0: ratios['부채비율'] = round((l / e) * 100, 2)
    if a != 0: ratios['자기자본비율'] = round((e / a) * 100, 2)
    if sales != 0: ratios['영업이익률'] = round((op_income / sales) * 100, 2)
    if e != 0: ratios['ROE'] = round((net_income / e) * 100, 2)

    # Input 텍스트 생성 (학습 데이터와 동일한 형식)
    fin_info = [f"{k}: {data[k]:,}원" for k in TARGET_MAPPING.keys() if data.get(k) != 0]
    ratio_info = [f"{k}: {v}%" for k, v in ratios.items()]
    
    input_text = f"{data['corp_name']}의 {data['year']}년도 주요 재무 실적 및 분석 정보: " 
    input_text += " | ".join(fin_info) + " [분석 지표] " + " | ".join(ratio_info)

    # Output JSON 구조 생성
    output_content = {
        "metadata": {
            "company": data['corp_name'],
            "fiscal_year": data['year']
        },
        "financial_metrics": {k: data[k] for k in TARGET_MAPPING.keys()},
        "analysis_ratios": ratios
    }

    return {
        "instruction": "제시된 재무 데이터를 바탕으로 핵심 지표 8종을 추출하고 주요 재무 비율(부채비율, 자기자본비율 등)을 분석하여 JSON으로 응답하세요.",
        "input": input_text,
        "output": json.dumps(output_content, ensure_ascii=False, indent=2)
    }