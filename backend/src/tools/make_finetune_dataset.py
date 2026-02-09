import pandas as pd
import json
import os
import re

# ==========================================
# 1. 설정 및 8대 핵심 지표 매핑 정의
# ==========================================
TARGET_MAPPING = {
    '매출액': {'id': 'ifrs-full_Revenue', 'nm': '매출액'},
    '영업이익': {'id': 'dart_OperatingIncomeLoss', 'nm': '영업이익'},
    '당기순이익': {'id': 'ifrs-full_ProfitLoss', 'nm': '당기순이익'},
    '자산총계': {'id': 'ifrs-full_Assets', 'nm': '자산총계'},
    '부채총계': {'id': 'ifrs-full_Liabilities', 'nm': '부채총계'},
    '자본총계': {'id': 'ifrs-full_Equity', 'nm': '자본총계'},
    '영업활동현금흐름': {'id': 'ifrs-full_CashFlowsFromUsedInOperatingActivities', 'nm': '영업활동현금흐름'},
    '자본금': {'id': 'ifrs-full_IssuedCapital', 'nm': '자본금'},
}

# CSV 파일 경로 (사용자 환경에 맞게 수정)
FILE_LIST = [
    '../../data/raw/fs_full_2023.csv',
    '../../data/raw/fs_full_2024.csv',
    '../../data/raw/fs_full_2025.csv'
]

def clean_amount(val):
    """금액 데이터에서 숫자만 추출하여 정수로 변환"""
    if pd.isna(val): return None
    cleaned = re.sub(r'[^0-9.\-]', '', str(val))
    if not cleaned or cleaned == '-': return None
    try:
        return int(float(cleaned))
    except:
        return None

def extract_comprehensive_data(file_list):
    """CSV 파일들에서 재무 데이터를 추출하고 회사/연도별로 통합"""
    all_data = []
    
    for file in file_list:
        if not os.path.exists(file):
            print(f"파일 없음: {file}")
            continue
            
        print(f"--- {file} 분석 시작 ---")
        df = None
        # 인코딩과 구분자 자동 시도
        for enc in ['utf-8-sig', 'cp949']:
            for s in [',', '\t']:
                try:
                    # 첫 줄 'corp_name_origin' 건너뛰기
                    temp_df = pd.read_csv(file, header=None, encoding=enc, sep=s, 
                                          skiprows=1, engine='python', on_bad_lines='skip')
                    if temp_df.shape[1] > 10:
                        df = temp_df
                        sep_name = 'TAB' if s == '\t' else 'COMMA'
                        print(f"로드 성공: 인코딩 {enc}, 구분자 '{sep_name}'")
                        break
                except:
                    continue
            if df is not None: break

        if df is None:
            print(f"실패: {file}을 읽을 수 없습니다.")
            continue

        # 컬럼 인덱스 설정 (표준 DART CSV 구조 기준)
        idx_id, idx_nm, idx_year, idx_amount = 6, 7, 2, 10
        
        match_count = 0
        for _, row in df.iterrows():
            try:
                acc_id = str(row[idx_id]).strip()
                acc_nm = str(row[idx_nm]).strip()
                year = str(row[idx_year]).strip()
                corp_name = str(row.iloc[-1]).strip()

                for key, target in TARGET_MAPPING.items():
                    is_match = False
                    # 1. ID가 정확히 일치할 때
                    if acc_id == target['id']:
                        is_match = True
                    # 2. 명칭 매칭 (부채 추출 시 '자본' 혼입 방지)
                    elif target['nm'] in acc_nm:
                        if key == '부채총계' and ('자본' in acc_nm or 'Equity' in acc_id):
                            is_match = False
                        else:
                            is_match = True
                    
                    if is_match:
                        val = clean_amount(row[idx_amount])
                        if val is not None:
                            all_data.append({
                                'corp_name': corp_name,
                                'year': year,
                                'indicator': key,
                                'amount': val
                            })
                            match_count += 1
            except:
                continue
        print(f"추출 완료: {match_count}건")

    if not all_data: return pd.DataFrame()
    
    raw_df = pd.DataFrame(all_data)
    # 중복 제거 및 피벗 (회사/연도별 한 줄 요약)
    return raw_df.drop_duplicates(['corp_name', 'year', 'indicator']).pivot_table(
        index=['corp_name', 'year'], columns='indicator', values='amount', aggfunc='first'
    ).reset_index()

def create_json_dataset(df):
    """분석 지표(ROE, 부채비율 등)를 계산하여 최종 JSONL 데이터셋 생성"""
    dataset = []
    for _, row in df.iterrows():
        # 주요 수치 로드
        a, l, e = row.get('자산총계'), row.get('부채총계'), row.get('자본총계')
        sales, op_income, net_income = row.get('매출액'), row.get('영업이익'), row.get('당기순이익')

        # 회계 등식 검증 (자산 = 부채 + 자본)
        if a and l and e:
            if abs(a - (l + e)) > abs(a * 0.01): # 1% 이상 차이 시 불량 데이터로 간주
                continue

        # --- 파생 분석 지표 계산 ---
        analysis_ratios = {}
        try:
            # 1. 부채비율 (부채 / 자본)
            if l and e and e != 0:
                analysis_ratios['부채비율'] = round((l / e) * 100, 2)
            
            # 2. 자기자본비율 (자본 / 자산)
            if e and a and a != 0:
                analysis_ratios['자기자본비율'] = round((e / a) * 100, 2)
                
            # 3. 영업이익률 (영업이익 / 매출액)
            if op_income and sales and sales != 0:
                analysis_ratios['영업이익률'] = round((op_income / sales) * 100, 2)
                
            # 4. ROE (당기순이익 / 자본)
            if net_income and e and e != 0:
                analysis_ratios['ROE'] = round((net_income / e) * 100, 2)
        except:
            pass

        # 텍스트 Input 구성
        financial_info = [f"{k}: {int(row[k]):,}원" for k in TARGET_MAPPING.keys() if pd.notnull(row.get(k))]
        ratio_info = [f"{k}: {v}%" for k, v in analysis_ratios.items()]
        
        # 지표가 너무 부족한 데이터는 제외
        if len(financial_info) < 5: continue

        input_text = f"{row['corp_name']}의 {row['year']}년도 주요 재무 실적 및 분석 정보: " 
        input_text += " | ".join(financial_info) + " [분석 지표] " + " | ".join(ratio_info)

        # Output 구성
        output_content = {
            "metadata": {"company": row['corp_name'], "fiscal_year": row['year']},
            "financial_metrics": {k: int(row[k]) for k in TARGET_MAPPING.keys() if pd.notnull(row.get(k))},
            "analysis_ratios": analysis_ratios
        }
        
        dataset.append({
            "instruction": "제시된 재무 데이터를 바탕으로 핵심 지표 8종을 추출하고 주요 재무 비율(부채비율, 자기자본비율 등)을 분석하여 JSON으로 응답하세요.",
            "input": input_text,
            "output": json.dumps(output_content, ensure_ascii=False, indent=2)
        })
    return dataset

if __name__ == "__main__":
    # 1. 데이터 추출
    final_df = extract_comprehensive_data(FILE_LIST)
    
    if not final_df.empty:
        # 2. 지표 계산 및 데이터셋 변환
        instruction_data = create_json_dataset(final_df)
        
        # 3. 결과 저장
        output_filename = 'dart_financial_analysis_dataset.jsonl'
        with open(output_filename, 'w', encoding='utf-8') as f:
            for item in instruction_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"✅ 최종 성공: 총 {len(instruction_data)}건의 분석 데이터셋이 {output_filename}에 저장되었습니다.")
    else:
        print("❌ 실패: 추출된 데이터가 없습니다.")