"""
fetch_financials.py — OpenDART API를 통한 재무제표 수집 모듈 (현재 미사용)

[역할]
  OpenDartReader 라이브러리를 사용하여 DART 전자공시 시스템에서
  상장 기업의 재무제표 데이터를 수집·정제하는 스크립트.

[주요 함수]
  - mass_collect_financials(): 시가총액 상위 기업 목록(corp_list.csv) 기반
    대량 재무제표 수집. 이어받기(resume) 로직 포함.
  - get_refined_financials(): 단일 기업의 재무제표를 조회하고,
    processing_financials.refine_dart_res()로 8대 핵심 지표를 정제하여 반환.

[의존]
  - processing_financials.py (같은 디렉토리) → refine_dart_res()
  - OpenDartReader, pandas, dotenv

[참조하는 곳]
  - models/dart_langgraph.py (실험용 LangGraph 파이프라인)
"""
import OpenDartReader
import pandas as pd
import os
import time
from dotenv import load_dotenv
from processing_financials import refine_dart_res

env_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', '.env')
load_dotenv(env_path)
dart = OpenDartReader(os.getenv("DART_API_KEY"))

def mass_collect_financials(target_year=2024):
    list_path = "../../data/raw/corp_list.csv"
    save_path = f"../../data/raw/fs_full_{target_year}.csv"
    
    # 1. 대상 리스트 로드
    df_listed = pd.read_csv(list_path)
    
    # 2. 이어받기 로직: 기존에 저장된 파일이 있다면 이미 수집된 기업 제외
    if os.path.exists(save_path):
        existing_df = pd.read_csv(save_path)
        done_corps = existing_df['corp_name_origin'].unique().tolist()
        df_todo = df_listed[~df_listed['corp_name'].isin(done_corps)]
        print(f"🔄 이어받기 모드: {len(done_corps)}개 완료, {len(df_todo)}개 남음")
    else:
        df_todo = df_listed
        print(f"🚀 신규 수집 모드: 총 {len(df_todo)}개 기업 대상")

    for idx, row in df_todo.iterrows():
        name = row['corp_name']
        code = str(row['corp_code']).zfill(8)
        
        try:
            # 연결(CFS) 시도 -> 없으면 별도(OFS) 시도
            fs = dart.finstate_all(code, target_year, reprt_code='11014', fs_div='CFS')
            if fs is None:
                fs = dart.finstate_all(code, target_year, reprt_code='11011', fs_div='OFS')
            
            if fs is not None:
                fs['corp_name_origin'] = name
                # 한 기업씩 바로 파일에 추가(Append) 모드로 저장하여 메모리 부담 감소 및 안정성 확보
                header = not os.path.exists(save_path)
                fs.to_csv(save_path, mode='a', index=False, header=header, encoding='utf-8-sig')
                print(f"[{idx+1}/{len(df_listed)}] {name} ✅")
            else:
                print(f"[{idx+1}/{len(df_listed)}] {name} ⚠️ 데이터 없음 (Skip)")
            
            time.sleep(0.3) # API 제한 준수
            
        except Exception as e:
            print(f"[{idx+1}/{len(df_listed)}] {name} ❌ 에러: {e}")
            continue

# 특정 회사 정보 뽑아오는 코드
def get_refined_financials(corp_name, target_year=2025):
    print(f"--- [TOOL] get_refined_financials 호출: {corp_name} ---")
    
    try:
        # [중요] finstate 호출 전, 기업 코드가 존재하는지 먼저 확인 (TypeError 방지)
        corp_code = dart.find_corp_code(corp_name)
        if not corp_code:
            print(f"❌ 기업 코드를 찾을 수 없음: {corp_name}")
            return None

        # 코드가 있을 때만 호출
        res = dart.finstate(corp_code, target_year, reprt_code='11014')
        
        if res is None or (isinstance(res, pd.DataFrame) and res.empty):
            return None
            
        return refine_dart_res(res, corp_name)
        
    except Exception as e:
        print(f"DART API Error: {e}")
        return None

if __name__ == "__main__":
    print('여기는 fetch_financials')
    # mass_collect_financials(2025)
    # get_refined_financials('삼성전자')