import OpenDartReader
import pandas as pd
import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()
api_key = os.getenv("DART_API_KEY")
dart = OpenDartReader(api_key)

def save_refined_corp_list():
    try:
        # 1. DART의 모든 기업 리스트 확보
        df_all = dart.corp_codes
        
        # 2. 상장사 필터링: stock_code가 있는 기업만 추출
        # NaN(결측치) 제거 및 빈 문자열 제거
        df_listed = df_all[df_all['stock_code'].notnull()].copy()
        df_listed = df_listed[df_listed['stock_code'].str.strip() != ''].copy()
        
        # 3. 데이터 정제: 종목코드(stock_code)를 6자리 문자열로 포맷팅
        # (앞자리가 0인 코드가 숫자로 인식되어 0이 사라지는 것을 방지)
        df_listed['stock_code'] = df_listed['stock_code'].astype(str).str.zfill(6)
        
        # 4. 저장 경로 설정
        save_path = "../../data/raw/corp_list.csv"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 5. CSV 저장 (한글 깨짐 방지 utf-8-sig)
        df_listed.to_csv(save_path, index=False, encoding='utf-8-sig')
        
        print("-" * 50)
        print(f"✅ 필터링 완료!")
        print(f"📦 전체 기업 수: {len(df_all):,}개")
        print(f"🎯 최종 상장사 수: {len(df_listed):,}개") # 코스피, 코스닥, 코넥스, 우선주 등
        print(f"📍 저장 위치: {os.path.abspath(save_path)}")
        print("-" * 50)

    except Exception as e:
        print(f"❌ 오류 발생: {e}")

if __name__ == "__main__":
    save_refined_corp_list()