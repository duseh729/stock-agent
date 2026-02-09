import OpenDartReader
import pandas as pd
import os
import time
from dotenv import load_dotenv

load_dotenv()
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
            fs = dart.finstate_all(code, target_year)
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

if __name__ == "__main__":
    mass_collect_financials(2023)