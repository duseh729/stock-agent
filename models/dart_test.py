# ollama를 이용해서 파인튜닝 slm 테스트 하는 코드

from langchain_ollama import ChatOllama

# 모델 초기화
# 위에서 'ollama create'로 등록한 이름을 그대로 사용해야 합니다.
llm = ChatOllama(
    model="dart_model_v1", 
    base_url="http://localhost:11434",
    temperature=0
)

# 분석할 데이터
sample_text = "(주)에이아이컴퍼니는 2025년 매출액 2억, 영업이익 3억을 기록하였다."

# 프롬프트 구성 (Modelfile의 TEMPLATE을 활용하므로 내용만 전달)
instruction = f"다음 기업 공시 텍스트에서 주요 재무 지표를 JSON 형식으로 추출해줘.\n텍스트: {sample_text}"
prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"

try:
    print(f"모델 '{llm.model}'에 연결 중...")
    response = llm.invoke(prompt)
    
    print("\n" + "="*50)
    print("[재무 지표 추출 결과]")
    print("="*50)
    print(response.content.strip())
    print("="*50)

except Exception as e:
    print(f"에러 발생: {e}")
    print("\nTIP: 'ollama list' 명령어를 통해 'dart_model_v1'이 목록에 있는지 확인해보세요.")