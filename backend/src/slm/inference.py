# 파인튜닝한 slm 테스트 하는 코드

from unsloth import FastLanguageModel
import torch

# 1. 저장된 모델 불러오기
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "dart_analysis_small_model",
    max_seq_length = 1024,
    dtype = None,
    load_in_4bit = True,
)
FastLanguageModel.for_inference(model)

# 2. 테스트 데이터 및 프롬프트 설정
test_input = "삼성전자의 2024년도 주요 재무 실적: 매출액: 300,000,000,000,000원 | 영업이익: 50,000,000,000,000원 | 당기순이익: 40,000,000,000,000원 | 자산총계: 450,000,000,000,000원 | 부채총계: 100,000,000,000,000원 | 자본총계: 350,000,000,000,000원 | 영업활동현금흐름: 60,000,000,000,000원 | 자본금: 800,000,000,000원 [분석 지표] 부채비율: 28.57% | 자기자본비율: 77.78% | 영업이익률: 16.67% | ROE: 11.43%"

alpaca_prompt = """아래는 작업을 설명하는 명령어와 추가 컨텍스트를 제공하는 입력이 쌍을 이루는 예제입니다. 요청을 적절히 완료하는 응답을 작성하세요.

### Instruction:
제시된 재무 데이터를 바탕으로 핵심 지표 8종을 추출하고 주요 재무 비율(부채비율, 자기자본비율 등)을 분석하여 JSON으로 응답하세요.

### Input:
{}

### Response:
{}"""

# 3. 토크나이징 (Response 시작 부분에 '{'를 넣어 가이드 제공)
inputs = tokenizer(
    [
        alpaca_prompt.format(
            "제시된 재무 데이터를 바탕으로 핵심 지표 8종을 추출하고 주요 재무 비율(부채비율, 자기자본비율 등)을 분석하여 JSON으로 응답하세요.",
            test_input,
            "{"  # 모델이 JSON 형식을 바로 시작하도록 유도
        )
    ], return_tensors = "pt").to("cuda")

# 4. 생성 설정 (반복 방지 및 정밀도 향상)
outputs = model.generate(
    **inputs, 
    max_new_tokens = 512, 
    use_cache = True,
    temperature = 0.1,         # 0에 가까울수록 모델이 헛소리를 덜 하고 훈련된 대로만 답함
    repetition_penalty = 1.2,  # 같은 단어나 문장을 반복하면 감점을 주어 억제
    eos_token_id = tokenizer.eos_token_id # 문장이 끝나면 확실히 멈추게 함
)

# 5. 결과 출력 및 가공
decoded_output = tokenizer.batch_decode(outputs)[0]
# Response 이후의 내용만 추출
response_part = decoded_output.split("### Response:")[1]

# 만약 프롬프트에서 강제로 넣은 "{"가 잘렸다면 다시 붙여줌
if not response_part.strip().startswith("{"):
    response_part = "{" + response_part

# 불필요한 종료 토큰(<|eot_id|> 등) 제거
final_json = response_part.split("<|")[0].strip()

print("=== 모델의 분석 결과 (JSON) ===")
print(final_json)