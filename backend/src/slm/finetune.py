# 파인튜닝 하는 코드
from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

# 1. 설정 (Llama-3.2-3B는 6GB에서 아주 여유롭습니다)
max_seq_length = 1024 # 1024까지도 충분히 가능하지만, 안전하게 512로 시작
dtype = None 
load_in_4bit = True 

# 2. 모델 및 토크나이저 불러오기 (3B 모델로 변경)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# 3. LoRA 설정
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, 
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth", 
    random_state = 3407,
)

# 4. 데이터셋 준비 (DART JSONL 데이터 적용)
alpaca_prompt = """아래는 작업을 설명하는 명령어와 추가 컨텍스트를 제공하는 입력이 쌍을 이루는 예제입니다. 요청을 적절히 완료하는 응답을 작성하세요.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token 

def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }

dataset = load_dataset("json", data_files="dart_financial_analysis_dataset.jsonl", split="train")
dataset = dataset.map(formatting_prompts_func, batched = True,)

# 5. 학습 설정 (VRAM 6GB 맞춤형)
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    args = TrainingArguments(
        per_device_train_batch_size = 1, # 모델이 작아져서 2도 가능할 겁니다
        gradient_accumulation_steps = 8,
        warmup_steps = 5,
        max_steps = 600,
        learning_rate = 2e-4,
        fp16 = True, 
        bf16 = False,
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)

# 6. 실행
trainer_stats = trainer.train()

# 7. 저장
model.save_pretrained("dart_analysis_small_model")
tokenizer.save_pretrained("dart_analysis_small_model")

print("작은 모델로 학습이 완료되었습니다!")