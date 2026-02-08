# 🚀 Stock Assistant AI: LangGraph & SLM 기반 주식 투자 도우미

본 프로젝트는 기업의 **재무제표 데이터(OpenDART)**를 수집하고, **SLM(Llama 3.2/4)** 파인튜닝과 **LangGraph**를 활용하여 고도화된 투자 분석 에이전트를 구축하는 것을 목표로 합니다.

## 🛠️ Tech Stack

- **Language:** Python 3.11 (Conda Environment)
- **Orchestration:** LangGraph, LangChain
- **AI Models:** \* **Main Reasoning:** Claude 4.6 / GPT-5 mini
- **Domain SLM:** Llama 3.2 3B (Target for 1660 Super VRAM 6GB)

- **Data:** OpenDART API (Financial Statements)
- **Database:** SQLite / Supabase (To be implemented)

## 📁 Project Structure

```text
stock-assistant-ai/
├── backend/
│   ├── data/
│   │   ├── raw/            # 원본 CSV (상장사 리스트 등)
│   │   └── dataset/        # SLM 학습용 JSONL
│   ├── src/
│   │   ├── agents/         # LangGraph 로직 (graph.py, nodes.py)
│   │   ├── slm/            # 파인튜닝 (train_lora.py)
│   │   └── tools/          # 데이터 수집기 (dart_collector.py)
│   ├── .env                # API Keys (GIT IGNORE 필수)
│   └── requirements.txt
└── frontend/               # React Dashboard

```

## ⚙️ Setup Instructions

### 1. 가상환경 설정 (Conda)

```bash
conda create -n stock-agent python=3.11 -y
conda activate stock-agent
pip install opendartreader pandas python-dotenv langgraph langchain-openai

```

### 2. 환경 변수 설정

OpenDART에서 api key 발급 후
`backend/` 폴더에 `.env` 파일을 생성하고 발급받은 키를 입력

```text
DART_API_KEY=your_api_key_here
OPENAI_API_KEY=your_openai_key_here

```

### 3. 데이터 수집 실행

```bash
cd backend/src/tools
python dart_collector.py

```

## 🎯 Current Milestone: Step 1 - Data Collection

- [x] 프로젝트 폴더 구조 설계
- [ ] 상장사 고유번호(corp_code) 리스트 수집 기능 구현
- [ ] **(Next)** 시가총액 상위 종목 대상 재무제표 텍스트 추출 루프 구현
- [ ] **(Next)** SLM 학습을 위한 재무 데이터-인사이트 페어 데이터셋 구축

---

## ⚠️ PC 작업 시 참고 (GPU 사양)

- **GPU:** NVIDIA GTX 1660 Super (VRAM 6GB)
- **Strategy:** 8B 모델보다는 **Llama 3.2 3B**급 모델을 선택하여 **Unsloth + QLoRA**로 로컬 파인튜닝 시도.
- **Optimization:** VRAM 부족 시 Google Colab 또는 RunPod을 활용한 클라우드 학습 병행.
