# gpt2obsidian

ChatGPT 대화 내보내기(JSON)를 **Obsidian 노트**로 정리해 주는 스크립트입니다.  
(옵션) OpenAI API를 이용해 **제목/요약** 파일을 자동 생성합니다.

## 기능

- `conversations.json`을 읽어 하루치 대화를 하나의 Markdown으로 병합
- 대화별 `title`을 소제목으로 포함
- 첨부파일(`files/`)을 Obsidian 폴더로 복사
- **(NEW)** 제목(`_title.md`) & 요약(`_summary.md`) 자동 생성 (OpenAI API)
- 날짜 필터링 (`--date YYYY-MM-DD`)

## 설치

```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install requests tiktoken
```

## OpenAI 요약 사용 시 환경변수 설정:

```bash
export OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxx
# Windows PowerShell
# $Env:OPENAI_API_KEY="sk-xxxxxxxxxxxxxxxx"
```

## 사용법
1) 기본 변환 (요약 없음)

```bash
python gpt2obsidian.py --input ./chatgpt_export --outdir ./vault/ChatGPT
```

출력 예시:

```
YYYY-MM-DD_conversations.md
files/  (있을 경우)
```

2) 특정 날짜만 병합

```bash
python gpt2obsidian.py --input ./chatgpt_export --outdir ./vault/ChatGPT --date 2025-08-15
```

3) 제목/요약 생성 포함

```bash
python gpt2obsidian.py --input ./chatgpt_export --outdir ./vault/ChatGPT \
  --date 2025-08-15 --summarize --model gpt-4o-mini
```

추가 출력:

```bash
YYYY-MM-DD_title.md      # 1줄 제목
YYYY-MM-DD_summary.md    # 5~8줄 요약
```
참고: OpenAI API 호출 실패 시 자동으로 로컬 간이 요약으로 폴백합니다.

동작 원리
conversations.json에서 메시지(author/text) 추출

사용자/어시스턴트 대화를 Markdown으로 정리

(옵션) 본문 일부를 모델에 입력해 제목/요약 생성

첨부(files/) 복사

자주 묻는 질문
Q. export 폴더에 conversations.json이 없어요.
A. 폴더 내 가장 큰 .json 하나를 conversations로 간주합니다. 필요시 --input에 파일을 직접 지정하세요.

Q. 요약이 너무 길어요.
A. 현재 요약은 5~8줄을 목표로 합니다. 더 줄이고 싶다면 summarize_text_kor의 max_tokens 값을 낮추세요.

Q. 모델 추천은?
A. 비용/속도 균형은 gpt-4o-mini. 더 높은 품질은 gpt-4o 계열을 권장합니다.

Q. 개인정보가 포함된 대화를 요약해도 안전한가요?
A. 모델 호출 전 민감 정보를 제거하는 것을 권장합니다. (이 스크립트는 원문 그대로 전송합니다.)