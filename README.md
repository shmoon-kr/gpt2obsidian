# gpt2obsidian

ChatGPT 내보내기 JSON을 **Obsidian 노트**로 변환하는 도구입니다.  

- 대화별 **작은 Markdown 파일**로 분할 저장 (합본 없음)  
- 첨부 파일(`.png`, `.pdf`, `.mp3` 등)은 Vault 내 `_attachments/` 폴더로 이동  
- (옵션) OpenAI API를 사용해 **제목 요약 / 본문 요약**을 프런트매터에 추가  
- (옵션) **자동 태그 / 수동 태그** 지원 → Obsidian 검색 및 정리에 활용  

---

## 주요 기능 ✨

- ✅ ChatGPT JSON → 개별 `.md` 파일 변환
- ✅ 프런트매터(`---`) 자동 생성 (`title`, `created`, `source`, `original_filename` 등)
- ✅ (옵션) OpenAI 기반 요약 → `summary` 필드에 기록  
- ✅ (옵션) 키워드 기반 **자동 태그** 추출  
- ✅ (옵션) 수동 태그(`--tags`) 추가 가능  
- ✅ 첨부 파일 자동 이동/복사, 임베드 노트 생성 옵션 제공  

---

## 설치 ⚙️

### 1. 클론 & 진입
```bash
git clone https://github.com/yourname/gpt2obsidian_simple.git
cd gpt2obsidian_simple
```
### 2. 가상환경 & 의존성 설치
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. OpenAI API Key 설정
루트에 .env 파일 생성:

```
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxx
```

## 사용법 🚀
### 기본 변환
```bash
python gpt2obsidian_simple.py \
  --src "E:\OneDrive\ChatbotGame\GPT\Threads\20250815" \
  --vault "G:\내 드라이브\Obsidian\DarkLord" \
  --subfolder "Archive\ChatGPT_20250815" \
  --attachments "_attachments" \
  --prefix-date --copy
```

* --src: ChatGPT JSON 내보내기 파일이 있는 폴더
* --vault: Obsidian Vault 루트 경로
* --subfolder: 결과 노트 저장 폴더
* --attachments: 첨부 파일 저장 폴더
* --prefix-date: 파일명 앞에 날짜 붙이기 (YYYY-MM-DD_)
* --copy: 이동 대신 복사

### 요약 & 태그 옵션
```bash
python gpt2obsidian_simple.py \
  --src "E:\export" --vault "D:\Vault" \
  --subfolder "ChatGPT" --attachments "_attachments" \
  --summarize --model gpt-4o-mini \
  --auto-tags --tags "work,diary"
```

* --summarize : OpenAI API로 요약 (제목·본문 요약 → summary:에 저장)
* --model : 요약에 사용할 모델 (gpt-4o-mini, gpt-4.1 등)
* --auto-tags : 본문 키워드 기반 자동 태그 생성
* --tags : 수동 태그 추가 (,로 구분)

### 결과 예시 📝

```markdown

---
title: 릴과 마왕 — 전투 스타일 논의
created: 2025-08-15T23:21:05+09:00
source: ChatGPT
original_filename: conversation_20250815.json
summary: 릴의 전투 무기와 마왕의 전략을 MTG 덱 스타일로 비유하며 캐릭터성을 구체화. 릴은 레드-블랙, 마왕은 블루로 정리. 무자비의 경기장 연출과 현실 리더십 비교까지 이어짐.
tags: [chatgpt, mtg, 무자비의경기장, 릴, novel]
---

# 릴과 마왕 — 전투 스타일 논의

### 🧑 user — 2025-08-15T23:18:22+09:00
채찍은 릴이 이미 들고 있는 무기라서 좀 다른 종류였으면 좋겠는데…

### 🤖 assistant — 2025-08-15T23:19:45+09:00
맞아요, 채찍은 릴의 시그니처라 다른 걸 주는 게 더 개성 살릴 수 있어요. …
```

### 태그 자동화 🔖

`--auto-tags` 사용 시, 아래 키워드를 탐지하면 자동으로 태그 추가됩니다.
(추가/수정은 `KEYWORD_TAGS` 리스트 수정)

키워드	태그
"MTG", "매직더개더링"	mtg
"무자비의 경기장"	무자비의경기장
"릴", "릴리스 녹타"	릴
"Obsidian", "옵시디언"	obsidian
"라노벨", "웹소설"	novel
그 외 AWS, VR/AR, Prompt 등	해당 태그 자동 생성

모든 노트에는 기본적으로 chatgpt 태그가 추가됩니다.

### 개발 메모 🛠
* Python 3.9 이상 권장
* Windows/WSL/macOS/Linux 테스트됨
* OpenAI API 호출 실패 시, 로컬 폴백 요약 제공
* 대규모 대화도 --chunk-chars 옵션으로 안전하게 분할 가능