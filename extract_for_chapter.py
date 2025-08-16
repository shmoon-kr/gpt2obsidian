#!/usr/bin/env python3
# extract_for_chapter.py
# 목적: Obsidian Vault에서 지정한 노트/섹션/태그에 해당하는 **핵심 발췌본**을 묶어
#       "이번 회차 프롬프트" 파일로 생성.
#
# 예)
#  python extract_for_chapter.py --vault "G:\Obsidian\DarkLord" --chapter "릴과 히로인 첫 대면" \
#    --char "릴리스 녹타:말투 & 서술 톤 가이드;능력 세트;전투 운영(메커닉 관점)" \
#    --char "발렌 아드로스:말투 & 서술 톤 가이드;능력 세트;전투 운영(메커닉 관점)" \
#    --note "무자비의-경기장:씬 템플릿;라노벨/성인 버전 스위치" \
#    --tags "novel,prompt" --max-chars 9000 --out ".\_build\2025-08-16_릴-첫대면_prompt.md" \
#    --summarize --model gpt-4o-mini
# python .\extract_for_chapter.py --vault "G:\\내 드라이브\\Obsidian\\DarkLord\\" --chapter "릴과 히로인 첫 대면" --char "릴리스 녹타" --char "발렌 아드로스" --note "무자비의 경기장" --max-chars 9000 --out ".\_build\2025-08-16_릴-첫대면_prompt.md
#
# 준비물: pip install python-dotenv requests (요약 옵션 쓸 때만 OPENAI_API_KEY 필요)

import argparse, os, re, sys, json
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Tuple, Dict

# --- .env 자동 로드(옵션) ---
try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(usecwd=True) or ".env")
except Exception:
    pass

# ---------- 유틸 ----------
def read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore")

def write_text(p: Path, s: str):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(s, encoding="utf-8")

def slugify(s: str) -> str:
    s = s.strip()
    s = re.sub(r"[\\/:*?\"<>|]", "_", s)
    s = re.sub(r"\s+", "-", s)
    return s.strip("-_") or "note"

# ---------- 프런트매터 파서 ----------
def parse_frontmatter(md: str) -> Tuple[Dict, str]:
    if md.startswith("---"):
        end = md.find("\n---", 3)
        if end != -1:
            raw = md[3:end].strip()
            body = md[end+4:]
            fm = _yaml_to_dict(raw)
            return fm, body
    return {}, md

def _yaml_to_dict(raw: str) -> Dict:
    # 가벼운 YAML 파싱(단순 key: value / key: [a, b]만)
    d = {}
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"): continue
        if ":" not in line: continue
        k, v = line.split(":", 1)
        k = k.strip()
        v = v.strip()
        if v.startswith("[") and v.endswith("]"):
            items = [x.strip() for x in v[1:-1].split(",") if x.strip()]
            d[k] = items
        else:
            d[k] = v
    return d

# ---------- 섹션 추출 ----------
def extract_section(md_body: str, title_substring: str) -> Optional[str]:
    """
    '## 제목' 형태의 섹션 중, 제목에 부분일치하는 섹션 하나를 추출.
    다음 동급(##) 또는 상위(#) 헤딩 전까지 포함.
    """
    # 헤딩 토큰 탐색
    pat = re.compile(r"^(#{1,6})\s*(.+?)\s*$", re.M)
    matches = list(pat.finditer(md_body))
    # 섹션 인덱싱
    for i, m in enumerate(matches):
        level = len(m.group(1))
        title = m.group(2).strip()
        if title_substring.strip().lower() in title.lower():
            start = m.start()
            # 같은 레벨 이하 다음 헤딩까지
            end = len(md_body)
            for j in range(i+1, len(matches)):
                next_level = len(matches[j].group(1))
                if next_level <= level:
                    end = matches[j].start()
                    break
            return md_body[start:end].strip()
    return None

# ---------- 노트 검색 ----------
def find_note_paths(vault: Path, query: str) -> List[Path]:
    """
    쿼리:
      - 파일명(확장자/하이픈 포함 가능) 부분일치
      - 프런트매터 title/aliases에 부분일치
    """
    q = query.lower()
    hits = []
    for p in vault.rglob("*.md"):
        name = p.stem.lower()
        if q in name:
            hits.append(p); continue
        try:
            md = read_text(p)
        except Exception:
            continue
        fm, _ = parse_frontmatter(md)
        title = (fm.get("title") or "").lower()
        aliases = fm.get("aliases") or []
        ali_join = " ".join(aliases).lower() if isinstance(aliases, list) else str(aliases).lower()
        if (q and (q in title or q in ali_join)):
            hits.append(p)
    # 우선순위: 파일명 정확도 > 프런트매터 일치
    hits = sorted(set(hits), key=lambda x: (len(x.name), str(x)))
    return hits

def find_by_tag(vault: Path, tag_names: List[str]) -> List[Path]:
    tags = [t.strip().lower() for t in tag_names if t.strip()]
    if not tags: return []
    hits = []
    for p in vault.rglob("*.md"):
        try:
            md = read_text(p)
        except Exception:
            continue
        fm, _ = parse_frontmatter(md)
        fm_tags = [str(x).lower() for x in (fm.get("tags") or [])]
        if any(t in fm_tags for t in tags):
            hits.append(p)
    return hits

# ---------- 요약(옵션) ----------
def _openai_chat_complete(prompt: str, model: str, max_tokens: int) -> Optional[str]:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None
    import requests, time
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a concise Korean writing assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
        "top_p": 0.9,
        "max_tokens": max_tokens
    }
    for k in range(3):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=60)
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"].strip()
            if r.status_code in (429, 500, 502, 503):
                time.sleep(1.2*(k+1)); continue
            break
        except Exception:
            time.sleep(1.2*(k+1))
    return None

def summarize_block(text: str, model: str) -> str:
    if len(text) > 12000:
        text = text[:9000] + "\n...\n" + text[-2500:]
    prompt = "다음 내용을 6~10줄로 간결히 요약하세요(불릿X, 단락형):\n\n" + text
    out = _openai_chat_complete(prompt, model=model, max_tokens=420)
    if out: return out
    # 폴백: 앞문장 몇 개
    parts = re.split(r"(?<=[.!?…])\s+", text.strip())
    return " ".join(parts[:7]) if parts else text[:800]

# ---------- 본문 제한 ----------
def trim_to_max(s: str, max_chars: Optional[int]) -> str:
    if not max_chars: return s
    if len(s) <= max_chars: return s
    head = s[: int(max_chars*0.7)]
    tail = s[- int(max_chars*0.25):]
    return head + "\n\n---\n(생략)\n---\n\n" + tail

# ---------- 메인 ----------
def main():
    ap = argparse.ArgumentParser(description="Obsidian 발췌 → 회차 프롬프트 생성기")
    ap.add_argument("--vault", required=True, help="Obsidian Vault 루트")
    ap.add_argument("--chapter", required=True, help="회차/챕터 제목")
    ap.add_argument("--out", default=None, help="출력 파일 경로(없으면 vault/_build/{date}_{slug}.md)")
    # 반복 옵션: --char "이름:섹션1;섹션2"  / --note "파일명or제목:섹션1;섹션2"
    ap.add_argument("--char", action="append", default=[], help="캐릭터 발췌 규칙 (이름:섹션1;섹션2;...)")
    ap.add_argument("--note", action="append", default=[], help="일반 노트 발췌 규칙 (파일명or제목:섹션1;섹션2;...)")
    ap.add_argument("--tags", default="", help="추가 포함할 태그들(쉼표 구분). 해당 노트는 '전체 본문' 포함")
    ap.add_argument("--summarize", action="store_true", help="각 블록에 1~2문단 요약 추가(OpenAI)")
    ap.add_argument("--model", default="gpt-4o-mini", help="요약 모델")
    ap.add_argument("--max-chars", type=int, default=9000, help="최대 글자수(초과시 앞/뒤만 남김)")
    args = ap.parse_args()

    vault = Path(os.path.expanduser(args.vault)).resolve()
    if not vault.exists():
        print(f"[에러] Vault 경로 없음: {vault}", file=sys.stderr); sys.exit(1)

    # 출력 경로 결정
    if args.out:
        out_path = Path(args.out).resolve()
    else:
        build_dir = vault / "_build"
        date = datetime.now().strftime("%Y-%m-%d")
        out_path = build_dir / f"{date}_{slugify(args.chapter)}.md"

    blocks = []
    hdr = f"# 회차 프롬프트 — {args.chapter}\n\n생성일: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    blocks.append(hdr)

    # ---- 캐릭터 규칙 처리
    for rule in args.char:
        # "이름:섹1;섹2"
        if ":" in rule:
            name, secs = rule.split(":", 1)
            sec_list = [s.strip() for s in secs.split(";") if s.strip()]
        else:
            name, sec_list = rule, []
        paths = find_note_paths(vault, name.strip())
        if not paths:
            blocks.append(f"\n> [경고] 캐릭터 노트 미발견: {name}\n")
            continue
        p = paths[0]
        md = read_text(p); fm, body = parse_frontmatter(md)
        title = fm.get("title") or p.stem
        blocks.append(f"\n---\n## 캐릭터: {title}\n> 파일: {p.relative_to(vault)}\n")
        if not sec_list:
            # 섹션 지정 없으면 본문 전체(길면 트림)
            content = trim_to_max(body, args.max_chars)
            blocks.append(content)
            if args.summarize:
                summ = summarize_block(content, args.model)
                blocks.append(f"\n**요약**:\n{summ}\n")
        else:
            for sname in sec_list:
                sec = extract_section(body, sname)
                if not sec:
                    blocks.append(f"\n> [참고] 섹션 미발견: {sname}\n")
                    continue
                content = trim_to_max(sec, args.max_chars)
                blocks.append(f"\n### 섹션 — {sname}\n{content}\n")
                if args.summarize:
                    summ = summarize_block(content, args.model)
                    blocks.append(f"\n**요약**:\n{summ}\n")

    # ---- 일반 노트 규칙 처리
    for rule in args.note:
        if ":" in rule:
            name, secs = rule.split(":", 1)
            sec_list = [s.strip() for s in secs.split(";") if s.strip()]
        else:
            name, sec_list = rule, []
        paths = find_note_paths(vault, name.strip())
        if not paths:
            blocks.append(f"\n> [경고] 노트 미발견: {name}\n")
            continue
        p = paths[0]
        md = read_text(p); fm, body = parse_frontmatter(md)
        title = fm.get("title") or p.stem
        blocks.append(f"\n---\n## 노트: {title}\n> 파일: {p.relative_to(vault)}\n")
        if not sec_list:
            content = trim_to_max(body, args.max_chars)
            blocks.append(content)
            if args.summarize:
                summ = summarize_block(content, args.model)
                blocks.append(f"\n**요약**:\n{summ}\n")
        else:
            for sname in sec_list:
                sec = extract_section(body, sname)
                if not sec:
                    blocks.append(f"\n> [참고] 섹션 미발견: {sname}\n")
                    continue
                content = trim_to_max(sec, args.max_chars)
                blocks.append(f"\n### 섹션 — {sname}\n{content}\n")
                if args.summarize:
                    summ = summarize_block(content, args.model)
                    blocks.append(f"\n**요약**:\n{summ}\n")

    # ---- 태그 포함 노트(전체)
    tag_list = [t.strip() for t in args.tags.split(",") if t.strip()]
    if tag_list:
        tagged = find_by_tag(vault, tag_list)
        if tagged:
            blocks.append(f"\n---\n## 태그 포함 노트 (tags={', '.join(tag_list)})\n")
            for p in sorted(tagged):
                md = read_text(p); fm, body = parse_frontmatter(md)
                title = fm.get("title") or p.stem
                blocks.append(f"\n### {title}\n> 파일: {p.relative_to(vault)}\n")
                content = trim_to_max(body, args.max_chars)
                blocks.append(content)
                if args.summarize:
                    summ = summarize_block(content, args.model)
                    blocks.append(f"\n**요약**:\n{summ}\n")

    # 출력
    result = "\n".join(blocks).strip() + "\n"
    write_text(out_path, result)
    print(f"[완료] 발췌 프롬프트 생성: {out_path}")

if __name__ == "__main__":
    main()
