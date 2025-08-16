#!/usr/bin/env python3
# gpt2obsidian_simple.py
# 목적: ChatGPT 내보내기 JSON → "대화별 작은 md 파일"로 쪼개어 Obsidian에 넣기
#      그 외 파일은 첨부 폴더로 이동. 거대 합본 파일 절대 생성 안 함.
# ex) python .\gpt2obsidian.py --src "E:\OneDrive\ChatbotGame\GPT\Threads\20250815" --prefix-date --copy --subfolder "Archive\\ChatGPT_20250815" --attachments "_attachments\\"
# ap.add_argument("--vault", default="G:\\내 드라이브\\Obsidian\\DarkLord\\", help="Obsidian Vault 루트 경로")

import argparse
import json
import os
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ATTACHMENT_EXTS = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg",
    ".pdf", ".mp3", ".wav", ".m4a", ".mp4", ".mov", ".webm",
    ".csv", ".pptx", ".xlsx", ".docx", ".zip", ".heic"
}
TEXTLIKE_EXTS = {".md", ".txt"}

def now_iso():
    return datetime.now().astimezone().isoformat(timespec="seconds")

def iso_from_epoch(ts):
    try:
        return datetime.fromtimestamp(float(ts), tz=timezone.utc).astimezone().isoformat(timespec="seconds")
    except Exception:
        return now_iso()

def slugify(name: str) -> str:
    base = Path(name).stem if "." in name else name
    base = base.strip()
    base = re.sub(r"[\s/]+", "-", base)
    base = re.sub(r"[^\w\-\u3131-\u318E\uAC00-\uD7A3]", "", base)  # 한글 허용
    base = re.sub(r"-{2,}", "-", base).strip("-_")
    return base or "note"

def ensure_unique(path: Path) -> Path:
    if not path.exists():
        return path
    stem, suf = path.stem, path.suffix
    i = 2
    while True:
        cand = path.with_name(f"{stem}-{i}{suf}")
        if not cand.exists():
            return cand
        i += 1

def build_fm(title, original_filename=None, extra=None):
    fm = {"title": title, "created": now_iso(), "source": "ChatGPT"}
    if original_filename: fm["original_filename"] = original_filename
    if extra: fm.update(extra)
    lines = ["---"]
    for k, v in fm.items():
        if isinstance(v, list):
            lines.append(f"{k}: [{', '.join(v)}]")
        else:
            lines.append(f"{k}: {v}")
    lines.append("---\n")
    return "\n".join(lines)

def content_to_text_any(content):
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        if "text" in content and isinstance(content["text"], str):
            return content["text"]
        if "parts" in content and isinstance(content["parts"], list):
            return "\n".join(str(p) for p in content["parts"])
        if "content" in content and isinstance(content["content"], list):
            parts = []
            for it in content["content"]:
                if isinstance(it, dict) and it.get("type") == "text":
                    parts.append(it.get("text",""))
            return "\n".join(parts)
    if isinstance(content, list):
        parts = []
        for it in content:
            if isinstance(it, dict) and it.get("type") == "text":
                parts.append(it.get("text",""))
            else:
                parts.append(str(it))
        return "\n".join(parts)
    return str(content)

def role_of(msg):
    role = msg.get("role")
    if not role and isinstance(msg.get("author"), dict):
        role = msg["author"].get("role")
    if not role and isinstance(msg.get("message"), dict):
        role = (msg["message"].get("author") or {}).get("role")
    if not role and "from" in msg:
        role = msg["from"]
    return role or "unknown"

def text_of(msg):
    if "content" in msg:
        return content_to_text_any(msg["content"]).strip()
    if isinstance(msg.get("message"), dict):
        return content_to_text_any(msg["message"].get("content")).strip()
    if "value" in msg and isinstance(msg["value"], str):
        return msg["value"].strip()
    if "choices" in msg and isinstance(msg["choices"], list) and msg["choices"]:
        ch = msg["choices"][0]
        if isinstance(ch, dict) and "message" in ch:
            return (ch["message"].get("content") or "").strip()
    return ""

def time_of(msg):
    for k in ("create_time","created","timestamp","created_at"):
        if k in msg and msg[k] is not None:
            return iso_from_epoch(msg[k])
    if isinstance(msg.get("message"), dict) and msg["message"].get("create_time") is not None:
        return iso_from_epoch(msg["message"]["create_time"])
    return now_iso()

def normalize_messages(seq):
    out = []
    for m in seq:
        t = text_of(m)
        out.append({"role": role_of(m), "text": t if t else "(no content)", "ts": time_of(m)})
    return out

def collect_from_mapping(mapping: dict):
    nodes = []
    for node in mapping.values():
        msg = node.get("message")
        if msg: nodes.append(msg)
    def key(m):
        ct = m.get("create_time")
        return (0 if ct is None else 1, ct or 0.0)
    nodes.sort(key=key)
    return normalize_messages(nodes)

def parse_chat_items(data):
    """dict or list → [{title, created, messages:[...]}, ...]"""
    out = []
    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict): continue
            title = item.get("title") or "chat"
            created = iso_from_epoch(item.get("create_time")) if item.get("create_time") else now_iso()
            if isinstance(item.get("mapping"), dict):
                msgs = collect_from_mapping(item["mapping"])
            elif isinstance(item.get("messages"), list):
                msgs = normalize_messages(item["messages"])
            else:
                msgs = []
            if msgs:
                out.append({"title": title, "created": created, "messages": msgs})
        return out

    if not isinstance(data, dict):
        return out

    if isinstance(data.get("messages"), list):
        out.append({"title": data.get("title") or "chat",
                    "created": iso_from_epoch(data.get("create_time")) if data.get("create_time") else now_iso(),
                    "messages": normalize_messages(data["messages"])})
        return out

    if isinstance(data.get("mapping"), dict):
        out.append({"title": data.get("title") or "chat",
                    "created": iso_from_epoch(data.get("create_time")) if data.get("create_time") else now_iso(),
                    "messages": collect_from_mapping(data["mapping"])})
        return out

    if isinstance(data.get("conversations"), list):
        for conv in data["conversations"]:
            title = conv.get("title") or "chat"
            created = iso_from_epoch(conv.get("create_time")) if conv.get("create_time") else now_iso()
            if isinstance(conv.get("mapping"), dict):
                msgs = collect_from_mapping(conv["mapping"])
            elif isinstance(conv.get("messages"), list):
                msgs = normalize_messages(conv["messages"])
            else:
                msgs = []
            if msgs:
                out.append({"title": title, "created": created, "messages": msgs})
        return out

    return out

def md_from_conv(conv):
    lines = [f"# {conv.get('title','chat')}\n"]
    for m in conv["messages"]:
        who = {"system":"⚙️ system","user":"🧑 user","assistant":"🤖 assistant"}.get(m["role"], m["role"])
        lines.append(f"### {who} — {m['ts']}")
        lines.append(m["text"] if m["text"] else "_(no content)_")
        lines.append("")
    return "\n".join(lines).strip() + "\n"

def split_text_chunks(text: str, max_chars: int):
    if max_chars is None:
        return [text]
    if len(text) <= max_chars:
        return [text]
    chunks = []
    paras = text.split("\n\n")
    cur = ""
    for p in paras:
        add = (p + "\n\n")
        if len(cur) + len(add) > max_chars and cur:
            chunks.append(cur); cur = add
        else:
            cur += add
        if len(cur) >= int(max_chars * 1.05):
            chunks.append(cur); cur = ""
    if cur: chunks.append(cur)
    final = []
    for c in chunks:
        if len(c) <= max_chars:
            final.append(c)
        else:
            for j in range(0, len(c), max_chars):
                final.append(c[j:j+max_chars])
    return final

def get_chunks(text: str, max_chars):
    """max_chars=None이면 분할 없이 반환."""
    return split_text_chunks(text, max_chars)

def main():
    ap = argparse.ArgumentParser(description="ChatGPT JSON → 개별 작은 MD, 첨부는 이동(심플).")
    ap.add_argument("--src", required=True, help="소스 폴더 (ChatGPT 내보내기 JSON/첨부)")
    ap.add_argument("--vault", default="G:\\내 드라이브\\Obsidian\\DarkLord\\", help="Obsidian Vault 루트")
    ap.add_argument("--subfolder", default="ChatGPT", help="노트 저장 하위 폴더")
    ap.add_argument("--attachments", default="_attachments", help="첨부 저장 폴더명")
    ap.add_argument("--copy", action="store_true", help="이동 대신 복사")
    ap.add_argument("--prefix-date", action="store_true", help="파일명 앞에 YYYY-MM-DD_")
    ap.add_argument("--glob", default="*.json", help="처리할 JSON 패턴 (기본: *.json)")
    ap.add_argument("--chunk-chars", type=int, default=None, help="노트 최대 글자 수(없으면 분할 없음)")
    ap.add_argument("--embed-for-attachments", action="store_true", help="첨부 임베드 노트 생성")
    args = ap.parse_args()

    src_dir = Path(os.path.expanduser(args.src)).resolve()
    vault_dir = Path(os.path.expanduser(args.vault)).resolve()
    notes_dir = vault_dir / args.subfolder
    attach_dir = vault_dir / args.attachments

    if not src_dir.exists():
        print(f"[에러] 소스 폴더 없음: {src_dir}", file=sys.stderr); sys.exit(1)
    if not vault_dir.exists():
        print(f"[에러] Vault 폴더 없음: {vault_dir}", file=sys.stderr); sys.exit(1)

    notes_dir.mkdir(parents=True, exist_ok=True)
    attach_dir.mkdir(parents=True, exist_ok=True)

    date_prefix = datetime.now().strftime("%Y-%m-%d_") if args.prefix_date else ""

    # 1) JSON 처리
    json_files = list(src_dir.glob(args.glob))
    processed = 0
    for jf in json_files:
        try:
            raw = jf.read_text(encoding="utf-8")
            data = json.loads(raw)
        except Exception as e:
            print(f"[경고] JSON 파싱 실패: {jf.name} ({e})")
            continue

        convs = parse_chat_items(data)
        if not convs:
            # 구조 미인식: 원문을 코드블록으로라도 보존 (여기서도 chunk None-safe)
            title = slugify(jf.stem)
            body = "```json\n" + raw + "\n```\n"
            for idx, ch in enumerate(get_chunks(body, args.chunk_chars), 1):
                name = f"{date_prefix}{title}" + (f"-{idx}" if idx > 1 else "") + ".md"
                target = ensure_unique(notes_dir / name)
                target.write_text(build_fm(title, jf.name) + ch, encoding="utf-8")
                print(f"[대화원문] {jf.name} → {target.relative_to(vault_dir)}")
                processed += 1
            if not args.copy:
                jf.unlink(missing_ok=True)
            continue

        for ci, conv in enumerate(convs, 1):
            base_title = slugify(conv["title"]) or slugify(jf.stem)
            md = md_from_conv(conv)
            parts = get_chunks(md, args.chunk_chars)
            for si, ch in enumerate(parts, 1):
                name = f"{date_prefix}{base_title}"
                if len(convs) > 1: name += f"-{ci}"
                if len(parts) > 1: name += f"-part{si}"
                name += ".md"
                target = ensure_unique(notes_dir / name)
                extra = {"created": conv.get("created", now_iso())}
                target.write_text(build_fm(conv["title"], jf.name, extra) + ch, encoding="utf-8")
                print(f"[대화노트] {jf.name} → {target.relative_to(vault_dir)}")
                processed += 1
        if not args.copy:
            jf.unlink(missing_ok=True)

    # 2) 첨부/기타 파일 이동 (이 단계까지 오면 JSON 에러로 죽지 않음)
    for src in src_dir.iterdir():
        if src.is_dir(): continue
        ext = src.suffix.lower()
        if ext == ".json":  # 이미 처리함
            continue
        if ext in ATTACHMENT_EXTS or ext not in TEXTLIKE_EXTS:
            dst = ensure_unique(attach_dir / f"{date_prefix}{src.name}")
            if args.copy: shutil.copy2(src, dst)
            else: shutil.move(src, dst)
            print(f"[첨부] {src.name} → {dst.relative_to(vault_dir)}")
            if args.embed_for_attachments:
                stub = ensure_unique(notes_dir / f"{date_prefix}{slugify(src.stem)}.md")
                body = build_fm(slugify(src.stem), src.name, {"tags":["chatgpt","attachment"]})
                rel = str(dst.relative_to(vault_dir)).replace("\\","/")
                stub.write_text(body + f"![[{rel}]]\n", encoding="utf-8")
                print(f"[임베드노트] {stub.relative_to(vault_dir)}")
                processed += 1
        else:
            # .md/.txt 등은 손대지 않음
            pass

    print(f"\n완료. 생성/이동된 항목: {processed}")

if __name__ == "__main__":
    main()
