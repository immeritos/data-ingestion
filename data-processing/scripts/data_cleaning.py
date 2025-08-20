# -*- coding: utf-8 -*-
"""
prep_guideline_for_rag.py
清洗、结构化并按段落切块 ADHD 指南 JSONL -> 适合向量嵌入的 JSONL

输入:  原始 jsonl（每行一个 JSON 对象）
输出:  预处理好的 jsonl（每行一个“块”）
字段:  id, source, section, breadcrumb, page_start, page_end,
       side_labels, refs, text, highlighted_text
"""
import argparse
import json
import os
import re
import uuid
from typing import List, Dict, Any, Iterable

BULLETS = ("•", "◦", "▪", "‣", "·", "●", "*", "–", "—", "-", "·")

def normalize_whitespace(s: str) -> str:
    if not s:
        return ""
    # 1) 处理断字（de-hyphenation）：把行尾连字符+换行连接成一个词
    s = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", s)
    # 2) 把奇怪的换行压成段落换行
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    # 3) 连续空行最多保留 2 个
    s = re.sub(r"\n{3,}", "\n\n", s)
    # 4) 行内多空白压缩
    s = re.sub(r"[ \t]{2,}", " ", s)
    # 5) 统一项目符号到 "- "
    lines = []
    for line in s.split("\n"):
        l = line.strip()
        if l.startswith(BULLETS):
            # 去掉所有前导 bullet 符号，只保留一个标准 "- "
            l = re.sub(rf"^[{re.escape(''.join(BULLETS))}]+\s*", "", l)
            l = "- " + l
        lines.append(l)
    s = "\n".join(lines)
    # 6) 统一引号/破折号
    s = s.replace("’", "'").replace("“", '"').replace("”", '"')
    s = s.replace("–", "-").replace("—", "-")
    s = s.replace("•", "-")
    # 7) 去除首尾空白
    s = s.strip()
    return s

def extract_years(text: str) -> List[int]:
    if not text:
        return []
    years = set()
    # 匹配 [2018] 或 (2018) 或裸年份
    for m in re.finditer(r"[\[\(]?((19|20)\d{2})[\]\)]?", text):
        try:
            y = int(m.group(1))
            if 1900 <= y <= 2100:
                years.add(y)
        except Exception:
            pass
    return sorted(years)

def safe_list(x):
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]

def make_breadcrumb(section_path: Any, side_label: Any) -> str:
    parts = []
    if side_label:
        if isinstance(side_label, list):
            parts.append("[" + ",".join([str(s) for s in side_label]) + "]")
        else:
            parts.append(f"[{side_label}]")
    if section_path:
        if isinstance(section_path, list):
            parts.extend([str(p) for p in section_path if p])
        else:
            parts.append(str(section_path))
    return " > ".join([p for p in parts if p])

def chunk_paragraphs(paras: List[str], max_chars: int = 1000) -> List[List[str]]:
    """
    以段落为单位聚合，不跨越列表项边界，尽量在句子末尾收尾。
    """
    chunks, cur, cur_len = [], [], 0

    def is_bullet(p: str) -> bool:
        return p.strip().startswith("- ")

    i = 0
    n = len(paras)
    while i < n:
        p = paras[i].strip()
        if not p:
            i += 1
            continue

        # 将连续的 bullet 段落视作一个不可拆分块
        if is_bullet(p):
            bullet_block = [p]
            j = i + 1
            while j < n and is_bullet(paras[j]):
                bullet_block.append(paras[j].strip())
                j += 1
            block_text = "\n".join(bullet_block)
            if cur_len + len(block_text) + 1 > max_chars and cur:
                chunks.append(cur)
                cur, cur_len = [], 0
            cur.append(block_text)
            cur_len += len(block_text) + 1
            i = j
            continue

        # 普通段落
        if cur_len + len(p) + 1 > max_chars and cur:
            chunks.append(cur)
            cur, cur_len = [], 0
        cur.append(p)
        cur_len += len(p) + 1
        i += 1

    if cur:
        chunks.append(cur)

    return chunks

def yield_chunks(raw_item: Dict[str, Any],
                 max_chars: int = 1000,
                 source_name: str = "adhd_guideline") -> Iterable[Dict[str, Any]]:
    # 容错读取
    text_raw = raw_item.get("text") or raw_item.get("content") or ""
    text = normalize_whitespace(text_raw)

    section_path = raw_item.get("section_path") or raw_item.get("section") or []
    side_label = raw_item.get("side_label") or raw_item.get("side_labels")
    page = raw_item.get("page")
    page_start = raw_item.get("page_start") or page or raw_item.get("pageIndex")
    page_end = raw_item.get("page_end") or page or raw_item.get("pageIndex")
    refs_field = raw_item.get("refs") or raw_item.get("references") or []

    # 从正文和 refs 提取年份
    years = set(extract_years(text))
    for r in safe_list(refs_field):
        try:
            if isinstance(r, int):
                years.add(r)
            elif isinstance(r, str) and r.isdigit():
                y = int(r)
                if 1900 <= y <= 2100:
                    years.add(y)
        except Exception:
            pass
    refs = sorted(years)

    breadcrumb = make_breadcrumb(section_path, side_label)
    section_str = " > ".join(section_path) if isinstance(section_path, list) else (section_path or "")

    # 段落拆分与聚合
    paragraphs = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    paragraph_chunks = chunk_paragraphs(paragraphs, max_chars=max_chars)

    for idx, paras in enumerate(paragraph_chunks):
        clean_text = "\n\n".join(paras).strip()

        # highlighted_text 可保留 refs 的方括号之类用于前端渲染
        highlighted_text = clean_text

        # 稳定 id（用内容 + breadcrumb + 索引生成 UUID5）
        nid = uuid.uuid5(uuid.NAMESPACE_URL, f"{breadcrumb}::{idx}::{clean_text[:120]}")

        yield {
            "id": str(nid),
            "source": source_name,
            "section": section_str,
            "breadcrumb": breadcrumb,
            "page_start": page_start,
            "page_end": page_end,
            "side_labels": safe_list(side_label),
            "refs": refs,
            # 仅把干净正文用于嵌入
            "text": clean_text,
            # 可用于 UI 高亮
            "highlighted_text": highlighted_text
        }

def process_jsonl(input_path: str, output_path: str, max_chars: int = 1000):
    total_in, total_out = 0, 0
    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total_in += 1
            try:
                obj = json.loads(line)
            except Exception:
                # 跳过坏行
                continue
            for out_obj in yield_chunks(obj, max_chars=max_chars):
                fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
                total_out += 1
    print(f"Done. read_items={total_in}, wrote_chunks={total_out}, out={output_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", "-i", required=True, help="原始 jsonl 路径")
    ap.add_argument("--output", "-o", required=True, help="输出 jsonl 路径")
    ap.add_argument("--max-chars", type=int, default=1000, help="每块最大字符数（默认1000）")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    process_jsonl(args.input, args.output, max_chars=args.max_chars)

if __name__ == "__main__":
    main()
