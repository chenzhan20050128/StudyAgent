"""本项目内部使用的文档/网页解析工具。

参考 testParse 目录下已实现的逻辑，但在 src 内给出精简实现，
避免对 testParse 作为 Python 包的直接依赖。
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import List

import fitz  # PyMuPDF
import requests
from bs4 import BeautifulSoup
from docx import Document


# ========= PDF / Word =========


@dataclass
class TextChunk:
    content: str
    source: str
    chunk_index: int
    char_count: int
    metadata: dict


class DocumentParser:
    """PDF / Word 文档解析为纯文本。

    这里只复用解析思路，不在这里做分块，分块交给上层的
    RecursiveCharacterTextSplitter 保持全局一致策略。
    """

    def __init__(self, chunk_size: int = 800, overlap: int = 100) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def parse_pdf(self, path: str) -> str:
        text_parts: List[str] = []
        with fitz.open(path) as doc:
            for page_num, page in enumerate(doc):
                blocks = page.get_text("blocks")
                page_text = [b[4] for b in blocks if b[6] == 0]
                text_parts.append(f"--- 第{page_num + 1}页 ---")
                text_parts.extend(page_text)
        return "\n".join(text_parts)

    def parse_word(self, path: str) -> str:
        doc = Document(path)
        text_parts: List[str] = []

        for para in doc.paragraphs:
            if para.text.strip():
                text_parts.append(para.text)

        for table in doc.tables:
            text_parts.append("\n[表格]")
            for row in table.rows:
                row_text = " | ".join(c.text.strip() for c in row.cells)
                text_parts.append(row_text)

        return "\n".join(text_parts)

    def parse(self, path: str) -> str:
        ext = os.path.splitext(path)[1].lower()
        if ext == ".pdf":
            return self.parse_pdf(path)
        if ext in {".doc", ".docx"}:
            return self.parse_word(path)
        raise ValueError(f"不支持的格式: {ext}")


# ========= Web =========


class TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.text_parts: List[str] = []
        self.skip_content = False

    def handle_starttag(self, tag, attrs):  # type: ignore[override]
        if tag in ["script", "style"]:
            self.skip_content = True

    def handle_endtag(self, tag):  # type: ignore[override]
        if tag in ["script", "style"]:
            self.skip_content = False
        elif tag in {"p", "br"}:
            self.text_parts.append("\n")

    def handle_data(self, data):  # type: ignore[override]
        if not self.skip_content:
            text = data.strip()
            if text:
                self.text_parts.append(text)

    def get_text(self) -> str:
        return "".join(self.text_parts)


class WebPageParser:
    """网页解析器，处理泛网页 + 微信 + 知乎。"""

    def __init__(self) -> None:
        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 Chrome/91.0.4472.124 Safari/537.36"
            ),
            "Accept-Language": "zh-CN,zh;q=0.9",
        }

    def parse_weixin(self, url: str) -> str:
        headers = self.headers.copy()
        headers.update(
            {
                "Referer": "https://mp.weixin.qq.com/",
                "User-Agent": (
                    "Mozilla/5.0 (Linux; Android 11; SM-G9810) "
                    "AppleWebKit/537.36 Chrome/88.0.4324.152 "
                    "Mobile Safari/537.36"
                ),
            }
        )
        resp = requests.get(url, headers=headers, timeout=30)
        resp.encoding = "utf-8"

        title_match = re.search(r'"msg_title":"([^"]*)"', resp.text)
        title = title_match.group(1) if title_match else "微信文章"

        paragraphs = re.findall(r"<p[^>]*>(.*?)</p>", resp.text, re.DOTALL)
        texts: List[str] = []
        for p in paragraphs:
            text = re.sub(r"<[^>]+>", "", p).strip()
            if text and len(text) > 10:
                texts.append(text)

        content = "\n".join(texts[:100])
        return f"【标题】{title}\n【URL】{url}\n\n{content}"

    def parse_zhihu(self, url: str) -> str:
        headers = self.headers.copy()
        headers.update(
            {
                "Referer": "https://www.zhihu.com/",
                "Accept": "text/html,application/xhtml+xml",
            }
        )
        resp = requests.get(url, headers=headers, timeout=30)
        if resp.status_code == 403:
            return (
                f"【标题】知乎文章\n【URL】{url}\n"
                "【提示】知乎防反爬虫较强，建议使用浏览器访问或利用官方 API"
            )

        resp.encoding = resp.apparent_encoding or "utf-8"
        soup = BeautifulSoup(resp.text, "html.parser")

        selectors = [
            ".Post-RichText",
            ".Post-content",
            "article",
            ".content",
        ]

        title = soup.find("h1")
        title_text = title.get_text(strip=True) if title else "知乎文章"

        content_text = ""
        for sel in selectors:
            elem = soup.select_one(sel)
            if elem is not None:
                content_text = elem.get_text(separator="\n", strip=True)
                break

        if not content_text:
            content_text = "（无法提取内容，知乎反爬虫限制）"

        return f"【标题】{title_text}\n【URL】{url}\n\n{content_text[:5000]}"

    def parse_generic(self, url: str) -> str:
        resp = requests.get(url, headers=self.headers, timeout=30)
        if resp.encoding == "ISO-8859-1":
            resp.encoding = "utf-8"
        resp.encoding = resp.apparent_encoding or "utf-8"

        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        title = soup.find("h1") or soup.find("title")
        title_text = title.get_text(strip=True) if title else ""

        selectors = [
            "article",
            "main",
            ".content",
            ".post-content",
            ".entry-content",
            "#content",
            ".article-body",
        ]
        content_text = ""
        for sel in selectors:
            elem = soup.select_one(sel)
            if elem is not None:
                content_text = elem.get_text(separator="\n", strip=True)
                break

        if not content_text:
            paragraphs = soup.find_all("p")
            content_text = "\n\n".join(
                p.get_text(strip=True)
                for p in paragraphs
                if len(p.get_text().strip()) > 30
            )

        return f"【标题】{title_text}\n【URL】{url}\n\n{content_text[:5000]}"

    def parse(self, url: str) -> str:
        try:
            if "weixin.qq.com" in url or "mp.weixin.qq.com" in url:
                return self.parse_weixin(url)
            if "zhihu.com" in url:
                return self.parse_zhihu(url)
            return self.parse_generic(url)
        except Exception as exc:  # noqa: BLE001
            return f"【错误】{exc}"
