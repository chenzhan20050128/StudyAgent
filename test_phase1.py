"""阶段一基础能力自测：
- 创建表 (SQLAlchemy metadata.create_all)
- 文档上传 & 解析 & 入库
- 基础 RAG 问答 API
"""

from pathlib import Path

from fastapi.testclient import TestClient

from main import app


client = TestClient(app)


def _print_section(title: str) -> None:
    print("\n" + "=" * 20 + f" {title} " + "=" * 20)


def _rag_query(question: str) -> dict:
    """帮助函数：对给定问题做一次 RAG 查询，并打印关键信息。"""
    print(f"[RAG] 问题: {question}")
    resp = client.post("/api/rag/query", json={"query": question})
    print("[RAG] HTTP 状态:", resp.status_code)
    print("[RAG] 原始响应预览:", resp.text[:4000])
    assert resp.status_code == 200
    payload = resp.json()
    print("[RAG] Answer 预览:", payload.get("answer", "")[:3000])

    sources = payload.get("sources", [])
    print(f"[RAG] 命中 chunk 数: {len(sources)}")
    for i, s in enumerate(sources[:3]):
        preview = s.get("content", "")[:80].replace("\n", " ")
        print(
            f"  [src{i}] doc_id={s.get('doc_id')} "
            f"section={s.get('section_title')} -> {preview}"
        )
    return payload


def test_upload_and_rag_roundtrip_basic_txt():
    """基础自测：上传一个简单 txt，然后做一次 RAG 询问。

    覆盖链路：
    - /api/documents/upload：文件落盘 -> 解析 -> 分块 -> 自动打标（如启用）-> 写 Milvus
    - /api/rag/query：hybrid_search(top50) -> rerank(top10) -> 拼上下文 -> 生成回答
    """
    _print_section("基础 TXT 上传&问答")

    content = """Python是全球最流行、最简单易学、功能最强大的编程语言之一，由荷兰人吉多·范罗苏姆（Guido van Rossum）在1991年正式发布，设计哲学是优雅、明确、简单，核心标语：人生苦短，我用Python。一、Python是什么？Python是一种解释型、面向对象、动态类型的高级编程语言。简单说：不用编译：写完代码直接运行，不用像C/C++那样先编译；语法极简：代码像英语一样好读，新手最快1天就能入门；功能全能：几乎所有领域都能用；免费开源：所有人都能免费使用、修改、分发。二、Python的核心特点（为什么它这么火？）1.语法超级简单（最大优势）不用写复杂的大括号{}，不用严格的分号;，靠缩进区分代码块。对比：Java输出一句话：public class Test{public static void main(String[] args){System.out.println("Hello World");}}Python输出一句话：print("Hello World")一行搞定！2.跨平台（Windows/Mac/Linux都能跑）同一套代码，不修改就能在所有系统运行。3.拥有海量第三方库（工具包）别人写好的功能，你直接调用，不用重复造轮子。比如：爬取网页：requests；数据分析：pandas；人工智能：tensorflow/pytorch；自动化办公：openpyxl、python-docx；网站开发：Django、Flask。4.动态类型语言不用提前声明变量类型，Python自动识别：a=10 #整数 b="你好" #字符串 c=True #布尔值。5.解释型语言代码逐行执行，方便调试，开发速度极快。三、Python能做什么？（全领域覆盖）Python是全能语言，几乎没有它做不了的事：1.数据分析与办公自动化（最常用）自动处理Excel、Word、PDF；批量处理文件、报表；数据清洗、可视化（画图表）。2.人工智能与机器学习（王牌领域）深度学习；图像识别；语音识别；大模型开发。3.网络爬虫自动抓取网页数据；新闻、商品、评论、文档批量获取。4.网站开发知乎、豆瓣、Instagram都用Python开发。5.自动化测试/运维自动测试软件；服务器自动化管理。6.游戏开发、桌面软件、物联网、区块链…总结：学一门Python=掌握N种工作技能。四、Python基础语法（核心必学）1.第一个程序print("Hello, Python!")运行后直接输出文字。2.变量与数据类型#数字 age=20 height=1.75 #字符串 name="小明" #列表（数组） scores=[90,85,98] #字典（键值对） student={"name":"小明","age":20} #布尔值 is_student=True。3.条件判断score=85 if score>=60:print("及格") else:print("不及格")。4.循环#循环5次 for i in range(5):print(i)。5.函数def add(a,b):return a+b print(add(2,3)) #输出5。6.注释#单行注释 多行注释多行注释。五、Python的运行方式1.交互式（直接输入命令）打开终端输入python或python3，直接写代码运行。2.脚本式（写文件运行）新建test.py，写代码，然后运行：python test.py。3.常用开发工具VS Code（免费轻量，最推荐）；PyCharm（专业强大）；Jupyter Notebook（数据分析神器）。六、Python的优缺点优点✅极易上手，学习成本低 ✅代码简洁，开发效率极高 ✅库极多，几乎所有需求都有现成工具 ✅跨平台 ✅就业岗位多、薪资高 ✅全球第一大学习语言。缺点❌运行速度比C/C++、Java慢（但日常使用完全足够）❌移动端开发较弱。七、Python适合谁学？零基础编程新手；办公人士（自动化处理表格）；学生（考试、竞赛、毕业设计）；数据分析、运营、财务、产品经理；想转行IT、人工智能的人。它是最适合初学者的第一门编程语言，没有之一。八、Python学习路线（最简高效）1.基础语法（变量、循环、函数）2.数据结构（列表、字典、集合）3.文件操作、异常处理4.第三方库（爬虫、数据分析、自动化）5.进阶（面向对象、并发、框架）最快1～2个月就能做出实用项目。总结1.Python=简单+强大+全能2.代码像英语一样易读，开发速度极快3.能做办公自动化、数据分析、AI、爬虫、网站等几乎所有任务4.零基础也能快速学会，就业前景极好5.是2026年最值得学习的编程语言如果你愿意，我还可以：给你做Python入门教程；帮你配置开发环境；带你写第一个项目。你想继续深入哪一部分？"""
    files = {"file": ("sample.txt", content.encode("utf-8"), "text/plain")}
    print("[TEST] 上传 TXT /api/documents/upload ...")
    resp = client.post("/api/documents/upload", files=files)
    print("[TEST] 上传响应:", resp.status_code, resp.text)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "parsed"
    doc_id = data["doc_id"]
    print(f"[TEST] 文档已解析入库, doc_id={doc_id}")

    result = _rag_query("Python 是什么语言？")
    assert "answer" in result
    assert "sources" in result


def test_upload_and_rag_with_real_docs_and_urls():
    """扩展自测：遍历本地若干 pdf/docx + 若干 URL，上传并做 RAG。

    说明：
    - 文件类用例用于覆盖 PDF/DOCX/TXT 的解析分支。
    - URL 类用例用于覆盖网页解析分支（可能受网络波动影响，因此允许失败继续）。
    - 最后的 RAG 问题用于粗略验证：有返回 answer 即可。
    """

    _print_section("批量文档上传&问答 (来自 testParse)")

    # 本地测试材料目录（你机器上的路径）。如果不存在就直接 fail，提醒先准备资料。
    base_dir = Path("d:/czAgentProgram/testParse")
    assert base_dir.exists(), "testParse 目录不存在"

    # 选取若干 pdf/docx 文件
    file_candidates = [
        "Attention Residuals.pdf",
        "Deep Residual Learning for Image Recognition.pdf",
        "作业1示例 京东智能客户分析.pdf",
        "作业1示例识货APP分析.pdf",
        "Go GMP模型调度流程解析.docx",
        "Python从入门到精通.txt",
    ]

    uploaded_docs: list[int] = []

    for name in file_candidates:
        path = base_dir / name
        if not path.exists():
            print(f"[TEST] 跳过不存在的文件: {name}")
            continue

        print(f"[TEST] 上传文件: {name}")
        with open(path, "rb") as f:
            files = {"file": (name, f.read(), "application/octet-stream")}
        resp = client.post("/api/documents/upload", files=files)
        print("[TEST] 上传响应:", resp.status_code)
        print("[TEST] 响应体预览:", resp.text[:4000])
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "parsed"
        doc_id = data["doc_id"]
        uploaded_docs.append(doc_id)
        print(f"[TEST] -> doc_id={doc_id}")

    # 额外：用几个 URL 做网页解析&入库
    url_candidates = [
        "https://zhuanlan.zhihu.com/p/481029652",
        "https://developer.jdcloud.com/article/1271",
        "https://mp.weixin.qq.com/s/My9rfp5Z31iPHT6SRBGZZA",
    ]

    for url in url_candidates:
        print(f"[TEST] 上传 URL: {url}")
        resp = client.post(
            "/api/documents/upload",
            data={"url": url, "title": url[:50]},
        )
        print("[TEST] URL 上传响应:", resp.status_code)
        print("[TEST] 响应体预览:", resp.text[:4000])
        # 对于 URL，我们允许网络异常时失败，但会记录日志
        if resp.status_code == 200:
            data = resp.json()
            uploaded_docs.append(data["doc_id"])
            print(f"[TEST] -> doc_id={data['doc_id']}")
        else:
            print("[TEST] URL 解析失败，继续其他用例")

    # 对新解析入库的文档做一次面向“残差网络”的 RAG
    if uploaded_docs:
        _print_section("针对新文档做 RAG")
        question = "残差网络主要解决了什么问题？python如何实现？"
        result = _rag_query(question)
        # 只要能正常返回 answer 即认为通过
        assert "answer" in result
        assert isinstance(result.get("sources", []), list)
