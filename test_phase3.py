"""阶段三：测验与批改端到端自测。

用例覆盖：
- 调用 /api/quizzes/generate 生成一道题（不返回标准答案）；
- 调用 /api/quizzes/{quiz_id}/submit 提交答案并批改；
- 校验 MySQL 中 quizzes/quiz_attempts/weak_points 的落库行为。

说明：
- 依赖真实 LLM / Milvus / MySQL，若缺少 DASHSCOPE_API_KEY 则跳过。
"""

from __future__ import annotations

import os
import time

import pytest
from fastapi.testclient import TestClient

from main import app
from src.db import SessionLocal
from src.models import Quiz, QuizAttempt, WeakPoint


pytestmark = pytest.mark.skipif(
    not os.getenv("DASHSCOPE_API_KEY"),
    reason="需要 DASHSCOPE_API_KEY 才能跑阶段三端到端自测",
)

client = TestClient(app)


def _print_section(title: str) -> None:
    print("\n" + "=" * 20 + f" {title} " + "=" * 20)


def test_phase3_quiz_roundtrip() -> None:
    # 端到端目标：
    # 1) /api/quizzes/generate：生成题目（不回传 answer）
    # 2) MySQL quizzes：题目 question_json 中必须包含 answer（可批改）
    # 3) /api/quizzes/{id}/submit：错答触发低分路径；正答触发高分路径
    # 4) MySQL quiz_attempts：至少 2 条记录（错答+正答）
    # 5) 若错答低分：weak_points 应生成（后续复习链路依赖它）
    _print_section("阶段三：测验与批改")
    log = "[PHASE3_TEST]"

    body = {
        "description": "请基于 Python 基础语法给我出一道单选题",
        "question_type": "single_choice",
        "doc_ids": [26],
    }
    print(f"{log} 请求生成题目 body={body}")

    t0 = time.perf_counter()
    gen_resp = client.post("/api/quizzes/generate", json=body)
    total_cost = time.perf_counter() - t0

    assert gen_resp.status_code == 200
    payload = gen_resp.json()

    print(f"{log} /api/quizzes/generate status={gen_resp.status_code}")
    print(f"{log} /api/quizzes/generate total_cost={total_cost:.2f}s")
    print(f"{log} /api/quizzes/generate payload={payload}")

    quiz_id = payload.get("quiz_id")
    question = payload.get("question", {})
    assert quiz_id
    assert question.get("stem")
    assert "answer" not in question, "generate 接口不应返回标准答案"
    print(f"{log} 生成成功 quiz_id={quiz_id} question={question}")

    with SessionLocal() as session:
        quiz_row = session.get(Quiz, quiz_id)
        assert quiz_row is not None, "quizzes 中应存在该题目"
        assert isinstance(quiz_row.question_json, dict)
        assert "answer" in quiz_row.question_json, "标准答案应写入 MySQL"
        std_answer = str(quiz_row.question_json.get("answer") or "").strip()
        print(f"{log} DB 标准答案已落库 answer={std_answer}")
        assert std_answer, "标准答案不能为空，否则无法验证正确作答路径"

    # 提交一个明显错误答案，保证触发低分路径
    wrong_answer = "__definitely_wrong__"
    print(f"{log} 提交错误答案 answer={wrong_answer}")
    submit_resp = client.post(
        f"/api/quizzes/{quiz_id}/submit",
        json={"answer": wrong_answer},
    )
    assert submit_resp.status_code == 200
    submit_payload = submit_resp.json()
    print(f"{log} 错答批改结果 payload={submit_payload}")

    assert "score" in submit_payload
    assert "comment" in submit_payload
    wrong_score = float(submit_payload.get("score", 0))

    # 再提交一次正确答案，验证高分路径
    print(f"{log} 提交正确答案 answer={std_answer}")
    submit_ok_resp = client.post(
        f"/api/quizzes/{quiz_id}/submit",
        json={"answer": std_answer},
    )
    assert submit_ok_resp.status_code == 200
    submit_ok_payload = submit_ok_resp.json()
    print(f"{log} 正答批改结果 payload={submit_ok_payload}")

    assert "score" in submit_ok_payload
    assert "comment" in submit_ok_payload
    ok_score = float(submit_ok_payload.get("score", 0))
    assert ok_score >= wrong_score, "正确作答得分应不低于错误作答"

    with SessionLocal() as session:
        quiz_row = session.get(Quiz, quiz_id)
        assert quiz_row is not None, "quizzes 中应存在该题目"
        assert isinstance(quiz_row.question_json, dict)
        assert "answer" in quiz_row.question_json, "标准答案应写入 MySQL"
        print(
            f"{log} DB quiz row: "
            f"question_type={quiz_row.question_type}, "
            f"source_chunks={quiz_row.source_chunks}"
        )

        attempt_rows = (
            session.query(QuizAttempt)
            .filter(QuizAttempt.quiz_id == quiz_id, QuizAttempt.user_id == 1)
            .all()
        )
        assert attempt_rows, "quiz_attempts 中应存在作答记录"
        assert len(attempt_rows) >= 2, "应至少有两次作答记录（错答+正答）"

        ordered_attempts = sorted(attempt_rows, key=lambda x: x.id)
        latest = ordered_attempts[-1]
        print(f"{log} 当前题目作答总数 count={len(ordered_attempts)}")
        for idx, row in enumerate(ordered_attempts, 1):
            print(
                f"{log} attempt#{idx} id={row.id} score={row.score} "
                f"answer_json={row.answer_json}"
            )

        print(
            f"{log} latest attempt:",
            {
                "id": latest.id,
                "quiz_id": latest.quiz_id,
                "score": latest.score,
                "comment": latest.comment,
                "answer_json": latest.answer_json,
            },
        )

        # score < 4 触发薄弱点：此处不依赖“最新 attempt 一定是错答”，
        # 而是按 quiz_id + user_id 查询 weak_points，更稳。
        if wrong_score < 4:
            weak_rows = (
                session.query(WeakPoint)
                .filter(
                    WeakPoint.quiz_id == quiz_id,
                    WeakPoint.attempt_id == latest.id,
                    WeakPoint.user_id == 1,
                )
                .all()
            )
            # latest 可能是正答记录，因此这里按 quiz_id 查询并打印更稳妥
            weak_rows = (
                session.query(WeakPoint)
                .filter(WeakPoint.quiz_id == quiz_id, WeakPoint.user_id == 1)
                .all()
            )
            print(f"{log} weak_points count={len(weak_rows)}")
            for idx, wp in enumerate(sorted(weak_rows, key=lambda x: x.id), 1):
                print(
                    f"{log} weak_point#{idx} id={wp.id} level={wp.level} "
                    f"desc={wp.description} "
                    f"related_doc_id={wp.related_doc_id} "
                    f"related_chunk_ids={wp.related_chunk_ids}"
                )
            assert weak_rows, "低分时应生成 weak_points 记录"


def test_phase3_short_answer_excel() -> None:
    # 覆盖主观题批改路径（LLM 评分 + comment/weak_point），与客观题规则判卷不同。
    _print_section("阶段三：简答题（Python处理Excel）")
    log = "[PHASE3_TEST]"

    body = {
        "description": "请基于 Python 处理 Excel 相关知识给我出一道简答题",
        "question_type": "short_answer",
        "doc_ids": [26],
    }
    print(f"{log} 请求生成简答题 body={body}")

    t0 = time.perf_counter()
    gen_resp = client.post("/api/quizzes/generate", json=body)
    total_cost = time.perf_counter() - t0

    assert gen_resp.status_code == 200
    payload = gen_resp.json()

    print(f"{log} /api/quizzes/generate status={gen_resp.status_code}")
    print(f"{log} /api/quizzes/generate total_cost={total_cost:.2f}s")
    print(f"{log} /api/quizzes/generate payload={payload}")

    quiz_id = payload.get("quiz_id")
    question = payload.get("question", {})
    assert quiz_id
    assert question.get("stem")
    assert "answer" not in question, "generate 接口不应返回标准答案"
    print(f"{log} 简答题生成成功 quiz_id={quiz_id} question={question}")

    with SessionLocal() as session:
        quiz_row = session.get(Quiz, quiz_id)
        assert quiz_row is not None, "quizzes 中应存在该题目"
        assert isinstance(quiz_row.question_json, dict)
        assert "answer" in quiz_row.question_json, "标准答案应写入 MySQL"
        std_answer = str(quiz_row.question_json.get("answer") or "").strip()
        print(f"{log} DB 标准答案已落库 answer={std_answer[:50]}...")

    # 提交错误答案
    wrong_answer = "使用 print 打印数据"
    print(f"{log} 提交错误答案 answer={wrong_answer}")
    submit_resp = client.post(
        f"/api/quizzes/{quiz_id}/submit",
        json={"answer": wrong_answer},
    )
    assert submit_resp.status_code == 200
    submit_payload = submit_resp.json()
    print(
        f"{log} 错答批改结果 score={submit_payload.get('score')} "
        f"comment={submit_payload.get('comment')}"
    )

    wrong_score = float(submit_payload.get("score", 0))
    assert wrong_score < 5, "明显错误的答案应该得分不满分"

    # 提交正确答案
    correct_answer = "运用工具自动化完成 Excel 公式运算、数据透视、格式排版，Word 文档批量修订与模板套用，PDF 拆分合并、格式转换与内容提取；可批量统筹各类文件分类归档、自动化报表生成汇总；具备专业数据清洗、异常值剔除、数据规整能力，同时独立完成多维度数据分析与高清可视化图表制作。"
    "或使用 openpyxl/xlrd 库进行处理"
    print(f"{log} 提交正确答案 answer={correct_answer}")
    submit_ok_resp = client.post(
        f"/api/quizzes/{quiz_id}/submit",
        json={"answer": correct_answer},
    )
    assert submit_ok_resp.status_code == 200
    submit_ok_payload = submit_ok_resp.json()
    print(
        f"{log} 正答批改结果 score={submit_ok_payload.get('score')} "
        f"comment={submit_ok_payload.get('comment')}"
    )

    ok_score = float(submit_ok_payload.get("score", 0))
    assert ok_score >= wrong_score, "正确作答得分应不低于错误作答"

    with SessionLocal() as session:
        attempt_rows = (
            session.query(QuizAttempt)
            .filter(QuizAttempt.quiz_id == quiz_id, QuizAttempt.user_id == 1)
            .all()
        )
        assert len(attempt_rows) >= 2, "应至少有两次作答记录"

        print(f"{log} 简答题作答总数 count={len(attempt_rows)}")
        for idx, row in enumerate(attempt_rows, 1):
            print(
                f"{log} attempt#{idx} score={row.score} "
                f"comment={row.comment[:30] if row.comment else 'None'}..."
            )
