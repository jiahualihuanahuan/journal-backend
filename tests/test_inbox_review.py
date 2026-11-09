from datetime import date

import pytest

from app.inbox import parse_draft


def _drafts(vault):
    return sorted((vault / "Journal" / "Inbox").glob("*草稿.md"))


def test_inbox_text_to_draft_to_journal(client, vault):
    inbox = vault / "Journal" / "Inbox"
    inbox.mkdir(parents=True, exist_ok=True)
    (inbox / "voice.txt").write_text("今天纳指大跌，我买了一点。孩子说这是火箭。", "utf-8")

    r1 = client.post("/inbox/process").json()
    assert len(r1["drafts"]) == 1 and not r1["errors"]
    assert not (inbox / "voice.txt").exists()
    assert (inbox / "processed" / "voice.txt").exists()
    draft = _drafts(vault)[0]
    text = draft.read_text("utf-8")
    assert "status: needs-answers" in text and "### 1. 当时恐慌打几分？" in text

    # 没回答前不会生成
    assert client.post("/inbox/process").json()["composed"] == []

    text = text.replace("### 1. 当时恐慌打几分？\n回答：", "### 1. 当时恐慌打几分？\n回答：8 分")
    text = text.replace("- [ ] 我已回答完", "- [x] 我已回答完")
    draft.write_text(text, "utf-8")
    r3 = client.post("/inbox/process").json()
    assert len(r3["composed"]) == 1 and not r3["errors"]
    assert _drafts(vault) == []
    done = list((inbox / "processed").glob("*草稿.md"))[0].read_text("utf-8")
    assert "status: done" in done
    day = date.today().isoformat()
    journal = (vault / "Journal" / f"{day}.md").read_text("utf-8")
    assert "8 分" in journal

    # 幂等：再跑什么都不做
    r4 = client.post("/inbox/process").json()
    assert r4 == {"drafts": [], "composed": [], "skipped": [], "errors": []}


def test_inbox_duplicate_source_ignored(client, vault):
    inbox = vault / "Journal" / "Inbox"
    inbox.mkdir(parents=True, exist_ok=True)
    (inbox / "a.md").write_text("同一段话", "utf-8")
    client.post("/inbox/process")
    (inbox / "a.md").write_text("同一段话", "utf-8")  # 同步又冒出来
    r = client.post("/inbox/process").json()
    assert r["drafts"] == []
    assert len(_drafts(vault)) == 1


def test_inbox_audio_without_whisper_reports_error(client, vault):
    inbox = vault / "Journal" / "Inbox"
    inbox.mkdir(parents=True, exist_ok=True)
    (inbox / "a.m4a").write_bytes(b"fake")
    r = client.post("/inbox/process").json()
    assert r["errors"] and (inbox / "a.m4a").exists()  # 保留原文件，开了 whisper 后再处理


def test_inbox_status_answered_frontmatter(client, vault):
    inbox = vault / "Journal" / "Inbox"
    inbox.mkdir(parents=True, exist_ok=True)
    (inbox / "b.txt").write_text("今天学到复利", "utf-8")
    client.post("/inbox/process")
    d = _drafts(vault)[0]
    d.write_text(d.read_text("utf-8").replace("status: needs-answers", "status: answered"), "utf-8")
    assert len(client.post("/inbox/process").json()["composed"]) == 1


def test_inbox_not_indexed(client, vault):
    inbox = vault / "Journal" / "Inbox"
    inbox.mkdir(parents=True, exist_ok=True)
    (inbox / "c.md").write_text("独角兽关键词", "utf-8")
    client.post("/inbox/process")
    client.post("/reindex")
    assert client.post("/search", json={"query": "独角兽关键词"}).json()["results"] == []


def test_parse_draft():
    text = ("---\njournal-backend: draft\n---\n## 原始记录\n\n原话\n\n## 追问\n\n### 1. 问一\n回答：答一\n\n"
            "### 2. 问二\n回答：\n\n## 想再补充的\n\n补充\n\n---\n- [x] 我已回答完\n")
    raw, answers, extra = parse_draft(text)
    assert raw == "原话" and extra == "补充"
    assert answers == [{"question": "问一", "answer": "答一"}, {"question": "问二", "answer": ""}]


def test_invest_reviews_due(client, vault):
    client.post("/journal/compose", json={"text": "纳指大跌", "date": "2026-11-01"})
    r = client.get("/invest/reviews", params={"as_of": "2026-11-08"}).json()
    assert [i["slot"] for i in r["items"]] == ["一周后回看"]
    assert r["items"][0]["status"] == "due" and r["items"][0]["link"] == "[[Journal/2026-11-01#投资记录]]"
    r = client.get("/invest/reviews", params={"as_of": "2027-03-01"}).json()
    assert len(r["items"]) == 3 and all(i["status"] == "overdue" for i in r["items"])
    # 勾选后就不再提醒
    p = vault / "Journal" / "2026-11-01.md"
    p.write_text(p.read_text("utf-8").replace("- [ ] 一周后回看", "- [x] 一周后回看"), "utf-8")
    r = client.get("/invest/reviews", params={"as_of": "2026-11-08", "write": "true"}).json()
    assert r["items"] == []
    assert (vault / r["written"]).exists()


def test_digest_refuses_to_overwrite_user_file(client, vault):
    (vault / "Journal" / "投资回看清单.md").write_text("我自己的笔记", "utf-8")
    from app.review import write_digest
    with pytest.raises(RuntimeError):
        write_digest(client.app.state.settings, date(2026, 11, 9))
    assert (vault / "Journal" / "投资回看清单.md").read_text("utf-8") == "我自己的笔记"
