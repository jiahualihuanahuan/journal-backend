from datetime import date

from app.journal import add_months, write_entry


def test_followups_parsed_and_capped(client):
    r = client.post("/journal/followups", json={"text": "今天纳指跌了，我买了一点"})
    assert r.status_code == 200
    qs = r.json()["questions"]
    assert qs == ["当时恐慌打几分？", "孩子原话是什么？", "学到了什么？"]


def test_compose_creates_structured_note(client, vault):
    r = client.post("/journal/compose", json={
        "text": "今天纳指跌了六点几，我硬着头皮买了一点。晚上和孩子搭乐高。",
        "answers": [{"question": "恐慌几分？", "answer": "8 分吧"}],
        "date": "2026-11-09"})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["action"] == "created" and j["path"] == "Journal/2026-11-09"
    md = (vault / "Journal" / "2026-11-09.md").read_text("utf-8")
    assert md.startswith("---\ndate: 2026-11-09\n")
    assert "fear_level: 8" in md
    assert "themes: [learnings, investing, kid]" in md
    for h in ("## 今天学到的", "## 投资记录", "## 和孩子"):
        assert h in md
    assert "> [!ai] AI 补充" in md
    assert "- [ ] 一周后回看 📅 2026-11-16：" in md
    assert "- [ ] 一个月后回看 📅 2026-12-09：" in md
    assert "- [ ] 三个月后回看 📅 2027-02-09：" in md
    assert "“妈妈，这是火箭！”" in md
    assert "> [!quote]- 原始记录" in md and "8 分吧" in md
    # AI 补充只在 callout 里
    assert "- 可以提前写好分批加仓规则" not in md.replace("> - 可以提前写好分批加仓规则", "")


def test_compose_appends_never_overwrites(client, vault):
    p = vault / "Journal" / "2026-11-10.md"
    original = "---\ndate: 2026-11-10\n---\n# 我自己写的\n\n手写内容，不能丢。"  # 末尾没有换行
    p.write_text(original, "utf-8")
    r1 = client.post("/journal/compose", json={"text": "第一段", "date": "2026-11-10"})
    r2 = client.post("/journal/compose", json={"text": "第二段", "date": "2026-11-10"})
    assert r1.json()["action"] == "appended" and r2.json()["action"] == "appended"
    md = p.read_text("utf-8")
    assert md.startswith(original + "\n")
    assert md.count("补充记录") == 2
    assert md.count("---\ndate:") == 1          # 没有重复 frontmatter
    assert "### 投资记录" in md and "#journal/investing" in md


def test_compose_dry_run_writes_nothing(client, vault):
    r = client.post("/journal/compose", json={"text": "x", "date": "2026-11-11", "dry_run": True})
    assert r.json()["action"] == "dry_run"
    assert not (vault / "Journal" / "2026-11-11.md").exists()


def test_compose_without_investing_or_kid(client, fake, vault):
    fake.compose_json = {"title": "读书", "learnings": {"items": ["复利需要时间"]},
                         "investing": None, "kid": None, "other": {"items": ["散步"]}}
    client.post("/journal/compose", json={"text": "读书", "date": "2026-11-12"})
    md = (vault / "Journal" / "2026-11-12.md").read_text("utf-8")
    assert "投资记录" not in md and "和孩子" not in md and "回看" not in md
    assert "## 其他" in md


def test_write_entry_exclusive(tmp_path):
    p = tmp_path / "a.md"
    assert write_entry(p, "new\n", "block\n") == "created"
    assert write_entry(p, "SHOULD NOT\n", "block\n") == "appended"
    assert p.read_text() == "new\nblock\n"


def test_add_months_clamps():
    assert add_months(date(2026, 1, 31), 1) == date(2026, 2, 28)
    assert add_months(date(2026, 11, 30), 3) == date(2027, 2, 28)
