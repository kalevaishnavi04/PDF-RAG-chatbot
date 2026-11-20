"""Run with:  python -m pytest -q   (or just: python tests/test_utils.py)"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import utils  # noqa: E402


def test_greeting_parts_of_day():
    assert utils.greeting("Asha", 8) == "Good morning, Asha"
    assert utils.greeting("", 14) == "Good afternoon"
    assert utils.greeting("  ", 19) == "Good evening"
    assert utils.greeting("Asha", 2) == "Working late, Asha"


def test_make_title():
    assert utils.make_title("") == "New chat"
    assert utils.make_title("  hello   world ") == "hello world"
    long = "x" * 100
    assert len(utils.make_title(long, 20)) == 20 and utils.make_title(long, 20).endswith("…")


def test_extract_text_handles_gemini_blocks():
    assert utils.extract_text("hi") == "hi"
    assert utils.extract_text([{"type": "text", "text": "a"}, {"type": "thinking", "text": "x"}, "b"]) == "ab"
    assert utils.extract_text(None) == ""


def test_citations():
    text = "Water boils at 100 C [1]. See also [2][3] and [link](http://x)."
    assert utils.cited_numbers(text) == [1, 2, 3]
    out = utils.link_citations(text)
    assert ":yellow-badge[1]" in out and "[link](http://x)" in out
    fenced = "```\nx[1]\n```"
    assert utils.link_citations(fenced) == fenced


def test_sample_evenly_keeps_order_and_ends():
    items = list(range(100))
    picked = utils.sample_evenly(items, 5)
    assert picked[0] == 0 and picked[-1] == 99 and picked == sorted(picked) and len(picked) == 5
    assert utils.sample_evenly([1, 2], 5) == [1, 2]
    assert utils.sample_evenly([], 3) == []


def test_build_context_numbers_sources():
    ctx, sources = utils.build_context(
        [{"text": "alpha", "file": "a.pdf", "page": 2, "doc_id": "d1"}, {"text": "beta", "file": "b.pdf", "page": 5}]
    )
    assert "[1] (a.pdf, page 2)" in ctx and "[2] (b.pdf, page 5)" in ctx
    assert [s["n"] for s in sources] == [1, 2]


def test_prompt_modes():
    strict = utils.build_answer_prompt("Q?", "[1] x", "", "Concise", True)
    loose = utils.build_answer_prompt("Q?", "[1] x", "", "Detailed", False)
    assert "I could not find the answer in the documents." in strict
    assert "general knowledge" in loose and "Q?" in loose
    assert "untrusted" in strict


def test_friendly_errors():
    assert "quota" in utils.friendly_error(RuntimeError("429 RESOURCE_EXHAUSTED")).lower()
    assert "api key" in utils.friendly_error(RuntimeError("API_KEY_INVALID")).lower()
    assert utils.is_retryable(RuntimeError("503 UNAVAILABLE"))
    assert not utils.is_model_fallback_error(ValueError("bad input"))


def test_search_chats_matches_title_and_messages():
    chats = [
        {"title": "Physics notes", "messages": [{"content": "gravity"}]},
        {"title": "Cooking", "messages": [{"content": "pasta recipe"}]},
    ]
    assert [c["title"] for c in utils.search_chats(chats, "grav")] == ["Physics notes"]
    assert [c["title"] for c in utils.search_chats(chats, "COOK")] == ["Cooking"]
    assert len(utils.search_chats(chats, "")) == 2


def test_history_and_export():
    chat = {
        "title": "T",
        "messages": [
            {"role": "user", "content": "q", "kind": "text"},
            {"role": "assistant", "content": "a [1]", "kind": "text", "sources": [{"n": 1, "file": "f.pdf", "page": 3}]},
            {"role": "assistant", "content": "oops", "kind": "error"},
        ],
    }
    assert utils.history_text(chat["messages"]) == "user: q\nassistant: a [1]"
    md = utils.chat_to_markdown(chat, ["f.pdf"])
    assert "# T" in md and "f.pdf, page 3" in md


def test_artifact_title():
    assert utils.artifact_title("## **Summary**\nbody") == "Summary"
    assert utils.artifact_title("") == "New chat"


def test_model_tier_uses_whole_words():
    assert utils.model_tier("gemini-3.8-flash") == "Balanced"
    assert utils.model_tier("gemini-3.5-flash-lite") == "Fast"
    assert utils.model_tier("gemini-3.1-pro") == "Powerful"
    assert utils.model_tier("something-else") == "Standard"


def test_retry_hint_in_quota_message():
    assert utils.retry_seconds(RuntimeError("Please retry in 34.5s.")) == 35
    assert utils.retry_seconds(RuntimeError("nothing here")) is None
    assert utils.format_wait(45) == "45 s" and utils.format_wait(125) == "2 min" and utils.format_wait(4500) == "1 h 15 min"
    assert "Try again in about 1 min" in utils.friendly_error(RuntimeError("429 RESOURCE_EXHAUSTED retry in 59s"))


def test_language_rule():
    assert "language the user writes in" in utils.language_rule(["Auto"])
    assert utils.language_rule(["Hindi"]) == "Always reply in Hindi."
    assert "Hindi, English" in utils.language_rule(["Hindi", "English"])


def test_prompt_extras_and_web():
    p = utils.build_answer_prompt("Q?", "[1] x", "", "Balanced", True, extra="Be kind.")
    assert "Be kind." in p and "Additional instructions" in p
    assert "Additional instructions" not in utils.build_answer_prompt("Q?", "", "", "Balanced", True)
    assert "Google Search" in utils.build_answer_prompt("Q?", "", "", "Balanced", True, web=True)


def test_share_exports_are_safe():
    chat = {"title": "<b>t</b>", "messages": [
        {"role": "assistant", "content": "**hi** <script>x</script>", "kind": "text",
         "sources": [{"n": 1, "file": "a.pdf", "page": 2}], "web": [{"title": "T", "uri": "https://x.example"}]}]}
    page = utils.chat_to_html(chat, ["a.pdf"])
    assert "<script>" not in page and "<b>hi</b>" in page and "a.pdf, page 2" in page
    assert "Web sources:" in utils.chat_to_markdown(chat, [])
    import io, zipfile
    z = zipfile.ZipFile(io.BytesIO(utils.chats_zip([chat, chat], lambda c: [])))
    assert len(z.namelist()) == 2 and z.namelist()[0].endswith(".md")


def test_email_and_gstin_validation():
    assert utils.is_email("asha@example.com") and utils.is_email(" a.b@c.in ")
    assert not utils.is_email("asha@") and not utils.is_email("no spaces@x.com") and not utils.is_email("")
    assert utils.is_gstin("22AAAAA0000A1Z5") and utils.is_gstin("22aaaaa0000a1z5")
    assert not utils.is_gstin("BAD") and not utils.is_gstin("22AAAAA0000A1Z") and not utils.is_gstin("")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"{len(tests)} tests passed")
