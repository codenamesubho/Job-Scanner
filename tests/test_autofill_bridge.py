import json

from scanner import autofill_bridge, database


def test_context_path_tracks_rebound_db_path(isolated_db):
    assert autofill_bridge.context_path() == isolated_db.parent / "autofill_context.md"


def test_build_context_markdown_from_candidate_and_resume_extracted(isolated_db):
    candidate = {
        "name": "Ada Lovelace",
        "title": "Backend Engineer",
        "years_exp": 6,
        "email": "ada@example.com",
        "phone": "555-0100",
        "linkedin": "linkedin.com/in/ada",
        "summary": "Backend engineer focused on distributed systems.",
        "resume_extracted": json.dumps({
            "skills": ["Python", "Kubernetes"],
            "domains": ["fintech"],
            "education": ["BSc Computer Science"],
            "work_summary": ["Acme Corp — Backend Engineer (2020 - Present): built payments infra."],
            "location": "Remote",
            "github": "github.com/ada",
        }),
    }

    path = autofill_bridge.build_context_markdown(candidate)
    text = path.read_text()

    assert path == autofill_bridge.context_path()
    assert "# About Me" in text
    assert "- Name: Ada Lovelace" in text
    assert "- Current title: Backend Engineer" in text
    assert "- Location: Remote" in text
    assert "- Email: ada@example.com" in text
    assert "Backend engineer focused on distributed systems." in text
    assert "- Skills: Python, Kubernetes" in text
    assert "Acme Corp — Backend Engineer" in text
    # Never leaks salary/work-auth fields into the doc, even if a future
    # ResumeProfile extraction adds them.
    assert "salary" not in text.lower().split("<!--")[0]


def test_build_context_markdown_does_not_overwrite_without_force(isolated_db):
    autofill_bridge.build_context_markdown({"name": "Original"})
    path = autofill_bridge.context_path()
    path.write_text("hand-edited content\n")

    autofill_bridge.build_context_markdown({"name": "Should Not Appear"})
    assert path.read_text() == "hand-edited content\n"

    autofill_bridge.build_context_markdown({"name": "Regenerated"}, force=True)
    assert "Regenerated" in path.read_text()
    assert "hand-edited content" not in path.read_text()


def test_resolve_llm_env_derives_from_claude_config(monkeypatch):
    monkeypatch.delenv("AUTOFILL_LLM_API_KEY", raising=False)
    monkeypatch.setenv("CLAUDE_API_KEY", "proxy-key")
    monkeypatch.setenv("CLAUDE_MODEL", "claude-opus-5")

    overrides = autofill_bridge.resolve_llm_env()

    assert overrides["AUTOFILL_LLM_API_KEY"] == "proxy-key"
    assert overrides["AUTOFILL_LLM_PROVIDER"] == "litellm"
    assert overrides["AUTOFILL_LLM_MODEL"] == "anthropic/claude-opus-5"
    assert overrides["AUTOFILL_LLM_BASE_URL"] == "http://localhost:8317"


def test_resolve_llm_env_leaves_explicit_config_alone(monkeypatch):
    monkeypatch.setenv("AUTOFILL_LLM_API_KEY", "already-set")
    monkeypatch.setenv("CLAUDE_API_KEY", "proxy-key")
    monkeypatch.setenv("CLAUDE_MODEL", "claude-opus-5")

    assert autofill_bridge.resolve_llm_env() == {}


def test_resolve_llm_env_no_op_when_nothing_to_derive_from(monkeypatch):
    monkeypatch.delenv("AUTOFILL_LLM_API_KEY", raising=False)
    monkeypatch.delenv("CLAUDE_API_KEY", raising=False)
    monkeypatch.delenv("CLAUDE_MODEL", raising=False)

    assert autofill_bridge.resolve_llm_env() == {}


def test_summarize_fill_run_buckets_by_write_status():
    run = {
        "jobs": [{
            "fields": [
                {"label": "Email", "write_status": "written"},
                {"label": "Cover letter", "write_status": "escalated"},
                {"label": "Resume upload", "write_status": "failed"},
                {"label": "Unknown", "write_status": "something_else"},
            ]
        }]
    }
    summary = autofill_bridge._summarize_fill_run(run)
    assert summary == {
        "filled": ["Email"],
        "escalated": ["Cover letter"],
        "failed": ["Resume upload"],
    }


def test_summarize_fill_run_handles_none():
    assert autofill_bridge._summarize_fill_run(None) == {"filled": [], "escalated": [], "failed": []}
