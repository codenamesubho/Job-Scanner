"""Tests for save_jobs()'s write-time deduplication.

This is the most consequential rule in the app: a wrongly-merged job silently
disappears from the user's list. It had no direct coverage before — only the
migration tests touched it incidentally.
"""
import pandas as pd

from scanner import database

LONG_DESC = "A real job description. " * 20   # comfortably over the 200-char hash floor


def _row(job_id, title="Backend Engineer", company="Acme", description=LONG_DESC, **extra):
    return {"id": job_id, "site": "test", "title": title, "company": company,
            "description": description, **extra}


def _save(*rows, **kw):
    return database.save_jobs(pd.DataFrame(list(rows)), **kw)


# ------------------------------------------------------------- basic insertion

def test_new_jobs_are_inserted(isolated_db):
    assert _save(_row("a"), _row("b", title="Other", description="x" * 300)) == 2
    assert len(database.get_jobs()) == 2


def test_same_id_upserts_rather_than_duplicating(isolated_db):
    _save(_row("a"))
    assert _save(_row("a")) == 0
    assert len(database.get_jobs()) == 1


# ------------------------------------------------- signal 1: content_hash

def test_identical_description_under_a_different_id_is_a_resighting(isolated_db):
    """The same posting mirrored on another board, retitled — the (title, company)
    key misses this, which is why the hash is checked first."""
    _save(_row("linkedin-1", title="Backend Engineer", company="Acme"))
    inserted = _save(_row("greenhouse-1", title="Backend Engineer (Platform)", company="ACME Inc."))

    assert inserted == 0
    assert len(database.get_jobs()) == 1


def test_short_descriptions_do_not_collide_across_companies(isolated_db):
    """Below the 200-char floor the hash is not a reliable identity signal, so two
    unrelated stubs must not merge."""
    stub = "Apply on our careers page."
    _save(_row("a", title="Engineer", company="Acme", description=stub))
    inserted = _save(_row("b", title="Chef", company="Bistro", description=stub))

    assert inserted == 1
    assert len(database.get_jobs()) == 2


# --------------------------------------------- signal 2: (title, company)

def test_same_title_and_company_is_a_resighting(isolated_db):
    _save(_row("src1-1", description="one description " * 20))
    inserted = _save(_row("src2-1", description="a totally different description " * 20))

    assert inserted == 0
    assert len(database.get_jobs()) == 1


def test_title_company_matching_ignores_case_and_padding(isolated_db):
    _save(_row("a", title="Backend Engineer", company="Acme"))
    inserted = _save(_row("b", title="  backend engineer ", company="ACME",
                           description="different " * 40))

    assert inserted == 0


def test_genuinely_different_jobs_are_both_kept(isolated_db):
    _save(_row("a", title="Backend Engineer", company="Acme"))
    inserted = _save(_row("b", title="Data Scientist", company="Globex",
                           description="unrelated " * 40))

    assert inserted == 1
    assert len(database.get_jobs()) == 2


# ------------------------------------------------------- within one batch

def test_duplicates_inside_a_single_batch_are_collapsed(isolated_db):
    """Scanning several sources at once goes through one save_jobs() call."""
    inserted = _save(
        _row("linkedin-1", title="Backend Engineer", company="Acme"),
        _row("greenhouse-1", title="Backend Engineer", company="Acme",
             description="different text " * 30),
    )

    assert inserted == 1
    assert len(database.get_jobs()) == 1


def test_exact_duplicate_ids_in_one_batch_are_collapsed(isolated_db):
    assert _save(_row("a"), _row("a")) == 1


# ------------------------------------------------- what a re-sighting preserves

def test_a_resighting_never_resets_user_status(isolated_db):
    _save(_row("a"))
    job_id = database.get_jobs().iloc[0]["id"]
    database.update_status(job_id, "applied")

    _save(_row("b", title="Backend Engineer (Platform)", company="Acme"))

    assert database.get_jobs().iloc[0]["status"] == "applied"


def test_a_resighting_backfills_a_missing_direct_url(isolated_db):
    _save(_row("a", job_url_direct=""))
    _save(_row("b", title="Backend Engineer (Platform)", company="Acme",
                job_url_direct="https://careers.acme.com/1"))

    assert database.get_jobs().iloc[0]["job_url_direct"] == "https://careers.acme.com/1"


def test_a_resighting_does_not_overwrite_an_existing_direct_url(isolated_db):
    _save(_row("a", job_url_direct="https://original.example/1"))
    _save(_row("b", title="Backend Engineer (Platform)", company="Acme",
                job_url_direct="https://mirror.example/2"))

    assert database.get_jobs().iloc[0]["job_url_direct"] == "https://original.example/1"


def test_a_blank_field_on_rescrape_does_not_clobber_a_stored_value(isolated_db):
    """A transiently-failed description fetch must not wipe the stored one."""
    _save(_row("a"))
    _save(_row("a", description=""))

    assert database.get_jobs().iloc[0]["description"] == LONG_DESC


def test_default_status_applies_only_to_new_rows(isolated_db):
    _save(_row("a"))                                  # inserted as "new"
    _save(_row("a"), default_status="shortlisted")    # same id -> status untouched

    assert database.get_jobs().iloc[0]["status"] == "new"


def test_manual_adds_can_use_a_different_default_status(isolated_db):
    _save(_row("a"), default_status="shortlisted")

    assert database.get_jobs().iloc[0]["status"] == "shortlisted"


# ----------------------------------------------------------------- edge cases

def test_empty_frame_is_a_noop(isolated_db):
    assert database.save_jobs(pd.DataFrame()) == 0


def test_frame_without_an_id_column_is_a_noop(isolated_db):
    assert database.save_jobs(pd.DataFrame([{"title": "x"}])) == 0


def test_remote_flag_is_corrected_when_location_names_a_real_place(isolated_db):
    """Sources keyword-match "remote" in the description and over-report it."""
    _save(_row("a", is_remote=True, location="Bengaluru, India"))

    assert int(database.get_jobs().iloc[0]["is_remote"]) == 0


def test_remote_flag_is_kept_when_the_location_says_remote(isolated_db):
    _save(_row("a", is_remote=True, location="Remote"))

    assert int(database.get_jobs().iloc[0]["is_remote"]) == 1
