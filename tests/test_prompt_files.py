from scanner.llm._prompt_loader import load_prompt_file

_EXPECTED_KEYS = {
    "raw_scoring": {"score_rubric", "batch_score_system", "batch_score_user", "job_block"},
    "structured_scoring": {
        "structured_score_rubric",
        "structured_batch_score_system",
        "structured_batch_score_user",
    },
    "referral": {"referral_message", "referral_retry_nudge", "form_field_match"},
    "extraction": {"summary", "jd_extract", "resume_extract"},
}


def test_all_prompt_files_load_with_expected_keys():
    for module_name, expected_keys in _EXPECTED_KEYS.items():
        prompts = load_prompt_file(module_name)
        assert set(prompts.keys()) == expected_keys, module_name
        for key, entry in prompts.items():
            assert entry.get("version"), f"{module_name}.{key} missing version"
            assert entry.get("template"), f"{module_name}.{key} missing template"
            assert isinstance(entry["template"], str)
