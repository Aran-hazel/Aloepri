from src.stage_k_llama_release import default_stage_k_llama_profiles


def test_stage_k_llama_profiles_match_paper_consistent_mainline() -> None:
    profiles = default_stage_k_llama_profiles()
    names = [item.name for item in profiles]
    assert names == ["default", "reference"]
    assert {item.source_dir for item in profiles} == {"artifacts/stage_j_llama_instruct_paper_consistent"}
    assert profiles[0].correctness_evidence_file == "outputs/stage_k_llama_release/correctness/default.json"
    assert profiles[1].correctness_evidence_file == "outputs/stage_k_llama_release/correctness/reference.json"
