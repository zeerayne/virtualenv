from virtualenv.run import run_via_cli


def test_base_bootstrap_link_via_app_data(tmp_path):
    result = run_via_cli([str(tmp_path), "--no-venv"])
    assert result
