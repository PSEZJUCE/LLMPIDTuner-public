from llmpidtuner.training.config import ServerJobConfig
from llmpidtuner.training.server import render_sbatch


def test_render_sbatch_is_uv_first_without_conda_defaults() -> None:
    script = render_sbatch(
        ServerJobConfig(
            job_name="pid-test",
            command="uv run --no-sync llmpidtuner --help",
            output_dir="outputs/slurm/test",
            workdir="/opt/llmpidtuner",
        )
    )

    assert "conda activate" not in script
    assert "conda activate" not in script
    assert 'cd "/opt/llmpidtuner"' in script
    assert "uv run --no-sync llmpidtuner --help" in script


def test_render_sbatch_includes_optional_setup_commands() -> None:
    script = render_sbatch(
        ServerJobConfig(
            job_name="pid-test",
            command="uv run --no-sync pytest",
            output_dir="outputs/slurm/test",
            setup_commands=["module load cuda/12.4", "export CUDA_VISIBLE_DEVICES=0"],
        )
    )

    assert "module load cuda/12.4" in script
    assert "export CUDA_VISIBLE_DEVICES=0" in script
