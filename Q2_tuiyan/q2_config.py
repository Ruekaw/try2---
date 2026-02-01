from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Q2Config:
    root_dir: Path
    clean_csv: Path
    npz_dir: Path
    output_dir: Path
    tracker_names: tuple[str, ...]
    use_season_segments: bool
    png_dpi: int


def default_config() -> Q2Config:
    root = Path(__file__).resolve().parents[1]
    return Q2Config(
        root_dir=root,
        clean_csv=root / "outputs" / "dwts_long_clean.csv",
        npz_dir=root / "Q1_data_expo",
        output_dir=root / "outputs" / "q2_counterfactual",
        tracker_names=("Jerry Rice", "Bobby Bones"),
        use_season_segments=True,
        png_dpi=160,
    )


def season_segment_combine_method(season: int) -> str:
    if season <= 2:
        return "rank"
    if 3 <= season <= 27:
        return "percent"
    return "rank"