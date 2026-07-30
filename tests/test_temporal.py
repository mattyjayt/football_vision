"""Tests for exploration.temporal — Phase 1 observability core.

Models are mocked (no video, no inference); the stats math, summarizer,
CSV roundtrip, and report table are tested for real.
"""

from __future__ import annotations

import numpy as np
import pytest

from exploration.temporal import (
    FrameStats, TemporalReport, report_table_markdown, summarize, write_csv,
)
from exploration.temporal_plots import load_csv


def _stats_row(frame_id=0, time_s=0.0, n_players=20, ball=1, conf=0.8,
               kpts=12, hom_ok=1, rms=0.8):
    return FrameStats(
        frame_id=frame_id, time_s=time_s,
        n_players=n_players, n_goalkeepers=2, n_referees=1,
        n_total_detections=n_players + 3, ball_detected=ball,
        conf_mean=conf, conf_median=conf, conf_p10=conf - 0.2,
        conf_p90=conf + 0.1, conf_min=conf - 0.4, conf_max=conf + 0.15,
        bbox_area_mean=1500.0, bbox_area_p90=4000.0,
        n_keypoints=kpts, homography_ok=hom_ok, homography_rms_m=rms,
    )


class TestSummarize:
    def test_aggregates(self):
        stats = [_stats_row(frame_id=i, time_s=i / 25, n_players=18 + i,
                            ball=i % 2, conf=0.7 + 0.01 * i, kpts=10 + i,
                            hom_ok=1 if i < 3 else 0, rms=0.5 + 0.1 * i)
                 for i in range(4)]
        r = summarize("vid.mp4", stats, fps=25.0)
        assert r.frames_sampled == 4
        assert r.avg_players == pytest.approx(np.mean([18, 19, 20, 21]))
        assert r.ball_visibility_pct == pytest.approx(50.0)
        assert r.homography_success_pct == pytest.approx(75.0)
        # median RMS over successful frames only (first 3)
        assert r.median_homography_rms_m == pytest.approx(np.median([0.5, 0.6, 0.7]))

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            summarize("vid.mp4", [], fps=25.0)

    def test_no_successful_homography(self):
        stats = [_stats_row(hom_ok=0) for _ in range(3)]
        r = summarize("vid.mp4", stats, fps=25.0)
        assert r.homography_success_pct == 0.0
        assert r.median_homography_rms_m == 0.0


class TestCsvRoundtrip:
    def test_write_then_load(self, tmp_path):
        stats = [_stats_row(frame_id=i, time_s=i / 25) for i in range(5)]
        path = write_csv(stats, tmp_path / "stats.csv")
        rows = load_csv(path)
        assert len(rows) == 5
        assert rows[0]["frame_id"] == 0
        assert rows[3]["n_players"] == 20
        assert set(rows[0].keys()) == set(FrameStats.fieldnames())


class TestReportTable:
    def test_markdown_rows(self):
        stats = [_stats_row()]
        r = summarize("alpha.mp4", stats, fps=25.0)
        md = report_table_markdown([r])
        assert "| alpha.mp4 |" in md
        assert "avg players" in md.splitlines()[0]

    def test_two_videos_comparable(self):
        good = summarize("good.mp4", [_stats_row(n_players=24, conf=0.9)], 25.0)
        bad = summarize("bad.mp4", [_stats_row(n_players=8, conf=0.4)], 25.0)
        md = report_table_markdown([good, bad])
        assert md.count("\n") == 3  # header, separator... 2 data rows
        assert "good.mp4" in md and "bad.mp4" in md
