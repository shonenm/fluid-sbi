"""Tests for sda.console module."""

from __future__ import annotations

from sda.console import (
    console,
    create_progress,
    print_config_table,
    print_error,
    print_experiment_results,
    print_info,
    print_metrics_table,
    print_success,
    print_warning,
    track,
)


class TestConsole:
    def test_console_exists(self):
        assert console is not None

    def test_console_print(self, capsys):
        console.print("test message")
        captured = capsys.readouterr()
        assert "test message" in captured.out


class TestTrack:
    def test_track_list(self):
        result = list(track(range(5), description="Test", transient=True))
        assert result == [0, 1, 2, 3, 4]

    def test_track_with_total(self):
        items = iter(range(3))
        result = list(track(items, total=3, transient=True))
        assert result == [0, 1, 2]


class TestCreateProgress:
    def test_create_progress(self):
        progress = create_progress("Test", transient=True)
        assert progress is not None
        with progress:
            task = progress.add_task("Test task", total=10)
            for _ in range(10):
                progress.update(task, advance=1)


class TestPrintFunctions:
    def test_print_metrics_table(self, capsys):
        metrics = {"loss": 0.05, "accuracy": 0.95}
        print_metrics_table(metrics, title="Test")
        captured = capsys.readouterr()
        assert "loss" in captured.out
        assert "0.0500" in captured.out

    def test_print_config_table(self, capsys):
        config = {"epochs": 100, "nested": {"value": 42}}
        print_config_table(config, title="Test")
        captured = capsys.readouterr()
        assert "epochs" in captured.out
        assert "100" in captured.out

    def test_print_experiment_results(self, capsys):
        results = [
            {"loss": 0.1, "acc": 0.9},
            {"loss": 0.05, "acc": 0.95},
        ]
        print_experiment_results(
            results,
            metric_names=["loss", "acc"],
            row_names=["baseline", "improved"],
        )
        captured = capsys.readouterr()
        assert "baseline" in captured.out
        assert "improved" in captured.out

    def test_print_success(self, capsys):
        print_success("Success message")
        captured = capsys.readouterr()
        assert "Success message" in captured.out

    def test_print_error(self, capsys):
        print_error("Error message")
        captured = capsys.readouterr()
        assert "Error message" in captured.out

    def test_print_warning(self, capsys):
        print_warning("Warning message")
        captured = capsys.readouterr()
        assert "Warning message" in captured.out

    def test_print_info(self, capsys):
        print_info("Info message")
        captured = capsys.readouterr()
        assert "Info message" in captured.out
