"""Tests for the collection watchdog's draw-date logic."""

from datetime import date

from scripts.monitoring.collection_watchdog import most_recent_draw_date


class TestMostRecentDrawDate:
    def test_sunday_expects_saturday(self):
        assert most_recent_draw_date(date(2026, 7, 19)) == date(2026, 7, 18)

    def test_thursday_expects_wednesday(self):
        assert most_recent_draw_date(date(2026, 7, 16)) == date(2026, 7, 15)

    def test_saturday_expects_wednesday(self):
        # On draw day itself the previous draw is the expectation
        assert most_recent_draw_date(date(2026, 7, 18)) == date(2026, 7, 15)

    def test_wednesday_expects_previous_saturday(self):
        assert most_recent_draw_date(date(2026, 7, 15)) == date(2026, 7, 11)
