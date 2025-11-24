from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from neurons.validator.utils.common.interval import (
    AGGREGATION_INTERVAL_LENGTH_MINUTES,
    align_to_interval,
    get_interval_iso_datetime,
    get_interval_start_minutes,
    minutes_since_epoch,
    to_utc,
)


class TestInterval:
    def test_to_utc(self):
        assert to_utc(datetime(2024, 12, 27, 0, 1, 0, 0)) == datetime(
            2024, 12, 27, 0, 1, 0, 0, timezone.utc
        )
        assert to_utc(datetime(2024, 12, 27, 0, 1, 0, 0, timezone.utc)) == datetime(
            2024, 12, 27, 0, 1, 0, 0, timezone.utc
        )

        cet = timezone(timedelta(hours=1))
        dt_cet = datetime(2024, 12, 27, 0, 1, 0, 0, timezone.utc).astimezone(cet)
        assert dt_cet == datetime(2024, 12, 27, 1, 1, 0, 0, cet)
        assert to_utc(dt_cet) == datetime(2024, 12, 27, 0, 1, 0, 0, timezone.utc)

    def test_minutes_since_epoch(self):
        assert minutes_since_epoch(datetime(2024, 12, 27, 0, 1, 0, 0, timezone.utc)) == 519841

    def test_align_to_interval(self):
        # Slightly past an interval
        test_value_1 = 519841
        expected_1 = (
            test_value_1 // AGGREGATION_INTERVAL_LENGTH_MINUTES
        ) * AGGREGATION_INTERVAL_LENGTH_MINUTES
        assert align_to_interval(test_value_1) == expected_1

        # Near the end of an interval
        test_value_2 = 520079
        expected_2 = (
            test_value_2 // AGGREGATION_INTERVAL_LENGTH_MINUTES
        ) * AGGREGATION_INTERVAL_LENGTH_MINUTES
        assert align_to_interval(test_value_2) == expected_2

        # Exactly aligned
        exactly_aligned = AGGREGATION_INTERVAL_LENGTH_MINUTES * 100
        assert align_to_interval(exactly_aligned) == exactly_aligned

        # Same boundary
        interval_start = AGGREGATION_INTERVAL_LENGTH_MINUTES * 50
        within_interval = interval_start + 100
        assert align_to_interval(within_interval) == interval_start

    def test_get_interval_start_minutes(self):
        # Exact epoch start
        test_time = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        with patch("neurons.validator.utils.common.interval.datetime") as mock_datetime:
            mock_datetime.now.return_value = test_time
            result = get_interval_start_minutes()
            assert result == 0

        # Multiple days later
        test_time_2 = datetime(2025, 1, 3, 4, 0, 0, tzinfo=timezone.utc)
        with patch("neurons.validator.utils.common.interval.datetime") as mock_datetime:
            mock_datetime.now.return_value = test_time_2
            result = get_interval_start_minutes()
            mins_since = minutes_since_epoch(test_time_2)
            expected = (
                mins_since // AGGREGATION_INTERVAL_LENGTH_MINUTES
            ) * AGGREGATION_INTERVAL_LENGTH_MINUTES
            assert result == expected

        test_time_3 = datetime(2024, 1, 1, 4, 45, 30, tzinfo=timezone.utc)
        with patch("neurons.validator.utils.common.interval.datetime") as mock_datetime:
            mock_datetime.now.return_value = test_time_3
            result = get_interval_start_minutes()
            mins_since = minutes_since_epoch(test_time_3)
            expected = (
                mins_since // AGGREGATION_INTERVAL_LENGTH_MINUTES
            ) * AGGREGATION_INTERVAL_LENGTH_MINUTES
            assert result == expected

    def test_get_interval_iso_datetime(self):
        interval_start_minutes = 500000
        expected = "2024-12-13T05:20:00+00:00"

        result = get_interval_iso_datetime(interval_start_minutes)

        assert result == expected
