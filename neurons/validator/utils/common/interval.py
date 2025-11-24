from datetime import datetime, timedelta, timezone

# The base time epoch for clustering intervals.
SCORING_REFERENCE_DATE = datetime(2024, 1, 1, 0, 0, 0, 0, tzinfo=timezone.utc)

# Intervals are grouped in 24-hour blocks (1440 minutes).
AGGREGATION_INTERVAL_LENGTH_MINUTES = 60 * 24

# Number of intervals (days) used for scoring window and prediction window
# Miners only predict on events cutoffing in exactly this many days
# Scoring only considers the last N intervals before cutoff
SCORING_WINDOW_INTERVALS = 2

BLOCK_DURATION = 12  # 12 seconds block duration from bittensor


def to_utc(input_dt: datetime) -> datetime:
    return (
        input_dt.astimezone(timezone.utc)
        if input_dt.tzinfo
        else input_dt.replace(tzinfo=timezone.utc)
    )


def minutes_since_epoch(dt: datetime) -> int:
    """Convert a given datetime to the 'minutes since the reference date'."""
    return int((dt - SCORING_REFERENCE_DATE).total_seconds()) // 60


def align_to_interval(minutes_since: int) -> int:
    """
    Align a given number of minutes_since_epoch down to
    the nearest AGGREGATION_INTERVAL_LENGTH_MINUTES boundary.
    """
    return minutes_since - (minutes_since % AGGREGATION_INTERVAL_LENGTH_MINUTES)


def get_interval_start_minutes():
    now = datetime.now(timezone.utc)

    mins_since_epoch = minutes_since_epoch(now)
    interval_start_minutes = align_to_interval(mins_since_epoch)

    return interval_start_minutes


def get_interval_iso_datetime(interval_start_minutes: int):
    return (SCORING_REFERENCE_DATE + timedelta(minutes=interval_start_minutes)).isoformat()
