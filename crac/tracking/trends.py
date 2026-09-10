"""Phase 3: turn per-day observations and per-image labels into a timeline.

Keep this simple and explainable. Rolling windows and threshold crossings, not
ARIMA -- you need to be able to justify every detected event to a reader.

NOAA's own alert scale is the natural rule set:
    DHW >= 4   -> significant bleaching likely  (Alert Level 1)
    DHW >= 8   -> mortality likely              (Alert Level 2)

Definitions to settle before coding, and to write down in the README:
  - an EVENT starts on the first day DHW crosses the threshold, and ends after
    N consecutive days below it (N is a parameter; pick one and justify it).
  - RECOVERY is a rise in the healthy fraction of classifications at a site,
    sustained across at least two surveys after an event ends.
"""


def detect_thermal_events(observations, threshold=4.0, min_gap_days=14):
    """Find bleaching episodes in one site's observation series.

    Returns rows shaped for the events table. TODO.
    """
    raise NotImplementedError


def health_trajectory(classifications, freq="Q"):
    """Aggregate image labels per site into a class-fraction time series.

    Resample to quarters (surveys are sparse and irregular, so daily is noise).
    Carry the per-bucket image count through -- a 100% healthy bucket built from
    two images must not be plotted like one built from two hundred.

    TODO.
    """
    raise NotImplementedError


def correlate(observations, trajectory):
    """The project's central question: does thermal stress at a site precede a
    drop in the healthy fraction there?

    Lag the environmental series against the trajectory and report the
    association with its sample size. With a handful of sites this is
    descriptive, not inferential -- say so in the write-up.

    TODO.
    """
    raise NotImplementedError
