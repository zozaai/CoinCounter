"""Reuse one initialized counter across decoded RGB frames."""
from coincounter import CoinCounter


def count_frames(frames, parameters):
    with CoinCounter("random_guess", parameters) as counter:
        for frame in frames:
            yield counter.run(frame)
