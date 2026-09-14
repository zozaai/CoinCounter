"""Run after installing the package: python examples/count_image.py IMAGE."""
import sys
from coincounter import CoinCounter

with CoinCounter("random_guess", {"counts": [1, 2], "weights": [1, 1], "seed": 42}) as counter:
    print(counter.run(sys.argv[1]).count)
