import argparse
import json

from coincounter import CoinCounter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["count"])
    parser.add_argument("image")
    parser.add_argument("--engine", required=True)
    parser.add_argument("--parameters", required=True, help="JSON file containing engine parameters")
    args = parser.parse_args()
    try:
        with open(args.parameters) as source:
            parameters = json.load(source)
        with CoinCounter(args.engine, parameters) as counter:
            print(counter.run(args.image).count)
    except Exception as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
