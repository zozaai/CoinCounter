# 🪙 CoinCounter

Count coins in phone photos using traditional computer vision, deep neural regression, and vision-capable LLMs. The goal is to compare accuracy, speed, and cost, then build a mobile app backed by AWS.

## Planned workflow

```text
+-------------------+       +----------------------------+
| Phone app         | photo | AWS backend                |
| Capture & display | ----> | Count coins using:         |
|                   | <---- | - Computer vision          |
| Estimated count   | count | - Deep neural regression   |
+-------------------+       | - Vision-capable LLM       |
                            +----------------------------+
```

**Status:** Planning stage; models, backend, and app are not yet implemented.

**Roadmap:** Collect labeled photos → compare counting methods → build AWS backend → develop mobile app.

**License:** [Apache 2.0](LICENSE).
