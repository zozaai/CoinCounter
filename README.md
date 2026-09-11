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

## Dataset

117 phone photos of coins on a flat surface, each labeled with the number of coins it contains.

```text
dataset/
  images/        117 JPEGs, 480x480, converted from the original HEIC captures
  labels.json    { "IMG_4315.jpg": 7, ... }  coin count per image
  train/         82 images + labels.json
  val/           19 images + labels.json
  test/          16 images + labels.json
```

| Split | Images | Coin counts |
|-------|--------|-------------|
| train | 82     | 1-12        |
| val   | 19     | 1-12        |
| test  | 16     | 1-12        |
| **total** | **117** | 764 coins |

The 70/15/15 split uses seed 42 and is stratified by coin count, so every count from 1 to 12
appears in all three splits. Each split folder carries its own `labels.json` with the same
`{filename: count}` shape, so a split can be loaded without reading the global label file.

**Labels** are integers: the number of coins visible in the image. Nothing else is annotated -
no coin type, denomination, or position.

**Images** were resized to an exact 480x480 (a stretch, not a center crop), so the original
aspect ratio is not preserved.

## Archive

`archive/tools/` holds the one-off scripts used to build the dataset. They are kept for
reproducibility and are not part of the app:

- `convert_images.sh` - HEIC to 480x480 JPEG via macOS `sips`
- `label_tool.py` - browser labeler; shows one image at a time, autosaves to `labels.json`
- `split_dataset.py` - regenerates the train/val/test split (`--ratios`, `--seed`, `--move`)

**Status:** Planning stage; models, backend, and app are not yet implemented.

**Roadmap:** ~~Collect labeled photos~~ (done, see Dataset) → compare counting methods → build AWS backend → develop mobile app.

**License:** [Apache 2.0](LICENSE).
