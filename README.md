# 🪙 CoinCounter

Count coins in photos captured with a phone. CoinCounter will explore and compare traditional computer vision, deep neural regression models, and vision-capable large language models (LLMs) to estimate the number of coins in an image.

The goal is a mobile app that lets you capture a photo, send it to AWS for processing, and receive the estimated coin count.

## Counting approaches

| Approach | How it estimates the count |
| --- | --- |
| Traditional computer vision | Image processing techniques such as thresholding, contour detection, and circle detection identify individual coins. |
| Deep neural regression | A neural network learns to predict the coin count directly from an image. |
| Vision-capable LLMs | A multimodal model examines the photo and estimates how many coins are visible. |

The project will compare these approaches for accuracy, processing time, and cost across different lighting conditions, backgrounds, and coin arrangements, including touching or overlapping coins.

## Planned app workflow

1. Capture a photo of coins with your phone.
2. Upload the image to an AWS-hosted backend.
3. Process the image using a coin-counting model.
4. Return the estimated number of coins to the app.

### Planned architecture

```text
+--------------------------+
| Phone app                |
| Capture a photo of coins |
+------------+-------------+
             |
             | Upload image (HTTPS)
             v
+------------------------------------------------------+
| AWS backend                                          |
|                                                      |
|  +-----------------------------------------------+   |
|  | Receive image and prepare it for counting      |   |
|  +-----------------------+-----------------------+   |
|                          |                           |
|                          v                           |
|  +-----------------------------------------------+   |
|  | Run the selected counting approach            |   |
|  |                                               |   |
|  |  - Traditional computer vision                |   |
|  |  - Deep neural regression                     |   |
|  |  - Vision-capable LLM                         |   |
|  +-----------------------+-----------------------+   |
|                          |                           |
|                          v                           |
|  +-----------------------------------------------+   |
|  | Return estimated coin count                   |   |
|  +-----------------------+-----------------------+   |
+--------------------------+---------------------------+
                           |
                           | Response (HTTPS)
                           v
              +--------------------------+
              | Phone app                |
              | Display number of coins  |
              +--------------------------+
```

## Project status

This repository is at the planning stage. The counting pipelines, trained models, AWS backend, and mobile app are not yet implemented.

## Roadmap

- [ ] Collect phone photos and label each image with its coin count.
- [ ] Build a traditional computer vision baseline.
- [ ] Train and evaluate a deep neural regression model.
- [ ] Evaluate vision-capable LLMs on the same images.
- [ ] Compare accuracy, processing time, and cost.
- [ ] Build an AWS backend for image processing and count prediction.
- [ ] Develop a mobile app for photo capture and displaying results.

## License

Licensed under the [Apache License 2.0](LICENSE).
