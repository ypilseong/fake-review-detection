# fake-review-detection

[![pypi-image]][pypi-url]
[![version-image]][release-url]
[![release-date-image]][release-url]
[![license-image]][license-url]
[![codecov][codecov-image]][codecov-url]
[![jupyter-book-image]][docs-url]

<!-- Links: -->
[codecov-image]: https://codecov.io/gh/ypilseong/fake-review-detection/branch/main/graph/badge.svg?token=[REPLACE_ME]
[codecov-url]: https://codecov.io/gh/ypilseong/fake-review-detection
[pypi-image]: https://img.shields.io/pypi/v/fake-review-detection
[license-image]: https://img.shields.io/github/license/ypilseong/fake-review-detection
[license-url]: https://github.com/ypilseong/fake-review-detection/blob/main/LICENSE
[version-image]: https://img.shields.io/github/v/release/ypilseong/fake-review-detection?sort=semver
[release-date-image]: https://img.shields.io/github/release-date/ypilseong/fake-review-detection
[release-url]: https://github.com/ypilseong/fake-review-detection/releases
[jupyter-book-image]: https://jupyterbook.org/en/stable/_images/badge.svg

[repo-url]: https://github.com/ypilseong/fake-review-detection
[pypi-url]: https://pypi.org/project/fake-review-detection
[docs-url]: https://ypilseong.github.io/fake-review-detection
[changelog]: https://github.com/ypilseong/fake-review-detection/blob/main/CHANGELOG.md
[contributing guidelines]: https://github.com/ypilseong/fake-review-detection/blob/main/CONTRIBUTING.md
<!-- Links: -->

- Documentation: [https://ypilseong.github.io/fake-review-detection][docs-url]
- GitHub: [https://github.com/ypilseong/fake-review-detection][repo-url]
- PyPI: [https://pypi.org/project/fake-review-detection][pypi-url]

## Overview

The `fake-review-detection` project is focused on enhancing the detection of promotional reviews in online platforms, specifically targeting Naver restaurant reviews. By integrating large language models (LLMs) with location-based and image analysis data, the project aims to improve the accuracy of identifying promotional content that might not be detectable through simple text analysis alone.

### Key Features

- **Integration of Metadata:** The project utilizes various metadata fields such as the number of likes, hashtag counts, text length, and image counts to enrich the analysis beyond simple text content.

- **Textual Feature Extraction:** Using advanced embedding models like `dpr-longformer-4096` and `text-embedding-3-large`, the project extracts meaningful textual features that are then used to identify promotional content more accurately.

- **Image Analysis:** The project also incorporates image analysis features, such as the number and content of images in the reviews. By analyzing these images, the model can better assess whether the review contains promotional content, leveraging visual cues that complement textual analysis.

- **Sparse Autoencoders:** To prevent overfitting and extract key features efficiently, sparse autoencoders are employed. These features are critical in distinguishing between genuine and promotional reviews.

- **Location-Based Data Analysis:** The model incorporates location-based data, like the number of nearby stores and rental car visits, to add contextual information that enhances the detection accuracy. However, the quality and consistency of location data have shown to impact the model's performance.

### Research Outcomes

- **Improved Accuracy:** The combination of metadata, textual features, image analysis, and location data leads to a more accurate identification of promotional reviews compared to existing models.
  
- **Sparse Autoencoders' Effectiveness:** The use of sparse autoencoders significantly enhances the model's ability to learn and extract distinguishing features, leading to improved performance without overfitting.

- **Challenges with Location Data:** The inclusion of location data showed limitations due to poor data quality, impacting the model's overall effectiveness when relying heavily on this information.

### Model Performance

The models were evaluated using various metrics, with the following results:

- **SVM Model:** Different configurations, including basic metadata, location data, textual features, and image analysis features, were tested. The inclusion of textual features from `text-embedding-3-large` and image analysis yielded the highest accuracy and F1 Score.

## Project Documentation

For detailed information, please refer to the research poster:
[![PilseongYang_research_poster_update](https://github.com/user-attachments/assets/ed0d334c-06d6-4791-b499-6b57a9b397c4)](https://github.com/user-attachments/assets/ed0d334c-06d6-4791-b499-6b57a9b397c4)

## Changelog

See the [CHANGELOG] for more information.

## Contributing

Contributions are welcome! Please see the [contributing guidelines] for more information.

## License

This project is released under the [MIT License][license-url].
