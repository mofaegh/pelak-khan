# Third-Party Notices

Pelak-Khan uses third-party open-source libraries and machine-learning components. Those components remain subject to their own licenses and terms, independently of Pelak-Khan's AGPL-3.0 source-code license.

Dependency manifests in this repository include:

- `requirements-runtime.txt`
- `requirements-build.txt`
- `requirements-audit.txt`
- `pyproject.toml`

## Ultralytics / YOLO

The detector pipeline uses Ultralytics/YOLO technology. Users and redistributors are responsible for reviewing and complying with the license terms that apply to their use case, including any requirements that may apply to source distribution, models, services, or commercial deployment.

## Model weights and datasets

Trained model weights and training datasets are intentionally excluded from normal Git history. Dataset availability does not automatically grant redistribution rights for the dataset itself or for derived model artifacts. Before publishing or redistributing weights, verify the license and terms of every dataset, pretrained component, and dependency used in the training pipeline.

## No license substitution

This notice is informational and does not replace the license files or terms shipped by third-party projects. When redistributing a portable build, preserve any third-party notices and license materials required by the bundled dependencies.
