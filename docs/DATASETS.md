# Dataset Notes

Pelak-Khan development has evaluated or prepared data from multiple Iranian license-plate sources, including project-local copies or exports associated with datasets such as:

- Hezarai Persian Plate OCR
- IMLP
- IRANIS
- IR-LPR
- IR-LPR Corners
- SIVD

The repository does **not** redistribute these datasets.

## Detection dataset used in the project

The prepared v1 detection dataset contained 2,972 images and 3,075 labeled plate boxes, split into train/validation/test partitions. Structural QA included checks for malformed labels and exact cross-split duplication before training.

## OCR dataset used in the project

The OCR preparation pipeline used the Hezarai-based plate OCR corpus available in the project environment. The accepted OCR sample set contained 9,902 items after filtering, with train/validation/test splits prepared for CRNN/CTC training.

## Licensing and redistribution

Dataset names listed here are provenance notes, not a grant of redistribution rights. Before publishing data, derived annotations, trained weights, or a commercial product, verify the original license/terms for each source and confirm that the intended use and redistribution are permitted.

When adding a new dataset to the project documentation, record at minimum:

- canonical dataset name
- source/project URL
- version or retrieval date
- license or terms
- whether commercial use is allowed
- whether redistribution is allowed
- whether trained-weight redistribution is addressed
- preprocessing or filtering performed by Pelak-Khan

Do not commit raw datasets to this repository.
