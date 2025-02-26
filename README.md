# Eye Fundus Image Classification

This project is designed to classify eye fundus images, based on the environment and configurations provided by [DDA](https://github.com/shiyegao/DDA). Please refer to the DDA repository for all necessary setup details.

---

## Table of Contents
- [Environment Setup](#environment-setup)
- [Data Preparation](#data-preparation)
- [Generate Adapted Images](#generate-adapted-images)
- [Test-Time Adaptation (TTA)](#test-time-adaptation-tta)
- [TTA Parameter Modification](#tta-parameter-modification)
- [Acknowledgments](#acknowledgments)

---

## Environment Setup
Use the same environment configuration as described in the [DDA repository](https://github.com/shiyegao/DDA). Ensure all dependencies and prerequisites are installed correctly before proceeding.

---

## Data Preparation
Prepare your dataset according to your specific requirements. For more details on the recommended structure or format, refer to the [data preparation guidelines in DDA](https://github.com/shiyegao/DDA).

---

## Generate Adapted Images
1. Update the `optic_adapt.sh` script with the following parameters:
   - Original dataset path
   - Adapted dataset output path
   - `source_dataset`
   - `model_path`
   - `target_dataset` (edited in the associated Python script)

2. **Important:** Currently, it only supports single-GPU execution. Set `batchsize` to `1`. Using multiple GPUs may lead to missing samples.

3. Run the following command:
   ```bash
   bash optic_adapt.sh
