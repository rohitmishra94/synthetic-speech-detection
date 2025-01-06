# Cloned Voice Detection System

## Project Overview
This project implements a machine learning system for detecting synthetic/cloned voices from natural human speech. It uses a combination of acoustic features and deep speaker embeddings to achieve highly accurate detection of artificially generated speech.

## Key Features
- Dual feature extraction approach combining:
  - OpenSMILE acoustic features (eGeMAPSv02 feature set)
  - NVIDIA TitanNet speaker embeddings
- Multiple machine learning models evaluated:
  - Logistic Regression
  - Random Forest
  - Gradient Boosting
- Optimized feature selection combining both embedding and acoustic features
- High accuracy in synthetic speech detection

## Technical Implementation

### Feature Extraction
1. **OpenSMILE Features**
   - Uses eGeMAPSv02 feature set for acoustic analysis
   - Extracts functionals of various acoustic parameters
   - Provides interpretable voice characteristics

2. **TitanNet Embeddings**
   - Leverages NVIDIA's pretrained speaker verification model
   - Extracts high-dimensional speaker embeddings
   - Captures speaker identity characteristics

### Model Performance

Different feature combinations were evaluated:

1. **OpenSMILE Features Only**
   - Good baseline performance
   - More interpretable features
   - Slightly lower accuracy compared to combined approach

2. **TitanNet Embeddings Only**
   - Strong performance
   - Less interpretable features
   - Captures subtle voice characteristics

3. **Combined Features (Final Model)**
   - Best overall performance
   - Uses selected top OpenSMILE features with TitanNet embeddings
   - Gradient Boosting achieved the highest accuracy

## Results

The final model using Gradient Boosting with combined features achieved:
- Accuracy: >95%
- Balanced performance across both synthetic and natural speech detection
- Robust performance across different speakers and conditions

## Dependencies
- NeMo Toolkit (NVIDIA)
- OpenSMILE
- PyTorch
- scikit-learn
- pandas
- numpy
- audiofile

## Dataset
The model was trained on the ODSS (Open Dataset of Synthetic Speech) which includes:
- Paired natural and synthetic speech samples
- Multiple speakers and conditions
- High-quality synthetic speech generated using state-of-the-art TTS systems

## References
- ODSS Dataset: [Zenodo](https://zenodo.org/records/8370669)
- VeRa AI Project: [Website](https://www.veraai.eu/posts/odss-an-open-dataset-of-synthetic-speech)
- Research Paper: [arXiv:2307.07683v2](https://arxiv.org/pdf/2307.07683v2)
- OpenSMILE: [Official Website](https://www.audeering.com/research/opensmile/)
- NVIDIA TitanNet: [NGC Catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/titanet_large)

## Usage
1. Install dependencies
2. Extract features using provided functions
3. Load the pretrained model
4. Run inference on audio samples

The system provides probability scores for synthetic vs natural speech classification.
