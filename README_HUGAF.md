# HUGAF - Healthcare Unified Generalizable AI Framework

HUGAF (Healthcare Unified Generalizable AI Framework) is an enhanced version of the original HAIM (Holistic AI in Medicine) framework, featuring cutting-edge AI technologies including transformer architectures, large language models, Explainable AI (XAI), and privacy-preserving techniques like federated learning.

## Overview

This repository contains the code to replicate the data processing, modeling and reporting of our Holistic AI in Medicine (HAIM) in Nature's NPJ Digital Medicine. [Soenksen, L.R., Ma, Y., Zeng, C. et al. Integrated multimodal artificial intelligence framework for healthcare applications. npj Digit. Med. 5, 149 (2022). https://doi.org/10.1038/s41746-022-00689-4](https://www.nature.com/articles/s41746-022-00689-4).

## Enhanced Features in HUGAF Framework

### 1. Extended Data Modalities Support
- **Genomic Data**: Added support for genetic variant data and functional impact scores
- **Wearable Device Data**: Integrated time-series data from wearable health monitoring devices
- **Pathology Images**: Added support for histopathology image analysis

### 2. Advanced Model Architectures
- **Transformer-based Multimodal Fusion**: Replaced traditional XGBoost with a custom MultimodalTransformer model that uses attention mechanisms to effectively fuse multiple data modalities
- **Large Language Models Integration**: Integrated Bio_ClinicalBERT for enhanced text processing capabilities

### 3. Enhanced Explainability (XAI)
- **Attention Visualization**: Built-in attention mechanism visualization to understand model focus
- **SHAP Value Analysis**: Extended SHAP analysis for quantifying feature importance across modalities
- **Modality Contribution Analysis**: Detailed breakdown of each data modality's contribution to predictions

### 4. Privacy-Preserving Technologies
- **Federated Learning**: Implemented federated learning capabilities to enable collaborative model training across institutions without sharing sensitive patient data
- **Differential Privacy**: Added support for training with differential privacy guarantees

### 5. Continuous Learning and Optimization
- **Model Continual Learning**: Implemented continual learning mechanisms to automatically update models with new data
- **Hyperparameter Optimization**: Integrated automated hyperparameter tuning using both grid search and Bayesian optimization
- **Performance Monitoring**: Added comprehensive performance monitoring and model versioning

### 6. Scalability and Deployment
- **Self-Optimization**: Implemented automated model optimization for peak performance
- **Modular Design**: Enhanced modularity for easy extension and customization

## Code Structure

The code uses Python 3.6.9 and is separated into several components:

### Original HAIM Components:
0 - Software Package requirement

1 - Data Preprocessing. Noteevents.csv are public and available for download at Physionet.org; however, other "NOTES" data requires pre-release direct permission from Physionet.org for download as "discharge notes", "radiology notes", "ECG notes" and "ECHO notes" are not yet publicly released for MIMIC-IV as of Sep 2022, these files are: ds_icustay.csv, ecg_icustay.csv, echo_icustay.csv, rad_icustay.csv). To run our code without them just comment import and usage of these notes.

2 - Modeling of our three tasks: mortality prediction, length of stay prediction, chest pathology classification

3 - Result Generating: Including reporting of the AUROC, AUPRC, F1 scores, as well as code to generate the plots reported in the paper.

Please be advised that sufficient RAM or cluster access to parallel processing is needed to run these experiments.

### New HUGAF Components:
- `transformers_model.py`: Implementation of transformer-based multimodal models
- `xai_analysis.py`: Explainable AI analysis tools including SHAP and attention visualization
- `federated_learning.py`: Federated learning implementation with privacy-preserving capabilities
- `continual_learning.py`: Continuous learning and hyperparameter optimization modules
- `test_hugaf_simple.py`: Simplified test suite for validating HUGAF components

## How to Use the HUGAF Framework

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/HAIM.git
   cd HAIM
   ```

2. Install dependencies:
   ```bash
   pip install -r 0-requirements.txt
   pip install shap optuna
   ```
   
   For the full HUGAF experience, also install:
   ```bash
   pip install opencv-python
   ```

### Data Preparation

The HUGAF framework builds upon the original HAIM-MIMIC-MM dataset. Follow the original HAIM data preprocessing steps:

1. Download MIMIC-IV v1.0 from PhysioNet
2. Download MIMIC-CXR-JPG v2.0.0 from PhysioNet
3. Run data preprocessing scripts:
   ```bash
   python 1_1-Create_Pickle_Files.py
   python 1_2-Generate_Embeddings.py
   ```

### Using Enhanced Components

#### 1. Transformer-based Multimodal Model

```python
from transformers_model import MultimodalTransformer

# Define modality dimensions
modality_dims = {
    'demo': 10,      # Demographics
    'ts_ce': 50,     # Chart events time series
    'ts_le': 50,     # Lab events time series
    'ts_pe': 20,     # Procedure events time series
    'vd': 1024,      # Vision dense features
    'vp': 18,        # Vision predictions
    'vmd': 1024,     # Vision multi dense features
    'vmp': 18,       # Vision multi predictions
    'n_ecg': 768,    # ECG notes embeddings
    'n_ech': 768,    # Echocardiogram notes embeddings
    'n_rad': 768,    # Radiology notes embeddings
    'genomic': 100,  # Genomic data embeddings
    'wearable': 50,  # Wearable device data
    'pathology': 256 # Pathology image features
}

# Create model
model = MultimodalTransformer(
    modality_dims=modality_dims,
    hidden_dim=256,
    num_heads=4,
    num_layers=3
)

# Prepare input data (1D vectors for each modality)
modality_inputs = {
    'demo': torch.randn(10),
    'ts_ce': torch.randn(50),
    # ... add other modalities
}

# Get predictions
output = model(modality_inputs)
```

#### 2. Explainable AI Analysis

```python
from xai_analysis import XAIAnalyzer

# Create analyzer
analyzer = XAIAnalyzer(model, feature_names)

# Compute SHAP values
shap_values = analyzer.compute_shapley_values(X_train, X_test)

# Generate explanation report for a specific sample
report = analyzer.generate_explanation_report(shap_values, sample_idx=0)
print(report)

# Plot modality importance
modality_mapping = {
    'demo_0': 'demo', 'demo_1': 'demo',
    'ts_ce_0': 'ts_ce', 'ts_ce_1': 'ts_ce',
    # ... map all features to modalities
}
analyzer.plot_modality_importance(shap_values, modality_mapping)
```

#### 3. Federated Learning

```python
from federated_learning import FederatedClient, FederatedServer, FederatedLearningSystem

# Create global model
global_model = MultimodalTransformer(modality_dims)

# Create clients with local data
client1 = FederatedClient("hospital_1", global_model, data_loader_1)
client2 = FederatedClient("hospital_2", global_model, data_loader_2)

# Create server
server = FederatedServer(global_model)

# Create federated learning system
fl_system = FederatedLearningSystem(global_model, [client1, client2], server)

# Run federated training
results = fl_system.run_federated_training(num_rounds=10, local_epochs=5)
```

#### 4. Continual Learning

```python
from continual_learning import ContinualLearner, HyperparameterOptimizer

# Create continual learner
learner = ContinualLearner(model, performance_threshold=0.01, max_age_days=30)

# Evaluate and update model
updated = learner.evaluate_and_update(val_loader, current_performance)

# Check if retraining is needed
if learner.should_retrain():
    # Retrain model
    pass

# Get best model
best_model = learner.get_best_model()
```

### Running Tests

To validate the HUGAF framework components:

```bash
python test_hugaf_simple.py
```

This will run tests for:
- Transformer-based multimodal models
- Federated learning components
- Continual learning mechanisms

Note: Some tests may be skipped due to network limitations (e.g., LLM model downloads).

## Customization and Extension

The HUGAF framework is designed to be highly modular and extensible:

1. **Adding new data modalities**: Extend the `Patient_ICU` class and add corresponding feature extraction functions
2. **Custom model architectures**: Implement new models by extending the `MultimodalTransformer` class
3. **Additional XAI techniques**: Extend the `XAIAnalyzer` class with new explanation methods
4. **New privacy techniques**: Extend the federated learning components with additional privacy mechanisms

## Performance Considerations

- The transformer-based models may require significant computational resources
- For large-scale deployments, consider using GPU acceleration
- Federated learning can distribute computational load across multiple institutions

## Contributing

We welcome contributions to enhance the HUGAF framework. Please follow these steps:

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the Apache License, Version 2.0. See the LICENSE file for details.

## References

If you use this code in your research, please cite the original HAIM paper:

```
Soenksen, L.R., Ma, Y., Zeng, C. et al. Integrated multimodal artificial intelligence framework for healthcare applications. npj Digit. Med. 5, 149 (2022). https://doi.org/10.1038/s41746-022-00689-4
```

### UPDATE (Jan. 6, 2023)
The radiology and the discharge notes for MIMIC-IV have been officially released on:
https://physionet.org/content/mimic-iv-note/2.2/note/#files-panel

### UPDATE (Jun. 12, 2023)
For the publication, our team generated the file 'mimic-cxr-2.0.0-jpeg-txt.csv' by compiling an early-release version of participant notes and text from the images in CXR corresponding to MIMIC-IV. We wanted to add these to this repository, but the data policy from PhysioNet.org states we cannot directly share this compiled data via Git Hub. Physionet is the only one with permission to do so or subsets of the data. This means users need to generate their own mimic-cxr-2.0.0-jpeg-txt.csv based on the released notes and CXR files from Physionet.org once all notes are released. The dataset structure can be inferred from the code. As of June 12, 2023, Physionet has not fully released these notes, but it is likely they are planning to do so as part of their full release of MIMIC-IV. We are very sorry for any inconvenience this may cause.
