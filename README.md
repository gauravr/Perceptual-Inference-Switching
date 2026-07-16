# Perceptual Inference Switching

Modeling human perceptual estimation: Bayesian inference, switching heuristics, and history-dependent bias analysis.

## Project Overview
This repository contains Python implementations of computational models used to analyze human perceptual estimation behavior, based on the research presented in *“A Switching Observer for Human Perceptual Estimation”* (Laquitaine & Gardner, 2018). 

The project investigates the discrepancy between normative Bayesian inference—where sensory evidence and prior knowledge are multiplicatively integrated—and the heuristic strategies humans actually employ, such as switching between prior and sensory states.

## Key Model Logic
The models utilize circular statistics, specifically **von Mises distributions**, to represent motion directions and prior beliefs. The core functionality includes:

*   **Basic Bayesian Observer:** Models optimal integration of prior and likelihood using Maximum A Posteriori (MAP) readout.
*   **Switching Observer:** Implements a heuristic strategy that probabilistically switches between prior mean and sensory evidence, successfully predicting bimodal estimate distributions.

## File Structure
*   `data.csv`: Processed behavioral data.
*   `params.csv`: Fitted model parameters.

## Getting Started
1. **Prerequisites**: Ensure you have dependecies installed present in pip-requirements.txt file.
2. **Execution**: Run the provided plotting functions to simulate the models and generate the comparison charts:
   ```bash
   ipython models/switching.py 
   ipython models/bayesian.py  
   ```
## References
[Laquitaine, S., & Gardner, J. L. (2018). A Switching Observer for Human Perceptual Estimation. *Neuron*, 97(2), 462-474](https://pubmed.ncbi.nlm.nih.gov/29290551/).
