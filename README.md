# HyperSCA: Hyperbolic Spatiotemporal Causal Analysis

[![PyPI version](https://badge.fury.io/py/hypersca.svg)](https://badge.fury.io/py/hypersca)
[![Documentation Status](https://readthedocs.org/projects/hypersca/badge/?version=latest)](https://hypersca.readthedocs.io/en/latest/?badge=latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Unveiling Cell Communication and Causal Signaling in the Tumor Microenvironment via Hyperbolic Geometry.**

## Overview

**HyperSCA** is a computational framework designed to integrate Single-cell RNA-seq (scRNA-seq) and Spatial Transcriptomics (ST) data. Unlike traditional Euclidean methods, HyperSCA leverages **Hyperbolic Geometry (Poincaré Ball/Lorentz Model)** to faithfully embed complex hierarchical cellular lineages and spatial topologies.

It uniquely combines:
1.  **Topology-preserving Embedding:** Hyperbolic VAEs to reduce distortion in latent space.
2.  **Causal Disentanglement:** Separating intrinsic cell states from extrinsic microenvironmental signals.
3.  **In Silico Perturbation:** Simulating virtual knockouts to predict spatial remodeling.

![Model Architecture](docs/figures/model_workflow.png)
*(Note: Replace with your Figure 1: Technical Roadmap)*

## Key Features

* **Hyperbolic VAE (H-VAE):** Capture tree-like differentiation trajectories (e.g., T-cell exhaustion) with low distortion.
* **Spatial Neighbor Graph:** Construction of topology-encoding graphs from ST coordinates.
* **Causal Inference Engine:** Identification of directed signaling axes (e.g., CAF $\rightarrow$ TAM) using Conditional Mutual Information (CMI).
* **Counterfactual Generation:** Predict cell state shifts under specific gene perturbations.

## Installation

### Prerequisites
* Python >= 3.9
* PyTorch >= 2.0 (with CUDA support recommended)
* Scanpy & Squidpy

### Install from PyPI (Coming Soon)
```bash
pip install hypersca