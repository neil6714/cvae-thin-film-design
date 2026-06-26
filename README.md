# AI-Driven Inverse Design for Atomic Layer Deposition (ALD)

## Overview
This repository contains a complete, end-to-end Machine Learning pipeline for the inverse design of thin-film materials via Atomic Layer Deposition (ALD). In modern semiconductor fabrication, discovering the optimal machine parameters (Temperature, Pulse Time, Plasma Power) to achieve specific physical material properties (Growth Per Cycle, Refractive Index, Film Stress) requires highly expensive, iterative trial-and-error in a cleanroom. 

This project solves that bottleneck by reframing ALD recipe formulation as a **Conditional Generative AI** problem. It features a custom synthetic dataset, an ultra-fast Conditional Variational Autoencoder (CVAE), a highly precise Denoising Diffusion Probabilistic Model (DDPM) adapted for tabular data, and an independent ALD Digital Twin to scientifically benchmark generative accuracy.

## Project Architecture

### 1. The Generative Engines (Inverse Design)
The repository implements and contrasts two distinct generative architectures:
* **Conditional Variational Autoencoder (CVAE):** Engineered for immense inference speed. It compresses process parameters into a probabilistic latent space conditioned on target properties. It relies on a one-shot decoding process, enabling real-time, on-demand recipe calculation (~2ms inference time).
* **Tabular Diffusion Model (DDPM):** Engineered for absolute precision and strict boundary adherence. It utilizes a 100-step Gaussian noising schedule and learns to iteratively denoise a chaotic vector back into a highly optimized ALD recipe. It incorporates Sinusoidal Position Embeddings to manage timestep tracking.

### 2. The ALD Digital Twin (Evaluation Framework)
Because standard generative metrics (like KL-Divergence or Noise MSE) cannot measure physical viability, this project utilizes an independent **Digital Twin**. Built on a highly accurate Random Forest Regressor, the Digital Twin maps the forward problem (`Recipe -> Properties`). 
During benchmarking, both the CVAE and DDPM generate recipes for 100 complex target conditions. These generated recipes are fed into the Digital Twin, which simulates the ALD reactor to calculate the exact physical properties the recipes would yield in reality. The models are then evaluated based on Mean Absolute Error (MAE).

### 3. Inference & Optimization Strategies
While the generative models handle the underlying physics, the repository provides two distinct deployment scripts tailored for different engineering use cases:
* **Direct Inverse Design (`generate_recipe.py`):** Acts as an on-demand calculator. When an engineer requires a film with strict, exact specifications (e.g., exactly 1.45 GPC and -50 MPa stress), this script utilizes the trained network to instantaneously output the necessary hardware parameters.
* **Search Space Discovery (`optimize_recipe.py`):** Built for exploratory materials science. Rather than targeting a pre-known value, this script generates 10,000 theoretical possibilities and evaluates them against a mathematical fitness function (e.g., maximizing throughput while minimizing stress) to discover the absolute optimal processing boundaries.

## Repository Structure

```text
├── data/
│   ├── top_100_optimized_recipes.csv       # Curated output from CVAE optimization space
├── models/
│   ├── [Weights & Scalers]                 # Ignored binary files (.pth, .pkl)
├── src/
│   ├── dataset_generator.py                # ALD physics simulation and data generation
│   ├── train_cvae.py                       # CVAE architecture and training loop
│   ├── train_ddpm.py                       # Tabular Diffusion architecture and training
│   ├── generate_recipe.py                  # Direct on-demand CVAE inference script
│   ├── optimize_recipe.py                  # Large-scale search space exploration
│   ├── benchmark.py                        # Head-to-head Digital Twin evaluation
├── .gitignore
├── requirements.txt
└── README.md
