# Calibrated Federated Kalman Filter with Fault Detection

## Overview

This repository implements a **two-phase Federated Kalman Filter (FKF)** architecture that learns optimal sensor trust weights through a calibration phase using a high-accuracy reference sensor, then operates autonomously with fault detection and isolation capabilities.

## Algorithm Description

### Architecture Components
The repository is implemented using object-oriented programming principles with the following classes:

1. **MotionModel**: Defines a linear time-invariant (LTI) motion model (x_{k+1} = Fx_k + Gw_k, w_k ~ N(0,Q)). Used to define motion model parameters and propagate states.
2. **LinearSensor**: Implements a linear measurement model (z_k = Hx_k + v_k). Supplies measurements according to defined sensor characteristics.
3. **LocalKalmanFilter**: The primary implementation using the calibrated architecture. Orchestrates multiple LocalKalmanFilter objects and the fusion center. Implements a two-phase approach: a calibration phase to learn sensor weights using a reference filter, followed by an operational phase.
5. **FederatedKF_ERR_only**: Similar to FederatedKF but without the calibrated architecture. Based on the architecture proposed in Reference [2].
6. **FederatedKFTraditional**: The original implementation as proposed by Carlson (see Reference [1]). Used for comparison.
7. **CentralizedKF**: Standard Kalman filter implementation. Used for comparison.
8. **FusionCenter**: The fusion hub responsible for optimal fusion of multiple filter estimates.

### General Functions
1. **demo_fkf_linear**: Single-run evaluation of the different filter architectures.
2. **Monte_Carlo_Simulation**: Multi-run evaluation of the different filter architectures.

### Two-Phase Operation

#### Phase 1: Calibration

**Purpose**: Learn Information Sharing Factors (ISFs) that quantify relative sensor reliability.

**Process**:
1. All local filters process their sensor measurements independently
2. Reference filter processes high-accuracy measurements in parallel
3. Master filter fuses all estimates with fixed weights:
   - Reference filter: 60-70% weight
   - Local filters: Share remaining 30-40% equally
4. Accumulate squared errors between each local filter and reference filter
5. At end of calibration, calculate ISFs:

ISF_i = (1/MSE_i) / sum(1/MSE_j)

where ISFs satisfy: **sum(ISF_i) = 1**


#### Phase 2: Operational

**Purpose**: Use learned ISFs for fault-tolerant state estimation without reference sensor.

**Process**:
1. Local filters predict and update with their measurements
2. **Fault Detection** before each update:
   - Chi-square test: `λ = r'·S⁻¹·r > threshold`
   - Sliding window test: `η = tr(S_theoretical)/tr(S_actual) ∉ [0.4, 2.5]`
3. **Fault Handling**: Skip measurement update if fault detected
4. Master filter fuses local estimates using Carlson's formulation (equations 15,16 - see reference [1])
5. **Information Sharing**: Reset local filters with the ISFs


## Performance Characteristics

### Advantages
- **Adaptive weighting**: Automatically learns sensor reliability from data
- **Fault tolerance**: Maintains accuracy even with 10-15% fault rates
- **Distributed architecture**: Computational load distributed across local filters

### Limitations
- **Calibration dependency**: Performance relies on high-quality calibration phase
- **Static ISFs**: Weights don't adapt during operation
- **Moderate improvement**: 5-15% RMSE improvement over standard FKF (skip update) at high fault rates

## Comparison with Alternatives

| Method | Fault-Free RMSE | High Fault RMSE | Complexity |
|--------|-----------------|-----------------|------------|
| CKF | 0.76m (best) | 7.4m (worst) | Low |
| Traditional FKF | 1.1m | 7.4m | Medium |
| Standard FKF (Skip) | 1.1m | 3.6m | Medium |
| **Calibrated FKF** | **1.1m** | **3.7m** | **High** |

**Conclusion**: Calibrated FKF shows marginal performance gains over simpler Standard FKF (Skip Update), suggesting the added complexity may not be justified for most applications.

## References

[1] Neal A. Carlson. Federated square root filter for decentralized parallel processors. IEEE Trans-
actions on Aerospace and Electronic Systems, 26(3):517–525, 1990.
[2] X. Wu, Z. Su, L. Li, and Z. Bai. Improved adaptive federated kalman filtering for ins/gnss/vns
integrated navigation algorithm. Applied Sciences, 13(9):5790, 2023.
