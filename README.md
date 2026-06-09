# AGV Edge-Cost Decision Support

This repository contains the code, representative data files, graph files, preprocessing outputs, learned edge-cost outputs, temporal motion outputs, and evaluation scripts used for learning planner-compatible AGV edge costs and temporal trajectory realization in industrial logistics.

## Main purpose

The project learns traversal time and energy costs from real Automated Guided Vehicle (AGV) telemetry and integrates the learned edge costs with classical graph-based route planning. It also includes temporal motion modeling for per-second trajectory realization.

## Repository contents

### Main input files

- `Node_F3.csv`  
  Directed shop-floor graph node file. It contains the graph nodes used for AGV route planning and graph-based learning.

- `Edge_Distances3_.csv`  
  Directed edge-distance file. It defines feasible graph connections and edge distances.

- `wholetesting_nav2_.xlsx`  
  Representative raw AGV telemetry file used for preprocessing and evaluation.

## Main source code files

- `config.py`  
  Contains file paths, thresholds, model settings, and experiment configuration.

- `preprocess_nav2_to_1hz.py`  
  Converts raw AGV telemetry into a cleaned uniform 1 Hz trajectory representation.

- `preprocess.py`  
  Aligns AGV trajectory data with the directed graph and extracts edge traversal samples.

- `graph_data.py`  
  Loads graph node and edge files and prepares graph structures for learning and planning.

- `model.py`  
  Defines the graph-aligned edge-cost learning model.

- `edge_cost_train.py`  
  Trains the traversal time and energy prediction model.

- `planner.py`  
  Performs route planning using learned edge costs and classical graph search.

- `trajectory.py`  
  Generates per-second trajectory realization from planned routes.

- `templates.py`  
  Handles geometric and temporal templates used for trajectory realization.

- `temporal_mlp.py`  
  Contains temporal motion modeling components.

- `main_run.py`  
  Runs the main workflow.

- `evaluate_full_run.py`  
  Evaluates the full route and trajectory realization workflow.

- `utils_io.py`  
  Utility functions for input/output operations.

## Experiment scripts

- `holdout_trajectory_eval.py`  
  Evaluates hold-out mission performance and trajectory-level behavior.

- `baseline_compare_holdout_all.py`  
  Compares the learned edge-cost model with rule-based baselines.

- `ablation_eval.py`  
  Runs ablation analysis for learned edge costs, slowdown features, and temporal realization.

- `distance_only_ablation.py`  
  Runs a distance-only reference comparison.

- `alpha_sensitivity_operation_to_all_nodes.py`  
  Performs alpha-sensitivity analysis over operation-node starts and reachable graph destinations.

- `blocked_edge_reroute.py`  
  Runs static corridor-unavailability rerouting tests.

- `compare_mlp_xgb_lgbm_with_proposed_ggnn.py`  
  Compares non-graph learning baselines such as MLP, XGBoost, and LightGBM.

## Output folders

### `out_1hz/`

This folder contains preprocessed 1 Hz trajectory outputs.

Typical files:

- `nav_1hz_full.csv`  
  Full cleaned 1 Hz AGV trajectory.

- `nav_1hz_move_only.csv`  
  Movement-only 1 Hz AGV trajectory after stationary samples are removed.

- `nav_1hz_summary.json`  
  Summary of preprocessing statistics.

### `inductive_folder_new_data/`

This folder contains learned edge-cost outputs, temporal model outputs, hold-out results, baseline reports, ablation reports, alpha-sensitivity reports, and rerouting reports.

Important contents include:

- `edge_samples.csv`  
  Extracted edge traversal samples used for training and evaluation.

- `edge_costs_pred.csv`  
  Predicted traversal time and energy values for graph edges.

- `edge_cost_model.pt`  
  Trained edge-cost model checkpoint.

- `geom_templates.csv`  
  Geometric templates used for trajectory realization.

- `temporal_all_summary.csv`  
  Summary of temporal model evaluation.

- `ablation_reports/`  
  Ablation study outputs.

- `alpha_operation_to_all_nodes/`  
  Alpha-sensitivity analysis outputs.

- `baseline_reports_holdout_all/`  
  Hold-out baseline comparison outputs.

- `blocked_edge_reroute_reports/`  
  Static corridor-unavailability rerouting outputs.

- `holdout_traj_reports/`  
  Hold-out mission and trajectory evaluation outputs.

- `eswa_learning_baselines_safe/`  
  MLP, XGBoost, and LightGBM baseline outputs.

## Requirements

Install the main dependencies using:

```bash
pip install numpy pandas scikit-learn networkx matplotlib torch openpyxl xgboost lightgbm
