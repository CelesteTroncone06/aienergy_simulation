# Energy Use of AI Inference, Efficiency Pathways, and Test-Time Scaling

Analysis code accompanying the Joule manuscript:

Felipe Oviedo et al., *Energy Use of AI Inference, Efficiency Pathways, and Test-Time Scaling*, Joule, 2026

## Set-up environment

```bash
uv sync --frozen
source .venv/bin/activate
```

You can also run scripts directly without activating the environment:

```bash
uv run python simulation_traditional_queries.py
```

## Dataset

- Public token per second throughput (TPS) for open weight models running on H100 nodes: `model_throughput_DB.csv`.
- **Last update**: December, 2025.
- These TPS metrics are comparable to those of high-concurrency LLM services. We recommend using latest TPS benchmarks for other models, recent service engines, and other hardware platforms.

## Scripts

### Energy per query estimation (P5-P95) ***[Wh/query]***

The following Monte Carlo simulations estimate energy per query for open weight models running on H100 nodes in two regimes: traditional-query and long-query, as defined in the publication.

- `simulation_traditional_queries.py`
  Traditional / non-reasoning query regime. Produces results for each model, along with a blended 200B model and energy improvement pathways (Figure 1a and Figure 1b).

- `simulation_long_queries.py`
  Long-query / test-time-scaling regime. Produces results for each model, along with a blended 200B model and energy improvement pathways (Figure 2a and Figure 2b).

### Energy for serving 1 billion queries per day ***[GWh/day]***

The follow scrips estimate energy for serving 1 billion queries per day and efficiency pathways.
Then, these are aggregated to produce the various scenarios, includes traditional queries only and mixed scenarios with traditional and long queries, presented in Figure 3 in the manuscript.


```bash
uv run python simulation_1B_traditional_queries.py
uv run python simulation_1B_long_queries.py
uv run python simulation_1B_queries_aggregated.py
```

Script roles:

- `simulation_1B_traditional_queries.py`
  Generates `1B_queries_energy_consumption_short.csv`.

- `simulation_1B_long_queries.py`
  Generates `1B_queries_energy_consumption_reasoning.csv`.

- `simulation_1B_queries_aggregated.py`
  Aggregates and generates final results.

## Supplemental analyses

- `sensitivity_PU_proportionality_traditional_queries.py`
  Sensitivity analysis for power and utilization proportionality in the traditional-query regime.

- `estimate_per_mode_regression_reasoning_PUsens.py`
  Sensitivity analysis for power and utilization proportionality in the test-time-scaling regime.

- `sensitivity_linput_standard_queries.py`
  Sensitivity analysis for prompt input length in the traditional-query regime.

- `sensitivity_linput_long_queries.py`
  Sensitivity analysis for prompt input length in the test-time-scaling regime.

## License

This project is distributed under the MIT License. See `LICENSE`.

## Citation

If you find this work useful, please cite:

```bibtex
@article{oviedo2025energyuseaiinference,
      title={Energy Use of AI Inference: Efficiency Pathways and Test-Time Compute}, 
      author={Felipe Oviedo and Fiodar Kazhamiaka and Esha Choukse and Allen Kim and Amy Luers and Melanie Nakagawa and Ricardo Bianchini and Juan M. Lavista Ferres},
      year={2025},
      eprint={2509.20241},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2509.20241}, 
}
```

## Disclaimer

This repository is provided for research and informational purposes only and does not constitute legal, regulatory, compliance, or policy guidance.  Results should be interpreted as assumption-dependent and directional, not as definitive measurements of AI energy use across all systems or as guaranteed efficiency outcomes. The analysis focuses on per-query inference energy and efficiency pathways only and is not a full environmental or lifecycle assessment.

All code in this repository is released under the MIT License.

The data used in this repository are derived from publicly available sources
