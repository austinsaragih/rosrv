# rosrv

This repository contains the replication code and computational materials for:

**“Real Options for Systemic Resilience and Viability: Experimental Analysis of Process Flexibility Design.”**

The repository contains the code used to reproduce the numerical experiments, robustness analyses, figures, and industrial case study reported in the paper.

## Repository Structure

- `5.1.Baseline/`  
  Baseline numerical experiments and figures reported in Section 5.1, including the four experimental cases comparing the multi-period, start-period, long-chain, decision-rule, and perfect-information designs.

- `5.2and3.OFATandFullFac/`  
  One-factor-at-a-time (OFAT) analyses and figures reported in Section 5.2, together with the full factorial experimental design reported in Section 5.3.

- `6.CaseStudy/`  
  Industrial case-study analysis and figures reported in Section 6.

- `App.B.andC/`  
  Robust Optimization (RO), Distributionally Robust Optimization (DRO), and multi-stage analyses reported in Appendices B and C, together with their cross-evaluation tables and figures.

- `App.D.4.psi9/`  
  Rebalancing sensitivity analysis and figures for $\psi=9$ reported in Appendix D.4.

- `App.D.4.psi14/`  
  Rebalancing sensitivity analysis and figures for $\psi=14$ reported in Appendix D.4.

## Main Scripts

- `all_analyses_engaging.py`  
  Main analysis script used for the numerical experiments, including the OFAT and full factorial analyses.

- `rosr_extended_csv_engaging.py`  
  Extended analysis script used for the RO, DRO, and multi-stage experiments reported in Appendices B and C.

- `run_one_analysis.sh`  
  SLURM submission script for running one numerical analysis at a time.

- `run_rosr_extended_csv.sh`  
  SLURM submission script for running the extended analyses for Appendices B and C.

## Requirements

The computational experiments are implemented in Python and use Gurobi for mathematical optimization.

A valid Gurobi installation and license are required to reproduce the optimization results.

The optimization experiments reported in the paper were executed on a SLURM-based computing cluster. Plotting notebooks and scripts can also be run locally after the corresponding result files have been generated.

## Running the Section 5.2 and 5.3 Analyses

The analyses in `all_analyses_engaging.py` are designed to be submitted **one analysis at a time** using `run_one_analysis.sh`.

First create the output directories if needed:

```bash
mkdir -p logs results
```

Submit an individual analysis as:

```bash
sbatch --job-name=ROSR_A05 run_one_analysis.sh 5
```

where the final argument is the analysis number.

For example, analyses can be submitted individually as:

```bash
sbatch --job-name=ROSR_A05 run_one_analysis.sh 5
sbatch --job-name=ROSR_A06 run_one_analysis.sh 6
sbatch --job-name=ROSR_A07 run_one_analysis.sh 7
sbatch --job-name=ROSR_A08 run_one_analysis.sh 8
sbatch --job-name=ROSR_A09 run_one_analysis.sh 9
sbatch --job-name=ROSR_A10 run_one_analysis.sh 10
sbatch --job-name=ROSR_A11 run_one_analysis.sh 11
```

Each analysis is therefore run as a separate SLURM job rather than as a single combined job.

Analysis 11 corresponds to the full factorial experiment reported in Section 5.3.

## Running the Appendix B and C Extensions

The RO, DRO, and multi-stage extensions are generated using:

```text
rosr_extended_csv_engaging.py
```

and submitted through:

```text
run_rosr_extended_csv.sh
```

First create the output directories if needed:

```bash
mkdir -p logs rosr_extended_outputs
```

Then submit:

```bash
sbatch run_rosr_extended_csv.sh
```

The SLURM script is configured to process the four experimental cases **one at a time**. In particular, the array restriction

```bash
#SBATCH --array=1-4%1
```

ensures that only one case runs at a time.

The resulting files are written to:

```text
rosr_extended_outputs/
```

and provide the computational results used for the RO, DRO, and multi-stage analyses in Appendices B and C.

## Generating Tables and Figures

The repository folders are organized to correspond directly to the sections and appendices of the paper.

After the required computational outputs have been generated, the analysis and plotting notebooks or scripts can be run directly from their corresponding folders:

- **Section 5.1 analyses and figures:** `5.1.Baseline/`
- **Section 5.2 OFAT figures and Section 5.3 full factorial outputs:** `5.2and3.OFATandFullFac/`
- **Section 6 case-study analyses and figures:** `6.CaseStudy/`
- **Appendix B and C tables and figures:** `App.B.andC/`
- **Appendix D.4 sensitivity analyses and figures:**  
  - `App.D.4.psi9/`
  - `App.D.4.psi14/`

The plotting notebooks can therefore be executed directly within the corresponding repository folders once the required CSV result files are available.

Because optimization runtimes and numerical tolerances may depend on hardware, Gurobi version, and the computational environment, small numerical or runtime differences may occur when reproducing the experiments.

## Citation

If you use this code, data, or framework in your research, or if you plan to extend the methodology, please cite the foundational research paper using the following BibTeX entry:

```bibtex
@article{saragih2026real,
  title={Real Options for Systemic Resilience and Viability: Experimental Analysis of Process Flexibility Design},
  author={Saragih, Austin and Janjevic, Milena and Goentzel, Jarrod and de Neufville, Richard and Sheffi, Yossi},
  journal={MIT Center for Transportation \& Logistics Research Paper},
  number={2026},
  year={2026}
}
```

## License

This repository is released under the MIT License.
