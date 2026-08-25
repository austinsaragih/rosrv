from __future__ import annotations


"""MIT Engaging solver entry point for ROSR Cases 1-4.

This module intentionally has no plotting dependencies and creates CSV files only.
Generate figures later from the saved CSV files with the companion notebook.
"""

import itertools
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CaseConfig:
    """Inputs that distinguish Cases 1-4."""

    case_number: int
    established_products: int
    launch_products: int
    compete_products: int
    availability_floor: float
    base_capacity: float
    disruption_probability: float

    # Requested probabilities are fixed consistently across all four cases.
    launch_probability: float = 0.975


@dataclass(frozen=True)
class RuntimeConfig:
    """Solver, scenario, CSV-output, and reproducibility controls."""

    planning_exhaustive: bool = False
    planning_sample_size: int = 40
    planning_scenario_seed: int = 0
    capacity_seed: int = 73

    # The original evaluation code uses all 3^4 Increase/Flat/Decrease tuples.
    evaluation_exhaustive: bool = True
    evaluation_sample_size: int = 81
    evaluation_seed: int = 0

    # Executive evaluation convention: keep NoLaunch in probability-weighted
    # planning, but exclude it from the common policy-evaluation set.  With four
    # products this yields the original 3^4 = 81 equally weighted trajectory
    # combinations; for Growth portfolios this is conditional on successful launch.
    evaluation_include_no_launch: bool = False

    # Baseline SP models use the same limits as rosr_modular_cases.ipynb.
    # Extension-specific RO/DRO/multistage limits are defined separately below.
    planning_time_limit: Optional[float] = 900.0
    start_period_time_limit: Optional[float] = 900.0
    evaluation_time_limit: Optional[float] = 300.0
    perfect_information_time_limit: Optional[float] = 300.0

    output_flag: int = 1
    evaluation_output_flag: int = 0
    threads: int = 8

    run_decision_rule: bool = True
    run_perfect_information: bool = True


CASE_CONFIGS: Dict[int, CaseConfig] = {
    # Case 1: four established products, deterministic availability floor,
    # requested base capacity of 30.
    1: CaseConfig(
        case_number=1,
        established_products=4,
        launch_products=0,
        compete_products=0,
        availability_floor=1.0,
        base_capacity=30.0,
        disruption_probability=0,
    ),
    # Case 2: same product mix as Case 1, with availability uncertainty and 5% disruption probability.
    2: CaseConfig(
        case_number=2,
        established_products=4,
        launch_products=0,
        compete_products=0,
        availability_floor=0.8,
        base_capacity=35.0,
        disruption_probability=0.05,
    ),
    # Case 3: two established, one competed, and one launch product.
    3: CaseConfig(
        case_number=3,
        established_products=2,
        launch_products=1,
        compete_products=1,
        availability_floor=1.0,
        base_capacity=35.0,
        disruption_probability=0,
    ),
    # Case 4: same product mix as Case 3, with availability uncertainty and 5% disruption probability.
    4: CaseConfig(
        case_number=4,
        established_products=2,
        launch_products=1,
        compete_products=1,
        availability_floor=0.8, 
        base_capacity=35.0,
        disruption_probability=0.05,
    ),
}


def get_case_config(case_number: int) -> CaseConfig:
    """Return one validated case configuration."""

    if case_number not in CASE_CONFIGS:
        raise ValueError(f"CASE_NUMBER must be one of {sorted(CASE_CONFIGS)}.")
    return CASE_CONFIGS[case_number]

@dataclass
class ProblemData:
    """Deterministic sets, costs, demand parameters, and initial design data."""

    config: CaseConfig

    # Horizons and index sets.
    theta: int
    nu: int
    T_I: List[int]
    T_V: List[int]

    # Machine and product sets.
    M: List[str]
    E: List[str]
    C: List[str]
    L: List[str]
    P: List[str]
    product_type: Dict[str, str]

    # Demand states and trajectories before joint-scenario construction.
    base_states: List[str]
    launch_states: List[str]
    base_demand: Dict[Tuple[str, str, int], float]

    # Economic parameters.
    discount_rate: float
    unit_profit: Dict[Tuple[str, str], float]
    fixed_opex: Dict[str, float]
    machine_capex: Dict[str, float]
    arc_capex: Dict[Tuple[str, str], float]
    lifetime: Dict[str, int]

    # Installed system at the start of the horizon.
    pre_installed_machines: List[str]
    pre_installed_arcs: List[Tuple[str, str]]


@dataclass
class ScenarioSet:
    """A complete stochastic scenario set used by an optimization/evaluation."""

    scenario_ids: List[int]
    scenario_map: Dict[int, Tuple[str, ...]]
    probabilities: Dict[int, float]
    demand: Dict[Tuple[str, int, int], float]
    capacity: Dict[Tuple[str, int, int], float]


@dataclass
class MultiPeriodDesign:
    """Solved multi-period first-stage design."""

    objective_value: float
    machine_on: Dict[Tuple[str, int], float]
    machine_start: Dict[Tuple[str, int], float]
    arc_on: Dict[Tuple[str, str, int], float]
    production: Dict[Tuple[str, str, int, int], float]


@dataclass
class StartPeriodDesign:
    """Solved design in which all investments occur in period zero."""

    objective_value: float
    machine_on: Dict[str, float]
    machine_start: Dict[str, float]
    arc_on: Dict[Tuple[str, str], float]


@dataclass
class PolicyNPVs:
    """Per-scenario NPV arrays for all policies in the comparison."""

    multi_period: List[float]
    start_period: List[float]
    long_chain: List[float]
    decision_rule: List[float]
    perfect_information: List[float]

def read_colab_wls_options() -> Dict[str, object]:
    """Read optional Gurobi WLS credentials from Google Colab secrets.

    When the code is run locally, an empty dictionary is returned and Gurobi
    uses the local license configuration instead.
    """

    try:
        from google.colab import userdata  # type: ignore
    except ImportError:
        return {}

    access_id = userdata.get("secret_WLSACCESSID")
    secret = userdata.get("secret_WLSSECRET")
    license_id = userdata.get("secret_LICENSEID")

    if not (access_id and secret and license_id):
        return {}

    return {
        "WLSACCESSID": access_id,
        "WLSSECRET": secret,
        "LICENSEID": int(license_id),
    }


def create_gurobi_environment(output_dir: Path) -> gp.Env:
    """Create one reusable Gurobi environment for a complete case run."""

    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "gurobi.log"
    options = read_colab_wls_options()

    if options:
        return gp.Env(str(log_path), params=options)
    return gp.Env(str(log_path))


def configure_model(
    model: gp.Model,
    *,
    output_flag: int,
    threads: int,
    time_limit: Optional[float],
) -> None:
    """Apply solver controls consistently to every Gurobi model."""

    model.Params.OutputFlag = output_flag
    model.Params.Threads = threads
    if time_limit is not None:
        model.Params.TimeLimit = time_limit


def optimize_and_check(model: gp.Model) -> float:
    """Optimize a model and return its objective after validating a solution."""

    model.optimize()

    if model.SolCount <= 0:
        raise RuntimeError(
            f"Model {model.ModelName!r} produced no feasible solution; "
            f"Gurobi status={model.Status}."
        )

    return float(model.ObjVal)



def states_for_product(data: ProblemData, product: str) -> List[str]:
    """Return the valid demand states for one product."""

    if data.product_type[product] == "L":
        return data.launch_states
    return data.base_states


def build_problem_data(config: CaseConfig) -> ProblemData:
    """Construct all deterministic data shared by planning and evaluation."""

    # Planning and valuation horizons from the original code.
    theta = 4
    nu = 14
    T_I = list(range(theta + 1))
    T_V = list(range(nu + 1))

    # Product counts and timing parameters.
    launch_time = 3
    competition_time = 3
    launch_ramp_time = 1
    decline_ramp_time = 1

    # Demand levels and state-dependent slopes.
    base_level = 30.0
    growth_level = 45.0
    decline_level = 15.0
    base_slope = 1.0
    growth_slope = 2.0
    decline_slope = 1.0

    # Eight resources are used in every original case.
    M = [f"Resource {idx + 1}" for idx in range(8)]
    E = [f"Product E{idx + 1}" for idx in range(config.established_products)]
    C = [f"Product C{idx + 1}" for idx in range(config.compete_products)]
    L = [f"Product L{idx + 1}" for idx in range(config.launch_products)]
    P = E + C + L

    product_type = {
        product: ("E" if product in E else "L" if product in L else "C")
        for product in P
    }

    base_states = ["Increase", "Flat", "Decrease"]
    launch_states = ["Increase", "Flat", "Decrease", "NoLaunch"]

    # State slopes follow the original scripts.
    base_slopes = {
        "Increase": +base_slope,
        "Flat": 0.0,
        "Decrease": -base_slope,
    }
    growth_slopes = {
        "Increase": +growth_slope,
        "Flat": 0.0,
        "Decrease": -growth_slope,
    }
    decline_slopes = {
        "Increase": +decline_slope,
        "Flat": 0.0,
        "Decrease": -decline_slope,
    }

    # Build d_bar[j,state,t], the trajectory before products are combined into
    # joint scenarios.
    base_demand: Dict[Tuple[str, str, int], float] = {}

    for product in P:
        valid_states = launch_states if product_type[product] == "L" else base_states

        for state in valid_states:
            for t in T_V:
                if product_type[product] == "E":
                    demand_value = base_level + base_slopes[state] * t

                elif product_type[product] == "L":
                    if state == "NoLaunch":
                        demand_value = 0.0
                    elif t < launch_time:
                        demand_value = 0.0
                    elif t < launch_time + launch_ramp_time:
                        demand_value = (
                            growth_level
                            / launch_ramp_time
                            * (t - launch_time + 1)
                        )
                    else:
                        demand_value = (
                            growth_level
                            + growth_slopes[state]
                            * (t - launch_time - launch_ramp_time + 1)
                        )

                else:  # Competed/declining product.
                    if t < competition_time:
                        demand_value = base_level
                    elif t < competition_time + decline_ramp_time:
                        demand_value = (
                            base_level
                            - (base_level - decline_level)
                            / decline_ramp_time
                            * (t - competition_time + 1)
                        )
                    else:
                        demand_value = (
                            decline_level
                            + decline_slopes[state]
                            * (t - competition_time - decline_ramp_time + 1)
                        )

                # Demand cannot become negative.
                base_demand[product, state, t] = max(0.0, demand_value)

    # Core economics from the original notebooks.
    discount_rate = 0.075
    unit_profit = {(i, j): 300.0 for i in M for j in P}
    fixed_opex = {i: 1_000.0 for i in M}
    machine_capex = {i: 25_000.0 for i in M}
    arc_capex = {(i, j): 1_000.0 for i in M for j in P}
    lifetime = {i: 10 for i in M}

    # Established and competed products begin with one paired resource each.
    pre_installed_machines = M[: len(E) + len(C)]
    pre_installed_arcs = list(zip(pre_installed_machines, E + C))

    # Pre-installed resources and arcs have zero start-up CAPEX, exactly as in
    # the original scripts.
    for machine in pre_installed_machines:
        machine_capex[machine] = 0.0
    for arc in pre_installed_arcs:
        arc_capex[arc] = 0.0

    return ProblemData(
        config=config,
        theta=theta,
        nu=nu,
        T_I=T_I,
        T_V=T_V,
        M=M,
        E=E,
        C=C,
        L=L,
        P=P,
        product_type=product_type,
        base_states=base_states,
        launch_states=launch_states,
        base_demand=base_demand,
        discount_rate=discount_rate,
        unit_profit=unit_profit,
        fixed_opex=fixed_opex,
        machine_capex=machine_capex,
        arc_capex=arc_capex,
        lifetime=lifetime,
        pre_installed_machines=pre_installed_machines,
        pre_installed_arcs=pre_installed_arcs,
    )


def discount_factor(data: ProblemData, t: int) -> float:
    """Present-value factor used by the original model for period t."""

    return 1.0 / (1.0 + data.discount_rate) ** (t + 1)

def product_state_probabilities(
    data: ProblemData,
    *,
    include_no_launch: bool,
) -> Tuple[List[List[str]], List[List[float]]]:
    """Return each product's state space and state probabilities."""

    all_states: List[List[str]] = []
    all_probabilities: List[List[float]] = []

    for product in data.P:
        if data.product_type[product] == "L" and include_no_launch:
            all_states.append(list(data.launch_states))
            all_probabilities.append(
                [
                    data.config.launch_probability / 3.0,
                    data.config.launch_probability / 3.0,
                    data.config.launch_probability / 3.0,
                    1.0 - data.config.launch_probability,
                ]
            )
        else:
            all_states.append(list(data.base_states))
            all_probabilities.append([1.0 / 3.0] * 3)

    return all_states, all_probabilities


def build_joint_scenarios(
    data: ProblemData,
    *,
    include_no_launch: bool,
    exhaustive: bool,
    sample_size: int,
    random_seed: int,
    weighted_sampling: bool,
) -> Tuple[List[int], Dict[int, Tuple[str, ...]], Dict[int, float]]:
    """Build a joint scenario map and normalized scenario probabilities."""

    per_product_states, per_product_probabilities = product_state_probabilities(
        data,
        include_no_launch=include_no_launch,
    )

    all_tuples = list(itertools.product(*per_product_states))
    weights = np.empty(len(all_tuples), dtype=float)

    for tuple_index, state_tuple in enumerate(all_tuples):
        joint_weight = 1.0
        for product_index, state in enumerate(state_tuple):
            state_space = per_product_states[product_index]
            state_probabilities = per_product_probabilities[product_index]
            joint_weight *= state_probabilities[state_space.index(state)]
        weights[tuple_index] = joint_weight

    positive_indices = np.flatnonzero(weights > 0.0)
    if len(positive_indices) == 0:
        raise ValueError("All scenario weights are zero.")

    if exhaustive:
        kept_indices = positive_indices
    else:
        number_to_keep = min(sample_size, len(positive_indices))
        rng = np.random.default_rng(random_seed)

        if weighted_sampling:
            probabilities = weights / weights.sum()
        else:
            probabilities = None

        kept_indices = rng.choice(
            len(all_tuples),
            size=number_to_keep,
            replace=False,
            p=probabilities,
        )

    scenario_map = {
        new_id: all_tuples[old_id]
        for new_id, old_id in enumerate(kept_indices)
    }
    scenario_ids = list(scenario_map)

    kept_weights = weights[np.asarray(kept_indices, dtype=int)]
    kept_weights = kept_weights / kept_weights.sum()
    scenario_probabilities = {
        scenario_id: float(kept_weights[position])
        for position, scenario_id in enumerate(scenario_ids)
    }

    return scenario_ids, scenario_map, scenario_probabilities


def build_scenario_demand(
    data: ProblemData,
    scenario_map: Mapping[int, Tuple[str, ...]],
) -> Dict[Tuple[str, int, int], float]:
    """Expand product-state trajectories into joint-scenario demand."""

    demand: Dict[Tuple[str, int, int], float] = {}

    for scenario_id, state_tuple in scenario_map.items():
        for product_index, product in enumerate(data.P):
            state = state_tuple[product_index]
            for t in data.T_V:
                demand[product, scenario_id, t] = data.base_demand[product, state, t]

    return demand


def build_scenario_capacity(
    data: ProblemData,
    scenario_ids: Sequence[int],
    capacity_rng: random.Random,
) -> Dict[Tuple[str, int, int], float]:
    """Generate capacity using the original disruption/availability process."""

    capacity: Dict[Tuple[str, int, int], float] = {}

    for machine in data.M:
        for scenario_id in scenario_ids:
            for t in data.T_V:
                disrupted = (
                    capacity_rng.random()
                    < data.config.disruption_probability
                )

                if disrupted:
                    capacity[machine, scenario_id, t] = 0.0
                else:
                    availability = capacity_rng.uniform(
                        data.config.availability_floor,
                        1.0,
                    )
                    capacity[machine, scenario_id, t] = (
                        data.config.base_capacity * availability
                    )

    return capacity


def build_planning_scenarios(
    data: ProblemData,
    runtime: RuntimeConfig,
    capacity_rng: random.Random,
) -> ScenarioSet:
    """Create the probability-weighted scenario set used to choose designs."""

    scenario_ids, scenario_map, probabilities = build_joint_scenarios(
        data,
        include_no_launch=True,
        exhaustive=runtime.planning_exhaustive,
        sample_size=runtime.planning_sample_size,
        random_seed=runtime.planning_scenario_seed,
        weighted_sampling=True,
    )

    demand = build_scenario_demand(data, scenario_map)
    capacity = build_scenario_capacity(data, scenario_ids, capacity_rng)

    return ScenarioSet(
        scenario_ids=scenario_ids,
        scenario_map=scenario_map,
        probabilities=probabilities,
        demand=demand,
        capacity=capacity,
    )


def build_evaluation_scenarios(
    data: ProblemData,
    runtime: RuntimeConfig,
    capacity_rng: random.Random,
) -> ScenarioSet:
    """Create the scenario set used for CDF and policy comparisons.

    By default this reproduces the original evaluation logic: all products use
    Increase/Flat/Decrease states, so there are 3^4 = 81 equally weighted
    scenarios. `evaluation_include_no_launch=True` can be used for a fully
    probability-consistent launch evaluation.
    """

    scenario_ids, scenario_map, probabilities = build_joint_scenarios(
        data,
        include_no_launch=runtime.evaluation_include_no_launch,
        exhaustive=runtime.evaluation_exhaustive,
        sample_size=runtime.evaluation_sample_size,
        random_seed=runtime.evaluation_seed,
        weighted_sampling=runtime.evaluation_include_no_launch,
    )

    # Preserve the original uniform evaluation weighting when NoLaunch is not
    # included. This makes the CDF an empirical distribution over the 81 states.
    if not runtime.evaluation_include_no_launch:
        probabilities = {
            scenario_id: 1.0 / len(scenario_ids)
            for scenario_id in scenario_ids
        }

    demand = build_scenario_demand(data, scenario_map)
    capacity = build_scenario_capacity(data, scenario_ids, capacity_rng)

    return ScenarioSet(
        scenario_ids=scenario_ids,
        scenario_map=scenario_map,
        probabilities=probabilities,
        demand=demand,
        capacity=capacity,
    )

OKABE_ITO = [
    "#009E73",
    "#0072B2",
    "#E69F00",
    "#D55E00",
    "#0072B2",
    "#D55E00",
    "#CC79A7",
    "#000000",
]








def make_cdf(values: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    """Return sorted observations and their empirical cumulative probability."""

    sorted_values = np.sort(np.asarray(values, dtype=float))
    cumulative_probability = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
    return sorted_values, cumulative_probability

# Centralized model colors.  The five original models exactly preserve the
# palette used in rosr_modular_cases; reviewer extensions use distinct colors.
MODEL_COLORS = {
    # Original baseline models.
    "Multi-period": "#E69F00",
    "Stochastic multi-period": "#E69F00",
    "Two-stage multi-period": "#E69F00",
    "Two-stage multi-period (multistage recourse)": "#E69F00",
    "Start-period": "#56B4E9",
    "Stochastic start-period": "#56B4E9",
    "Two-stage start-period": "#56B4E9",
    "Two-stage start-period (multistage recourse)": "#56B4E9",
    "Long-chain": "#009E73",
    "Decision-rule": "#D55E00",
    "Perfect Information": "#0072B2",
    "Perfect information": "#0072B2",

    # Reviewer-requested extensions.
    "Multistage stochastic": "#6A3D9A",
    "Start-period RO": "#8C564B",
    "Multi-period RO": "#E7298A",
    "Start-period DRO": "#7F7F7F",
    "Multi-period DRO": "#17BECF",
}

BASELINE_MODEL_ORDER = [
    "Multi-period",
    "Start-period",
    "Long-chain",
    "Decision-rule",
    "Perfect Information",
]

EXTENSION_MODEL_ORDER = BASELINE_MODEL_ORDER + [
    "Start-period RO",
    "Multi-period RO",
    "Start-period DRO",
    "Multi-period DRO",
]

MULTISTAGE_COMPARISON_ORDER = [
    "Multistage stochastic",
    "Two-stage multi-period (multistage recourse)",
    "Two-stage start-period (multistage recourse)",
]


def solve_multi_period_design(
    data: ProblemData,
    scenarios: ScenarioSet,
    env: gp.Env,
    runtime: RuntimeConfig,
) -> MultiPeriodDesign:
    """Solve the multi-period two-stage flexibility-investment model."""

    model = gp.Model("Flexibility-Investment Multi-Period Two-Stage", env=env)
    configure_model(
        model,
        output_flag=runtime.output_flag,
        threads=runtime.threads,
        time_limit=runtime.planning_time_limit,
    )

    # First-stage machine, arc, and machine-start decisions.
    machine_on = model.addVars(data.M, data.T_I, vtype=GRB.BINARY, name="m")
    arc_on = model.addVars(data.M, data.P, data.T_I, vtype=GRB.BINARY, name="a")
    machine_start = model.addVars(data.M, data.T_I, vtype=GRB.BINARY, name="u")

    # Scenario-dependent operating decisions.
    production = model.addVars(
        data.M,
        data.P,
        scenarios.scenario_ids,
        data.T_V,
        vtype=GRB.CONTINUOUS,
        lb=0.0,
        name="x",
    )
    frozen_share = model.addVars(
        data.M,
        data.P,
        scenarios.scenario_ids,
        lb=0.0,
        ub=1.0,
        name="y",
    )

    # Expected discounted gross profit.
    profit_term = gp.quicksum(
        discount_factor(data, t)
        * scenarios.probabilities[scenario_id]
        * data.unit_profit[machine, product]
        * production[machine, product, scenario_id, t]
        for machine, product, scenario_id, t in itertools.product(
            data.M,
            data.P,
            scenarios.scenario_ids,
            data.T_V,
        )
    )

    # Present value of life-limited fixed OPEX for each possible start time.
    opex_coefficients: Dict[Tuple[str, int], float] = {}
    for machine in data.M:
        for start_time in data.T_I:
            last_active_time = min(
                start_time + data.lifetime[machine] - 1,
                data.nu,
            )
            opex_coefficients[machine, start_time] = (
                data.fixed_opex[machine]
                * sum(
                    discount_factor(data, t)
                    for t in range(start_time, last_active_time + 1)
                )
            )

    fixed_opex_term = gp.quicksum(
        opex_coefficients[machine, start_time]
        * machine_start[machine, start_time]
        for machine in data.M
        for start_time in data.T_I
    )

    # Machine CAPEX is paid when the machine starts.
    machine_capex_term = gp.quicksum(
        data.machine_capex[machine]
        / (1.0 + data.discount_rate) ** start_time
        * machine_start[machine, start_time]
        for machine in data.M
        for start_time in data.T_I
    )

    # Arc CAPEX is paid only when an arc changes from 0 to 1.
    arc_capex_term = gp.LinExpr()
    for machine in data.M:
        for product in data.P:
            for t in data.T_I:
                previous_arc = 0.0 if t == 0 else arc_on[machine, product, t - 1]
                arc_capex_term += (
                    data.arc_capex[machine, product]
                    / (1.0 + data.discount_rate) ** t
                    * (arc_on[machine, product, t] - previous_arc)
                )

    model.setObjective(
        profit_term - fixed_opex_term - machine_capex_term - arc_capex_term,
        GRB.MAXIMIZE,
    )

    # A machine starts exactly when its on/off indicator rises.
    model.addConstrs(
        (
            machine_on[machine, 0] == machine_start[machine, 0]
            for machine in data.M
        ),
        name="startup_period_0",
    )
    model.addConstrs(
        (
            machine_on[machine, t] - machine_on[machine, t - 1]
            == machine_start[machine, t]
            for machine in data.M
            for t in data.T_I
            if t > 0
        ),
        name="startup",
    )

    # Arcs persist once installed and require an installed machine.
    model.addConstrs(
        (
            arc_on[machine, product, t]
            >= arc_on[machine, product, t - 1]
            for machine in data.M
            for product in data.P
            for t in data.T_I
            if t > 0
        ),
        name="arc_monotonicity",
    )
    model.addConstrs(
        (
            arc_on[machine, product, t] <= machine_on[machine, t]
            for machine in data.M
            for product in data.P
            for t in data.T_I
        ),
        name="arc_requires_machine",
    )

    # Beyond the investment window, a resource may produce only while a start
    # remains within its physical lifetime.
    for machine in data.M:
        for scenario_id in scenarios.scenario_ids:
            for t in range(data.theta + 1, data.nu + 1):
                valid_starts = range(
                    max(0, t - data.lifetime[machine] + 1),
                    min(t, data.theta) + 1,
                )
                model.addConstr(
                    gp.quicksum(
                        production[machine, product, scenario_id, t]
                        for product in data.P
                    )
                    <= scenarios.capacity[machine, scenario_id, t]
                    * gp.quicksum(
                        machine_start[machine, start_time]
                        for start_time in valid_starts
                    ),
                    name=f"lifetime_capacity_{machine}_{scenario_id}_{t}",
                )

    # Product demand is an upper bound; unmet demand is allowed.
    model.addConstrs(
        (
            gp.quicksum(
                production[machine, product, scenario_id, t]
                for machine in data.M
            )
            <= scenarios.demand[product, scenario_id, t]
            for product in data.P
            for scenario_id in scenarios.scenario_ids
            for t in data.T_V
        ),
        name="demand",
    )

    # Resource capacity follows the time-phased machine installation decision.
    model.addConstrs(
        (
            gp.quicksum(
                production[machine, product, scenario_id, t]
                for product in data.P
            )
            <= scenarios.capacity[machine, scenario_id, t]
            * (
                machine_on[machine, t]
                if t <= data.theta
                else machine_on[machine, data.theta]
            )
            for machine in data.M
            for scenario_id in scenarios.scenario_ids
            for t in data.T_V
        ),
        name="capacity",
    )

    # Production requires an installed machine-product arc.
    model.addConstrs(
        (
            production[machine, product, scenario_id, t]
            <= scenarios.capacity[machine, scenario_id, t]
            * (
                arc_on[machine, product, t]
                if t <= data.theta
                else arc_on[machine, product, data.theta]
            )
            for machine in data.M
            for product in data.P
            for scenario_id in scenarios.scenario_ids
            for t in data.T_V
        ),
        name="arc_link",
    )

    # The original model freezes each resource's product share after theta.
    model.addConstrs(
        (
            production[machine, product, scenario_id, data.theta]
            <= frozen_share[machine, product, scenario_id]
            * scenarios.demand[product, scenario_id, data.theta]
            for machine in data.M
            for product in data.P
            for scenario_id in scenarios.scenario_ids
        ),
        name="share_definition",
    )
    model.addConstrs(
        (
            production[machine, product, scenario_id, t]
            <= frozen_share[machine, product, scenario_id]
            * scenarios.demand[product, scenario_id, t]
            for machine in data.M
            for product in data.P
            for scenario_id in scenarios.scenario_ids
            for t in data.T_V
            if t > data.theta
        ),
        name="share_frozen",
    )
    model.addConstrs(
        (
            gp.quicksum(
                frozen_share[machine, product, scenario_id]
                for machine in data.M
            )
            == 1.0
            for product in data.P
            for scenario_id in scenarios.scenario_ids
        ),
        name="share_sum",
    )

    objective_value = optimize_and_check(model)

    return MultiPeriodDesign(
        objective_value=objective_value,
        machine_on={key: float(variable.X) for key, variable in machine_on.items()},
        machine_start={key: float(variable.X) for key, variable in machine_start.items()},
        arc_on={key: float(variable.X) for key, variable in arc_on.items()},
        production={key: float(variable.X) for key, variable in production.items()},
    )

def solve_start_period_design(
    data: ProblemData,
    scenarios: ScenarioSet,
    env: gp.Env,
    runtime: RuntimeConfig,
) -> StartPeriodDesign:
    """Solve the design model when all investments must occur in period zero."""

    model = gp.Model("Flexibility-Investment Start-Period Two-Stage", env=env)
    configure_model(
        model,
        output_flag=runtime.output_flag,
        threads=runtime.threads,
        time_limit=runtime.start_period_time_limit,
    )

    machine_on = model.addVars(data.M, vtype=GRB.BINARY, name="m")
    arc_on = model.addVars(data.M, data.P, vtype=GRB.BINARY, name="a")
    machine_start = model.addVars(data.M, vtype=GRB.BINARY, name="u")
    production = model.addVars(
        data.M,
        data.P,
        scenarios.scenario_ids,
        data.T_V,
        vtype=GRB.CONTINUOUS,
        lb=0.0,
        name="x",
    )
    frozen_share = model.addVars(
        data.M,
        data.P,
        scenarios.scenario_ids,
        lb=0.0,
        ub=1.0,
        name="y",
    )

    profit_term = gp.quicksum(
        discount_factor(data, t)
        * scenarios.probabilities[scenario_id]
        * data.unit_profit[machine, product]
        * production[machine, product, scenario_id, t]
        for machine, product, scenario_id, t in itertools.product(
            data.M,
            data.P,
            scenarios.scenario_ids,
            data.T_V,
        )
    )

    fixed_opex_coefficients: Dict[str, float] = {}
    for machine in data.M:
        last_active_time = min(data.lifetime[machine] - 1, data.nu)
        fixed_opex_coefficients[machine] = (
            data.fixed_opex[machine]
            * sum(
                discount_factor(data, t)
                for t in range(last_active_time + 1)
            )
        )

    fixed_opex_term = gp.quicksum(
        fixed_opex_coefficients[machine] * machine_start[machine]
        for machine in data.M
    )
    machine_capex_term = gp.quicksum(
        data.machine_capex[machine] * machine_start[machine]
        for machine in data.M
    )
    arc_capex_term = gp.quicksum(
        data.arc_capex[machine, product] * arc_on[machine, product]
        for machine in data.M
        for product in data.P
    )

    model.setObjective(
        profit_term - fixed_opex_term - machine_capex_term - arc_capex_term,
        GRB.MAXIMIZE,
    )

    model.addConstrs(
        (machine_on[machine] == machine_start[machine] for machine in data.M),
        name="startup_period_0",
    )
    model.addConstrs(
        (
            arc_on[machine, product] <= machine_on[machine]
            for machine in data.M
            for product in data.P
        ),
        name="arc_requires_machine",
    )

    # Start-period resources are inactive after their physical lifetime.
    for machine in data.M:
        for scenario_id in scenarios.scenario_ids:
            for t in data.T_V:
                if t > data.lifetime[machine] - 1:
                    model.addConstr(
                        gp.quicksum(
                            production[machine, product, scenario_id, t]
                            for product in data.P
                        )
                        <= 0.0,
                        name=f"lifetime_capacity_{machine}_{scenario_id}_{t}",
                    )

    model.addConstrs(
        (
            gp.quicksum(
                production[machine, product, scenario_id, t]
                for machine in data.M
            )
            <= scenarios.demand[product, scenario_id, t]
            for product in data.P
            for scenario_id in scenarios.scenario_ids
            for t in data.T_V
        ),
        name="demand",
    )
    model.addConstrs(
        (
            gp.quicksum(
                production[machine, product, scenario_id, t]
                for product in data.P
            )
            <= scenarios.capacity[machine, scenario_id, t] * machine_on[machine]
            for machine in data.M
            for scenario_id in scenarios.scenario_ids
            for t in data.T_V
        ),
        name="capacity",
    )
    model.addConstrs(
        (
            production[machine, product, scenario_id, t]
            <= scenarios.capacity[machine, scenario_id, t]
            * arc_on[machine, product]
            for machine in data.M
            for product in data.P
            for scenario_id in scenarios.scenario_ids
            for t in data.T_V
        ),
        name="arc_link",
    )
    model.addConstrs(
        (
            production[machine, product, scenario_id, data.theta]
            <= frozen_share[machine, product, scenario_id]
            * scenarios.demand[product, scenario_id, data.theta]
            for machine in data.M
            for product in data.P
            for scenario_id in scenarios.scenario_ids
        ),
        name="share_definition",
    )
    model.addConstrs(
        (
            production[machine, product, scenario_id, t]
            <= frozen_share[machine, product, scenario_id]
            * scenarios.demand[product, scenario_id, t]
            for machine in data.M
            for product in data.P
            for scenario_id in scenarios.scenario_ids
            for t in data.T_V
            if t > data.theta
        ),
        name="share_frozen",
    )
    model.addConstrs(
        (
            gp.quicksum(
                frozen_share[machine, product, scenario_id]
                for machine in data.M
            )
            == 1.0
            for product in data.P
            for scenario_id in scenarios.scenario_ids
        ),
        name="share_sum",
    )

    objective_value = optimize_and_check(model)

    return StartPeriodDesign(
        objective_value=objective_value,
        machine_on={key: float(variable.X) for key, variable in machine_on.items()},
        machine_start={key: float(variable.X) for key, variable in machine_start.items()},
        arc_on={key: float(variable.X) for key, variable in arc_on.items()},
    )

def build_long_chain_design(
    data: ProblemData,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[Tuple[str, str], float]]:
    """Return the fixed four-resource chain/cycle used in the original scripts."""

    if len(data.P) != 4:
        raise ValueError("The fixed long-chain benchmark requires exactly four products.")

    machine_on = {machine: 0.0 for machine in data.M}
    machine_start = {machine: 0.0 for machine in data.M}
    arc_on = {(machine, product): 0.0 for machine in data.M for product in data.P}

    # The first four resources are installed at t=0.
    for machine in data.M[:4]:
        machine_on[machine] = 1.0
        machine_start[machine] = 1.0

    # Preserve the exact architecture hard-coded in all four original files.
    long_chain_arcs = [
        (data.M[0], data.P[0]),
        (data.M[0], data.P[3]),
        (data.M[1], data.P[0]),
        (data.M[1], data.P[1]),
        (data.M[2], data.P[1]),
        (data.M[2], data.P[2]),
        (data.M[3], data.P[2]),
        (data.M[3], data.P[3]),
    ]
    for arc in long_chain_arcs:
        arc_on[arc] = 1.0

    return machine_on, machine_start, arc_on

def multi_period_cash_flow(
    data: ProblemData,
    scenarios: ScenarioSet,
    design: MultiPeriodDesign,
) -> Dict[str, Dict[int, float]]:
    """Compute the same undiscounted period cash-flow components as the originals."""

    gross_profit = {t: 0.0 for t in range(data.nu + 1)}
    fixed_operating_cost = {t: 0.0 for t in range(data.nu + 1)}
    machine_investment_cost = {t: 0.0 for t in range(data.nu + 1)}
    arc_investment_cost = {t: 0.0 for t in range(data.nu + 1)}

    # The original code reports period-0 gross profit as zero and shifts
    # production period t to displayed period t+1 when available.
    for (machine, product, scenario_id, t), quantity in design.production.items():
        display_period = t + 1
        if display_period <= data.nu:
            gross_profit[display_period] += (
                scenarios.probabilities[scenario_id]
                * data.unit_profit[machine, product]
                * quantity
            )

    # A resource incurs fixed OPEX in each period covered by any start decision.
    for t in data.T_V:
        for machine in data.M:
            active = any(
                design.machine_start[machine, start_time] > 0.5
                and start_time <= t < start_time + data.lifetime[machine]
                for start_time in data.T_I
            )
            if active:
                fixed_operating_cost[t] -= data.fixed_opex[machine]

    # Machine and arc investment costs occur in their installation period.
    for machine in data.M:
        for t in data.T_I:
            machine_investment_cost[t] -= (
                data.machine_capex[machine]
                * design.machine_start[machine, t]
            )

    for machine in data.M:
        for product in data.P:
            previous_arc = 0.0
            for t in data.T_I:
                current_arc = design.arc_on[machine, product, t]
                arc_investment_cost[t] -= (
                    data.arc_capex[machine, product]
                    * (current_arc - previous_arc)
                )
                previous_arc = current_arc

    net_profit = {
        t: (
            gross_profit[t]
            + fixed_operating_cost[t]
            + machine_investment_cost[t]
            + arc_investment_cost[t]
        )
        for t in range(data.nu + 1)
    }

    return {
        "Gross Profit": gross_profit,
        "Fixed Operating Cost": fixed_operating_cost,
        "Machine Investment Cost": machine_investment_cost,
        "Arc Investment Cost": arc_investment_cost,
        "Net Profit": net_profit,
    }




def final_multi_period_arcs(
    data: ProblemData,
    design: MultiPeriodDesign,
) -> List[Tuple[str, str]]:
    """Return installed arcs at the end of the investment window."""

    return [
        (machine, product)
        for machine in data.M
        for product in data.P
        if design.arc_on[machine, product, data.theta] > 0.5
    ]


def report_new_investments(data: ProblemData, design: MultiPeriodDesign) -> None:
    """Print newly added machines and arcs by investment period."""

    baseline_arcs = set(data.pre_installed_arcs)
    baseline_machines = set(data.pre_installed_machines)

    arcs_by_time = {
        t: {
            (machine, product)
            for machine in data.M
            for product in data.P
            if design.arc_on[machine, product, t] > 0.5
        }
        for t in data.T_I
    }
    machines_by_time = {
        t: {
            machine
            for machine in data.M
            if design.machine_on[machine, t] > 0.5
        }
        for t in data.T_I
    }

    print("\nNew multi-period arc investments")
    for t in data.T_I:
        previous_arcs = baseline_arcs if t == 0 else arcs_by_time[t - 1]
        print(f"  Period {t}: {sorted(arcs_by_time[t] - previous_arcs)}")

    print("\nNew multi-period machine investments")
    for t in data.T_I:
        previous_machines = baseline_machines if t == 0 else machines_by_time[t - 1]
        print(f"  Period {t}: {sorted(machines_by_time[t] - previous_machines)}")

def fixed_costs_multi_period(
    data: ProblemData,
    design: MultiPeriodDesign,
) -> Tuple[float, float, float]:
    """PV OPEX, machine CAPEX, and arc CAPEX for a multi-period design."""

    opex = 0.0
    for machine in data.M:
        for start_time in data.T_I:
            last_active_time = min(
                start_time + data.lifetime[machine] - 1,
                data.nu,
            )
            coefficient = data.fixed_opex[machine] * sum(
                discount_factor(data, t)
                for t in range(start_time, last_active_time + 1)
            )
            opex += coefficient * design.machine_start[machine, start_time]

    machine_capex = sum(
        data.machine_capex[machine]
        / (1.0 + data.discount_rate) ** start_time
        * design.machine_start[machine, start_time]
        for machine in data.M
        for start_time in data.T_I
    )

    arc_capex = 0.0
    for machine in data.M:
        for product in data.P:
            previous_arc = 0.0
            for t in data.T_I:
                current_arc = design.arc_on[machine, product, t]
                arc_capex += (
                    data.arc_capex[machine, product]
                    / (1.0 + data.discount_rate) ** t
                    * (current_arc - previous_arc)
                )
                previous_arc = current_arc

    return opex, machine_capex, arc_capex


def fixed_costs_start_period(
    data: ProblemData,
    machine_start: Mapping[str, float],
    arc_on: Mapping[Tuple[str, str], float],
) -> Tuple[float, float, float]:
    """PV fixed costs for a design installed entirely in period zero."""

    opex = 0.0
    for machine in data.M:
        last_active_time = min(data.lifetime[machine] - 1, data.nu)
        coefficient = data.fixed_opex[machine] * sum(
            discount_factor(data, t)
            for t in range(last_active_time + 1)
        )
        opex += coefficient * machine_start[machine]

    machine_capex = sum(
        data.machine_capex[machine] * machine_start[machine]
        for machine in data.M
    )
    arc_capex = sum(
        data.arc_capex[machine, product] * arc_on[machine, product]
        for machine in data.M
        for product in data.P
    )

    return opex, machine_capex, arc_capex


def evaluate_multi_period_scenario(
    data: ProblemData,
    scenarios: ScenarioSet,
    scenario_id: int,
    design: MultiPeriodDesign,
    env: gp.Env,
    runtime: RuntimeConfig,
) -> float:
    """Re-optimize operations for one scenario with the multi-period design fixed."""

    model = gp.Model(f"Evaluate Multi-Period Scenario {scenario_id}", env=env)
    configure_model(
        model,
        output_flag=runtime.evaluation_output_flag,
        threads=runtime.threads,
        time_limit=runtime.evaluation_time_limit,
    )

    production = model.addVars(data.M, data.P, data.T_V, lb=0.0, name="x")
    frozen_share = model.addVars(data.M, data.P, lb=0.0, ub=1.0, name="y")

    revenue = gp.quicksum(
        discount_factor(data, t)
        * data.unit_profit[machine, product]
        * production[machine, product, t]
        for machine in data.M
        for product in data.P
        for t in data.T_V
    )
    opex, machine_capex, arc_capex = fixed_costs_multi_period(data, design)
    model.setObjective(revenue - opex - machine_capex - arc_capex, GRB.MAXIMIZE)

    model.addConstrs(
        (
            gp.quicksum(production[machine, product, t] for machine in data.M)
            <= scenarios.demand[product, scenario_id, t]
            for product in data.P
            for t in data.T_V
        ),
        name="demand",
    )
    model.addConstrs(
        (
            gp.quicksum(production[machine, product, t] for product in data.P)
            <= scenarios.capacity[machine, scenario_id, t]
            * (
                design.machine_on[machine, t]
                if t <= data.theta
                else design.machine_on[machine, data.theta]
            )
            for machine in data.M
            for t in data.T_V
        ),
        name="capacity",
    )

    for machine in data.M:
        for t in range(data.theta + 1, data.nu + 1):
            valid_start_sum = sum(
                design.machine_start[machine, start_time]
                for start_time in range(
                    max(0, t - data.lifetime[machine] + 1),
                    min(t, data.theta) + 1,
                )
            )
            model.addConstr(
                gp.quicksum(
                    production[machine, product, t]
                    for product in data.P
                )
                <= scenarios.capacity[machine, scenario_id, t]
                * valid_start_sum,
                name=f"lifetime_capacity_{machine}_{t}",
            )

    model.addConstrs(
        (
            production[machine, product, t]
            <= scenarios.capacity[machine, scenario_id, t]
            * (
                design.arc_on[machine, product, t]
                if t <= data.theta
                else design.arc_on[machine, product, data.theta]
            )
            for machine in data.M
            for product in data.P
            for t in data.T_V
        ),
        name="arc_link",
    )
    model.addConstrs(
        (
            production[machine, product, data.theta]
            <= frozen_share[machine, product]
            * scenarios.demand[product, scenario_id, data.theta]
            for machine in data.M
            for product in data.P
        ),
        name="share_definition",
    )
    model.addConstrs(
        (
            production[machine, product, t]
            <= frozen_share[machine, product]
            * scenarios.demand[product, scenario_id, t]
            for machine in data.M
            for product in data.P
            for t in data.T_V
            if t > data.theta
        ),
        name="share_frozen",
    )
    model.addConstrs(
        (
            gp.quicksum(frozen_share[machine, product] for machine in data.M)
            == 1.0
            for product in data.P
        ),
        name="share_sum",
    )

    return optimize_and_check(model)


def evaluate_start_period_scenario(
    data: ProblemData,
    scenarios: ScenarioSet,
    scenario_id: int,
    machine_on: Mapping[str, float],
    machine_start: Mapping[str, float],
    arc_on: Mapping[Tuple[str, str], float],
    env: gp.Env,
    runtime: RuntimeConfig,
    *,
    tag: str,
) -> float:
    """Evaluate one fixed start-at-zero design in one scenario."""

    model = gp.Model(f"Evaluate {tag} Scenario {scenario_id}", env=env)
    configure_model(
        model,
        output_flag=runtime.evaluation_output_flag,
        threads=runtime.threads,
        time_limit=runtime.evaluation_time_limit,
    )

    production = model.addVars(data.M, data.P, data.T_V, lb=0.0, name="x")
    frozen_share = model.addVars(data.M, data.P, lb=0.0, ub=1.0, name="y")

    revenue = gp.quicksum(
        discount_factor(data, t)
        * data.unit_profit[machine, product]
        * production[machine, product, t]
        for machine in data.M
        for product in data.P
        for t in data.T_V
    )
    opex, machine_capex, arc_capex = fixed_costs_start_period(
        data,
        machine_start,
        arc_on,
    )
    model.setObjective(revenue - opex - machine_capex - arc_capex, GRB.MAXIMIZE)

    model.addConstrs(
        (
            gp.quicksum(production[machine, product, t] for machine in data.M)
            <= scenarios.demand[product, scenario_id, t]
            for product in data.P
            for t in data.T_V
        ),
        name="demand",
    )

    for machine in data.M:
        for t in data.T_V:
            active = 1.0 if t <= data.lifetime[machine] - 1 else 0.0
            model.addConstr(
                gp.quicksum(
                    production[machine, product, t]
                    for product in data.P
                )
                <= scenarios.capacity[machine, scenario_id, t]
                * machine_on[machine]
                * active,
                name=f"capacity_{machine}_{t}",
            )

    model.addConstrs(
        (
            production[machine, product, t]
            <= scenarios.capacity[machine, scenario_id, t]
            * arc_on[machine, product]
            for machine in data.M
            for product in data.P
            for t in data.T_V
        ),
        name="arc_link",
    )
    model.addConstrs(
        (
            production[machine, product, data.theta]
            <= frozen_share[machine, product]
            * scenarios.demand[product, scenario_id, data.theta]
            for machine in data.M
            for product in data.P
        ),
        name="share_definition",
    )
    model.addConstrs(
        (
            production[machine, product, t]
            <= frozen_share[machine, product]
            * scenarios.demand[product, scenario_id, t]
            for machine in data.M
            for product in data.P
            for t in data.T_V
            if t > data.theta
        ),
        name="share_frozen",
    )
    model.addConstrs(
        (
            gp.quicksum(frozen_share[machine, product] for machine in data.M)
            == 1.0
            for product in data.P
        ),
        name="share_sum",
    )

    return optimize_and_check(model)

def machine_alive_at(
    data: ProblemData,
    machine: str,
    t: int,
    machine_start: Mapping[Tuple[str, int], float],
) -> bool:
    """Return whether a machine is alive at period t under a start schedule."""

    return any(
        machine_start.get((machine, start_time), 0.0) > 0.5
        and start_time <= t < start_time + data.lifetime[machine]
        for start_time in data.T_I
    )


def machine_path_from_starts(
    data: ProblemData,
    machine_start: Mapping[Tuple[str, int], float],
) -> Dict[Tuple[str, int], float]:
    """Convert machine starts to the machine-alive path within the investment window."""

    return {
        (machine, t): float(machine_alive_at(data, machine, t, machine_start))
        for machine in data.M
        for t in data.T_I
    }


def product_capacity_at_time(
    data: ProblemData,
    scenarios: ScenarioSet,
    scenario_id: int,
    t: int,
    arcs: Iterable[Tuple[str, str]],
    alive_machines: Iterable[str],
) -> Dict[str, float]:
    """Available capacity for each product from alive, connected resources."""

    alive_set = set(alive_machines)
    capacity_by_product = {product: 0.0 for product in data.P}

    for machine, product in arcs:
        if machine in alive_set:
            capacity_by_product[product] += scenarios.capacity[
                machine,
                scenario_id,
                t,
            ]

    return capacity_by_product


def least_flexible_resource(
    data: ProblemData,
    alive_machines: Iterable[str],
    arcs: Iterable[Tuple[str, str]],
) -> Optional[str]:
    """Choose the alive resource with the fewest installed product arcs."""

    alive_list = list(alive_machines)
    if not alive_list:
        return None

    arc_counts = {machine: 0 for machine in alive_list}
    for machine, _ in arcs:
        if machine in arc_counts:
            arc_counts[machine] += 1

    machine_order = {machine: index for index, machine in enumerate(data.M)}
    return min(
        alive_list,
        key=lambda machine: (arc_counts[machine], machine_order[machine]),
    )


def most_constrained_product(
    data: ProblemData,
    scenarios: ScenarioSet,
    scenario_id: int,
    t: int,
    arcs: Iterable[Tuple[str, str]],
    alive_machines: Iterable[str],
) -> Tuple[Optional[str], float]:
    """Return the product with the largest positive demand-capacity gap."""

    capacity_by_product = product_capacity_at_time(
        data,
        scenarios,
        scenario_id,
        t,
        arcs,
        alive_machines,
    )

    selected_product: Optional[str] = None
    selected_gap = 0.0

    for product in data.P:
        gap = (
            scenarios.demand[product, scenario_id, t]
            - capacity_by_product[product]
        )
        if gap > selected_gap:
            selected_product = product
            selected_gap = gap

    return selected_product, selected_gap


def build_decision_rule_schedule(
    data: ProblemData,
    scenarios: ScenarioSet,
    scenario_id: int,
    *,
    verbose: bool = False,
) -> Tuple[
    Dict[Tuple[str, int], float],
    Dict[Tuple[str, int], float],
    Dict[Tuple[str, str, int], float],
]:
    """Build the same reactive investment schedule as the original code."""

    tolerance = 1e-9

    machine_start = {
        (machine, t): 0.0
        for machine in data.M
        for t in data.T_I
    }
    for machine in data.pre_installed_machines:
        machine_start[machine, 0] = 1.0

    arc_on = {
        (machine, product, t): 0.0
        for machine in data.M
        for product in data.P
        for t in data.T_I
    }
    for machine, product in data.pre_installed_arcs:
        arc_on[machine, product, 0] = 1.0

    arcs_by_time: Dict[int, set[Tuple[str, str]]] = {
        0: set(data.pre_installed_arcs)
    }
    next_machine_index = len(data.pre_installed_machines)

    for t in data.T_I:
        alive_machines = {
            machine
            for machine in data.M
            if machine_alive_at(data, machine, t, machine_start)
        }

        if t > 0:
            arcs_by_time[t] = set(arcs_by_time[t - 1])
            for machine in data.M:
                for product in data.P:
                    arc_on[machine, product, t] = arc_on[
                        machine,
                        product,
                        t - 1,
                    ]

            previous_alive_machines = {
                machine
                for machine in data.M
                if machine_alive_at(data, machine, t - 1, machine_start)
            }
            previous_capacity = product_capacity_at_time(
                data,
                scenarios,
                scenario_id,
                t - 1,
                arcs_by_time[t - 1],
                previous_alive_machines,
            )

            # Add one arc for every product that was short in the prior period.
            for product in data.P:
                prior_shortage = (
                    scenarios.demand[product, scenario_id, t - 1]
                    > previous_capacity[product] + tolerance
                )
                if prior_shortage:
                    candidate_machine = least_flexible_resource(
                        data,
                        alive_machines,
                        arcs_by_time[t - 1],
                    )
                    candidate_arc = (candidate_machine, product)
                    if (
                        candidate_machine is not None
                        and candidate_arc not in arcs_by_time[t]
                    ):
                        arcs_by_time[t].add(candidate_arc)
                        arc_on[candidate_machine, product, t] = 1.0
                        if verbose:
                            print(
                                f"t={t}: added arc {candidate_arc} "
                                "after prior-period shortage"
                            )

        total_demand = sum(
            scenarios.demand[product, scenario_id, t]
            for product in data.P
        )
        total_capacity = sum(
            scenarios.capacity[machine, scenario_id, t]
            for machine in alive_machines
        )

        # Start at most one new resource in each investment period.
        if (
            total_demand > total_capacity + tolerance
            and next_machine_index < len(data.M)
        ):
            new_machine = data.M[next_machine_index]
            machine_start[new_machine, t] = 1.0
            next_machine_index += 1
            alive_machines.add(new_machine)

            constrained_product, gap = most_constrained_product(
                data,
                scenarios,
                scenario_id,
                t,
                arcs_by_time[t],
                alive_machines,
            )
            if constrained_product is not None and gap > tolerance:
                arcs_by_time[t].add((new_machine, constrained_product))
                arc_on[new_machine, constrained_product, t] = 1.0

            if verbose:
                print(
                    f"t={t}: started {new_machine} and connected it to "
                    f"{constrained_product}"
                )

    machine_on = machine_path_from_starts(data, machine_start)
    return machine_start, machine_on, arc_on


def fixed_costs_timed_schedule(
    data: ProblemData,
    machine_start: Mapping[Tuple[str, int], float],
    arc_on: Mapping[Tuple[str, str, int], float],
) -> Tuple[float, float, float]:
    """PV fixed costs for an arbitrary time-phased schedule."""

    opex = 0.0
    for machine in data.M:
        for start_time in data.T_I:
            if machine_start.get((machine, start_time), 0.0) > 0.5:
                last_active_time = min(
                    start_time + data.lifetime[machine] - 1,
                    data.nu,
                )
                opex += data.fixed_opex[machine] * sum(
                    discount_factor(data, t)
                    for t in range(start_time, last_active_time + 1)
                )

    machine_capex = sum(
        data.machine_capex[machine]
        / (1.0 + data.discount_rate) ** start_time
        * machine_start.get((machine, start_time), 0.0)
        for machine in data.M
        for start_time in data.T_I
    )

    arc_capex = 0.0
    for machine in data.M:
        for product in data.P:
            previous_arc = 0.0
            for t in data.T_I:
                current_arc = arc_on.get((machine, product, t), 0.0)
                arc_capex += (
                    data.arc_capex[machine, product]
                    / (1.0 + data.discount_rate) ** t
                    * (current_arc - previous_arc)
                )
                previous_arc = current_arc

    return opex, machine_capex, arc_capex


def evaluate_timed_schedule_scenario(
    data: ProblemData,
    scenarios: ScenarioSet,
    scenario_id: int,
    machine_start: Mapping[Tuple[str, int], float],
    arc_on: Mapping[Tuple[str, str, int], float],
    env: gp.Env,
    runtime: RuntimeConfig,
    *,
    tag: str,
) -> float:
    """Evaluate one arbitrary time-phased schedule in one scenario."""

    machine_on = machine_path_from_starts(data, machine_start)
    opex, machine_capex, arc_capex = fixed_costs_timed_schedule(
        data,
        machine_start,
        arc_on,
    )

    model = gp.Model(f"Evaluate {tag} Scenario {scenario_id}", env=env)
    configure_model(
        model,
        output_flag=runtime.evaluation_output_flag,
        threads=runtime.threads,
        time_limit=runtime.evaluation_time_limit,
    )

    production = model.addVars(data.M, data.P, data.T_V, lb=0.0, name="x")
    frozen_share = model.addVars(data.M, data.P, lb=0.0, ub=1.0, name="y")

    revenue = gp.quicksum(
        discount_factor(data, t)
        * data.unit_profit[machine, product]
        * production[machine, product, t]
        for machine in data.M
        for product in data.P
        for t in data.T_V
    )
    model.setObjective(revenue - opex - machine_capex - arc_capex, GRB.MAXIMIZE)

    model.addConstrs(
        (
            gp.quicksum(production[machine, product, t] for machine in data.M)
            <= scenarios.demand[product, scenario_id, t]
            for product in data.P
            for t in data.T_V
        ),
        name="demand",
    )
    model.addConstrs(
        (
            gp.quicksum(production[machine, product, t] for product in data.P)
            <= scenarios.capacity[machine, scenario_id, t]
            * (
                machine_on[machine, t]
                if t <= data.theta
                else machine_on[machine, data.theta]
            )
            for machine in data.M
            for t in data.T_V
        ),
        name="capacity",
    )

    for machine in data.M:
        for t in range(data.theta + 1, data.nu + 1):
            valid_start_sum = sum(
                machine_start.get((machine, start_time), 0.0)
                for start_time in range(
                    max(0, t - data.lifetime[machine] + 1),
                    min(t, data.theta) + 1,
                )
            )
            model.addConstr(
                gp.quicksum(
                    production[machine, product, t]
                    for product in data.P
                )
                <= scenarios.capacity[machine, scenario_id, t]
                * valid_start_sum,
                name=f"lifetime_capacity_{machine}_{t}",
            )

    model.addConstrs(
        (
            production[machine, product, t]
            <= scenarios.capacity[machine, scenario_id, t]
            * (
                arc_on[machine, product, t]
                if t <= data.theta
                else arc_on[machine, product, data.theta]
            )
            for machine in data.M
            for product in data.P
            for t in data.T_V
        ),
        name="arc_link",
    )
    model.addConstrs(
        (
            production[machine, product, data.theta]
            <= frozen_share[machine, product]
            * scenarios.demand[product, scenario_id, data.theta]
            for machine in data.M
            for product in data.P
        ),
        name="share_definition",
    )
    model.addConstrs(
        (
            production[machine, product, t]
            <= frozen_share[machine, product]
            * scenarios.demand[product, scenario_id, t]
            for machine in data.M
            for product in data.P
            for t in data.T_V
            if t > data.theta
        ),
        name="share_frozen",
    )
    model.addConstrs(
        (
            gp.quicksum(frozen_share[machine, product] for machine in data.M)
            == 1.0
            for product in data.P
        ),
        name="share_sum",
    )

    return optimize_and_check(model)

def optimize_perfect_information_scenario(
    data: ProblemData,
    scenarios: ScenarioSet,
    scenario_id: int,
    env: gp.Env,
    runtime: RuntimeConfig,
) -> float:
    """Optimize investment and production with perfect foresight of one scenario."""

    model = gp.Model(f"Perfect Information Scenario {scenario_id}", env=env)
    configure_model(
        model,
        output_flag=runtime.evaluation_output_flag,
        threads=runtime.threads,
        time_limit=runtime.perfect_information_time_limit,
    )

    machine_on = model.addVars(data.M, data.T_I, vtype=GRB.BINARY, name="m")
    arc_on = model.addVars(data.M, data.P, data.T_I, vtype=GRB.BINARY, name="a")
    machine_start = model.addVars(data.M, data.T_I, vtype=GRB.BINARY, name="u")
    production = model.addVars(data.M, data.P, data.T_V, lb=0.0, name="x")
    frozen_share = model.addVars(data.M, data.P, lb=0.0, ub=1.0, name="y")

    revenue = gp.quicksum(
        discount_factor(data, t)
        * data.unit_profit[machine, product]
        * production[machine, product, t]
        for machine in data.M
        for product in data.P
        for t in data.T_V
    )

    opex_coefficients: Dict[Tuple[str, int], float] = {}
    for machine in data.M:
        for start_time in data.T_I:
            last_active_time = min(
                start_time + data.lifetime[machine] - 1,
                data.nu,
            )
            opex_coefficients[machine, start_time] = (
                data.fixed_opex[machine]
                * sum(
                    discount_factor(data, t)
                    for t in range(start_time, last_active_time + 1)
                )
            )

    opex = gp.quicksum(
        opex_coefficients[machine, start_time]
        * machine_start[machine, start_time]
        for machine in data.M
        for start_time in data.T_I
    )
    machine_capex = gp.quicksum(
        data.machine_capex[machine]
        / (1.0 + data.discount_rate) ** t
        * machine_start[machine, t]
        for machine in data.M
        for t in data.T_I
    )

    arc_capex = gp.LinExpr()
    for machine in data.M:
        for product in data.P:
            for t in data.T_I:
                previous_arc = 0.0 if t == 0 else arc_on[machine, product, t - 1]
                arc_capex += (
                    data.arc_capex[machine, product]
                    / (1.0 + data.discount_rate) ** t
                    * (arc_on[machine, product, t] - previous_arc)
                )

    model.setObjective(revenue - opex - machine_capex - arc_capex, GRB.MAXIMIZE)

    model.addConstrs(
        (machine_on[machine, 0] == machine_start[machine, 0] for machine in data.M),
        name="startup_period_0",
    )
    model.addConstrs(
        (
            machine_on[machine, t] - machine_on[machine, t - 1]
            == machine_start[machine, t]
            for machine in data.M
            for t in data.T_I
            if t > 0
        ),
        name="startup",
    )
    model.addConstrs(
        (
            arc_on[machine, product, t]
            >= arc_on[machine, product, t - 1]
            for machine in data.M
            for product in data.P
            for t in data.T_I
            if t > 0
        ),
        name="arc_monotonicity",
    )
    model.addConstrs(
        (
            arc_on[machine, product, t] <= machine_on[machine, t]
            for machine in data.M
            for product in data.P
            for t in data.T_I
        ),
        name="arc_requires_machine",
    )

    for machine in data.M:
        for t in range(data.theta + 1, data.nu + 1):
            valid_starts = range(
                max(0, t - data.lifetime[machine] + 1),
                min(t, data.theta) + 1,
            )
            model.addConstr(
                gp.quicksum(
                    production[machine, product, t]
                    for product in data.P
                )
                <= scenarios.capacity[machine, scenario_id, t]
                * gp.quicksum(
                    machine_start[machine, start_time]
                    for start_time in valid_starts
                ),
                name=f"lifetime_capacity_{machine}_{t}",
            )

    model.addConstrs(
        (
            gp.quicksum(production[machine, product, t] for machine in data.M)
            <= scenarios.demand[product, scenario_id, t]
            for product in data.P
            for t in data.T_V
        ),
        name="demand",
    )
    model.addConstrs(
        (
            gp.quicksum(production[machine, product, t] for product in data.P)
            <= scenarios.capacity[machine, scenario_id, t]
            * (
                machine_on[machine, t]
                if t <= data.theta
                else machine_on[machine, data.theta]
            )
            for machine in data.M
            for t in data.T_V
        ),
        name="capacity",
    )
    model.addConstrs(
        (
            production[machine, product, t]
            <= scenarios.capacity[machine, scenario_id, t]
            * (
                arc_on[machine, product, t]
                if t <= data.theta
                else arc_on[machine, product, data.theta]
            )
            for machine in data.M
            for product in data.P
            for t in data.T_V
        ),
        name="arc_link",
    )
    model.addConstrs(
        (
            production[machine, product, data.theta]
            <= frozen_share[machine, product]
            * scenarios.demand[product, scenario_id, data.theta]
            for machine in data.M
            for product in data.P
        ),
        name="share_definition",
    )
    model.addConstrs(
        (
            production[machine, product, t]
            <= frozen_share[machine, product]
            * scenarios.demand[product, scenario_id, t]
            for machine in data.M
            for product in data.P
            for t in data.T_V
            if t > data.theta
        ),
        name="share_frozen",
    )
    model.addConstrs(
        (
            gp.quicksum(frozen_share[machine, product] for machine in data.M)
            == 1.0
            for product in data.P
        ),
        name="share_sum",
    )

    return optimize_and_check(model)

def evaluate_all_policies(
    data: ProblemData,
    evaluation_scenarios: ScenarioSet,
    multi_period_design: MultiPeriodDesign,
    start_period_design: StartPeriodDesign,
    env: gp.Env,
    runtime: RuntimeConfig,
) -> PolicyNPVs:
    """Evaluate all policies scenario by scenario, as in the original scripts."""

    long_machine_on, long_machine_start, long_arc_on = build_long_chain_design(data)

    multi_period_npvs: List[float] = []
    start_period_npvs: List[float] = []
    long_chain_npvs: List[float] = []
    decision_rule_npvs: List[float] = []
    perfect_information_npvs: List[float] = []

    total_scenarios = len(evaluation_scenarios.scenario_ids)

    for position, scenario_id in enumerate(
        evaluation_scenarios.scenario_ids,
        start=1,
    ):
        if position == 1 or position % 10 == 0 or position == total_scenarios:
            print(f"Evaluating scenario {position}/{total_scenarios}")

        multi_period_npvs.append(
            evaluate_multi_period_scenario(
                data,
                evaluation_scenarios,
                scenario_id,
                multi_period_design,
                env,
                runtime,
            )
        )
        start_period_npvs.append(
            evaluate_start_period_scenario(
                data,
                evaluation_scenarios,
                scenario_id,
                start_period_design.machine_on,
                start_period_design.machine_start,
                start_period_design.arc_on,
                env,
                runtime,
                tag="Start-Period",
            )
        )
        long_chain_npvs.append(
            evaluate_start_period_scenario(
                data,
                evaluation_scenarios,
                scenario_id,
                long_machine_on,
                long_machine_start,
                long_arc_on,
                env,
                runtime,
                tag="Long-Chain",
            )
        )

        if runtime.run_decision_rule:
            decision_start, _, decision_arcs = build_decision_rule_schedule(
                data,
                evaluation_scenarios,
                scenario_id,
            )
            decision_rule_npvs.append(
                evaluate_timed_schedule_scenario(
                    data,
                    evaluation_scenarios,
                    scenario_id,
                    decision_start,
                    decision_arcs,
                    env,
                    runtime,
                    tag="Decision-Rule",
                )
            )

        if runtime.run_perfect_information:
            perfect_information_npvs.append(
                optimize_perfect_information_scenario(
                    data,
                    evaluation_scenarios,
                    scenario_id,
                    env,
                    runtime,
                )
            )

    # When an optional benchmark is disabled, use NaN placeholders so that the
    # result object and summary table keep a consistent schema.
    if not runtime.run_decision_rule:
        decision_rule_npvs = [float("nan")] * total_scenarios
    if not runtime.run_perfect_information:
        perfect_information_npvs = [float("nan")] * total_scenarios

    return PolicyNPVs(
        multi_period=multi_period_npvs,
        start_period=start_period_npvs,
        long_chain=long_chain_npvs,
        decision_rule=decision_rule_npvs,
        perfect_information=perfect_information_npvs,
    )






def summarize_policy_npvs(policy_npvs: PolicyNPVs) -> pd.DataFrame:
    """Return the original descriptive statistics as a tidy DataFrame."""

    def summarize(values: Sequence[float]) -> Dict[str, float]:
        array = np.asarray(values, dtype=float)
        array = array[np.isfinite(array)]
        if len(array) == 0:
            return {
                "mean": np.nan,
                "median": np.nan,
                "q1": np.nan,
                "q3": np.nan,
                "min": np.nan,
                "max": np.nan,
                "std": np.nan,
            }

        return {
            "mean": float(np.mean(array)),
            "median": float(np.median(array)),
            "q1": float(np.percentile(array, 25)),
            "q3": float(np.percentile(array, 75)),
            "min": float(np.min(array)),
            "max": float(np.max(array)),
            "std": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
        }

    summaries = {
        "Multi-period": summarize(policy_npvs.multi_period),
        "Start-period": summarize(policy_npvs.start_period),
        "Long-chain": summarize(policy_npvs.long_chain),
        "Decision-rule": summarize(policy_npvs.decision_rule),
        "Perfect Information": summarize(policy_npvs.perfect_information),
    }
    return pd.DataFrame(summaries).T


# =============================================================================
# ROSR EXTENSIONS: multistage, robust optimization, and DRO
# =============================================================================

@dataclass(frozen=True)
class ExtensionRuntimeConfig:
    """Controls for the five reviewer-requested extensions."""

    # Multistage scenario-tree model.
    multistage_time_limit: Optional[float] = 900.0
    multistage_keep_frozen_shares: bool = False
    history_round_digits: int = 6

    # Single systemic stress path used by the two robust models.
    robust_time_limit: Optional[float] = 900.0
    worst_capacity_multiplier: Optional[float] = None

    # Polyhedral DRO ambiguity set and constraint generation.
    dro_probability_radius: float = 0.50
    dro_moment_radius: float = 0.10
    dro_max_iterations: int = 25
    dro_tolerance: float = 1e-4
    dro_time_limit: Optional[float] = 900.0
    dro_adversary_time_limit: Optional[float] = 60.0

    # Keep the RO/DRO start-period feasible region identical to the baseline
    # start-period model: investments occur at t=0, but long-run allocation-share
    # bounds are introduced at the common investment-horizon boundary t=theta.
    # None means "use data.theta"; an explicit integer is retained only for
    # controlled sensitivity checks.
    start_period_share_anchor: Optional[int] = None

    # Evaluation and output controls.
    evaluate_out_of_sample: bool = True
    include_original_benchmarks: bool = True
    output_flag: int = 1


@dataclass
class MultistageResult:
    """Solution of the finite-scenario multistage stochastic program."""

    objective_value: float
    scenario_npvs: Dict[int, float]
    machine_on: Dict[Tuple[str, int, int], float]
    machine_start: Dict[Tuple[str, int, int], float]
    arc_on: Dict[Tuple[str, str, int, int], float]
    history_cluster_count: Dict[int, int]
    solver_status: int
    mip_gap: float
    objective_bound: float


@dataclass
class RobustResult:
    """Solution of the single-scenario stress-test robust model."""

    objective_value: float
    design: object
    stress_npvs: Dict[str, float]
    solver_status: int
    mip_gap: float
    objective_bound: float


@dataclass
class DROResult:
    """Solution of a finite-scenario DRO model solved by constraint generation."""

    objective_value: float
    master_upper_bound: float
    design: object
    scenario_npvs: Dict[int, float]
    worst_probabilities: Dict[int, float]
    iterations: int
    converged: bool
    probability_radius: float
    moment_radius: float
    master_status: int
    master_mip_gap: float
    design_gap: float


def _model_mip_gap(model: gp.Model) -> float:
    """Return a finite MIP gap when available, otherwise NaN."""

    if model.SolCount <= 0:
        return float("nan")
    try:
        return float(model.MIPGap)
    except (AttributeError, gp.GurobiError):
        return 0.0


def build_ro_stress_scenario(
    data: ProblemData,
    extension: ExtensionRuntimeConfig,
) -> Tuple[ScenarioSet, Dict[int, str]]:
    """Create the single systemic stress-test scenario used by RO.

    Demand stress:
      * every product uses its ``Decrease`` trajectory;
      * launch products therefore launch, but subsequently follow the decreasing
        launch trajectory rather than the ``NoLaunch`` trajectory.

    Capacity stress:
      * every resource is persistently limited to the case availability floor,
        unless ``worst_capacity_multiplier`` supplies another multiplier;
      * zero capacity is intentionally avoided because it would make the robust
        solution trivially install nothing.
    """

    stress_states = tuple("Decrease" for _ in data.P)
    scenario_map = {0: stress_states}
    scenario_ids = [0]
    probabilities = {0: 1.0}
    demand = build_scenario_demand(data, scenario_map)

    stress_multiplier = extension.worst_capacity_multiplier
    if stress_multiplier is None:
        stress_multiplier = data.config.availability_floor
    if not 0.0 <= stress_multiplier <= 1.0:
        raise ValueError("worst_capacity_multiplier must lie in [0, 1].")

    capacity: Dict[Tuple[str, int, int], float] = {}
    for machine in data.M:
        for t in data.T_V:
            capacity[machine, 0, t] = (
                data.config.base_capacity * stress_multiplier
            )

    return (
        ScenarioSet(
            scenario_ids=scenario_ids,
            scenario_map=scenario_map,
            probabilities=probabilities,
            demand=demand,
            capacity=capacity,
        ),
        {0: "Stress"},
    )


def _resolve_start_period_share_anchor(
    data: ProblemData,
    extension: ExtensionRuntimeConfig,
) -> int:
    """Return the start-period share anchor used by the baseline feasible region."""

    anchor = (
        data.theta
        if extension.start_period_share_anchor is None
        else int(extension.start_period_share_anchor)
    )
    if anchor not in data.T_V:
        raise ValueError("start_period_share_anchor must be in the valuation horizon.")
    return anchor


def _start_period_components(
    model: gp.Model,
    data: ProblemData,
    scenarios: ScenarioSet,
    *,
    share_anchor: int,
):
    """Add the common start-period feasible region and return NPV expressions."""

    if share_anchor not in data.T_V:
        raise ValueError("share_anchor must be in the valuation horizon.")

    machine_on = model.addVars(data.M, vtype=GRB.BINARY, name="m")
    arc_on = model.addVars(data.M, data.P, vtype=GRB.BINARY, name="a")
    machine_start = model.addVars(data.M, vtype=GRB.BINARY, name="u")
    production = model.addVars(
        data.M,
        data.P,
        scenarios.scenario_ids,
        data.T_V,
        lb=0.0,
        name="x",
    )
    frozen_share = model.addVars(
        data.M,
        data.P,
        scenarios.scenario_ids,
        lb=0.0,
        ub=1.0,
        name="y",
    )

    fixed_opex_coefficients: Dict[str, float] = {}
    for machine in data.M:
        last_active_time = min(data.lifetime[machine] - 1, data.nu)
        fixed_opex_coefficients[machine] = (
            data.fixed_opex[machine]
            * sum(discount_factor(data, t) for t in range(last_active_time + 1))
        )

    common_cost = (
        gp.quicksum(
            fixed_opex_coefficients[machine] * machine_start[machine]
            for machine in data.M
        )
        + gp.quicksum(
            data.machine_capex[machine] * machine_start[machine]
            for machine in data.M
        )
        + gp.quicksum(
            data.arc_capex[machine, product] * arc_on[machine, product]
            for machine in data.M
            for product in data.P
        )
    )

    scenario_npv = {}
    for scenario_id in scenarios.scenario_ids:
        revenue = gp.quicksum(
            discount_factor(data, t)
            * data.unit_profit[machine, product]
            * production[machine, product, scenario_id, t]
            for machine in data.M
            for product in data.P
            for t in data.T_V
        )
        scenario_npv[scenario_id] = revenue - common_cost

    model.addConstrs(
        (machine_on[machine] == machine_start[machine] for machine in data.M),
        name="startup_period_0",
    )
    model.addConstrs(
        (
            arc_on[machine, product] <= machine_on[machine]
            for machine in data.M
            for product in data.P
        ),
        name="arc_requires_machine",
    )

    model.addConstrs(
        (
            gp.quicksum(
                production[machine, product, scenario_id, t]
                for machine in data.M
            )
            <= scenarios.demand[product, scenario_id, t]
            for product in data.P
            for scenario_id in scenarios.scenario_ids
            for t in data.T_V
        ),
        name="demand",
    )

    for machine in data.M:
        for scenario_id in scenarios.scenario_ids:
            for t in data.T_V:
                alive = 1.0 if t <= data.lifetime[machine] - 1 else 0.0
                model.addConstr(
                    gp.quicksum(
                        production[machine, product, scenario_id, t]
                        for product in data.P
                    )
                    <= scenarios.capacity[machine, scenario_id, t]
                    * machine_on[machine]
                    * alive,
                    name=f"capacity_{machine}_{scenario_id}_{t}",
                )

    model.addConstrs(
        (
            production[machine, product, scenario_id, t]
            <= scenarios.capacity[machine, scenario_id, t]
            * arc_on[machine, product]
            for machine in data.M
            for product in data.P
            for scenario_id in scenarios.scenario_ids
            for t in data.T_V
        ),
        name="arc_link",
    )

    model.addConstrs(
        (
            production[machine, product, scenario_id, share_anchor]
            <= frozen_share[machine, product, scenario_id]
            * scenarios.demand[product, scenario_id, share_anchor]
            for machine in data.M
            for product in data.P
            for scenario_id in scenarios.scenario_ids
        ),
        name="share_definition",
    )
    model.addConstrs(
        (
            production[machine, product, scenario_id, t]
            <= frozen_share[machine, product, scenario_id]
            * scenarios.demand[product, scenario_id, t]
            for machine in data.M
            for product in data.P
            for scenario_id in scenarios.scenario_ids
            for t in data.T_V
            if t > share_anchor
        ),
        name="share_frozen",
    )
    model.addConstrs(
        (
            gp.quicksum(
                frozen_share[machine, product, scenario_id]
                for machine in data.M
            )
            == 1.0
            for product in data.P
            for scenario_id in scenarios.scenario_ids
        ),
        name="share_sum",
    )

    return {
        "machine_on": machine_on,
        "machine_start": machine_start,
        "arc_on": arc_on,
        "production": production,
        "frozen_share": frozen_share,
        "scenario_npv": scenario_npv,
    }


def _multi_period_components(
    model: gp.Model,
    data: ProblemData,
    scenarios: ScenarioSet,
):
    """Add the common two-stage multi-period feasible region."""

    machine_on = model.addVars(data.M, data.T_I, vtype=GRB.BINARY, name="m")
    arc_on = model.addVars(
        data.M, data.P, data.T_I, vtype=GRB.BINARY, name="a"
    )
    machine_start = model.addVars(
        data.M, data.T_I, vtype=GRB.BINARY, name="u"
    )
    production = model.addVars(
        data.M,
        data.P,
        scenarios.scenario_ids,
        data.T_V,
        lb=0.0,
        name="x",
    )
    frozen_share = model.addVars(
        data.M,
        data.P,
        scenarios.scenario_ids,
        lb=0.0,
        ub=1.0,
        name="y",
    )

    opex_coefficients: Dict[Tuple[str, int], float] = {}
    for machine in data.M:
        for start_time in data.T_I:
            last_active = min(
                start_time + data.lifetime[machine] - 1,
                data.nu,
            )
            opex_coefficients[machine, start_time] = (
                data.fixed_opex[machine]
                * sum(
                    discount_factor(data, t)
                    for t in range(start_time, last_active + 1)
                )
            )

    common_cost = (
        gp.quicksum(
            opex_coefficients[machine, start_time]
            * machine_start[machine, start_time]
            for machine in data.M
            for start_time in data.T_I
        )
        + gp.quicksum(
            data.machine_capex[machine]
            / (1.0 + data.discount_rate) ** start_time
            * machine_start[machine, start_time]
            for machine in data.M
            for start_time in data.T_I
        )
    )

    arc_cost = gp.LinExpr()
    for machine in data.M:
        for product in data.P:
            for t in data.T_I:
                previous = 0.0 if t == 0 else arc_on[machine, product, t - 1]
                arc_cost += (
                    data.arc_capex[machine, product]
                    / (1.0 + data.discount_rate) ** t
                    * (arc_on[machine, product, t] - previous)
                )
    common_cost += arc_cost

    scenario_npv = {}
    for scenario_id in scenarios.scenario_ids:
        revenue = gp.quicksum(
            discount_factor(data, t)
            * data.unit_profit[machine, product]
            * production[machine, product, scenario_id, t]
            for machine in data.M
            for product in data.P
            for t in data.T_V
        )
        scenario_npv[scenario_id] = revenue - common_cost

    model.addConstrs(
        (
            machine_on[machine, 0] == machine_start[machine, 0]
            for machine in data.M
        ),
        name="startup_period_0",
    )
    model.addConstrs(
        (
            machine_on[machine, t] - machine_on[machine, t - 1]
            == machine_start[machine, t]
            for machine in data.M
            for t in data.T_I
            if t > 0
        ),
        name="startup",
    )
    model.addConstrs(
        (
            arc_on[machine, product, t] >= arc_on[machine, product, t - 1]
            for machine in data.M
            for product in data.P
            for t in data.T_I
            if t > 0
        ),
        name="arc_monotonicity",
    )
    model.addConstrs(
        (
            arc_on[machine, product, t] <= machine_on[machine, t]
            for machine in data.M
            for product in data.P
            for t in data.T_I
        ),
        name="arc_requires_machine",
    )

    model.addConstrs(
        (
            gp.quicksum(
                production[machine, product, scenario_id, t]
                for machine in data.M
            )
            <= scenarios.demand[product, scenario_id, t]
            for product in data.P
            for scenario_id in scenarios.scenario_ids
            for t in data.T_V
        ),
        name="demand",
    )
    model.addConstrs(
        (
            gp.quicksum(
                production[machine, product, scenario_id, t]
                for product in data.P
            )
            <= scenarios.capacity[machine, scenario_id, t]
            * (
                machine_on[machine, t]
                if t <= data.theta
                else machine_on[machine, data.theta]
            )
            for machine in data.M
            for scenario_id in scenarios.scenario_ids
            for t in data.T_V
        ),
        name="capacity",
    )

    for machine in data.M:
        for scenario_id in scenarios.scenario_ids:
            for t in range(data.theta + 1, data.nu + 1):
                valid_starts = range(
                    max(0, t - data.lifetime[machine] + 1),
                    min(t, data.theta) + 1,
                )
                model.addConstr(
                    gp.quicksum(
                        production[machine, product, scenario_id, t]
                        for product in data.P
                    )
                    <= scenarios.capacity[machine, scenario_id, t]
                    * gp.quicksum(
                        machine_start[machine, tau] for tau in valid_starts
                    ),
                    name=f"lifetime_{machine}_{scenario_id}_{t}",
                )

    model.addConstrs(
        (
            production[machine, product, scenario_id, t]
            <= scenarios.capacity[machine, scenario_id, t]
            * (
                arc_on[machine, product, t]
                if t <= data.theta
                else arc_on[machine, product, data.theta]
            )
            for machine in data.M
            for product in data.P
            for scenario_id in scenarios.scenario_ids
            for t in data.T_V
        ),
        name="arc_link",
    )

    model.addConstrs(
        (
            production[machine, product, scenario_id, data.theta]
            <= frozen_share[machine, product, scenario_id]
            * scenarios.demand[product, scenario_id, data.theta]
            for machine in data.M
            for product in data.P
            for scenario_id in scenarios.scenario_ids
        ),
        name="share_definition",
    )
    model.addConstrs(
        (
            production[machine, product, scenario_id, t]
            <= frozen_share[machine, product, scenario_id]
            * scenarios.demand[product, scenario_id, t]
            for machine in data.M
            for product in data.P
            for scenario_id in scenarios.scenario_ids
            for t in data.T_V
            if t > data.theta
        ),
        name="share_frozen",
    )
    model.addConstrs(
        (
            gp.quicksum(
                frozen_share[machine, product, scenario_id]
                for machine in data.M
            )
            == 1.0
            for product in data.P
            for scenario_id in scenarios.scenario_ids
        ),
        name="share_sum",
    )

    return {
        "machine_on": machine_on,
        "machine_start": machine_start,
        "arc_on": arc_on,
        "production": production,
        "frozen_share": frozen_share,
        "scenario_npv": scenario_npv,
    }


def _extract_start_period_design(
    objective_value: float,
    components,
) -> StartPeriodDesign:
    return StartPeriodDesign(
        objective_value=float(objective_value),
        machine_on={k: float(v.X) for k, v in components["machine_on"].items()},
        machine_start={
            k: float(v.X) for k, v in components["machine_start"].items()
        },
        arc_on={k: float(v.X) for k, v in components["arc_on"].items()},
    )


def _extract_multi_period_design(
    objective_value: float,
    components,
) -> MultiPeriodDesign:
    return MultiPeriodDesign(
        objective_value=float(objective_value),
        machine_on={k: float(v.X) for k, v in components["machine_on"].items()},
        machine_start={
            k: float(v.X) for k, v in components["machine_start"].items()
        },
        arc_on={k: float(v.X) for k, v in components["arc_on"].items()},
        production={
            k: float(v.X) for k, v in components["production"].items()
        },
    )


def solve_start_period_robust(
    data: ProblemData,
    stress_scenarios: ScenarioSet,
    stress_labels: Mapping[int, str],
    env: gp.Env,
    runtime: RuntimeConfig,
    extension: ExtensionRuntimeConfig,
) -> RobustResult:
    """Maximize NPV in the single systemic stress-test scenario."""

    model = gp.Model("Start-Period Minimax Robust", env=env)
    configure_model(
        model,
        output_flag=extension.output_flag,
        threads=runtime.threads,
        time_limit=extension.robust_time_limit,
    )
    components = _start_period_components(
        model,
        data,
        stress_scenarios,
        share_anchor=_resolve_start_period_share_anchor(data, extension),
    )

    worst_npv = model.addVar(lb=-GRB.INFINITY, name="eta")
    model.addConstrs(
        (
            worst_npv <= components["scenario_npv"][scenario_id]
            for scenario_id in stress_scenarios.scenario_ids
        ),
        name="minimax_epigraph",
    )
    model.setObjective(worst_npv, GRB.MAXIMIZE)
    objective = optimize_and_check(model)

    stress_npvs = {
        stress_labels[s]: float(components["scenario_npv"][s].getValue())
        for s in stress_scenarios.scenario_ids
    }
    design = _extract_start_period_design(objective, components)
    return RobustResult(
        objective_value=objective,
        design=design,
        stress_npvs=stress_npvs,
        solver_status=int(model.Status),
        mip_gap=_model_mip_gap(model),
        objective_bound=float(model.ObjBound),
    )


def solve_multi_period_robust(
    data: ProblemData,
    stress_scenarios: ScenarioSet,
    stress_labels: Mapping[int, str],
    env: gp.Env,
    runtime: RuntimeConfig,
    extension: ExtensionRuntimeConfig,
) -> RobustResult:
    """Multi-period design optimized for the single stress-test scenario."""

    model = gp.Model("Multi-Period Minimax Robust", env=env)
    configure_model(
        model,
        output_flag=extension.output_flag,
        threads=runtime.threads,
        time_limit=extension.robust_time_limit,
    )
    components = _multi_period_components(model, data, stress_scenarios)

    worst_npv = model.addVar(lb=-GRB.INFINITY, name="eta")
    model.addConstrs(
        (
            worst_npv <= components["scenario_npv"][scenario_id]
            for scenario_id in stress_scenarios.scenario_ids
        ),
        name="minimax_epigraph",
    )
    model.setObjective(worst_npv, GRB.MAXIMIZE)
    objective = optimize_and_check(model)

    stress_npvs = {
        stress_labels[s]: float(components["scenario_npv"][s].getValue())
        for s in stress_scenarios.scenario_ids
    }
    design = _extract_multi_period_design(objective, components)
    return RobustResult(
        objective_value=objective,
        design=design,
        stress_npvs=stress_npvs,
        solver_status=int(model.Status),
        mip_gap=_model_mip_gap(model),
        objective_bound=float(model.ObjBound),
    )


def _history_signature(
    data: ProblemData,
    scenarios: ScenarioSet,
    scenario_id: int,
    decision_time: int,
    digits: int,
) -> Tuple[float, ...]:
    """Demand history observed before a strategic decision at ``decision_time``."""

    if decision_time == 0:
        return tuple()

    # Strategic learning follows the manuscript's scenario tree: only realized
    # demand histories define information nodes. Capacity variability and outages
    # are transient operational uncertainty and must not identify future scenario
    # paths. Including continuous realized capacities here would almost always
    # split the sampled paths after one period and create spurious foresight.
    values: List[float] = []
    for t in range(decision_time):
        for product in data.P:
            values.append(round(scenarios.demand[product, scenario_id, t], digits))
    return tuple(values)


def build_history_clusters(
    data: ProblemData,
    scenarios: ScenarioSet,
    digits: int,
) -> Dict[int, List[List[int]]]:
    """Partition scenarios by their observed history before each decision time."""

    clusters: Dict[int, List[List[int]]] = {}
    for t in data.T_I:
        groups: Dict[Tuple[float, ...], List[int]] = {}
        for scenario_id in scenarios.scenario_ids:
            signature = _history_signature(
                data,
                scenarios,
                scenario_id,
                t,
                digits,
            )
            groups.setdefault(signature, []).append(scenario_id)
        clusters[t] = list(groups.values())
    return clusters


def solve_multistage_scenario_tree(
    data: ProblemData,
    scenarios: ScenarioSet,
    env: gp.Env,
    runtime: RuntimeConfig,
    extension: ExtensionRuntimeConfig,
) -> MultistageResult:
    """Solve a finite-scenario multistage model with non-anticipativity.

    Investment decisions are scenario indexed but tied across scenarios with the
    same observed demand history. A decision in period t can use demand observations
    through t-1, not the realization in period t. Transient capacity variability
    and outages do not define strategic history nodes. Production is fully adaptive each
    period.  Set ``multistage_keep_frozen_shares=True`` to retain the original
    post-theta routing-share convention.
    """

    model = gp.Model("Multistage Scenario-Tree ROSR", env=env)
    configure_model(
        model,
        output_flag=extension.output_flag,
        threads=runtime.threads,
        time_limit=extension.multistage_time_limit,
    )

    S = scenarios.scenario_ids
    machine_on = model.addVars(data.M, S, data.T_I, vtype=GRB.BINARY, name="m")
    machine_start = model.addVars(
        data.M, S, data.T_I, vtype=GRB.BINARY, name="u"
    )
    arc_on = model.addVars(
        data.M, data.P, S, data.T_I, vtype=GRB.BINARY, name="a"
    )
    production = model.addVars(
        data.M, data.P, S, data.T_V, lb=0.0, name="x"
    )

    frozen_share = None
    if extension.multistage_keep_frozen_shares:
        frozen_share = model.addVars(
            data.M, data.P, S, lb=0.0, ub=1.0, name="y"
        )

    opex_coefficients: Dict[Tuple[str, int], float] = {}
    for machine in data.M:
        for start_time in data.T_I:
            last_active = min(
                start_time + data.lifetime[machine] - 1,
                data.nu,
            )
            opex_coefficients[machine, start_time] = (
                data.fixed_opex[machine]
                * sum(
                    discount_factor(data, t)
                    for t in range(start_time, last_active + 1)
                )
            )

    scenario_npv_expr = {}
    for s in S:
        revenue = gp.quicksum(
            discount_factor(data, t)
            * data.unit_profit[machine, product]
            * production[machine, product, s, t]
            for machine in data.M
            for product in data.P
            for t in data.T_V
        )
        opex = gp.quicksum(
            opex_coefficients[machine, tau] * machine_start[machine, s, tau]
            for machine in data.M
            for tau in data.T_I
        )
        machine_capex = gp.quicksum(
            data.machine_capex[machine]
            / (1.0 + data.discount_rate) ** tau
            * machine_start[machine, s, tau]
            for machine in data.M
            for tau in data.T_I
        )
        arc_capex = gp.LinExpr()
        for machine in data.M:
            for product in data.P:
                for t in data.T_I:
                    previous = 0.0 if t == 0 else arc_on[machine, product, s, t - 1]
                    arc_capex += (
                        data.arc_capex[machine, product]
                        / (1.0 + data.discount_rate) ** t
                        * (arc_on[machine, product, s, t] - previous)
                    )
        scenario_npv_expr[s] = revenue - opex - machine_capex - arc_capex

    model.setObjective(
        gp.quicksum(
            scenarios.probabilities[s] * scenario_npv_expr[s] for s in S
        ),
        GRB.MAXIMIZE,
    )

    model.addConstrs(
        (
            machine_on[machine, s, 0] == machine_start[machine, s, 0]
            for machine in data.M
            for s in S
        ),
        name="startup_period_0",
    )
    model.addConstrs(
        (
            machine_on[machine, s, t] - machine_on[machine, s, t - 1]
            == machine_start[machine, s, t]
            for machine in data.M
            for s in S
            for t in data.T_I
            if t > 0
        ),
        name="startup",
    )
    model.addConstrs(
        (
            arc_on[machine, product, s, t]
            >= arc_on[machine, product, s, t - 1]
            for machine in data.M
            for product in data.P
            for s in S
            for t in data.T_I
            if t > 0
        ),
        name="arc_monotonicity",
    )
    model.addConstrs(
        (
            arc_on[machine, product, s, t] <= machine_on[machine, s, t]
            for machine in data.M
            for product in data.P
            for s in S
            for t in data.T_I
        ),
        name="arc_requires_machine",
    )

    # Scenario-indexed equivalent of node-based non-anticipativity.
    history_clusters = build_history_clusters(
        data,
        scenarios,
        extension.history_round_digits,
    )
    for t, clusters_at_t in history_clusters.items():
        for cluster_number, cluster in enumerate(clusters_at_t):
            representative = cluster[0]
            for s in cluster[1:]:
                for machine in data.M:
                    model.addConstr(
                        machine_on[machine, s, t]
                        == machine_on[machine, representative, t],
                        name=f"na_m_{t}_{cluster_number}_{machine}_{s}",
                    )
                    model.addConstr(
                        machine_start[machine, s, t]
                        == machine_start[machine, representative, t],
                        name=f"na_u_{t}_{cluster_number}_{machine}_{s}",
                    )
                    for product in data.P:
                        model.addConstr(
                            arc_on[machine, product, s, t]
                            == arc_on[machine, product, representative, t],
                            name=(
                                f"na_a_{t}_{cluster_number}_{machine}_"
                                f"{product}_{s}"
                            ),
                        )

    model.addConstrs(
        (
            gp.quicksum(production[machine, product, s, t] for machine in data.M)
            <= scenarios.demand[product, s, t]
            for product in data.P
            for s in S
            for t in data.T_V
        ),
        name="demand",
    )
    model.addConstrs(
        (
            gp.quicksum(production[machine, product, s, t] for product in data.P)
            <= scenarios.capacity[machine, s, t]
            * (
                machine_on[machine, s, t]
                if t <= data.theta
                else machine_on[machine, s, data.theta]
            )
            for machine in data.M
            for s in S
            for t in data.T_V
        ),
        name="capacity",
    )

    for machine in data.M:
        for s in S:
            for t in range(data.theta + 1, data.nu + 1):
                valid_starts = range(
                    max(0, t - data.lifetime[machine] + 1),
                    min(t, data.theta) + 1,
                )
                model.addConstr(
                    gp.quicksum(
                        production[machine, product, s, t]
                        for product in data.P
                    )
                    <= scenarios.capacity[machine, s, t]
                    * gp.quicksum(
                        machine_start[machine, s, tau] for tau in valid_starts
                    ),
                    name=f"lifetime_{machine}_{s}_{t}",
                )

    model.addConstrs(
        (
            production[machine, product, s, t]
            <= scenarios.capacity[machine, s, t]
            * (
                arc_on[machine, product, s, t]
                if t <= data.theta
                else arc_on[machine, product, s, data.theta]
            )
            for machine in data.M
            for product in data.P
            for s in S
            for t in data.T_V
        ),
        name="arc_link",
    )

    if frozen_share is not None:
        model.addConstrs(
            (
                production[machine, product, s, data.theta]
                <= frozen_share[machine, product, s]
                * scenarios.demand[product, s, data.theta]
                for machine in data.M
                for product in data.P
                for s in S
            ),
            name="share_definition",
        )
        model.addConstrs(
            (
                production[machine, product, s, t]
                <= frozen_share[machine, product, s]
                * scenarios.demand[product, s, t]
                for machine in data.M
                for product in data.P
                for s in S
                for t in data.T_V
                if t > data.theta
            ),
            name="share_frozen",
        )
        model.addConstrs(
            (
                gp.quicksum(
                    frozen_share[machine, product, s] for machine in data.M
                )
                == 1.0
                for product in data.P
                for s in S
            ),
            name="share_sum",
        )

    objective = optimize_and_check(model)
    scenario_npvs = {
        s: float(scenario_npv_expr[s].getValue()) for s in S
    }

    return MultistageResult(
        objective_value=objective,
        scenario_npvs=scenario_npvs,
        machine_on={k: float(v.X) for k, v in machine_on.items()},
        machine_start={k: float(v.X) for k, v in machine_start.items()},
        arc_on={k: float(v.X) for k, v in arc_on.items()},
        history_cluster_count={t: len(v) for t, v in history_clusters.items()},
        solver_status=int(model.Status),
        mip_gap=_model_mip_gap(model),
        objective_bound=float(model.ObjBound),
    )


def build_dro_features(
    data: ProblemData,
    scenarios: ScenarioSet,
) -> Dict[str, Dict[int, float]]:
    """Build normalized first-moment features for demand and capacity."""

    raw_demand = {}
    raw_capacity = {}
    for s in scenarios.scenario_ids:
        raw_demand[s] = sum(
            discount_factor(data, t) * scenarios.demand[product, s, t]
            for product in data.P
            for t in data.T_V
        )
        raw_capacity[s] = sum(
            discount_factor(data, t) * scenarios.capacity[machine, s, t]
            for machine in data.M
            for t in data.T_V
        )

    nominal_demand = sum(
        scenarios.probabilities[s] * raw_demand[s]
        for s in scenarios.scenario_ids
    )
    nominal_capacity = sum(
        scenarios.probabilities[s] * raw_capacity[s]
        for s in scenarios.scenario_ids
    )

    if nominal_demand <= 0.0 or nominal_capacity <= 0.0:
        raise ValueError(
            "DRO moment normalization requires positive nominal discounted "
            "demand and capacity."
        )

    # Normalization makes both nominal moments equal to one and gives the
    # moment-radius parameter a transparent percentage interpretation.
    return {
        "discounted_total_demand": {
            s: raw_demand[s] / nominal_demand for s in scenarios.scenario_ids
        },
        "discounted_total_capacity": {
            s: raw_capacity[s] / nominal_capacity for s in scenarios.scenario_ids
        },
    }


def _optimize_adversary_to_optimality(model: gp.Model) -> float:
    """Solve the DRO probability LP and require an optimal adversarial value."""

    model.optimize()
    if model.Status != GRB.OPTIMAL:
        raise RuntimeError(
            "DRO probability adversary did not solve to optimality; "
            f"status={model.Status}, solutions={model.SolCount}. "
            "An inexact adversary cannot certify a DRO cut or convergence."
        )
    return float(model.ObjVal)


def solve_dro_adversary(
    data: ProblemData,
    scenarios: ScenarioSet,
    scenario_npvs: Mapping[int, float],
    env: gp.Env,
    runtime: RuntimeConfig,
    extension: ExtensionRuntimeConfig,
) -> Tuple[Dict[int, float], float]:
    """Find the worst probability distribution in the ambiguity set."""

    rho = extension.dro_probability_radius
    epsilon = extension.dro_moment_radius
    if rho < 0.0:
        raise ValueError("dro_probability_radius must be nonnegative.")
    if epsilon < 0.0:
        raise ValueError("dro_moment_radius must be nonnegative.")

    # Radius zero fixes q to the nominal distribution exactly.
    if rho == 0.0 and epsilon == 0.0:
        q = dict(scenarios.probabilities)
        value = sum(q[s] * scenario_npvs[s] for s in scenarios.scenario_ids)
        return q, float(value)

    model = gp.Model("DRO probability adversary", env=env)
    configure_model(
        model,
        output_flag=0,
        threads=runtime.threads,
        time_limit=extension.dro_adversary_time_limit,
    )

    q = model.addVars(scenarios.scenario_ids, lb=0.0, name="q")
    model.addConstr(
        gp.quicksum(q[s] for s in scenarios.scenario_ids) == 1.0,
        name="probability_sum",
    )

    for s in scenarios.scenario_ids:
        nominal = scenarios.probabilities[s]
        lower = max(0.0, (1.0 - rho) * nominal)
        upper = (1.0 + rho) * nominal
        model.addConstr(q[s] >= lower, name=f"probability_lower_{s}")
        model.addConstr(q[s] <= upper, name=f"probability_upper_{s}")

    features = build_dro_features(data, scenarios)
    for feature_name, feature_values in features.items():
        moment = gp.quicksum(
            feature_values[s] * q[s] for s in scenarios.scenario_ids
        )
        model.addConstr(
            moment >= 1.0 - epsilon,
            name=f"moment_lower_{feature_name}",
        )
        model.addConstr(
            moment <= 1.0 + epsilon,
            name=f"moment_upper_{feature_name}",
        )

    model.setObjective(
        gp.quicksum(
            scenario_npvs[s] * q[s] for s in scenarios.scenario_ids
        ),
        GRB.MINIMIZE,
    )
    value = _optimize_adversary_to_optimality(model)
    return (
        {s: float(q[s].X) for s in scenarios.scenario_ids},
        value,
    )


def _solve_dro_master(
    *,
    design_kind: str,
    data: ProblemData,
    scenarios: ScenarioSet,
    env: gp.Env,
    runtime: RuntimeConfig,
    extension: ExtensionRuntimeConfig,
) -> DROResult:
    """Generic finite-scenario DRO solver using probability-cut generation."""

    if design_kind not in {"start", "multi"}:
        raise ValueError("design_kind must be 'start' or 'multi'.")
    if extension.dro_max_iterations < 1:
        raise ValueError("dro_max_iterations must be at least 1.")
    if extension.dro_tolerance < 0.0:
        raise ValueError("dro_tolerance must be nonnegative.")

    model = gp.Model(f"{design_kind.title()}-Period DRO Master", env=env)
    configure_model(
        model,
        output_flag=extension.output_flag,
        threads=runtime.threads,
        time_limit=extension.dro_time_limit,
    )

    if design_kind == "start":
        components = _start_period_components(
            model,
            data,
            scenarios,
            share_anchor=_resolve_start_period_share_anchor(data, extension),
        )
    else:
        components = _multi_period_components(model, data, scenarios)

    robust_value = model.addVar(lb=-GRB.INFINITY, name="eta_DRO")
    model.setObjective(robust_value, GRB.MAXIMIZE)

    cut_distributions: List[Dict[int, float]] = [
        dict(scenarios.probabilities)
    ]
    model.addConstr(
        robust_value
        <= gp.quicksum(
            scenarios.probabilities[s] * components["scenario_npv"][s]
            for s in scenarios.scenario_ids
        ),
        name="dro_cut_0_nominal",
    )

    converged = False
    worst_probabilities = dict(scenarios.probabilities)
    worst_value = float("nan")
    master_upper = float("nan")
    master_incumbent = float("nan")
    scenario_values: Dict[int, float] = {}

    for iteration in range(1, extension.dro_max_iterations + 1):
        model.optimize()
        if model.SolCount <= 0:
            raise RuntimeError(
                f"DRO master has no incumbent; status={model.Status}."
            )

        master_incumbent = float(model.ObjVal)
        master_upper = float(model.ObjBound)
        scenario_values = {
            s: float(components["scenario_npv"][s].getValue())
            for s in scenarios.scenario_ids
        }
        worst_probabilities, worst_value = solve_dro_adversary(
            data,
            scenarios,
            scenario_values,
            env,
            runtime,
            extension,
        )

        design_gap = master_incumbent - worst_value
        tolerance = extension.dro_tolerance * max(1.0, abs(master_incumbent))
        print(
            f"DRO {design_kind} iteration {iteration}: "
            f"incumbent={master_incumbent:,.4f}, "
            f"bound={master_upper:,.4f}, "
            f"adversary={worst_value:,.4f}, "
            f"design_gap={design_gap:,.4f}"
        )

        # Constraint-generation convergence is certified only when the MILP
        # master itself is optimal. If a time limit stops the master early, the
        # current design and adversarial value are still returned, but the
        # result is marked non-converged.
        if model.Status == GRB.OPTIMAL and design_gap <= tolerance:
            converged = True
            break

        duplicate = any(
            max(
                abs(q[s] - worst_probabilities[s])
                for s in scenarios.scenario_ids
            )
            <= 1e-9
            for q in cut_distributions
        )
        if duplicate:
            print("DRO stopped because the adversarial distribution repeated.")
            break

        cut_distributions.append(dict(worst_probabilities))
        model.addConstr(
            robust_value
            <= gp.quicksum(
                worst_probabilities[s] * components["scenario_npv"][s]
                for s in scenarios.scenario_ids
            ),
            name=f"dro_cut_{iteration}",
        )

    if design_kind == "start":
        design = _extract_start_period_design(worst_value, components)
    else:
        design = _extract_multi_period_design(worst_value, components)

    return DROResult(
        objective_value=float(worst_value),
        master_upper_bound=float(master_upper),
        design=design,
        scenario_npvs=scenario_values,
        worst_probabilities=worst_probabilities,
        iterations=iteration,
        converged=converged,
        probability_radius=extension.dro_probability_radius,
        moment_radius=extension.dro_moment_radius,
        master_status=int(model.Status),
        master_mip_gap=_model_mip_gap(model),
        design_gap=float(master_incumbent - worst_value),
    )


def solve_start_period_dro(
    data: ProblemData,
    scenarios: ScenarioSet,
    env: gp.Env,
    runtime: RuntimeConfig,
    extension: ExtensionRuntimeConfig,
) -> DROResult:
    return _solve_dro_master(
        design_kind="start",
        data=data,
        scenarios=scenarios,
        env=env,
        runtime=runtime,
        extension=extension,
    )


def solve_multi_period_dro(
    data: ProblemData,
    scenarios: ScenarioSet,
    env: gp.Env,
    runtime: RuntimeConfig,
    extension: ExtensionRuntimeConfig,
) -> DROResult:
    return _solve_dro_master(
        design_kind="multi",
        data=data,
        scenarios=scenarios,
        env=env,
        runtime=runtime,
        extension=extension,
    )


def evaluate_start_period_extension_scenario(
    data: ProblemData,
    scenarios: ScenarioSet,
    scenario_id: int,
    design: StartPeriodDesign,
    env: gp.Env,
    runtime: RuntimeConfig,
    *,
    share_anchor: int,
    tag: str,
) -> float:
    """Evaluate a start-period extension using its stated share anchor."""

    model = gp.Model(f"Evaluate {tag} Scenario {scenario_id}", env=env)
    configure_model(
        model,
        output_flag=runtime.evaluation_output_flag,
        threads=runtime.threads,
        time_limit=runtime.evaluation_time_limit,
    )

    production = model.addVars(data.M, data.P, data.T_V, lb=0.0, name="x")
    share = model.addVars(data.M, data.P, lb=0.0, ub=1.0, name="y")
    revenue = gp.quicksum(
        discount_factor(data, t)
        * data.unit_profit[machine, product]
        * production[machine, product, t]
        for machine in data.M
        for product in data.P
        for t in data.T_V
    )
    opex, machine_capex, arc_capex = fixed_costs_start_period(
        data,
        design.machine_start,
        design.arc_on,
    )
    model.setObjective(revenue - opex - machine_capex - arc_capex, GRB.MAXIMIZE)

    model.addConstrs(
        (
            gp.quicksum(production[machine, product, t] for machine in data.M)
            <= scenarios.demand[product, scenario_id, t]
            for product in data.P
            for t in data.T_V
        ),
        name="demand",
    )
    for machine in data.M:
        for t in data.T_V:
            alive = 1.0 if t <= data.lifetime[machine] - 1 else 0.0
            model.addConstr(
                gp.quicksum(production[machine, product, t] for product in data.P)
                <= scenarios.capacity[machine, scenario_id, t]
                * design.machine_on[machine]
                * alive,
                name=f"capacity_{machine}_{t}",
            )
    model.addConstrs(
        (
            production[machine, product, t]
            <= scenarios.capacity[machine, scenario_id, t]
            * design.arc_on[machine, product]
            for machine in data.M
            for product in data.P
            for t in data.T_V
        ),
        name="arc_link",
    )
    model.addConstrs(
        (
            production[machine, product, share_anchor]
            <= share[machine, product]
            * scenarios.demand[product, scenario_id, share_anchor]
            for machine in data.M
            for product in data.P
        ),
        name="share_definition",
    )
    model.addConstrs(
        (
            production[machine, product, t]
            <= share[machine, product]
            * scenarios.demand[product, scenario_id, t]
            for machine in data.M
            for product in data.P
            for t in data.T_V
            if t > share_anchor
        ),
        name="share_frozen",
    )
    model.addConstrs(
        (
            gp.quicksum(share[machine, product] for machine in data.M) == 1.0
            for product in data.P
        ),
        name="share_sum",
    )
    return optimize_and_check(model)


def evaluate_extension_designs(
    data: ProblemData,
    scenarios: ScenarioSet,
    designs: Mapping[str, Tuple[str, object]],
    env: gp.Env,
    runtime: RuntimeConfig,
    extension: ExtensionRuntimeConfig,
) -> Dict[str, List[float]]:
    """Evaluate all fixed RO/DRO designs on a common scenario set."""

    results = {name: [] for name in designs}
    total = len(scenarios.scenario_ids)
    for position, s in enumerate(scenarios.scenario_ids, start=1):
        if position == 1 or position % 10 == 0 or position == total:
            print(f"Extension evaluation scenario {position}/{total}")
        for name, (kind, design) in designs.items():
            if kind == "multi":
                value = evaluate_multi_period_scenario(
                    data, scenarios, s, design, env, runtime
                )
            elif kind == "start":
                value = evaluate_start_period_extension_scenario(
                    data,
                    scenarios,
                    s,
                    design,
                    env,
                    runtime,
                    share_anchor=_resolve_start_period_share_anchor(data, extension),
                    tag=name,
                )
            else:
                raise ValueError(f"Unknown design kind {kind!r}.")
            results[name].append(value)
    return results



def evaluate_all_extension_methods(
    data: ProblemData,
    scenarios: ScenarioSet,
    stochastic_multi: MultiPeriodDesign,
    stochastic_start: StartPeriodDesign,
    robust_start: RobustResult,
    robust_multi: RobustResult,
    dro_start: DROResult,
    dro_multi: DROResult,
    env: gp.Env,
    runtime: RuntimeConfig,
    extension: ExtensionRuntimeConfig,
) -> Dict[str, List[float]]:
    """Evaluate the nine baseline/RO/DRO policies on one common scenario set."""

    baseline_npvs = evaluate_all_policies(
        data,
        scenarios,
        stochastic_multi,
        stochastic_start,
        env,
        runtime,
    )
    series: Dict[str, List[float]] = {
        "Multi-period": baseline_npvs.multi_period,
        "Start-period": baseline_npvs.start_period,
        "Long-chain": baseline_npvs.long_chain,
        "Decision-rule": baseline_npvs.decision_rule,
        "Perfect Information": baseline_npvs.perfect_information,
    }
    series.update(
        evaluate_extension_designs(
            data,
            scenarios,
            {
                "Start-period RO": ("start", robust_start.design),
                "Multi-period RO": ("multi", robust_multi.design),
                "Start-period DRO": ("start", dro_start.design),
                "Multi-period DRO": ("multi", dro_multi.design),
            },
            env,
            runtime,
            extension,
        )
    )
    return series


def evaluate_named_npvs_under_dro(
    data: ProblemData,
    scenarios: ScenarioSet,
    series: Mapping[str, Sequence[float]],
    env: gp.Env,
    runtime: RuntimeConfig,
    extension: ExtensionRuntimeConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the same DRO ambiguity set separately to every evaluated policy.

    Each policy is first evaluated scenario by scenario on the DRO planning
    support. The probability adversary is then solved for that policy's NPV
    vector. This produces a fair method-specific worst distribution rather than
    applying the DRO design's distribution to unrelated policies.
    """

    summary_rows: List[Dict[str, object]] = []
    probability_frame = scenario_metadata_frame(data, scenarios).rename(
        columns={"probability": "nominal_probability"}
    )
    scenario_position = {
        scenario_id: position
        for position, scenario_id in enumerate(scenarios.scenario_ids)
    }

    for design_name, values in series.items():
        array = np.asarray(values, dtype=float)
        if len(array) != len(scenarios.scenario_ids):
            raise ValueError(
                f"Series {design_name!r} has {len(array)} observations; "
                f"expected {len(scenarios.scenario_ids)}."
            )

        scenario_npvs = {
            scenario_id: float(array[position])
            for scenario_id, position in scenario_position.items()
        }
        if not all(np.isfinite(value) for value in scenario_npvs.values()):
            summary_rows.append(
                {
                    "design": design_name,
                    "nominal_expected_npv": np.nan,
                    "dro_worst_expected_npv": np.nan,
                    "dro_loss_vs_nominal": np.nan,
                }
            )
            probability_frame[
                f"worst_probability__{_safe_column_name(design_name)}"
            ] = np.nan
            continue

        worst_probabilities, worst_value = solve_dro_adversary(
            data,
            scenarios,
            scenario_npvs,
            env,
            runtime,
            extension,
        )
        nominal_value = sum(
            scenarios.probabilities[scenario_id] * scenario_npvs[scenario_id]
            for scenario_id in scenarios.scenario_ids
        )
        summary_rows.append(
            {
                "design": design_name,
                "nominal_expected_npv": float(nominal_value),
                "dro_worst_expected_npv": float(worst_value),
                "dro_loss_vs_nominal": float(nominal_value - worst_value),
            }
        )
        probability_frame[
            f"worst_probability__{_safe_column_name(design_name)}"
        ] = [
            worst_probabilities[scenario_id]
            for scenario_id in scenarios.scenario_ids
        ]

    return (
        pd.DataFrame(summary_rows).set_index("design"),
        probability_frame,
    )


def summarize_named_npvs(series: Mapping[str, Sequence[float]]) -> pd.DataFrame:
    """Descriptive statistics for an arbitrary collection of policies."""

    records = []
    for name, values in series.items():
        array = np.asarray(values, dtype=float)
        array = array[np.isfinite(array)]
        records.append(
            {
                "design": name,
                "mean": float(np.mean(array)) if len(array) else np.nan,
                "median": float(np.median(array)) if len(array) else np.nan,
                "q1": float(np.percentile(array, 25)) if len(array) else np.nan,
                "q3": float(np.percentile(array, 75)) if len(array) else np.nan,
                "min": float(np.min(array)) if len(array) else np.nan,
                "max": float(np.max(array)) if len(array) else np.nan,
                "std": (
                    float(np.std(array, ddof=1)) if len(array) > 1 else 0.0
                ),
                "n": int(len(array)),
            }
        )
    return pd.DataFrame(records).set_index("design")



def _weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    quantile: float,
) -> float:
    """Return a probability-weighted quantile for finite observations."""

    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must lie in [0, 1].")
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights)
    return float(sorted_values[np.searchsorted(cumulative, quantile, side="left")])


def summarize_named_npvs_weighted(
    series: Mapping[str, Sequence[float]],
    scenarios: ScenarioSet,
) -> pd.DataFrame:
    """Probability-weighted descriptive statistics on a common scenario set.

    Values in every sequence must follow ``scenarios.scenario_ids`` order.
    The mean therefore matches the expected-NPV convention used in stochastic
    optimization rather than an unweighted arithmetic mean over sampled paths.
    """

    scenario_ids = list(scenarios.scenario_ids)
    base_weights = np.asarray(
        [scenarios.probabilities[s] for s in scenario_ids],
        dtype=float,
    )
    if len(base_weights) == 0 or not np.all(np.isfinite(base_weights)):
        raise ValueError("Scenario probabilities must be finite and nonempty.")
    if np.any(base_weights < 0.0) or base_weights.sum() <= 0.0:
        raise ValueError("Scenario probabilities must be nonnegative with positive sum.")
    base_weights = base_weights / base_weights.sum()

    records = []
    for name, values in series.items():
        array = np.asarray(values, dtype=float)
        if len(array) != len(scenario_ids):
            raise ValueError(
                f"Series {name!r} has {len(array)} observations; "
                f"expected {len(scenario_ids)}."
            )

        finite = np.isfinite(array)
        if not finite.any():
            records.append(
                {
                    "design": name,
                    "mean": np.nan,
                    "median": np.nan,
                    "q1": np.nan,
                    "q3": np.nan,
                    "min": np.nan,
                    "max": np.nan,
                    "std": np.nan,
                    "n": 0,
                }
            )
            continue

        x = array[finite]
        w = base_weights[finite]
        w = w / w.sum()
        mean = float(np.dot(w, x))
        variance = float(np.dot(w, (x - mean) ** 2))
        records.append(
            {
                "design": name,
                "mean": mean,
                "median": _weighted_quantile(x, w, 0.50),
                "q1": _weighted_quantile(x, w, 0.25),
                "q3": _weighted_quantile(x, w, 0.75),
                "min": float(np.min(x)),
                "max": float(np.max(x)),
                "std": float(np.sqrt(max(0.0, variance))),
                "n": int(len(x)),
            }
        )

    return pd.DataFrame(records).set_index("design")


def _safe_column_name(text: str) -> str:
    """Convert a product label to a stable CSV column suffix."""

    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()


def scenario_metadata_frame(
    data: ProblemData,
    scenarios: ScenarioSet,
) -> pd.DataFrame:
    """Return scenario IDs, probabilities, and product-state labels."""

    rows: List[Dict[str, object]] = []
    for scenario_id in scenarios.scenario_ids:
        row: Dict[str, object] = {
            "scenario_id": scenario_id,
            "probability": scenarios.probabilities[scenario_id],
        }
        state_tuple = scenarios.scenario_map.get(scenario_id, tuple())
        for product, state in zip(data.P, state_tuple):
            row[f"state__{_safe_column_name(product)}"] = state
        rows.append(row)
    return pd.DataFrame(rows)


def write_evaluation_csv(
    output_path: str | Path,
    data: ProblemData,
    scenarios: ScenarioSet,
    series: Mapping[str, Sequence[float]],
) -> pd.DataFrame:
    """Write one wide, plot-ready evaluation CSV and return its DataFrame."""

    frame = scenario_metadata_frame(data, scenarios)
    expected = len(frame)
    for label, values in series.items():
        if len(values) != expected:
            raise ValueError(
                f"Series {label!r} has {len(values)} observations; expected {expected}."
            )
        frame[label] = np.asarray(values, dtype=float)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return frame


def evaluate_multi_period_with_multistage_recourse_scenario(
    data: ProblemData,
    scenarios: ScenarioSet,
    scenario_id: int,
    design: MultiPeriodDesign,
    env: gp.Env,
    runtime: RuntimeConfig,
) -> float:
    """Evaluate a fixed two-stage multi-period design with adaptive operations.

    Investment timing and capability decisions are fixed to the two-stage
    solution. Production is re-optimized separately in every period/scenario,
    without the post-theta frozen-share rule. This places the fixed design in
    the same operational-recourse environment as the multistage model.
    """

    model = gp.Model(
        f"Evaluate Two-Stage Multi-Period with Multistage Recourse {scenario_id}",
        env=env,
    )
    configure_model(
        model,
        output_flag=runtime.evaluation_output_flag,
        threads=runtime.threads,
        time_limit=runtime.evaluation_time_limit,
    )

    production = model.addVars(data.M, data.P, data.T_V, lb=0.0, name="x")
    revenue = gp.quicksum(
        discount_factor(data, t)
        * data.unit_profit[machine, product]
        * production[machine, product, t]
        for machine in data.M
        for product in data.P
        for t in data.T_V
    )
    opex, machine_capex, arc_capex = fixed_costs_multi_period(data, design)
    model.setObjective(revenue - opex - machine_capex - arc_capex, GRB.MAXIMIZE)

    model.addConstrs(
        (
            gp.quicksum(production[machine, product, t] for machine in data.M)
            <= scenarios.demand[product, scenario_id, t]
            for product in data.P
            for t in data.T_V
        ),
        name="demand",
    )

    for machine in data.M:
        for t in data.T_V:
            if t <= data.theta:
                active = design.machine_on[machine, t]
            else:
                active = sum(
                    design.machine_start[machine, tau]
                    for tau in range(
                        max(0, t - data.lifetime[machine] + 1),
                        min(t, data.theta) + 1,
                    )
                )
            model.addConstr(
                gp.quicksum(
                    production[machine, product, t] for product in data.P
                )
                <= scenarios.capacity[machine, scenario_id, t] * active,
                name=f"capacity_{machine}_{t}",
            )

    model.addConstrs(
        (
            production[machine, product, t]
            <= scenarios.capacity[machine, scenario_id, t]
            * (
                design.arc_on[machine, product, t]
                if t <= data.theta
                else design.arc_on[machine, product, data.theta]
            )
            for machine in data.M
            for product in data.P
            for t in data.T_V
        ),
        name="arc_link",
    )

    return optimize_and_check(model)


def evaluate_start_period_with_multistage_recourse_scenario(
    data: ProblemData,
    scenarios: ScenarioSet,
    scenario_id: int,
    design: StartPeriodDesign,
    env: gp.Env,
    runtime: RuntimeConfig,
) -> float:
    """Evaluate a fixed start-period design with fully adaptive operations."""

    model = gp.Model(
        f"Evaluate Two-Stage Start-Period with Multistage Recourse {scenario_id}",
        env=env,
    )
    configure_model(
        model,
        output_flag=runtime.evaluation_output_flag,
        threads=runtime.threads,
        time_limit=runtime.evaluation_time_limit,
    )

    production = model.addVars(data.M, data.P, data.T_V, lb=0.0, name="x")
    revenue = gp.quicksum(
        discount_factor(data, t)
        * data.unit_profit[machine, product]
        * production[machine, product, t]
        for machine in data.M
        for product in data.P
        for t in data.T_V
    )
    opex, machine_capex, arc_capex = fixed_costs_start_period(
        data,
        design.machine_start,
        design.arc_on,
    )
    model.setObjective(revenue - opex - machine_capex - arc_capex, GRB.MAXIMIZE)

    model.addConstrs(
        (
            gp.quicksum(production[machine, product, t] for machine in data.M)
            <= scenarios.demand[product, scenario_id, t]
            for product in data.P
            for t in data.T_V
        ),
        name="demand",
    )

    for machine in data.M:
        for t in data.T_V:
            alive = 1.0 if t <= data.lifetime[machine] - 1 else 0.0
            model.addConstr(
                gp.quicksum(
                    production[machine, product, t] for product in data.P
                )
                <= scenarios.capacity[machine, scenario_id, t]
                * design.machine_on[machine]
                * alive,
                name=f"capacity_{machine}_{t}",
            )

    model.addConstrs(
        (
            production[machine, product, t]
            <= scenarios.capacity[machine, scenario_id, t]
            * design.arc_on[machine, product]
            for machine in data.M
            for product in data.P
            for t in data.T_V
        ),
        name="arc_link",
    )

    return optimize_and_check(model)


def evaluate_two_stage_designs_on_multistage_tree(
    data: ProblemData,
    planning_scenarios: ScenarioSet,
    multistage: MultistageResult,
    stochastic_multi: MultiPeriodDesign,
    stochastic_start: StartPeriodDesign,
    env: gp.Env,
    runtime: RuntimeConfig,
) -> Dict[str, List[float]]:
    """Create an apples-to-apples CDF comparison on the multistage tree.

    All three policies use exactly the same planning scenarios and probabilities.
    The multistage policy retains history-adaptive investment decisions. The two
    two-stage designs keep their investments fixed but receive the same fully
    adaptive operational allocation recourse as the multistage formulation.
    """

    multi_values: List[float] = []
    start_values: List[float] = []
    scenario_ids = planning_scenarios.scenario_ids

    for position, scenario_id in enumerate(scenario_ids, start=1):
        if position == 1 or position % 10 == 0 or position == len(scenario_ids):
            print(
                "Common multistage-recourse evaluation "
                f"scenario {position}/{len(scenario_ids)}"
            )
        multi_values.append(
            evaluate_multi_period_with_multistage_recourse_scenario(
                data,
                planning_scenarios,
                scenario_id,
                stochastic_multi,
                env,
                runtime,
            )
        )
        start_values.append(
            evaluate_start_period_with_multistage_recourse_scenario(
                data,
                planning_scenarios,
                scenario_id,
                stochastic_start,
                env,
                runtime,
            )
        )

    return {
        "Multistage stochastic": [
            multistage.scenario_npvs[scenario_id] for scenario_id in scenario_ids
        ],
        "Two-stage multi-period (multistage recourse)": multi_values,
        "Two-stage start-period (multistage recourse)": start_values,
    }


def _evaluate_original_two_stage_only(
    data: ProblemData,
    scenarios: ScenarioSet,
    stochastic_multi: MultiPeriodDesign,
    stochastic_start: StartPeriodDesign,
    env: gp.Env,
    runtime: RuntimeConfig,
) -> Dict[str, List[float]]:
    """Evaluate just the original two stochastic designs."""

    multi_values: List[float] = []
    start_values: List[float] = []
    for scenario_id in scenarios.scenario_ids:
        multi_values.append(
            evaluate_multi_period_scenario(
                data, scenarios, scenario_id, stochastic_multi, env, runtime
            )
        )
        start_values.append(
            evaluate_start_period_scenario(
                data,
                scenarios,
                scenario_id,
                stochastic_start.machine_on,
                stochastic_start.machine_start,
                stochastic_start.arc_on,
                env,
                runtime,
                tag="Start-Period",
            )
        )
    return {
        "Multi-period": multi_values,
        "Start-period": start_values,
    }


def run_extended_case(
    case_number: int,
    *,
    runtime: Optional[RuntimeConfig] = None,
    extension: Optional[ExtensionRuntimeConfig] = None,
    output_root: str | Path = "rosr_extended_outputs",
) -> Dict[str, object]:
    """Solve one case and save plot-ready CSV files; never create plots here."""

    runtime = runtime or RuntimeConfig()
    extension = extension or ExtensionRuntimeConfig()
    data = build_problem_data(get_case_config(case_number))
    output_dir = Path(output_root) / f"case_{case_number}"
    output_dir.mkdir(parents=True, exist_ok=True)

    capacity_rng = random.Random(runtime.capacity_seed)
    planning_scenarios = build_planning_scenarios(data, runtime, capacity_rng)
    evaluation_scenarios = build_evaluation_scenarios(data, runtime, capacity_rng)
    stress_scenarios, stress_labels = build_ro_stress_scenario(
        data, extension
    )

    env = create_gurobi_environment(output_dir)
    try:
        print(f"Case {case_number}: solving original stochastic designs...")
        stochastic_multi = solve_multi_period_design(
            data, planning_scenarios, env, runtime
        )
        stochastic_start = solve_start_period_design(
            data, planning_scenarios, env, runtime
        )

        print(f"Case {case_number}: solving multistage scenario tree...")
        multistage = solve_multistage_scenario_tree(
            data, planning_scenarios, env, runtime, extension
        )

        print(f"Case {case_number}: solving start/multi robust models...")
        robust_start = solve_start_period_robust(
            data, stress_scenarios, stress_labels, env, runtime, extension
        )
        robust_multi = solve_multi_period_robust(
            data, stress_scenarios, stress_labels, env, runtime, extension
        )

        print(f"Case {case_number}: solving start/multi DRO models...")
        dro_start = solve_start_period_dro(
            data, planning_scenarios, env, runtime, extension
        )
        dro_multi = solve_multi_period_dro(
            data, planning_scenarios, env, runtime, extension
        )

        fixed_extension_designs: Dict[str, Tuple[str, object]] = {
            "Start-period RO": ("start", robust_start.design),
            "Multi-period RO": ("multi", robust_multi.design),
            "Start-period DRO": ("start", dro_start.design),
            "Multi-period DRO": ("multi", dro_multi.design),
        }

        planning_objectives = pd.DataFrame(
            [
                ("Multi-period", stochastic_multi.objective_value),
                ("Start-period", stochastic_start.objective_value),
                ("Multistage stochastic", multistage.objective_value),
                ("Start-period RO", robust_start.objective_value),
                ("Multi-period RO", robust_multi.objective_value),
                ("Start-period DRO", dro_start.objective_value),
                ("Multi-period DRO", dro_multi.objective_value),
            ],
            columns=["design", "planning_objective"],
        ).set_index("design")
        planning_objectives.to_csv(output_dir / "extension_planning_objectives.csv")

        solver_diagnostics = pd.DataFrame(
            [
                {
                    "design": "Multistage stochastic",
                    "status": multistage.solver_status,
                    "mip_gap": multistage.mip_gap,
                    "objective_bound": multistage.objective_bound,
                    "converged": np.nan,
                    "constraint_generation_gap": np.nan,
                },
                {
                    "design": "Start-period RO",
                    "status": robust_start.solver_status,
                    "mip_gap": robust_start.mip_gap,
                    "objective_bound": robust_start.objective_bound,
                    "converged": np.nan,
                    "constraint_generation_gap": np.nan,
                },
                {
                    "design": "Multi-period RO",
                    "status": robust_multi.solver_status,
                    "mip_gap": robust_multi.mip_gap,
                    "objective_bound": robust_multi.objective_bound,
                    "converged": np.nan,
                    "constraint_generation_gap": np.nan,
                },
                {
                    "design": "Start-period DRO",
                    "status": dro_start.master_status,
                    "mip_gap": dro_start.master_mip_gap,
                    "objective_bound": dro_start.master_upper_bound,
                    "converged": dro_start.converged,
                    "constraint_generation_gap": dro_start.design_gap,
                },
                {
                    "design": "Multi-period DRO",
                    "status": dro_multi.master_status,
                    "mip_gap": dro_multi.master_mip_gap,
                    "objective_bound": dro_multi.master_upper_bound,
                    "converged": dro_multi.converged,
                    "constraint_generation_gap": dro_multi.design_gap,
                },
            ]
        ).set_index("design")
        solver_diagnostics.to_csv(output_dir / "extension_solver_diagnostics.csv")

        pd.DataFrame(
            {
                "decision_time": list(data.T_I),
                "history_nodes": [
                    multistage.history_cluster_count[t] for t in data.T_I
                ],
            }
        ).to_csv(output_dir / "multistage_history_nodes.csv", index=False)

        stress_table = pd.DataFrame(
            {
                "Start-period RO": robust_start.stress_npvs,
                "Multi-period RO": robust_multi.stress_npvs,
            }
        )
        stress_table.to_csv(output_dir / "robust_stress_npvs.csv")

        dro_probabilities = pd.DataFrame(
            {
                "nominal": planning_scenarios.probabilities,
                "start_period_DRO_worst": dro_start.worst_probabilities,
                "multi_period_DRO_worst": dro_multi.worst_probabilities,
            }
        )
        dro_probabilities.index.name = "scenario_id"
        dro_probabilities.to_csv(output_dir / "dro_worst_probabilities.csv")

        # ---------------------------------------------------------------
        # DRO evaluation: all nine fixed/benchmark policies are evaluated
        # on the DRO planning support, then each receives its own worst
        # probability distribution from the common ambiguity set.
        # ---------------------------------------------------------------
        print(f"Case {case_number}: evaluating all methods under DRO...")
        dro_evaluation_series = evaluate_all_extension_methods(
            data,
            planning_scenarios,
            stochastic_multi,
            stochastic_start,
            robust_start,
            robust_multi,
            dro_start,
            dro_multi,
            env,
            runtime,
            extension,
        )
        dro_evaluation_frame = write_evaluation_csv(
            output_dir / "dro_evaluation_npvs.csv",
            data,
            planning_scenarios,
            dro_evaluation_series,
        )
        dro_evaluation_summary, dro_evaluation_probability_frame = (
            evaluate_named_npvs_under_dro(
                data,
                planning_scenarios,
                dro_evaluation_series,
                env,
                runtime,
                extension,
            )
        )
        dro_evaluation_summary.to_csv(
            output_dir / "dro_evaluation_summary.csv"
        )
        dro_evaluation_probability_frame.to_csv(
            output_dir / "dro_evaluation_worst_probabilities.csv",
            index=False,
        )

        # ---------------------------------------------------------------
        # RO evaluation: all nine policies are evaluated on the single
        # systemic stress scenario (Decrease demand for every product).
        # ---------------------------------------------------------------
        print(f"Case {case_number}: evaluating all methods under RO stress...")
        ro_stress_series = evaluate_all_extension_methods(
            data,
            stress_scenarios,
            stochastic_multi,
            stochastic_start,
            robust_start,
            robust_multi,
            dro_start,
            dro_multi,
            env,
            runtime,
            extension,
        )
        ro_stress_evaluation_frame = write_evaluation_csv(
            output_dir / "ro_stress_evaluation_npvs.csv",
            data,
            stress_scenarios,
            ro_stress_series,
        )
        ro_stress_evaluation_summary = summarize_named_npvs(ro_stress_series)
        ro_stress_evaluation_summary.to_csv(
            output_dir / "ro_stress_evaluation_summary.csv"
        )

        # ---------------------------------------------------------------
        # CSV 1: common multistage-tree comparison.
        # ---------------------------------------------------------------
        multistage_common_series = evaluate_two_stage_designs_on_multistage_tree(
            data,
            planning_scenarios,
            multistage,
            stochastic_multi,
            stochastic_start,
            env,
            runtime,
        )
        multistage_common_frame = write_evaluation_csv(
            output_dir / "multistage_common_evaluation_npvs.csv",
            data,
            planning_scenarios,
            multistage_common_series,
        )
        multistage_common_summary = summarize_named_npvs_weighted(
            multistage_common_series,
            planning_scenarios,
        )
        # Internal audit: the probability-weighted mean of the multistage
        # scenario NPVs must reproduce the multistage planning objective.
        weighted_ms_mean = float(
            multistage_common_summary.loc["Multistage stochastic", "mean"]
        )
        if not np.isclose(
            weighted_ms_mean,
            multistage.objective_value,
            rtol=1e-7,
            atol=1e-4,
        ):
            raise RuntimeError(
                "Multistage probability-weighted summary does not match the "
                f"planning objective: summary={weighted_ms_mean}, "
                f"objective={multistage.objective_value}."
            )
        multistage_common_summary.to_csv(
            output_dir / "multistage_common_evaluation_summary.csv"
        )

        baseline_series: Dict[str, List[float]] = {}
        extension_series: Dict[str, List[float]] = {}
        baseline_summary = pd.DataFrame()
        evaluation_summary = pd.DataFrame()
        baseline_frame = pd.DataFrame()
        extension_frame = pd.DataFrame()

        if extension.evaluate_out_of_sample:
            # -----------------------------------------------------------
            # CSV 2: exact original five-model baseline evaluation.
            # -----------------------------------------------------------
            if extension.include_original_benchmarks:
                baseline_npvs = evaluate_all_policies(
                    data,
                    evaluation_scenarios,
                    stochastic_multi,
                    stochastic_start,
                    env,
                    runtime,
                )
                baseline_series = {
                    "Multi-period": baseline_npvs.multi_period,
                    "Start-period": baseline_npvs.start_period,
                    "Long-chain": baseline_npvs.long_chain,
                    "Decision-rule": baseline_npvs.decision_rule,
                    "Perfect Information": baseline_npvs.perfect_information,
                }
            else:
                baseline_series = _evaluate_original_two_stage_only(
                    data,
                    evaluation_scenarios,
                    stochastic_multi,
                    stochastic_start,
                    env,
                    runtime,
                )

            baseline_frame = write_evaluation_csv(
                output_dir / "baseline_evaluation_npvs.csv",
                data,
                evaluation_scenarios,
                baseline_series,
            )
            baseline_summary = summarize_named_npvs(baseline_series)
            baseline_summary.to_csv(
                output_dir / "baseline_evaluation_summary.csv"
            )

            # -----------------------------------------------------------
            # CSV 3: baseline plus RO/DRO fixed-design extensions.
            # -----------------------------------------------------------
            extension_series = dict(baseline_series)
            extension_series.update(
                evaluate_extension_designs(
                    data,
                    evaluation_scenarios,
                    fixed_extension_designs,
                    env,
                    runtime,
                    extension,
                )
            )
            extension_frame = write_evaluation_csv(
                output_dir / "extension_evaluation_npvs.csv",
                data,
                evaluation_scenarios,
                extension_series,
            )
            evaluation_summary = summarize_named_npvs(extension_series)
            evaluation_summary.to_csv(
                output_dir / "extension_evaluation_summary.csv"
            )

            # Long format is convenient for statistical packages.
            long_rows: List[Dict[str, object]] = []
            for design_name, values in extension_series.items():
                for scenario_id, value in zip(
                    evaluation_scenarios.scenario_ids,
                    values,
                ):
                    long_rows.append(
                        {
                            "design": design_name,
                            "scenario_id": scenario_id,
                            "probability": evaluation_scenarios.probabilities[
                                scenario_id
                            ],
                            "npv": value,
                        }
                    )
            pd.DataFrame(long_rows).to_csv(
                output_dir / "extension_evaluation_npvs_long.csv",
                index=False,
            )

        print(f"Case {case_number}: CSV export complete in {output_dir}")
        return {
            "data": data,
            "planning_scenarios": planning_scenarios,
            "evaluation_scenarios": evaluation_scenarios,
            "stress_scenarios": stress_scenarios,
            "stochastic_multi": stochastic_multi,
            "stochastic_start": stochastic_start,
            "multistage": multistage,
            "robust_start": robust_start,
            "robust_multi": robust_multi,
            "dro_start": dro_start,
            "dro_multi": dro_multi,
            "planning_objectives": planning_objectives,
            "solver_diagnostics": solver_diagnostics,
            "baseline_summary": baseline_summary,
            "evaluation_summary": evaluation_summary,
            "multistage_common_summary": multistage_common_summary,
            "dro_evaluation_summary": dro_evaluation_summary,
            "ro_stress_evaluation_summary": ro_stress_evaluation_summary,
            "baseline_frame": baseline_frame,
            "extension_frame": extension_frame,
            "multistage_common_frame": multistage_common_frame,
            "dro_evaluation_frame": dro_evaluation_frame,
            "dro_evaluation_probability_frame": dro_evaluation_probability_frame,
            "ro_stress_evaluation_frame": ro_stress_evaluation_frame,
            "output_dir": output_dir,
        }
    finally:
        env.dispose()


def run_all_extended_cases(
    *,
    runtime: Optional[RuntimeConfig] = None,
    extension: Optional[ExtensionRuntimeConfig] = None,
    output_root: str | Path = "rosr_extended_outputs",
) -> Dict[int, Dict[str, object]]:
    """Run Cases 1-4 and save combined summary tables; never create plots."""

    runtime = runtime or RuntimeConfig()
    extension = extension or ExtensionRuntimeConfig()
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)

    results: Dict[int, Dict[str, object]] = {}
    planning_frames: List[pd.DataFrame] = []
    baseline_frames: List[pd.DataFrame] = []
    extension_frames: List[pd.DataFrame] = []
    multistage_frames: List[pd.DataFrame] = []
    dro_evaluation_frames: List[pd.DataFrame] = []
    ro_stress_frames: List[pd.DataFrame] = []

    for case_number in range(1, 5):
        case_result = run_extended_case(
            case_number,
            runtime=runtime,
            extension=extension,
            output_root=root,
        )
        results[case_number] = case_result

        for key, destination in [
            ("planning_objectives", planning_frames),
            ("baseline_summary", baseline_frames),
            ("evaluation_summary", extension_frames),
            ("multistage_common_summary", multistage_frames),
            ("dro_evaluation_summary", dro_evaluation_frames),
        ]:
            frame = case_result[key]
            if isinstance(frame, pd.DataFrame) and not frame.empty:
                temp = frame.reset_index()
                temp.insert(0, "case", case_number)
                destination.append(temp)

        ro_frame = case_result["ro_stress_evaluation_frame"]
        if isinstance(ro_frame, pd.DataFrame) and not ro_frame.empty:
            temp = ro_frame.copy()
            temp.insert(0, "case", case_number)
            ro_stress_frames.append(temp)

    if planning_frames:
        pd.concat(planning_frames, ignore_index=True).to_csv(
            root / "all_cases_extension_planning_objectives.csv", index=False
        )
    if baseline_frames:
        pd.concat(baseline_frames, ignore_index=True).to_csv(
            root / "all_cases_baseline_evaluation_summary.csv", index=False
        )
    if extension_frames:
        pd.concat(extension_frames, ignore_index=True).to_csv(
            root / "all_cases_extension_evaluation_summary.csv", index=False
        )
    if multistage_frames:
        pd.concat(multistage_frames, ignore_index=True).to_csv(
            root / "all_cases_multistage_common_evaluation_summary.csv",
            index=False,
        )
    if dro_evaluation_frames:
        pd.concat(dro_evaluation_frames, ignore_index=True).to_csv(
            root / "all_cases_dro_evaluation_summary.csv",
            index=False,
        )
    if ro_stress_frames:
        pd.concat(ro_stress_frames, ignore_index=True).to_csv(
            root / "all_cases_ro_stress_evaluation_npvs.csv",
            index=False,
        )

    return results


# =============================================================================
# Command-line entry point for MIT Engaging (CSV output only; no plotting)
# =============================================================================

def _build_cli_parser():
    import argparse
    import os

    parser = argparse.ArgumentParser(
        description=(
            "Solve ROSR Cases 1-4 and save plot-ready evaluation CSV files. "
            "This script never creates figures."
        )
    )
    parser.add_argument(
        "--case",
        default="all",
        choices=["1", "2", "3", "4", "all"],
        help="Case to solve. Use 'all' only when the wall-time is sufficient.",
    )
    parser.add_argument(
        "--output-root",
        default="rosr_extended_outputs",
        help="Directory receiving case_N CSV folders.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=int(os.environ.get("SLURM_CPUS_PER_TASK", "8")),
    )
    parser.add_argument("--planning-sample-size", type=int, default=40)
    parser.add_argument("--evaluation-sample-size", type=int, default=81)
    parser.add_argument("--planning-time-limit", type=float, default=300.0)
    parser.add_argument("--start-period-time-limit", type=float, default=180.0)
    parser.add_argument("--evaluation-time-limit", type=float, default=60.0)
    parser.add_argument("--perfect-information-time-limit", type=float, default=120.0)
    parser.add_argument("--multistage-time-limit", type=float, default=900.0)
    parser.add_argument("--robust-time-limit", type=float, default=900.0)
    parser.add_argument("--dro-time-limit", type=float, default=900.0)
    parser.add_argument("--dro-max-iterations", type=int, default=25)
    parser.add_argument("--skip-decision-rule", action="store_true")
    parser.add_argument("--skip-perfect-information", action="store_true")
    parser.add_argument(
        "--sampled-evaluation",
        action="store_true",
        help="Sample evaluation scenarios instead of using the original exhaustive set.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Use small scenario sets and short limits to test the installation.",
    )
    return parser


def main() -> None:
    args = _build_cli_parser().parse_args()

    if args.smoke_test:
        planning_sample_size = min(args.planning_sample_size, 6)
        evaluation_sample_size = min(args.evaluation_sample_size, 6)
        planning_limit = min(args.planning_time_limit, 60.0)
        start_limit = min(args.start_period_time_limit, 60.0)
        evaluation_limit = min(args.evaluation_time_limit, 20.0)
        perfect_limit = min(args.perfect_information_time_limit, 30.0)
        multistage_limit = min(args.multistage_time_limit, 120.0)
        robust_limit = min(args.robust_time_limit, 120.0)
        dro_limit = min(args.dro_time_limit, 120.0)
        dro_iterations = min(args.dro_max_iterations, 5)
        evaluation_exhaustive = False
    else:
        planning_sample_size = args.planning_sample_size
        evaluation_sample_size = args.evaluation_sample_size
        planning_limit = args.planning_time_limit
        start_limit = args.start_period_time_limit
        evaluation_limit = args.evaluation_time_limit
        perfect_limit = args.perfect_information_time_limit
        multistage_limit = args.multistage_time_limit
        robust_limit = args.robust_time_limit
        dro_limit = args.dro_time_limit
        dro_iterations = args.dro_max_iterations
        evaluation_exhaustive = not args.sampled_evaluation

    runtime = RuntimeConfig(
        planning_exhaustive=False,
        planning_sample_size=planning_sample_size,
        evaluation_exhaustive=evaluation_exhaustive,
        evaluation_sample_size=evaluation_sample_size,
        evaluation_include_no_launch=False,
        planning_time_limit=planning_limit,
        start_period_time_limit=start_limit,
        evaluation_time_limit=evaluation_limit,
        perfect_information_time_limit=perfect_limit,
        threads=args.threads,
        run_decision_rule=not args.skip_decision_rule,
        run_perfect_information=not args.skip_perfect_information,
    )
    extension = ExtensionRuntimeConfig(
        multistage_time_limit=multistage_limit,
        multistage_keep_frozen_shares=False,
        robust_time_limit=robust_limit,
        dro_time_limit=dro_limit,
        dro_max_iterations=dro_iterations,
        start_period_share_anchor=None,
        evaluate_out_of_sample=True,
        include_original_benchmarks=True,
    )

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if args.case == "all":
        run_all_extended_cases(
            runtime=runtime,
            extension=extension,
            output_root=output_root,
        )
    else:
        run_extended_case(
            int(args.case),
            runtime=runtime,
            extension=extension,
            output_root=output_root,
        )


if __name__ == "__main__":
    main()
