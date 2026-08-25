"""
The optimization model is shared across the analyses. Each analysis is represented
as an ordered design of independent parameter combinations. The command-line driver
can run one design point, a cyclic slice assigned to a Slurm array worker, or merge
checkpoint files into the original analysis-level CSV outputs.

Analysis 11 supports two modes:
- full: the stated 15 x 2^7 = 1,920 point factorial design;
- conditional: factors for absent product types are held at their first level.
"""

"""
Engaging cluster driver for ROSR OFAT and factorial analyses.

Shared model assumptions are aligned with rosr_modular_cases.ipynb:
- discount rate = 7.5%;
- machine CapEx = 25,000;
- process-flexibility arc CapEx = 1,000;
- annual OpEx = 1,000;
- machine lifetime = 10;
- baseline Growth slope = 2;
- baseline launch probability = 0.975;
- 40 probability-weighted planning scenarios by default;
- 81-state Increase/Flat/Decrease evaluation by default (NoLaunch excluded),
  matching the modular notebook's published evaluation convention.

Experiment-specific DOE settings (for example, the four-machine Analysis 11
factorial and its factor levels) remain explicit in build_analysis_design().
"""

import os
import itertools
import random

import numpy as np
import pandas as pd

try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError:
    gp = None
    GRB = None

# ===============================================================
# Helper: build stochastic capacity tensor with disruptions
# ===============================================================
def build_capacity_tensor(M, S, T_V, machine_capacity, availability, prob_disruption, rng):
    """
    Build one stochastic capacity tensor using a caller-supplied RNG stream.

    The shared RNG stream is intentional: the modular case notebook generates
    planning capacities first and then continues the same stream for evaluation.
    """
    c_ist = {}
    for i in M:
        for sid in S:
            for t in T_V:
                if rng.random() < prob_disruption:
                    c_ist[i, sid, t] = 0.0
                else:
                    c_ist[i, sid, t] = machine_capacity * rng.uniform(availability, 1.0)
    return c_ist


def run_flexibility_investment_analysis(
    # Product portfolio configuration
    established_products=4,
    launch_products=0,
    decline_products=0,          # primary name
    compete_products=None,       # optional alias; if set, overrides decline_products

    # Time horizons
    investment_window=4,
    valuation_horizon=14,

    # Resource configuration
    n_machines=8,
    machine_capacity=30,
    machine_lifetime=10,

    # Economic parameters
    discount_rate=0.075,
    net_revenue=300,
    net_revenue_E=None,
    net_revenue_L=None,
    net_revenue_C=None,
    annual_opex=1000,
    machine_capex=25000,
    arc_capex=1000,

    # Demand trajectory parameters
    base_demand_level=30,
    growth_demand_level=45,
    decline_demand_level=15,

    # Demand volatility slopes
    base_slope=1,
    growth_slope=2,
    decline_slope=1,

    # Product lifecycle timing
    launch_time=3,
    competition_time=3,
    growth_duration=1,
    decline_duration=1,

    # Uncertainty knobs
    availability=0.8,            # Uniform[availability,1] multiplier (no disruption)
    probability_launch=0.975,    # P(launch)  (NoLaunch prob = 1 - pL)
    prob_disruption=0.0,         # NEW: with prob_disruption, capacity = 0 for (i,s,t)

    # Scenario and solver parameters
    n_scenarios_sampled=40,
    time_limit_multi=300,
    time_limit_single=180,
    time_limit_long=180,
    seed=0,
    capacity_seed=73,
    evaluation_include_no_launch=False,
    suppress_output=True,
    solver_threads=8,
    time_limit_evaluation=60
):
    """
    Multi-Period Stochastic Flexibility Network Design and Resource Investment Model
    with capacity uncertainty via availability *and* disruption:

      c_{i,s,t} = 0                            w.p. prob_disruption
                  machine_capacity * U[a,1]    otherwise, where a = availability

    The rest of the logic is unchanged. Launch uncertainty: {Increase, Flat, Decrease, NoLaunch}
    with probs {pL/3, pL/3, pL/3, 1-pL}.
    """

    if gp is None or GRB is None:
        raise RuntimeError(
            "gurobipy is not installed. Activate the project environment and install "
            "the 'gurobipy' package before running these analyses."
        )

    if solver_threads < 1:
        raise ValueError("solver_threads must be at least 1.")
    if not 0.0 <= prob_disruption <= 1.0:
        raise ValueError("prob_disruption must be between 0 and 1.")
    if not 0.0 <= availability <= 1.0:
        raise ValueError("availability must be between 0 and 1.")
    if n_scenarios_sampled < 1:
        raise ValueError("n_scenarios_sampled must be at least 1.")

    # ------- alias handling (back-compat for Analysis 10 code) -------
    if compete_products is not None:
        decline_products = compete_products

    # =============================================================================
    # INTERNAL SETUP - Build sets and parameters from inputs
    # =============================================================================
    T_I = list(range(investment_window + 1))   # 0..θ
    T_V = list(range(valuation_horizon + 1))   # 0..ν

    M = [f"Resource {i+1}" for i in range(n_machines)]
    E = [f"Product E{i+1}" for i in range(established_products)]
    C = [f"Product C{i+1}" for i in range(decline_products)]
    L = [f"Product L{i+1}" for i in range(launch_products)]
    P = E + C + L
    if len(P) == 0:
        raise ValueError("At least one product is required.")

    P_type = {j: ("E" if j in E else "L" if j in L else "C") for j in P}

    # Optional type-specific unit revenues are required by Analysis 11.
    revenue_by_type = {
        "E": net_revenue if net_revenue_E is None else net_revenue_E,
        "L": net_revenue if net_revenue_L is None else net_revenue_L,
        "C": net_revenue if net_revenue_C is None else net_revenue_C,
    }
    p_ij = {(i, j): revenue_by_type[P_type[j]] for i in M for j in P}
    f_i = {i: annual_opex for i in M}
    g_i = {i: machine_capex for i in M}
    h_ij = {(i, j): arc_capex for i in M for j in P}
    L_i = {i: machine_lifetime for i in M}

    # Pre-install
    n_pre = len(E) + len(C)
    pre_installed_machines = M[:n_pre]
    pre_installed_arcs = list(zip(pre_installed_machines, E + C))
    for i, j in pre_installed_arcs:
        h_ij[i, j] = 0
    for i in pre_installed_machines:
        g_i[i] = 0

    BASE_STATES = ["Increase", "Flat", "Decrease"]
    LAUNCH_STATES = BASE_STATES + ["NoLaunch"]
    disc = lambda t: 1 / (1 + discount_rate) ** (t + 1)

    # =============================================================================
    # DEMAND TRAJECTORY GENERATION (unchanged)
    # =============================================================================
    def generate_demand_trajectories_internal():
        d_E = {j: base_demand_level for j in E}
        d_L = {j: growth_demand_level for j in L}
        t_L = {j: launch_time for j in L}
        q_L = {j: growth_duration for j in L}

        d_C_B = {j: base_demand_level for j in C}
        d_C = {j: decline_demand_level for j in C}
        t_C = {j: competition_time for j in C}
        q_C = {j: decline_duration for j in C}

        w_vals_base    = {"Increase": +base_slope,    "Flat": 0, "Decrease": -base_slope}
        w_vals_growth  = {"Increase": +growth_slope,  "Flat": 0, "Decrease": -growth_slope}
        w_vals_decline = {"Increase": +decline_slope, "Flat": 0, "Decrease": -decline_slope}

        def states_for_product(j):
            return LAUNCH_STATES if P_type[j] == "L" else BASE_STATES

        base_d_j_s_t = {}
        for j in P:
            for s in states_for_product(j):
                for t in T_V:
                    if P_type[j] == "E":
                        w = w_vals_base[s]
                        d = d_E[j] + w * t
                    elif P_type[j] == "L":
                        if s == "NoLaunch":
                            d = 0
                        else:
                            w = w_vals_growth[s]
                            if t < t_L[j]:
                                d = 0
                            elif t < t_L[j] + q_L[j]:
                                d = (d_L[j] / q_L[j]) * (t - t_L[j] + 1)
                            else:
                                d = d_L[j] + w * (t - t_L[j] - q_L[j] + 1)
                    else:
                        w = w_vals_decline[s]
                        if t < t_C[j]:
                            d = d_C_B[j]
                        elif t < t_C[j] + q_C[j]:
                            d = d_C_B[j] - (d_C_B[j] - d_C[j]) / q_C[j] * (t - t_C[j] + 1)
                        else:
                            d = d_C[j] + w * (t - t_C[j] - q_C[j] + 1)
                    base_d_j_s_t[j, s, t] = max(0, d)
        return base_d_j_s_t

    base_d_j_s_t = generate_demand_trajectories_internal()

    # =============================================================================
    # SCENARIO GENERATION (now uses prob_disruption)
    # =============================================================================
    def generate_joint_scenarios_internal(
        *,
        exhaustive=False,
        n_sampled=40,
        include_no_launch=True,
        weighted_sampling=True,
        capacity_rng=None,
    ):
        """
        Build joint demand/capacity scenarios using the same information structure
        as rosr_modular_cases.ipynb.

        Planning includes NoLaunch and uses probability-weighted scenario sampling.
        By default, evaluation excludes NoLaunch and uses the 3^4 equally weighted
        Increase/Flat/Decrease support, matching the modular notebook.
        """
        per_prod_states, per_prod_probs = [], []
        for j in P:
            if P_type[j] == "L" and include_no_launch:
                per_prod_states.append(LAUNCH_STATES)
                per_prod_probs.append(
                    [probability_launch / 3.0] * 3
                    + [max(0.0, 1.0 - probability_launch)]
                )
            else:
                per_prod_states.append(BASE_STATES)
                per_prod_probs.append([1.0 / 3.0] * 3)

        all_tuples = list(itertools.product(*per_prod_states))
        weights = np.empty(len(all_tuples), dtype=float)
        for idx, tup in enumerate(all_tuples):
            joint_weight = 1.0
            for product_index, state in enumerate(tup):
                states = per_prod_states[product_index]
                probs = per_prod_probs[product_index]
                joint_weight *= probs[states.index(state)]
            weights[idx] = joint_weight

        positive_idx = np.flatnonzero(weights > 0.0)
        if len(positive_idx) == 0:
            raise ValueError("All scenario weights are zero; check probability_launch.")

        if exhaustive:
            kept_indices = positive_idx
        else:
            k = min(n_sampled, len(positive_idx))
            rng = np.random.default_rng(seed)
            if weighted_sampling:
                sample_probabilities = weights / weights.sum()
            else:
                sample_probabilities = None
            kept_indices = rng.choice(
                len(all_tuples),
                size=k,
                replace=False,
                p=sample_probabilities,
            )

        new_scenario_map = {
            new_id: all_tuples[old_id]
            for new_id, old_id in enumerate(kept_indices)
        }
        S = list(new_scenario_map)

        kept_weights = weights[np.asarray(kept_indices, dtype=int)]
        kept_weights = kept_weights / kept_weights.sum()
        π_s = {
            sid: float(kept_weights[position])
            for position, sid in enumerate(S)
        }

        # The modular notebook uses an empirical uniform 81-state evaluation
        # distribution when NoLaunch is excluded.
        if exhaustive and not include_no_launch:
            π_s = {sid: 1.0 / len(S) for sid in S}

        d_j_s_t = {}
        for sid, tup in new_scenario_map.items():
            for p_idx, j in enumerate(P):
                state = tup[p_idx]
                for t in T_V:
                    d_j_s_t[j, sid, t] = base_d_j_s_t[j, state, t]

        if capacity_rng is None:
            capacity_rng = random.Random(capacity_seed)
        c_ist = build_capacity_tensor(
            M=M,
            S=S,
            T_V=T_V,
            machine_capacity=machine_capacity,
            availability=availability,
            prob_disruption=prob_disruption,
            rng=capacity_rng,
        )

        return S, new_scenario_map, π_s, d_j_s_t, c_ist

    # =============================================================================
    # OPEX COEFFICIENTS + MODELS (unchanged)
    # =============================================================================
    def build_opex_coeff_multi():
        coeff = {}
        for i in M:
            Li = L_i[i]
            for tau in T_I:
                t_end = min(tau + Li - 1, valuation_horizon)
                coeff[i, tau] = (f_i[i] * sum(disc(t) for t in range(tau, t_end + 1))) if tau <= t_end else 0.0
        return coeff

    def build_opex_coeff_single():
        coeff_i = {}
        for i in M:
            t_end = min(L_i[i] - 1, valuation_horizon)
            coeff_i[i] = f_i[i] * sum(disc(t) for t in range(0, t_end + 1)) if t_end >= 0 else 0.0
        return coeff_i

    # =============================================================================
    # OPTIMIZATION MODELS
    # =============================================================================

    def solve_multi_period_model_internal(S, π_s, d_j_s_t, c_ist):
        model = gp.Model('Multi-Period-Two-Stage')
        if suppress_output:
            model.Params.OutputFlag = 0
        model.Params.Threads = solver_threads
        model.Params.TimeLimit = time_limit_multi

        # Decision variables
        m_it = model.addVars(M, T_I, vtype=GRB.BINARY, name="machine_on")
        a_ijt = model.addVars(M, P, T_I, vtype=GRB.BINARY, name="arc_installed")
        u_it = model.addVars(M, T_I, vtype=GRB.BINARY, name="machine_startup")
        x_ijst = model.addVars(M, P, S, T_V, vtype=GRB.CONTINUOUS, name="production")
        y_ijs = model.addVars(M, P, S, lb=0, ub=1, name="market_share")

        # Objective components
        profit_term = gp.quicksum(
            disc(t) * π_s[s] * p_ij[i, j] * x_ijst[i, j, s, t]
            for i, j, s, t in itertools.product(M, P, S, T_V)
        )

        # Lifetime-block OPEX based on u_it
        opex_coeff = build_opex_coeff_multi()
        opex = gp.quicksum(opex_coeff[i, tau] * u_it[i, tau] for i, tau in itertools.product(M, T_I))

        # Machine CAPEX (discounted to install time t)
        capex_M = gp.quicksum(
            g_i[i] / (1 + discount_rate) ** t * u_it[i, t]
            for i, t in itertools.product(M, T_I)
        )

        # Arc CAPEX on increments (discounted to install time)
        capex_A = gp.LinExpr()
        for i, j in itertools.product(M, P):
            for t in T_I:
                prev_arc = 0 if t == 0 else a_ijt[i, j, t - 1]
                capex_A += h_ij[i, j] / (1 + discount_rate) ** t * (a_ijt[i, j, t] - prev_arc)

        model.setObjective(profit_term - opex - capex_M - capex_A, GRB.MAXIMIZE)

        # Constraints
        model.addConstrs((m_it[i, 0] == u_it[i, 0] for i in M), name="startup_t0")
        model.addConstrs((m_it[i, t] - m_it[i, t - 1] == u_it[i, t]
                          for i in M for t in T_I if t > 0), name="startup_def")

        model.addConstrs((a_ijt[i, j, t] >= a_ijt[i, j, t - 1]
                          for i in M for j in P for t in T_I if t > 0), name="arc_monotonic")
        model.addConstrs((a_ijt[i, j, t] <= m_it[i, t]
                          for i in M for j in P for t in T_I), name="arc_requires_machine")

        # Lifetime-based availability: production only if started within last L_i periods
        for i in M:
            for s in S:
                for t in range(investment_window + 1, valuation_horizon + 1):
                    startup_sum = gp.quicksum(
                        u_it[i, tau] for tau in range(
                            max(0, t - L_i[i] + 1),
                            min(t, investment_window) + 1
                        )
                    )
                    model.addConstr(
                        gp.quicksum(x_ijst[i, j, s, t] for j in P) <= c_ist[i, s, t] * startup_sum,
                        name=f"lifetime_{i}_{s}_{t}"
                    )

        # Standard constraints
        model.addConstrs((gp.quicksum(x_ijst[i, j, s, t] for i in M) <= d_j_s_t[j, s, t]
                          for j, s, t in itertools.product(P, S, T_V)), name="demand_limit")

        model.addConstrs((gp.quicksum(x_ijst[i, j, s, t] for j in P) <=
                          c_ist[i, s, t] * (m_it[i, t] if t <= investment_window else m_it[i, investment_window])
                          for i, s, t in itertools.product(M, S, T_V)), name="capacity_limit")

        model.addConstrs((x_ijst[i, j, s, t] <=
                          c_ist[i, s, t] * (a_ijt[i, j, t] if t <= investment_window else a_ijt[i, j, investment_window])
                          for i, j, s, t in itertools.product(M, P, S, T_V)), name="arc_enablement")

        model.addConstrs((x_ijst[i, j, s, investment_window] <= y_ijs[i, j, s] * d_j_s_t[j, s, investment_window]
                          for i, j, s in itertools.product(M, P, S)), name="share_definition")

        model.addConstrs((x_ijst[i, j, s, t] <= y_ijs[i, j, s] * d_j_s_t[j, s, t]
                          for i, j, s, t in itertools.product(M, P, S, T_V) if t > investment_window),
                         name="frozen_shares")

        model.addConstrs((gp.quicksum(y_ijs[i, j, s] for i in M) == 1
                          for j, s in itertools.product(P, S)), name="complete_shares")

        model.optimize()

        if model.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
            return model, {
                'm_it': {key: var.X for key, var in m_it.items()},
                'a_ijt': {key: var.X for key, var in a_ijt.items()},
                'u_it': {key: var.X for key, var in u_it.items()},
            }
        else:
            return model, None

    def solve_single_period_model_internal(S, π_s, d_j_s_t, c_ist):
        model = gp.Model('Single-Period-Two-Stage')
        if suppress_output:
            model.Params.OutputFlag = 0
        model.Params.Threads = solver_threads
        model.Params.TimeLimit = time_limit_single

        # Decision variables
        m_i = model.addVars(M, vtype=GRB.BINARY, name="machine_on")
        a_ij = model.addVars(M, P, vtype=GRB.BINARY, name="arc_installed")
        u_i = model.addVars(M, vtype=GRB.BINARY, name="machine_startup")
        x_ijst = model.addVars(M, P, S, T_V, vtype=GRB.CONTINUOUS, name="production")
        y_ijs = model.addVars(M, P, S, lb=0, ub=1, name="market_share")

        # Objective components
        profit_term = gp.quicksum(
            disc(t) * π_s[s] * p_ij[i, j] * x_ijst[i, j, s, t]
            for i, j, s, t in itertools.product(M, P, S, T_V)
        )

        # Lifetime-block OPEX with start at t=0
        opex_coeff_i = build_opex_coeff_single()
        opex = gp.quicksum(opex_coeff_i[i] * u_i[i] for i in M)

        # CAPEX
        capex_M = gp.quicksum(g_i[i] * u_i[i] for i in M)  # occurs at t=0
        capex_A = gp.quicksum(h_ij[i, j] * a_ij[i, j] for i, j in itertools.product(M, P))

        model.setObjective(profit_term - opex - capex_M - capex_A, GRB.MAXIMIZE)

        # Constraints
        model.addConstrs((m_i[i] == u_i[i] for i in M), name="startup_def")
        model.addConstrs((a_ij[i, j] <= m_i[i] for i in M for j in P), name="arc_requires_machine")

        # Lifetime constraints
        for i in M:
            for s in S:
                for t in T_V:
                    if t > L_i[i] - 1:
                        model.addConstr(
                            gp.quicksum(x_ijst[i, j, s, t] for j in P) <= 0,
                            name=f"lifetime_{i}_{s}_{t}"
                        )

        # Standard constraints
        model.addConstrs((gp.quicksum(x_ijst[i, j, s, t] for i in M) <= d_j_s_t[j, s, t]
                          for j, s, t in itertools.product(P, S, T_V)), name="demand_limit")

        model.addConstrs((gp.quicksum(x_ijst[i, j, s, t] for j in P) <= c_ist[i, s, t] * m_i[i]
                          for i, s, t in itertools.product(M, S, T_V)), name="capacity_limit")

        model.addConstrs((x_ijst[i, j, s, t] <= c_ist[i, s, t] * a_ij[i, j]
                          for i, j, s, t in itertools.product(M, P, S, T_V)), name="arc_enablement")

        model.addConstrs((x_ijst[i, j, s, investment_window] <= y_ijs[i, j, s] * d_j_s_t[j, s, investment_window]
                          for i, j, s in itertools.product(M, P, S)), name="share_definition")

        model.addConstrs((x_ijst[i, j, s, t] <= y_ijs[i, j, s] * d_j_s_t[j, s, t]
                          for i, j, s, t in itertools.product(M, P, S, T_V) if t > investment_window),
                         name="frozen_shares")

        model.addConstrs((gp.quicksum(y_ijs[i, j, s] for i in M) == 1
                          for j, s in itertools.product(P, S)), name="complete_shares")

        model.optimize()

        if model.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
            return model, {
                'm_i': {key: var.X for key, var in m_i.items()},
                'a_ij': {key: var.X for key, var in a_ij.items()},
                'u_i': {key: var.X for key, var in u_i.items()},
            }
        else:
            return model, None

    def evaluate_long_chain_solution_internal(S, π_s, d_j_s_t, c_ist):
        # Fixed investment pattern (requires exactly 4 products)
        def remap_products_internal(P_list):
            fixed_pattern = {
                ('Resource 1', 'Product 1'): 1, ('Resource 1', 'Product 2'): 0,
                ('Resource 1', 'Product 3'): 0, ('Resource 1', 'Product 4'): 1,
                ('Resource 2', 'Product 1'): 1, ('Resource 2', 'Product 2'): 1,
                ('Resource 2', 'Product 3'): 0, ('Resource 2', 'Product 4'): 0,
                ('Resource 3', 'Product 1'): 0, ('Resource 3', 'Product 2'): 1,
                ('Resource 3', 'Product 3'): 1, ('Resource 3', 'Product 4'): 0,
                ('Resource 4', 'Product 1'): 0, ('Resource 4', 'Product 2'): 0,
                ('Resource 4', 'Product 3'): 1, ('Resource 4', 'Product 4'): 1,
                ('Resource 5', 'Product 1'): 0, ('Resource 5', 'Product 2'): 0,
                ('Resource 5', 'Product 3'): 0, ('Resource 5', 'Product 4'): 0,
                ('Resource 6', 'Product 1'): 0, ('Resource 6', 'Product 2'): 0,
                ('Resource 6', 'Product 3'): 0, ('Resource 6', 'Product 4'): 0,
                ('Resource 7', 'Product 1'): 0, ('Resource 7', 'Product 2'): 0,
                ('Resource 7', 'Product 3'): 0, ('Resource 7', 'Product 4'): 0,
                ('Resource 8', 'Product 1'): 0, ('Resource 8', 'Product 2'): 0,
                ('Resource 8', 'Product 3'): 0, ('Resource 8', 'Product 4'): 0
            }
            if len(P_list) != 4:
                raise ValueError("Long chain pattern requires exactly 4 products")
            prod_map = {f"Product {i+1}": P_list[i] for i in range(4)}
            return {(r, prod_map[p]): v for (r, p), v in fixed_pattern.items() if p in prod_map}

        a_ij_long = remap_products_internal(P)
        m_i_long = {'Resource 1': 1, 'Resource 2': 1, 'Resource 3': 1, 'Resource 4': 1,
                    'Resource 5': 0, 'Resource 6': 0, 'Resource 7': 0, 'Resource 8': 0}
        u_i_long = m_i_long.copy()  # start at t=0 if on

        # Evaluation model
        model = gp.Model('Long-Chain-Evaluation')
        if suppress_output:
            model.Params.OutputFlag = 0
        model.Params.Threads = solver_threads
        model.Params.TimeLimit = time_limit_long

        x_ijst = model.addVars(M, P, S, T_V, vtype=GRB.CONTINUOUS, name="production")
        y_ijs = model.addVars(M, P, S, lb=0, ub=1, name="market_share")

        # Objective with fixed investments
        profit_term = gp.quicksum(
            disc(t) * π_s[s] * p_ij[i, j] * x_ijst[i, j, s, t]
            for i, j, s, t in itertools.product(M, P, S, T_V)
        )

        # Lifetime-block OPEX at t=0
        opex_coeff_i = build_opex_coeff_single()
        opex = gp.quicksum(opex_coeff_i[i] * u_i_long[i] for i in M)

        # CAPEX at t=0 (fixed)
        capex_M = gp.quicksum(g_i[i] * u_i_long[i] for i in M)
        capex_A = gp.quicksum(h_ij[i, j] * a_ij_long[i, j] for i, j in itertools.product(M, P))

        model.setObjective(profit_term - opex - capex_M - capex_A, GRB.MAXIMIZE)

        # Constraints with fixed infrastructure
        for i in M:
            for s in S:
                for t in T_V:
                    if t > L_i[i] - 1:
                        model.addConstr(
                            gp.quicksum(x_ijst[i, j, s, t] for j in P) <= 0,
                            name=f"lifetime_{i}_{s}_{t}"
                        )

        model.addConstrs((gp.quicksum(x_ijst[i, j, s, t] for i in M) <= d_j_s_t[j, s, t]
                          for j, s, t in itertools.product(P, S, T_V)), name="demand_limit")

        model.addConstrs((gp.quicksum(x_ijst[i, j, s, t] for j in P) <= c_ist[i, s, t] * m_i_long[i]
                          for i, s, t in itertools.product(M, S, T_V)), name="capacity_limit")

        model.addConstrs((x_ijst[i, j, s, t] <= c_ist[i, s, t] * a_ij_long[i, j]
                          for i, j, s, t in itertools.product(M, P, S, T_V)), name="arc_enablement")

        model.addConstrs((x_ijst[i, j, s, investment_window] <= y_ijs[i, j, s] * d_j_s_t[j, s, investment_window]
                          for i, j, s in itertools.product(M, P, S)), name="share_definition")

        model.addConstrs((x_ijst[i, j, s, t] <= y_ijs[i, j, s] * d_j_s_t[j, s, t]
                          for i, j, s, t in itertools.product(M, P, S, T_V) if t > investment_window),
                         name="frozen_shares")

        model.addConstrs((gp.quicksum(y_ijs[i, j, s] for i in M) == 1
                          for j, s in itertools.product(P, S)), name="complete_shares")

        model.optimize()

        if model.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
            return model.ObjVal, a_ij_long, m_i_long
        else:
            return 0, a_ij_long, m_i_long

    # =============================================================================
    # MAIN EXECUTION WORKFLOW
    # =============================================================================

    # # Generate training scenarios (probability-weighted sample)
    # S_train, _, π_s_train, d_j_s_t_train, c_ist_train = generate_joint_scenarios_internal(
    #     exhaustive=False, n_sampled=n_scenarios_sampled
    # )

    # # Solve models with training scenarios
    # multi_model, multi_solution = solve_multi_period_model_internal(S_train, π_s_train, d_j_s_t_train, c_ist_train)
    # single_model, single_solution = solve_single_period_model_internal(S_train, π_s_train, d_j_s_t_train, c_ist_train)

    # # Generate evaluation scenarios (exhaustive over per-product states)
    # S_eval, _, π_s_eval, d_j_s_t_eval, c_ist_eval = generate_joint_scenarios_internal(
    #     exhaustive=True, n_sampled=n_scenarios_sampled
    # )
    # Match the modular notebook: planning and evaluation share one capacity RNG
    # stream. Planning includes NoLaunch; the default evaluation does not.
    capacity_rng = random.Random(capacity_seed)

    S_train, _, π_s_train, d_j_s_t_train, c_ist_train = generate_joint_scenarios_internal(
        exhaustive=False,
        n_sampled=n_scenarios_sampled,
        include_no_launch=True,
        weighted_sampling=True,
        capacity_rng=capacity_rng,
    )
    multi_model, multi_solution = solve_multi_period_model_internal(
        S_train, π_s_train, d_j_s_t_train, c_ist_train
    )
    single_model, single_solution = solve_single_period_model_internal(
        S_train, π_s_train, d_j_s_t_train, c_ist_train
    )

    S_eval, _, π_s_eval, d_j_s_t_eval, c_ist_eval = generate_joint_scenarios_internal(
        exhaustive=True,
        n_sampled=n_scenarios_sampled,
        include_no_launch=evaluation_include_no_launch,
        weighted_sampling=evaluation_include_no_launch,
        capacity_rng=capacity_rng,
    )
    # Extract solution characteristics
    def analyze_solution_internal(solution_dict):
        if solution_dict is None:
            return [], 0
        if 'm_it' in solution_dict:  # Multi-period
            m_data = solution_dict['m_it']
            a_data = solution_dict['a_ijt']
            final_arcs = [(i, j) for (i, j, tt), val in a_data.items()
                          if tt == investment_window and val > 0.5]
            n_machines = len([i for i in M if m_data.get((i, investment_window), 0) > 0.5])
        else:  # Single-period
            m_data = solution_dict['m_i']
            a_data = solution_dict['a_ij']
            final_arcs = [(i, j) for (i, j), val in a_data.items() if val > 0.5]
            n_machines = len([m for m in m_data.values() if m > 0.5])
        return final_arcs, n_machines

    multi_arcs, multi_machines = analyze_solution_internal(multi_solution) if multi_solution else ([], 0)
    single_arcs, single_machines = analyze_solution_internal(single_solution) if single_solution else ([], 0)

    # Evaluate solutions on exhaustive scenarios
    def evaluate_solution_exhaustive_internal(solution_dict, model_type):
        if solution_dict is None:
            return 0

        model = gp.Model(f'{model_type}-Evaluation')
        if suppress_output:
            model.Params.OutputFlag = 0
        model.Params.Threads = solver_threads
        model.Params.TimeLimit = time_limit_evaluation

        x_ijst = model.addVars(M, P, S_eval, T_V, vtype=GRB.CONTINUOUS, name="x")
        y_ijs = model.addVars(M, P, S_eval, lb=0, ub=1, name="y")

        profit_term = gp.quicksum(
            disc(t) * π_s_eval[s] * p_ij[i, j] * x_ijst[i, j, s, t]
            for i, j, s, t in itertools.product(M, P, S_eval, T_V)
        )

        # Lifetime-block OPEX & CAPEX with fixed decisions
        if model_type == 'Multi-Period':
            m_data = solution_dict['m_it']
            a_data = solution_dict['a_ijt']
            u_data = solution_dict['u_it']

            # OPEX from u_data
            opex_coeff = {}
            for i in M:
                for tau in T_I:
                    t_end = min(tau + L_i[i] - 1, valuation_horizon)
                    opex_coeff[i, tau] = (f_i[i] * sum(disc(t) for t in range(tau, t_end + 1))) if tau <= t_end else 0.0
            opex = gp.quicksum(opex_coeff[i, tau] * u_data[i, tau] for i, tau in itertools.product(M, T_I))

            # CAPEX_M discounted by start time
            capex_M = gp.quicksum(g_i[i] / (1 + discount_rate) ** t * u_data[i, t]
                                  for i, t in itertools.product(M, T_I))

            # CAPEX_A on increments discounted to t
            capex_A = 0
            for i, j in itertools.product(M, P):
                for t in T_I:
                    prev = 0 if t == 0 else a_data[i, j, t - 1]
                    capex_A += h_ij[i, j] / (1 + discount_rate) ** t * (a_data[i, j, t] - prev)

            # Constraints
            for i in M:
                for s in S_eval:
                    for t in range(investment_window + 1, valuation_horizon + 1):
                        startup_sum = sum(u_data[i, tau] for tau in range(
                            max(0, t - L_i[i] + 1), min(t, investment_window) + 1))
                        model.addConstr(
                            gp.quicksum(x_ijst[i, j, s, t] for j in P) <= c_ist_eval[i, s, t] * startup_sum,
                            name=f"lifetime_{i}_{s}_{t}")

            model.addConstrs((gp.quicksum(x_ijst[i, j, s, t] for j in P) <=
                              c_ist_eval[i, s, t] * (m_data[i, t] if t <= investment_window else m_data[i, investment_window])
                              for i, s, t in itertools.product(M, S_eval, T_V)), name="capacity")

            model.addConstrs((x_ijst[i, j, s, t] <=
                              c_ist_eval[i, s, t] * (a_data[i, j, t] if t <= investment_window else a_data[i, j, investment_window])
                              for i, j, s, t in itertools.product(M, P, S_eval, T_V)), name="arc")

        else:  # Single-period
            m_data = solution_dict['m_i']
            a_data = solution_dict['a_ij']
            u_data = solution_dict['u_i']

            # OPEX from u_data at t=0
            opex_coeff_i = {}
            for i in M:
                t_end = min(L_i[i] - 1, valuation_horizon)
                opex_coeff_i[i] = f_i[i] * sum(disc(t) for t in range(0, t_end + 1)) if t_end >= 0 else 0.0
            opex = gp.quicksum(opex_coeff_i[i] * u_data[i] for i in M)

            capex_M = gp.quicksum(g_i[i] * u_data[i] for i in M)  # at t=0
            capex_A = gp.quicksum(h_ij[i, j] * a_data[i, j] for i, j in itertools.product(M, P))

            # Lifetime constraints
            for i in M:
                for s in S_eval:
                    for t in T_V:
                        if t > L_i[i] - 1:
                            model.addConstr(gp.quicksum(x_ijst[i, j, s, t] for j in P) <= 0,
                                            name=f"lifetime_{i}_{s}_{t}")

            model.addConstrs((gp.quicksum(x_ijst[i, j, s, t] for j in P) <= c_ist_eval[i, s, t] * m_data[i]
                              for i, s, t in itertools.product(M, S_eval, T_V)), name="capacity")

            model.addConstrs((x_ijst[i, j, s, t] <= c_ist_eval[i, s, t] * a_data[i, j]
                              for i, j, s, t in itertools.product(M, P, S_eval, T_V)), name="arc")

        # Common constraints
        model.addConstrs((gp.quicksum(x_ijst[i, j, s, t] for i in M) <= d_j_s_t_eval[j, s, t]
                          for j, s, t in itertools.product(P, S_eval, T_V)), name="demand")

        model.addConstrs((x_ijst[i, j, s, investment_window] <= y_ijs[i, j, s] * d_j_s_t_eval[j, s, investment_window]
                          for i, j, s in itertools.product(M, P, S_eval)), name="share_def")

        model.addConstrs((x_ijst[i, j, s, t] <= y_ijs[i, j, s] * d_j_s_t_eval[j, s, t]
                          for i, j, s, t in itertools.product(M, P, S_eval, T_V) if t > investment_window),
                         name="frozen_shares")

        model.addConstrs((gp.quicksum(y_ijs[i, j, s] for i in M) == 1
                          for j, s in itertools.product(P, S_eval)), name="complete_shares")

        # Objective
        model.setObjective(profit_term - opex - capex_M - capex_A, GRB.MAXIMIZE)
        model.optimize()

        return model.ObjVal if model.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT] else 0

    # Evaluate all solutions
    multi_profit = evaluate_solution_exhaustive_internal(multi_solution, 'Multi-Period') if multi_solution else 0
    single_profit = evaluate_solution_exhaustive_internal(single_solution, 'Single-Period') if single_solution else 0
    long_profit, long_arcs_dict, long_machines_dict = (0, None, None)
    # Long-chain only defined if exactly 4 products
    if len(P) == 4:
        long_profit, long_arcs_dict, long_machines_dict = evaluate_long_chain_solution_internal(
            S_eval, π_s_eval, d_j_s_t_eval, c_ist_eval
        )

    # Calculate final metrics
    n_arcs_multi = len(multi_arcs) if (multi_solution and (multi_arcs := [a for a in multi_solution['a_ijt'] if a[2] == investment_window and multi_solution['a_ijt'][a] > 0.5])) else 0
    # For multi_machines we already computed earlier; recompute safely:
    if multi_solution:
        n_multi_machines = len([i for i in M if multi_solution['m_it'].get((i, investment_window), 0) > 0.5])
    else:
        n_multi_machines = 0

    if single_solution:
        n_arcs_single = len([k for k, v in single_solution['a_ij'].items() if v > 0.5])
        n_single_machines = len([v for v in single_solution['m_i'].values() if v > 0.5])
    else:
        n_arcs_single = 0
        n_single_machines = 0

    n_arcs_long = sum(long_arcs_dict.values()) if long_arcs_dict else 0
    n_machines_long = sum(long_machines_dict.values()) if long_machines_dict else 0

    VMPSP = (multi_profit - single_profit) if (multi_profit and single_profit) else 0
    VMPLC = (multi_profit - long_profit) if (multi_profit and long_profit) else 0

    percent_VMPSP = (VMPSP / single_profit * 100) if single_profit > 0 else 0
    percent_VMPLC = (VMPLC / long_profit * 100) if long_profit and long_profit > 0 else 0

    return {
        'multi_profit': multi_profit,
        'single_profit': single_profit,
        'long_profit': long_profit if long_profit else 0,
        'n_arcs_multi': n_arcs_multi,
        'n_arcs_single': n_arcs_single,
        'n_arcs_long': n_arcs_long,
        'multi_machines': n_multi_machines,
        'single_machines': n_single_machines,
        'n_machines_long': n_machines_long,
        'VMPSP': VMPSP,
        'VMPLC': VMPLC,
        'percent_vmpsp': percent_VMPSP,
        'percent_vmplc': percent_VMPLC,
        'planning_scenarios': len(S_train),
        'evaluation_scenarios': len(S_eval),
        'evaluation_include_no_launch': bool(evaluation_include_no_launch),
        'capacity_seed': int(capacity_seed),
    }


# ======================================================================================
# ANALYSES 1-13: DESIGN DEFINITIONS AND CLUSTER DRIVER
# ======================================================================================

import argparse
import json
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


ALL_PORTFOLIOS = (
    (4, 0, 0), (3, 1, 0), (3, 0, 1),
    (2, 2, 0), (2, 1, 1), (2, 0, 2),
    (1, 3, 0), (1, 2, 1), (1, 1, 2),
    (1, 0, 3), (0, 4, 0), (0, 3, 1),
    (0, 2, 2), (0, 1, 3), (0, 0, 4),
)
LAUNCH_PORTFOLIOS = (
    (4, 0, 0), (3, 1, 0), (2, 2, 0), (1, 3, 0), (0, 4, 0),
)
DECLINE_PORTFOLIOS = (
    (4, 0, 0), (3, 0, 1), (2, 0, 2), (1, 0, 3), (0, 0, 4),
)

ANALYSIS_NAMES = {
    1: "All product portfolio combinations",
    2: "Launch products vs net revenue",
    3: "Launch products vs growth demand level",
    4: "Launch products vs growth slope",
    5: "Launch products vs launch time",
    6: "Decline products vs competition time",
    7: "Decline products vs net revenue",
    8: "Decline products vs decline demand level",
    9: "Decline products vs decline slope",
    10: "Probability of launch OFAT",
    11: "Product-mix + seven-factor full factorial design",
    12: "Established-only availability vs base slope",
    13: "Established-only disruption probability vs base slope",
}

OUTPUT_FILENAMES = {
    1: "analysis_1_all_combinations.csv",
    2: "analysis_2_launches_vs_net_revenue.csv",
    3: "analysis_3_launches_vs_growth_demand_level.csv",
    4: "analysis_4_launches_vs_growth_slope.csv",
    5: "analysis_5_launches_vs_launch_time.csv",
    6: "analysis_6_declines_vs_competition_time.csv",
    7: "analysis_7_declines_vs_net_revenue.csv",
    8: "analysis_8_declines_vs_decline_demand_level.csv",
    9: "analysis_9_declines_vs_decline_slope.csv",
    10: "analysis_10_ofat_probability_launch.csv",
    11: "analysis_11_factorial_design.csv",
    12: "analysis_12_established_availability_slope.csv",
    13: "analysis_13_probdisruption_base_slope.csv",
}

DEFAULT_SCENARIOS = {
    **{analysis: 40 for analysis in range(1, 12)},
    12: 25,
    13: 30,
}


@dataclass(frozen=True)
class DesignPoint:
    analysis: int
    local_index: int
    metadata: dict[str, Any]
    model_kwargs: dict[str, Any]

    @property
    def run_id(self) -> int:
        return self.local_index + 1


def parse_float_list(value: str) -> tuple[float, ...]:
    """Parse comma-separated floating-point values."""
    try:
        parsed = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Expected comma-separated numbers: {value!r}") from exc
    if not parsed:
        raise argparse.ArgumentTypeError("At least one number is required.")
    return parsed


def parse_analysis_selection(value: str) -> tuple[int, ...]:
    """Parse 'all', a single number, or a comma-separated analysis list."""
    text = value.strip().lower()
    if text == "all":
        return tuple(range(1, 14))
    try:
        analyses = tuple(dict.fromkeys(int(item.strip()) for item in text.split(",") if item.strip()))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use 'all' or analysis numbers 1-13.") from exc
    if not analyses or any(number not in ANALYSIS_NAMES for number in analyses):
        raise argparse.ArgumentTypeError("Analysis numbers must be between 1 and 13.")
    return analyses


def _point(
    analysis: int,
    local_index: int,
    portfolio: tuple[int, int, int],
    metadata: dict[str, Any] | None = None,
    model_kwargs: dict[str, Any] | None = None,
) -> DesignPoint:
    established, launch, decline = portfolio
    base_metadata = {
        "established_products": established,
        "launch_products": launch,
        "decline_products": decline,
    }
    base_kwargs = {
        "established_products": established,
        "launch_products": launch,
        "decline_products": decline,
    }
    if metadata:
        base_metadata.update(metadata)
    if model_kwargs:
        base_kwargs.update(model_kwargs)
    return DesignPoint(analysis, local_index, base_metadata, base_kwargs)


def build_analysis_design(
    analysis: int,
    *,
    analysis11_mode: str = "full",
    availability_1to9: float = 0.8,
    probability_launch_1to9: float = 0.975,
    availability_a10: float = 1.0,
    availability_a11: float = 0.8,
    a10_probabilities: Sequence[float] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    a12_availabilities: Sequence[float] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    a12_base_slopes: Sequence[float] = (0.0, 0.25, 0.5, 1.0),
    a13_prob_disruptions: Sequence[float] = (
        0.0, 0.05, 0.10, 0.15, 0.20, 0.25,
        0.30, 0.35, 0.40, 0.45, 0.50,
    ),
    a13_base_slopes: Sequence[float] = (0.0, 0.25, 0.5, 1.0),
) -> list[DesignPoint]:
    """Build one analysis in the exact loop order used by the notebook."""
    if analysis not in ANALYSIS_NAMES:
        raise ValueError(f"Unknown analysis: {analysis}")
    if analysis11_mode not in {"full", "conditional"}:
        raise ValueError("analysis11_mode must be 'full' or 'conditional'.")

    design: list[DesignPoint] = []

    def add(portfolio, metadata=None, kwargs=None):
        design.append(_point(analysis, len(design), portfolio, metadata, kwargs))

    common_1to9 = {
        "availability": float(availability_1to9),
        "probability_launch": float(probability_launch_1to9),
        "prob_disruption": 0.0,
    }

    if analysis == 1:
        for portfolio in ALL_PORTFOLIOS:
            add(
                portfolio,
                {
                    "availability": availability_1to9,
                    "probability_launch": probability_launch_1to9,
                },
                common_1to9,
            )

    elif analysis == 2:
        for portfolio in LAUNCH_PORTFOLIOS:
            for value in (100, 200, 300, 400, 500, 600, 700, 800, 900, 1000):
                add(
                    portfolio,
                    {
                        "net_revenue": value,
                        "availability": availability_1to9,
                        "probability_launch": probability_launch_1to9,
                    },
                    {**common_1to9, "net_revenue": value},
                )

    elif analysis == 3:
        for portfolio in LAUNCH_PORTFOLIOS:
            for value in (30, 35, 40, 45, 50, 55, 60):
                add(
                    portfolio,
                    {
                        "growth_demand_level": value,
                        "availability": availability_1to9,
                        "probability_launch": probability_launch_1to9,
                    },
                    {**common_1to9, "growth_demand_level": value},
                )

    elif analysis == 4:
        for portfolio in LAUNCH_PORTFOLIOS:
            for value in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0):
                add(
                    portfolio,
                    {
                        "growth_slope": value,
                        "availability": availability_1to9,
                        "probability_launch": probability_launch_1to9,
                    },
                    {**common_1to9, "growth_slope": value},
                )

    elif analysis == 5:
        for portfolio in LAUNCH_PORTFOLIOS:
            for value in (1, 2, 3, 4, 5):
                add(
                    portfolio,
                    {
                        "launch_time": value,
                        "availability": availability_1to9,
                        "probability_launch": probability_launch_1to9,
                    },
                    {**common_1to9, "launch_time": value},
                )

    elif analysis == 6:
        for portfolio in DECLINE_PORTFOLIOS:
            for value in (1, 2, 3, 4, 5):
                add(
                    portfolio,
                    {
                        "competition_time": value,
                        "availability": availability_1to9,
                        "probability_launch": probability_launch_1to9,
                    },
                    {**common_1to9, "competition_time": value},
                )

    elif analysis == 7:
        for portfolio in DECLINE_PORTFOLIOS:
            for value in (100, 200, 300, 400, 500, 600, 700, 800, 900, 1000):
                add(
                    portfolio,
                    {
                        "net_revenue": value,
                        "availability": availability_1to9,
                        "probability_launch": probability_launch_1to9,
                    },
                    {**common_1to9, "net_revenue": value},
                )

    elif analysis == 8:
        for portfolio in DECLINE_PORTFOLIOS:
            for value in (5, 10, 15, 20, 25, 30):
                add(
                    portfolio,
                    {
                        "decline_demand_level": value,
                        "availability": availability_1to9,
                        "probability_launch": probability_launch_1to9,
                    },
                    {**common_1to9, "decline_demand_level": value},
                )

    elif analysis == 9:
        for portfolio in DECLINE_PORTFOLIOS:
            for value in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0):
                add(
                    portfolio,
                    {
                        "decline_slope": value,
                        "availability": availability_1to9,
                        "probability_launch": probability_launch_1to9,
                    },
                    {**common_1to9, "decline_slope": value},
                )

    elif analysis == 10:
        for portfolio in LAUNCH_PORTFOLIOS:
            for value in a10_probabilities:
                add(
                    portfolio,
                    {
                        "availability": availability_a10,
                        "probability_launch": float(value),
                    },
                    {
                        "availability": float(availability_a10),
                        "probability_launch": float(value),
                        "prob_disruption": 0.0,
                    },
                )

    elif analysis == 11:
        launch_demand_levels = (30, 60)
        compete_demand_levels = (10, 25)
        established_demand_levels = (15, 30)
        profit_launch_levels = (300, 600)
        profit_established_levels = (300, 600)
        profit_compete_levels = (300, 600)
        probability_launch_levels = (0.50, 0.75)
        full = analysis11_mode == "full"

        for established, launch, decline in ALL_PORTFOLIOS:
            portfolio = (established, launch, decline)
            gdl_values = launch_demand_levels if (full or launch > 0) else launch_demand_levels[:1]
            ddl_values = compete_demand_levels if (full or decline > 0) else compete_demand_levels[:1]
            bdl_values = established_demand_levels if (full or established > 0) else established_demand_levels[:1]
            p_l_values = profit_launch_levels if (full or launch > 0) else profit_launch_levels[:1]
            p_e_values = profit_established_levels if (full or established > 0) else profit_established_levels[:1]
            p_c_values = profit_compete_levels if (full or decline > 0) else profit_compete_levels[:1]
            prob_values = probability_launch_levels if (full or launch > 0) else probability_launch_levels[:1]

            for gdl in gdl_values:
                for ddl in ddl_values:
                    for bdl in bdl_values:
                        for p_l in p_l_values:
                            for p_e in p_e_values:
                                for p_c in p_c_values:
                                    for probability in prob_values:
                                        add(
                                            portfolio,
                                            {
                                                "compete_products": decline,
                                                "established_demand": bdl,
                                                "launch_demand": gdl,
                                                "compete_demand": ddl,
                                                "profit_established": p_e,
                                                "profit_launch": p_l,
                                                "profit_compete": p_c,
                                                "probability_launch": probability,
                                                "availability": availability_a11,
                                                "analysis11_mode": analysis11_mode,
                                            },
                                            {
                                                "n_machines": 4,
                                                "machine_capacity": 30,
                                                "machine_lifetime": 10,
                                                "discount_rate": 0.075,
                                                "annual_opex": 1000,
                                                "machine_capex": 25000,
                                                "arc_capex": 1000,
                                                "base_slope": 1,
                                                "growth_slope": 2,
                                                "decline_slope": 1,
                                                "base_demand_level": bdl,
                                                "growth_demand_level": gdl,
                                                "decline_demand_level": ddl,
                                                "net_revenue_E": p_e,
                                                "net_revenue_L": p_l,
                                                "net_revenue_C": p_c,
                                                "probability_launch": probability,
                                                "availability": float(availability_a11),
                                                "prob_disruption": 0.0,
                                            },
                                        )

    elif analysis == 12:
        for availability in a12_availabilities:
            for slope in a12_base_slopes:
                add(
                    (4, 0, 0),
                    {
                        "availability": float(availability),
                        "base_slope": float(slope),
                    },
                    {
                        "availability": float(availability),
                        "base_slope": float(slope),
                        "prob_disruption": 0.0,
                    },
                )

    elif analysis == 13:
        for probability in a13_prob_disruptions:
            if not 0.0 <= float(probability) <= 1.0:
                raise ValueError(f"Invalid disruption probability: {probability}")
            for slope in a13_base_slopes:
                add(
                    (4, 0, 0),
                    {
                        "availability": 1.0,
                        "prob_disruption": float(probability),
                        "base_slope": float(slope),
                    },
                    {
                        "availability": 1.0,
                        "prob_disruption": float(probability),
                        "base_slope": float(slope),
                    },
                )

    return design


def build_selected_design(analyses: Sequence[int], args: argparse.Namespace) -> list[DesignPoint]:
    points: list[DesignPoint] = []
    for analysis in analyses:
        points.extend(
            build_analysis_design(
                analysis,
                analysis11_mode=args.analysis11_mode,
                availability_1to9=args.availability_1to9,
                probability_launch_1to9=args.probability_launch_1to9,
                availability_a10=args.availability_a10,
                availability_a11=args.availability_a11,
                a10_probabilities=args.a10_probabilities,
                a12_availabilities=args.a12_availabilities,
                a12_base_slopes=args.a12_base_slopes,
                a13_prob_disruptions=args.a13_prob_disruptions,
                a13_base_slopes=args.a13_base_slopes,
            )
        )
    return points


def analysis_directory(base_dir: Path, analysis: int) -> Path:
    return base_dir / f"analysis_{analysis:02d}"


def checkpoint_path(base_dir: Path, point: DesignPoint) -> Path:
    return analysis_directory(base_dir, point.analysis) / "tasks" / f"task_{point.local_index:05d}.csv"


def atomic_write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame(rows).to_csv(temporary, index=False)
    temporary.replace(path)


def successful_checkpoint(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        frame = pd.read_csv(path)
    except Exception:
        return False
    return (
        len(frame) == 1
        and "status" in frame.columns
        and str(frame.iloc[0]["status"]).lower() == "success"
    )


def run_design_point(
    point: DesignPoint,
    *,
    output_dir: Path,
    n_scenarios_override: int | None,
    seed: int,
    capacity_seed: int,
    evaluation_include_no_launch: bool,
    threads: int,
    time_limit_multi: float,
    time_limit_single: float,
    time_limit_long: float,
    time_limit_evaluation: float,
    suppress_output: bool,
    force: bool,
) -> dict[str, Any]:
    path = checkpoint_path(output_dir, point)
    if not force and successful_checkpoint(path):
        print(
            f"[A{point.analysis:02d} point {point.local_index:05d}] already complete; skipping.",
            flush=True,
        )
        return pd.read_csv(path).iloc[0].to_dict()

    scenario_count = (
        int(n_scenarios_override)
        if n_scenarios_override is not None
        else DEFAULT_SCENARIOS[point.analysis]
    )
    start = time.perf_counter()

    base_row: dict[str, Any] = {
        "analysis": point.analysis,
        "analysis_name": ANALYSIS_NAMES[point.analysis],
        "analysis_task_index": point.local_index,
        "run_id": point.run_id,
        **point.metadata,
        "n_scenarios_sampled": scenario_count,
        "seed": int(seed),
        "capacity_seed": int(capacity_seed),
        "evaluation_include_no_launch": bool(evaluation_include_no_launch),
        "solver_threads": int(threads),
        "time_limit_multi": float(time_limit_multi),
        "time_limit_single": float(time_limit_single),
        "time_limit_long": float(time_limit_long),
        "time_limit_evaluation": float(time_limit_evaluation),
    }

    print(
        f"[A{point.analysis:02d} point {point.local_index:05d}/{point.run_id}] starting: "
        f"{json.dumps(point.metadata, sort_keys=True)}",
        flush=True,
    )

    try:
        result = run_flexibility_investment_analysis(
            **point.model_kwargs,
            n_scenarios_sampled=scenario_count,
            time_limit_multi=time_limit_multi,
            time_limit_single=time_limit_single,
            time_limit_long=time_limit_long,
            time_limit_evaluation=time_limit_evaluation,
            seed=seed,
            capacity_seed=capacity_seed,
            evaluation_include_no_launch=evaluation_include_no_launch,
            suppress_output=suppress_output,
            solver_threads=threads,
        )
        elapsed = time.perf_counter() - start
        row = {
            **base_row,
            **result,
            "status": "success",
            "runtime_seconds": elapsed,
            "error_type": "",
            "error_message": "",
        }
        atomic_write_csv([row], path)
        print(
            f"[A{point.analysis:02d} point {point.local_index:05d}] complete in {elapsed:.1f}s; "
            f"Multi={result['multi_profit']:.2f}, Single={result['single_profit']:.2f}, "
            f"Long={result['long_profit']:.2f}",
            flush=True,
        )
        return row
    except Exception as exc:
        elapsed = time.perf_counter() - start
        row = {
            **base_row,
            "status": "failed",
            "runtime_seconds": elapsed,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }
        atomic_write_csv([row], path)
        print(
            f"[A{point.analysis:02d} point {point.local_index:05d}] FAILED after {elapsed:.1f}s: "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )
        traceback.print_exc()
        return row


def merge_analysis_results(
    analysis: int,
    *,
    output_dir: Path,
    expected_points: int | None = None,
) -> Path:
    directory = analysis_directory(output_dir, analysis)
    task_files = sorted((directory / "tasks").glob("task_*.csv"))
    if not task_files:
        raise FileNotFoundError(f"No task files found for Analysis {analysis} in {directory}.")

    frames: list[pd.DataFrame] = []
    for path in task_files:
        try:
            frames.append(pd.read_csv(path))
        except Exception as exc:
            print(f"WARNING: could not read {path}: {exc}", flush=True)

    if not frames:
        raise RuntimeError(f"No readable task files found for Analysis {analysis}.")

    merged = pd.concat(frames, ignore_index=True, sort=False)
    merged = (
        merged.sort_values("analysis_task_index")
        .drop_duplicates(subset=["analysis_task_index"], keep="last")
        .reset_index(drop=True)
    )
    final_path = directory / OUTPUT_FILENAMES[analysis]
    atomic_write_csv(merged.to_dict(orient="records"), final_path)

    success = int((merged["status"] == "success").sum()) if "status" in merged else 0
    failed = int((merged["status"] == "failed").sum()) if "status" in merged else 0
    print(
        f"Analysis {analysis}: merged {len(merged)} rows -> {final_path} "
        f"(success={success}, failed={failed})",
        flush=True,
    )
    if expected_points is not None and len(merged) != expected_points:
        print(
            f"WARNING: Analysis {analysis} expected {expected_points} points but found {len(merged)}.",
            flush=True,
        )
    return final_path


def merge_selected_analyses(
    analyses: Sequence[int],
    *,
    output_dir: Path,
    args: argparse.Namespace,
) -> list[Path]:
    paths: list[Path] = []
    summary_rows: list[dict[str, Any]] = []
    for analysis in analyses:
        expected = len(build_analysis_design(
            analysis,
            analysis11_mode=args.analysis11_mode,
            availability_1to9=args.availability_1to9,
            probability_launch_1to9=args.probability_launch_1to9,
            availability_a10=args.availability_a10,
            availability_a11=args.availability_a11,
            a10_probabilities=args.a10_probabilities,
            a12_availabilities=args.a12_availabilities,
            a12_base_slopes=args.a12_base_slopes,
            a13_prob_disruptions=args.a13_prob_disruptions,
            a13_base_slopes=args.a13_base_slopes,
        ))
        try:
            path = merge_analysis_results(
                analysis,
                output_dir=output_dir,
                expected_points=expected,
            )
            paths.append(path)
            frame = pd.read_csv(path)
            summary_rows.append({
                "analysis": analysis,
                "analysis_name": ANALYSIS_NAMES[analysis],
                "expected_points": expected,
                "rows_found": len(frame),
                "successful": int((frame.get("status") == "success").sum()),
                "failed": int((frame.get("status") == "failed").sum()),
                "output_file": str(path),
            })
        except FileNotFoundError as exc:
            print(f"WARNING: {exc}", flush=True)
            summary_rows.append({
                "analysis": analysis,
                "analysis_name": ANALYSIS_NAMES[analysis],
                "expected_points": expected,
                "rows_found": 0,
                "successful": 0,
                "failed": 0,
                "output_file": "",
            })

    if summary_rows:
        summary_path = output_dir / "all_analyses_summary.csv"
        atomic_write_csv(summary_rows, summary_path)
        print(f"Wrote summary: {summary_path}", flush=True)
    return paths


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run or merge Analyses 1-13 from the ROSR notebook on MIT Engaging."
    )
    parser.add_argument(
        "--analysis",
        type=parse_analysis_selection,
        default=(1,),
        help="Analysis number, comma-separated numbers, or 'all'. Default: 1.",
    )
    parser.add_argument(
        "--analysis11-mode",
        choices=("full", "conditional"),
        default="full",
        help="Analysis 11 full=1920 points; conditional holds absent-type factors fixed.",
    )
    parser.add_argument("--task-index", type=int, help="Run one point from the selected design.")
    parser.add_argument("--worker-index", type=int, help="Zero-based Slurm worker index.")
    parser.add_argument("--worker-count", type=int, help="Total Slurm workers in the array.")
    parser.add_argument("--serial", action="store_true", help="Run every selected point serially.")
    parser.add_argument("--count", action="store_true", help="Print selected design size and exit.")
    parser.add_argument("--summary", action="store_true", help="Print per-analysis point counts and exit.")
    parser.add_argument("--list-design", action="store_true", help="Print design metadata and exit.")
    parser.add_argument("--merge", action="store_true", help="Merge checkpoint CSVs and exit.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os.environ.get("ROSR_OUTPUT_DIR", "rosr_all_analysis_results")),
    )
    parser.add_argument("--force", action="store_true", help="Rerun successful checkpoints.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--capacity-seed",
        type=int,
        default=73,
        help="Capacity RNG seed; 73 matches rosr_modular_cases.ipynb.",
    )
    parser.add_argument(
        "--evaluation-include-no-launch",
        action="store_true",
        help=(
            "Include NoLaunch in exhaustive evaluation. Default is False to "
            "match the modular notebook's 81-state evaluation convention."
        ),
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=int(os.environ.get("SLURM_CPUS_PER_TASK", "8")),
    )
    parser.add_argument("--n-scenarios", type=int, default=None, help="Override per-analysis defaults.")
    parser.add_argument("--time-limit-multi", type=float, default=300)
    parser.add_argument("--time-limit-single", type=float, default=180)
    parser.add_argument("--time-limit-long", type=float, default=180)
    parser.add_argument("--time-limit-evaluation", type=float, default=60)
    parser.add_argument("--show-gurobi-output", action="store_true")
    parser.add_argument(
        "--fail-job-on-error",
        action="store_true",
        help="Return a nonzero worker exit code when any assigned point fails.",
    )
    parser.add_argument("--gurobi-license-file", type=Path)

    # Notebook-level constants exposed as optional overrides.
    parser.add_argument("--availability-1to9", type=float, default=0.8)
    parser.add_argument("--probability-launch-1to9", type=float, default=0.975)
    parser.add_argument("--availability-a10", type=float, default=1.0)
    parser.add_argument("--availability-a11", type=float, default=0.8)
    parser.add_argument(
        "--a10-probabilities",
        type=parse_float_list,
        default=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    )
    parser.add_argument(
        "--a12-availabilities",
        type=parse_float_list,
        default=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    )
    parser.add_argument(
        "--a12-base-slopes",
        type=parse_float_list,
        default=(0.0, 0.25, 0.5, 1.0),
    )
    parser.add_argument(
        "--a13-prob-disruptions",
        type=parse_float_list,
        default=(0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50),
    )
    parser.add_argument(
        "--a13-base-slopes",
        type=parse_float_list,
        default=(0.0, 0.25, 0.5, 1.0),
    )
    return parser


def validate_arguments(args: argparse.Namespace) -> None:
    if args.threads < 1:
        raise ValueError("--threads must be at least 1.")
    if args.n_scenarios is not None and args.n_scenarios < 1:
        raise ValueError("--n-scenarios must be at least 1.")
    if (args.worker_index is None) != (args.worker_count is None):
        raise ValueError("Use --worker-index and --worker-count together.")
    if args.worker_count is not None:
        if args.worker_count < 1:
            raise ValueError("--worker-count must be at least 1.")
        if not 0 <= args.worker_index < args.worker_count:
            raise ValueError("--worker-index must satisfy 0 <= index < worker-count.")
    execution_modes = sum([
        args.task_index is not None,
        args.worker_index is not None,
        args.serial,
    ])
    if execution_modes > 1:
        raise ValueError("Choose only one of --task-index, --worker-index, or --serial.")


def main(argv: Sequence[str] | None = None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)
    validate_arguments(args)
    analyses = args.analysis

    if args.gurobi_license_file:
        license_path = args.gurobi_license_file.expanduser().resolve()
        if not license_path.is_file():
            raise FileNotFoundError(f"Gurobi license file not found: {license_path}")
        os.environ["GRB_LICENSE_FILE"] = str(license_path)

    if args.summary:
        total = 0
        for analysis in analyses:
            count = len(build_analysis_design(
                analysis,
                analysis11_mode=args.analysis11_mode,
                availability_1to9=args.availability_1to9,
                probability_launch_1to9=args.probability_launch_1to9,
                availability_a10=args.availability_a10,
                availability_a11=args.availability_a11,
                a10_probabilities=args.a10_probabilities,
                a12_availabilities=args.a12_availabilities,
                a12_base_slopes=args.a12_base_slopes,
                a13_prob_disruptions=args.a13_prob_disruptions,
                a13_base_slopes=args.a13_base_slopes,
            ))
            total += count
            print(f"A{analysis:02d}: {count:4d}  {ANALYSIS_NAMES[analysis]}")
        print(f"TOTAL: {total}")
        return 0

    design = build_selected_design(analyses, args)

    if args.count:
        print(len(design))
        return 0

    if args.list_design:
        for global_index, point in enumerate(design):
            print(json.dumps({
                "global_index": global_index,
                "analysis": point.analysis,
                "analysis_task_index": point.local_index,
                **point.metadata,
            }, sort_keys=True))
        return 0

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.merge:
        merge_selected_analyses(analyses, output_dir=output_dir, args=args)
        return 0

    if args.task_index is not None:
        if not 0 <= args.task_index < len(design):
            parser.error(f"--task-index must be between 0 and {len(design)-1}.")
        assigned = [design[args.task_index]]
    elif args.worker_index is not None:
        assigned = design[args.worker_index::args.worker_count]
        print(
            f"Worker {args.worker_index}/{args.worker_count} assigned {len(assigned)} "
            f"of {len(design)} selected design points.",
            flush=True,
        )
    elif args.serial:
        assigned = design
    else:
        parser.error(
            "Choose --task-index, --worker-index/--worker-count, --serial, "
            "--count, --summary, --list-design, or --merge."
        )

    failures = 0
    for point in assigned:
        row = run_design_point(
            point,
            output_dir=output_dir,
            n_scenarios_override=args.n_scenarios,
            seed=args.seed,
            capacity_seed=args.capacity_seed,
            evaluation_include_no_launch=args.evaluation_include_no_launch,
            threads=args.threads,
            time_limit_multi=args.time_limit_multi,
            time_limit_single=args.time_limit_single,
            time_limit_long=args.time_limit_long,
            time_limit_evaluation=args.time_limit_evaluation,
            suppress_output=not args.show_gurobi_output,
            force=args.force,
        )
        if str(row.get("status", "")).lower() == "failed":
            failures += 1

    print(
        f"Worker finished: assigned={len(assigned)}, failures={failures}, output={output_dir}",
        flush=True,
    )
    return 1 if (failures and args.fail_job_on_error) else 0


if __name__ == "__main__":
    raise SystemExit(main())
