import math
import random
import time

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# PAGE SETUP
# ============================================================

st.set_page_config(
    page_title="Transformation Strategy Lab",
    layout="wide",
)

st.markdown(
    """
<style>
.block-container {
    padding-top: 1.4rem;
    padding-bottom: 2rem;
    max-width: 1500px;
}
div[data-testid="stMetric"] {
    border: 1px solid rgba(128,128,128,0.22);
    border-radius: 10px;
    padding: 0.8rem;
}
.scenario-card {
    border: 1px solid rgba(128,128,128,0.28);
    border-radius: 14px;
    padding: 1.15rem 1.2rem 1.0rem 1.2rem;
    min-height: 235px;
    margin-bottom: 0.65rem;
}
.small-note {
    color: rgba(128,128,128,0.95);
    font-size: 0.92rem;
}
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# MODEL CONSTANTS
# ============================================================

ROLE_COUNTS = {
    "Referring clinicians": 20,
    "Home-care nurses": 20,
    "Care coordinators": 20,
    "Clinical / operational leaders": 10,
}

ROLE_SHORT = {
    "Referring clinicians": "Clinician",
    "Home-care nurses": "Nurse",
    "Care coordinators": "Coordinator",
    "Clinical / operational leaders": "Leader",
}

ROLE_SYMBOLS = {
    "Referring clinicians": "circle",
    "Home-care nurses": "diamond",
    "Care coordinators": "square",
    "Clinical / operational leaders": "triangle-up",
}

STATE_COLORS = {
    "Not engaged": "#9CA3AF",
    "Considering": "#FACC15",
    "Trial": "#FB923C",
    "Adopted": "#22C55E",
    "Abandoned": "#EF4444",
}

STATE_ORDER = [
    "Not engaged",
    "Considering",
    "Trial",
    "Adopted",
    "Abandoned",
]

DEFAULT_ROLE_CONDITIONS = pd.DataFrame(
    [
        ["Referring clinicians", 50, 60, 40, 45],
        ["Home-care nurses", 60, 70, 50, 55],
        ["Care coordinators", 55, 65, 45, 60],
        ["Clinical / operational leaders", 70, 40, 75, 70],
    ],
    columns=[
        "Role",
        "Trust",
        "Burnout",
        "Workflow confidence",
        "Coordination readiness",
    ],
).set_index("Role")


# ============================================================
# HELPERS
# ============================================================

def clamp(x, low=0.0, high=100.0):
    return float(max(low, min(high, x)))


def logistic(x):
    return 1.0 / (1.0 + math.exp(-x))


def readiness_score(row):
    return (
        0.30 * row["trust"]
        + 0.25 * row["coordination_readiness"]
        + 0.30 * row["workflow_confidence"]
        + 0.15 * (100 - row["burnout"])
    )


def role_adoption_rates(df):
    rows = []
    for role in ROLE_COUNTS:
        group = df[df["role"] == role]
        rows.append(
            {
                "role": role,
                "adoption": 100 * (group["state"] == "Adopted").mean(),
            }
        )
    return pd.DataFrame(rows)


def overall_metrics(df, peak):
    adoption = 100 * (df["state"] == "Adopted").mean()
    peak = max(float(peak), float(adoption))
    persistence = 100 * adoption / peak if peak > 0 else 0.0
    return adoption, peak, persistence


# ============================================================
# NETWORK + MODEL SETUP
# ============================================================

def build_network(agent_df, seed):
    rng = random.Random(seed)
    G = nx.Graph()

    for agent_id in agent_df["id"]:
        G.add_node(int(agent_id))

    ids = agent_df["id"].tolist()

    for agent_id in ids:
        desired_degree = rng.randint(3, 5)
        attempts = 0

        while G.degree(agent_id) < desired_degree and attempts < 200:
            partner = rng.choice(ids)

            if partner != agent_id:
                G.add_edge(agent_id, partner)

            attempts += 1

    role_lookup = agent_df.set_index("id")["role"].to_dict()

    for agent_id in ids:
        candidates = [
            other
            for other in ids
            if other != agent_id and role_lookup[other] != role_lookup[agent_id]
        ]

        if candidates:
            G.add_edge(agent_id, rng.choice(candidates))

    # Fixed positions: nodes do not jump around while the simulation runs.
    pos = nx.spring_layout(
        G,
        seed=seed,
        k=1.6 / math.sqrt(max(len(G), 1)),
        iterations=200,
        scale=1.45,
    )

    return G, pos


def initialize_model(seed, role_conditions):
    rng = np.random.default_rng(seed)
    records = []
    agent_id = 0

    for role, count in ROLE_COUNTS.items():
        assumptions = role_conditions.loc[role]

        for _ in range(count):
            records.append(
                {
                    "id": agent_id,
                    "role": role,
                    "state": "Not engaged",
                    "ever_adopted": False,
                    "state_duration": 0,
                    "trust": clamp(rng.normal(assumptions["Trust"], 8)),
                    "burnout": clamp(rng.normal(assumptions["Burnout"], 8)),
                    "workflow_confidence": clamp(
                        rng.normal(assumptions["Workflow confidence"], 8)
                    ),
                    "coordination_readiness": clamp(
                        rng.normal(assumptions["Coordination readiness"], 8)
                    ),
                }
            )
            agent_id += 1

    df = pd.DataFrame(records)
    G, pos = build_network(df, seed)

    history = pd.DataFrame(
        [
            {
                "tick": 0,
                "overall": 0.0,
                **{role: 0.0 for role in ROLE_COUNTS},
            }
        ]
    )

    state_history = pd.DataFrame(
        [
            {
                "tick": 0,
                **{state: int((df["state"] == state).sum()) for state in STATE_ORDER},
            }
        ]
    )

    return {
        "agents": df,
        "network": G,
        "positions": pos,
        "tick": 0,
        "peak_adoption": 0.0,
        "staffing_shortage": False,
        "sponsor_left": False,
        "history": history,
        "state_history": state_history,
        "seed": seed,
    }


# ============================================================
# MODEL STEP
# ============================================================

def simulation_step(model, budget):
    df = model["agents"].copy()
    rng = np.random.default_rng(model["seed"] + model["tick"] * 1009 + 17)

    staffing = budget["Staffing & capacity"]
    workflow = budget["Workflow & escalation redesign"]
    leadership = budget["Leadership & change support"]
    coordination = budget["Coordination infrastructure"]

    effective_leadership = leadership
    if model["sponsor_left"]:
        effective_leadership *= 0.25

    # Strategy effects. Because the four strategies share one 100-point budget,
    # improving one necessarily means allocating less somewhere else.
    df["trust"] += 0.040 * effective_leadership / 25
    df["workflow_confidence"] += 0.045 * workflow / 25
    df["coordination_readiness"] += 0.045 * coordination / 25
    df["burnout"] -= 0.030 * staffing / 25

    # Workflow redesign also modestly supports coordination.
    df["coordination_readiness"] += 0.015 * workflow / 25

    # Active shocks
    if model["staffing_shortage"]:
        shock = {
            "Referring clinicians": (0.10, -0.02),
            "Home-care nurses": (0.24, -0.10),
            "Care coordinators": (0.18, -0.08),
            "Clinical / operational leaders": (0.05, -0.01),
        }

        for role, (burnout_delta, confidence_delta) in shock.items():
            mask = df["role"] == role
            df.loc[mask, "burnout"] += burnout_delta
            df.loc[mask, "workflow_confidence"] += confidence_delta

    if model["sponsor_left"]:
        df["trust"] -= 0.11
        df["coordination_readiness"] -= 0.03

    # Learning through use
    trial_mask = df["state"] == "Trial"
    adopted_mask = df["state"] == "Adopted"

    df.loc[trial_mask, "workflow_confidence"] += 0.04
    df.loc[adopted_mask, "workflow_confidence"] += 0.06
    df.loc[adopted_mask, "coordination_readiness"] += 0.04

    for col in [
        "trust",
        "burnout",
        "workflow_confidence",
        "coordination_readiness",
    ]:
        df[col] = df[col].clip(0, 100)

    # State transitions
    for idx in df.index:
        row = df.loc[idx]
        readiness = readiness_score(row)
        state = row["state"]
        duration = int(row["state_duration"]) + 1

        if state == "Not engaged":
            p = 0.02 + 0.12 * logistic((readiness - 45) / 8)
            if rng.random() < p:
                state = "Considering"
                duration = 0

        elif state == "Considering":
            p = 0.02 + 0.12 * logistic((readiness - 50) / 8)
            if rng.random() < p:
                state = "Trial"
                duration = 0
            elif readiness < 35 and rng.random() < 0.03:
                state = "Not engaged"
                duration = 0

        elif state == "Trial":
            if duration >= 3:
                p = 0.02 + 0.13 * logistic((readiness - 52) / 7)

                if rng.random() < p:
                    state = "Adopted"
                    duration = 0
                    df.at[idx, "ever_adopted"] = True
                elif readiness < 35 and rng.random() < 0.04:
                    state = "Not engaged"
                    duration = 0

        elif state == "Adopted":
            abandon_p = 0.003 + 0.07 * (
                1 - logistic((readiness - 42) / 7)
            )

            if rng.random() < abandon_p:
                state = "Abandoned"
                duration = 0

        elif state == "Abandoned":
            if duration >= 5:
                reconsider_p = 0.05 * logistic((readiness - 55) / 8)

                if rng.random() < reconsider_p:
                    state = "Considering"
                    duration = 0

        df.at[idx, "state"] = state
        df.at[idx, "state_duration"] = duration

    model["agents"] = df
    model["tick"] += 1

    adoption, peak, persistence = overall_metrics(df, model["peak_adoption"])
    model["peak_adoption"] = peak

    rates = role_adoption_rates(df).set_index("role")["adoption"].to_dict()

    model["history"] = pd.concat(
        [
            model["history"],
            pd.DataFrame(
                [
                    {
                        "tick": model["tick"],
                        "overall": adoption,
                        **rates,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    model["state_history"] = pd.concat(
        [
            model["state_history"],
            pd.DataFrame(
                [
                    {
                        "tick": model["tick"],
                        **{
                            state: int((df["state"] == state).sum())
                            for state in STATE_ORDER
                        },
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    return model


# ============================================================
# FIGURES
# ============================================================

def make_network_figure(model):
    df = model["agents"]
    G = model["network"]
    pos = model["positions"]

    edge_x, edge_y = [], []

    for a, b in G.edges():
        x0, y0 = pos[a]
        x1, y1 = pos[b]

        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    traces = [
        go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line=dict(width=0.7, color="rgba(120,120,120,0.20)"),
            hoverinfo="none",
            showlegend=False,
        )
    ]

    for role in ROLE_COUNTS:
        for state in STATE_ORDER:
            subset = df[(df["role"] == role) & (df["state"] == state)]

            if subset.empty:
                continue

            xs = [pos[int(i)][0] for i in subset["id"]]
            ys = [pos[int(i)][1] for i in subset["id"]]
            hover_text = []

            for _, row in subset.iterrows():
                hover_text.append(
                    "<b>{}</b><br>"
                    "State: {}<br>"
                    "Trust: {:.0f}<br>"
                    "Burnout: {:.0f}<br>"
                    "Workflow confidence: {:.0f}<br>"
                    "Coordination readiness: {:.0f}<br>"
                    "Readiness score: {:.0f}".format(
                        ROLE_SHORT[row["role"]],
                        row["state"],
                        row["trust"],
                        row["burnout"],
                        row["workflow_confidence"],
                        row["coordination_readiness"],
                        readiness_score(row),
                    )
                )

            traces.append(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="markers",
                    marker=dict(
                        size=13,
                        color=STATE_COLORS[state],
                        symbol=ROLE_SYMBOLS[role],
                        line=dict(width=1.0, color="white"),
                    ),
                    name=f"{ROLE_SHORT[role]} — {state}",
                    hovertext=hover_text,
                    hoverinfo="text",
                    showlegend=False,
                )
            )

    fig = go.Figure(data=traces)

    fig.update_layout(
        height=610,
        margin=dict(l=5, r=5, t=45, b=5),
        title="Hospital-at-Home coordination network",
        uirevision="fixed-network",
        transition={"duration": 180, "easing": "cubic-in-out"},
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[-1.65, 1.65],
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[-1.65, 1.65],
            scaleanchor="x",
            scaleratio=1,
        ),
        hovermode="closest",
    )

    return fig


def make_adoption_figure(model):
    hist = model["history"]
    fig = go.Figure()

    for role in ROLE_COUNTS:
        fig.add_trace(
            go.Scatter(
                x=hist["tick"],
                y=hist[role],
                mode="lines",
                name=ROLE_SHORT[role],
                line=dict(width=2),
            )
        )

    fig.add_trace(
        go.Scatter(
            x=hist["tick"],
            y=hist["overall"],
            mode="lines",
            name="Overall",
            line=dict(width=4, dash="dash"),
        )
    )

    fig.update_layout(
        height=350,
        margin=dict(l=20, r=10, t=45, b=25),
        title="Adoption by role",
        xaxis_title="Simulation time",
        yaxis_title="Adoption (%)",
        yaxis=dict(range=[0, 100]),
        uirevision="adoption-chart",
        transition={"duration": 180, "easing": "linear"},
        legend=dict(orientation="h"),
    )

    return fig


# ============================================================
# SESSION STATE
# ============================================================

if "page" not in st.session_state:
    st.session_state.page = "landing"

if "role_conditions" not in st.session_state:
    st.session_state.role_conditions = DEFAULT_ROLE_CONDITIONS.copy()

if "model" not in st.session_state:
    st.session_state.model = None

if "budget" not in st.session_state:
    st.session_state.budget = {
        "Staffing & capacity": 25,
        "Workflow & escalation redesign": 25,
        "Leadership & change support": 25,
        "Coordination infrastructure": 25,
    }

if "assumption_notes" not in st.session_state:
    st.session_state.assumption_notes = ""


# ============================================================
# LANDING PAGE
# ============================================================

if st.session_state.page == "landing":
    st.title("Transformation Strategy Lab")
    st.write(
        "Choose a transformation challenge. Each scenario asks your group to make "
        "explicit assumptions about stakeholders, allocate a limited transformation "
        "budget, test a strategy, and interpret whether the result looks like "
        "innovation, transformation, or something in between."
    )

    st.markdown("### Choose a scenario")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown(
            """
<div class="scenario-card">
<h3>Hospital at Home</h3>
<p>Your health system wants to move eligible acute care from an inpatient bed
to a coordinated home-based model.</p>
<p><b>Core challenge:</b> Can roles, workflows, capacity, and coordination be
reorganized well enough for the model to persist?</p>
</div>
""",
            unsafe_allow_html=True,
        )

        if st.button("Open Hospital at Home", use_container_width=True):
            st.session_state.page = "hospital_at_home"
            st.rerun()

        st.markdown(
            """
<div class="scenario-card">
<h3>Chart Hero</h3>
<p>Embed AI-supported documentation into routine clinical work while preserving
human judgment, trust, and workflow fit.</p>
<p class="small-note">Prototype placeholder — coming next.</p>
</div>
""",
            unsafe_allow_html=True,
        )
        st.button("Chart Hero — coming soon", disabled=True, use_container_width=True)

    with c2:
        st.markdown(
            """
<div class="scenario-card">
<h3>Coordn8</h3>
<p>Redesign a burdensome referral-processing workflow around automation while
retaining human oversight.</p>
<p class="small-note">Prototype placeholder — coming next.</p>
</div>
""",
            unsafe_allow_html=True,
        )
        st.button("Coordn8 — coming soon", disabled=True, use_container_width=True)

        st.markdown(
            """
<div class="scenario-card">
<h3>Collaborative Care Redesign</h3>
<p>Transform a care process involving multiple professional groups with unequal
authority, voice, and decision rights.</p>
<p class="small-note">Prototype placeholder — coming next.</p>
</div>
""",
            unsafe_allow_html=True,
        )
        st.button(
            "Collaborative Care Redesign — coming soon",
            disabled=True,
            use_container_width=True,
        )

    st.stop()


# ============================================================
# HOSPITAL AT HOME SCENARIO
# ============================================================

top_left, top_right = st.columns([5, 1])

with top_left:
    st.title("Hospital at Home")
    st.caption(
        "Scenario prototype: Big T transformation through a redesigned site of care"
    )

with top_right:
    if st.button("Back to scenarios", use_container_width=True):
        st.session_state.page = "landing"
        st.rerun()

st.write(
    "Your health system has piloted acute care at home for eligible patients. "
    "The question is no longer whether the idea can work in isolated cases. "
    "The question is whether the organization can reorganize around it so the "
    "new model becomes routine and persists under stress."
)

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "1. Assumptions",
        "2. Strategy budget",
        "3. Simulation",
        "4. Debrief",
    ]
)


# ------------------------------------------------------------
# TAB 1: ROLE ASSUMPTIONS
# ------------------------------------------------------------

with tab1:
    st.subheader("Set role-specific starting conditions")
    st.write(
        "Each row represents a stakeholder group. The numbers are your group's "
        "assumptions about the starting organization, not empirical estimates. "
        "Set the conditions you think are reasonable and be prepared to justify them."
    )

    edited = st.data_editor(
        st.session_state.role_conditions,
        use_container_width=True,
        num_rows="fixed",
        column_config={
            "Trust": st.column_config.NumberColumn(
                "Trust", min_value=0, max_value=100, step=5
            ),
            "Burnout": st.column_config.NumberColumn(
                "Burnout", min_value=0, max_value=100, step=5
            ),
            "Workflow confidence": st.column_config.NumberColumn(
                "Workflow confidence", min_value=0, max_value=100, step=5
            ),
            "Coordination readiness": st.column_config.NumberColumn(
                "Coordination readiness", min_value=0, max_value=100, step=5
            ),
        },
    )

    st.session_state.role_conditions = edited

    st.markdown(
        """
**Condition meanings**

- **Trust:** Do people trust the care model, the people leading it, and whether it will last?
- **Burnout:** How much workload strain or capacity pressure is the role already experiencing?
- **Workflow confidence:** Do people understand their new responsibilities and feel able to perform them?
- **Coordination readiness:** How prepared is the role to work across the new home-based care pathway?
"""
    )

    st.session_state.assumption_notes = st.text_area(
        "Briefly justify your starting assumptions",
        value=st.session_state.assumption_notes,
        height=120,
        placeholder=(
            "Example: We assume home-care nurses begin with relatively high burnout "
            "because the new model adds capacity demands, while leaders begin with "
            "higher trust because they sponsored the pilot."
        ),
    )

    if st.button("Restore prototype assumptions"):
        st.session_state.role_conditions = DEFAULT_ROLE_CONDITIONS.copy()
        st.session_state.assumption_notes = ""
        st.rerun()


# ------------------------------------------------------------
# TAB 2: 100-POINT STRATEGY BUDGET
# ------------------------------------------------------------

with tab2:
    st.subheader("Allocate a 100-point transformation budget")
    st.write(
        "Resources are limited. Your four strategy investments must add up to exactly 100. "
        "Spending more in one area means spending less somewhere else."
    )

    b1, b2 = st.columns(2)

    with b1:
        staffing = st.slider(
            "Staffing & capacity",
            0,
            100,
            int(st.session_state.budget["Staffing & capacity"]),
            5,
            help="Protects capacity and reduces workload pressure.",
        )

        workflow = st.slider(
            "Workflow & escalation redesign",
            0,
            100,
            int(st.session_state.budget["Workflow & escalation redesign"]),
            5,
            help="Clarifies roles, workflows, and escalation pathways.",
        )

    with b2:
        leadership = st.slider(
            "Leadership & change support",
            0,
            100,
            int(st.session_state.budget["Leadership & change support"]),
            5,
            help="Builds trust and helps the change survive organizational uncertainty.",
        )

        coordination = st.slider(
            "Coordination infrastructure",
            0,
            100,
            int(st.session_state.budget["Coordination infrastructure"]),
            5,
            help="Supports cross-role coordination in the home-based care pathway.",
        )

    total = staffing + workflow + leadership + coordination
    remaining = 100 - total

    st.session_state.budget = {
        "Staffing & capacity": staffing,
        "Workflow & escalation redesign": workflow,
        "Leadership & change support": leadership,
        "Coordination infrastructure": coordination,
    }

    m1, m2 = st.columns(2)
    m1.metric("Allocated", f"{total} / 100")

    if remaining >= 0:
        m2.metric("Remaining", remaining)
    else:
        m2.metric("Over budget", abs(remaining))

    st.progress(min(total, 100) / 100)

    if total == 100:
        st.success("Budget is balanced. You can initialize the simulation.")
    elif total < 100:
        st.info(f"Allocate {100 - total} more points.")
    else:
        st.error(f"Reduce the strategy budget by {total - 100} points.")


# ------------------------------------------------------------
# TAB 3: SIMULATION
# ------------------------------------------------------------

with tab3:
    st.subheader("Run the transformation")

    total = sum(st.session_state.budget.values())

    controls = st.columns([1.2, 1, 1, 1.3, 1.3, 1.3])

    with controls[0]:
        initialize_clicked = st.button(
            "Initialize",
            use_container_width=True,
            disabled=(total != 100),
        )

    with controls[1]:
        step_clicked = st.button(
            "Step",
            use_container_width=True,
            disabled=(st.session_state.model is None),
        )

    with controls[2]:
        run10_clicked = st.button(
            "Run 10",
            use_container_width=True,
            disabled=(st.session_state.model is None),
        )

    with controls[3]:
        animate_clicked = st.button(
            "Animate to 100",
            use_container_width=True,
            disabled=(st.session_state.model is None),
        )

    with controls[4]:
        shortage_clicked = st.button(
            "Staffing shortage",
            use_container_width=True,
            disabled=(st.session_state.model is None),
        )

    with controls[5]:
        sponsor_clicked = st.button(
            "Sponsor leaves",
            use_container_width=True,
            disabled=(st.session_state.model is None),
        )

    c_restore, c_speed, _ = st.columns([1.3, 2.2, 2.5])

    with c_restore:
        restore_staffing_clicked = st.button(
            "Restore staffing",
            use_container_width=True,
            disabled=(st.session_state.model is None),
        )

    with c_speed:
        animation_delay = st.slider(
            "Animation speed",
            min_value=0.05,
            max_value=0.60,
            value=0.18,
            step=0.05,
            format="%.2f sec / tick",
        )

    if initialize_clicked:
        st.session_state.model = initialize_model(
            seed=42,
            role_conditions=st.session_state.role_conditions,
        )

    if st.session_state.model is not None:
        if shortage_clicked:
            st.session_state.model["staffing_shortage"] = True

        if sponsor_clicked:
            st.session_state.model["sponsor_left"] = True

        if restore_staffing_clicked:
            st.session_state.model["staffing_shortage"] = False

        if step_clicked and st.session_state.model["tick"] < 100:
            st.session_state.model = simulation_step(
                st.session_state.model,
                st.session_state.budget,
            )

        if run10_clicked:
            for _ in range(10):
                if st.session_state.model["tick"] >= 100:
                    break

                st.session_state.model = simulation_step(
                    st.session_state.model,
                    st.session_state.budget,
                )

    # Persistent placeholders let us redraw in-place during animation.
    # This avoids full-page reruns between ticks and removes most of the
    # "blinking" effect from the earlier prototype.
    status_placeholder = st.empty()
    metric_placeholder = st.empty()
    network_placeholder = st.empty()
    adoption_placeholder = st.empty()

    def render_simulation():
        model = st.session_state.model

        if model is None:
            status_placeholder.info(
                "Allocate exactly 100 strategy points, then select Initialize."
            )
            return

        adoption, peak, persistence = overall_metrics(
            model["agents"], model["peak_adoption"]
        )

        active = []

        if model["staffing_shortage"]:
            active.append("Staffing shortage active")

        if model["sponsor_left"]:
            active.append("Executive sponsor absent")

        if active:
            status_placeholder.warning(" | ".join(active))
        else:
            status_placeholder.success("No active organizational shocks.")

        with metric_placeholder.container():
            mm1, mm2, mm3, mm4 = st.columns(4)
            mm1.metric("Overall adoption", f"{adoption:.1f}%")
            mm2.metric("Peak adoption", f"{peak:.1f}%")
            mm3.metric("Persistence", f"{persistence:.1f}%")
            mm4.metric("Simulation time", f"{model['tick']} / 100")

        network_placeholder.plotly_chart(
            make_network_figure(model),
            use_container_width=True,
            config={"displayModeBar": False},
            key=f"network-{model['tick']}",
        )

        adoption_placeholder.plotly_chart(
            make_adoption_figure(model),
            use_container_width=True,
            config={"displayModeBar": False},
            key=f"adoption-{model['tick']}",
        )

    render_simulation()

    if animate_clicked and st.session_state.model is not None:
        while st.session_state.model["tick"] < 100:
            st.session_state.model = simulation_step(
                st.session_state.model,
                st.session_state.budget,
            )

            render_simulation()
            time.sleep(animation_delay)

    if st.session_state.model is not None:
        with st.expander("Inspect current role outcomes"):
            rates = role_adoption_rates(
                st.session_state.model["agents"]
            ).copy()

            rates["Role"] = rates["role"].map(ROLE_SHORT)
            rates["Adoption (%)"] = rates["adoption"].round(1)

            st.dataframe(
                rates[["Role", "Adoption (%)"]],
                hide_index=True,
                use_container_width=True,
            )

        st.caption(
            "Shape identifies role. Color identifies transformation state: "
            "gray = not engaged, yellow = considering, orange = trial, "
            "green = adopted, red = abandoned."
        )

        st.info(
            "The network represents coordination requirements across roles. "
            "Links do not directly spread adoption."
        )


# ------------------------------------------------------------
# TAB 4: DEBRIEF
# ------------------------------------------------------------

with tab4:
    st.subheader("Debrief")

    st.write(
        "Use the simulation outcome as evidence for discussion rather than treating "
        "the model as a prediction."
    )

    st.markdown(
        """
1. **What assumptions did your group make about the stakeholder roles?** Which assumption seems to have mattered most?
2. **How did you allocate your 100-point transformation budget, and why?** What did you deliberately choose *not* to invest in?
3. **Which role adopted most slowly or abandoned the model most often?** What organizational explanation would you give for that pattern?
4. **What happened when the system was stressed?** Did the change persist, weaken, or collapse?
5. **Which parts of your strategy look like little t improvement, and which require Big T transformation?**
6. **If you could take only one action before launching Hospital at Home, what would you do first?**
7. **What structural changes would be necessary for the model to continue after the original executive sponsor or champions were gone?**
8. **What important real-world factor is missing from this simulation?** How might adding it change the outcome?
"""
    )

    if st.session_state.assumption_notes.strip():
        st.markdown("#### Your group's starting rationale")
        st.write(st.session_state.assumption_notes)

    if st.session_state.model is not None:
        adoption, peak, persistence = overall_metrics(
            st.session_state.model["agents"],
            st.session_state.model["peak_adoption"],
        )

        st.markdown("#### Current outcome summary")

        d1, d2, d3 = st.columns(3)
        d1.metric("Current adoption", f"{adoption:.1f}%")
        d2.metric("Peak adoption", f"{peak:.1f}%")
        d3.metric("Persistence", f"{persistence:.1f}%")

        abandoned = int(
            (st.session_state.model["agents"]["state"] == "Abandoned").sum()
        )

        st.write(f"Agents currently in the abandoned state: **{abandoned}**")
