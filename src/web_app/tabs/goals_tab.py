"""Goals tab — financial goal setting with compound interest projections."""
from __future__ import annotations

import streamlit as st

from src.web_app.components.charts import goal_projection_chart
from src.web_app.components.disclaimer_banner import show_disclaimer_banner, show_inline_disclaimer
from src.web_app.components.source_attribution import show_sources
from src.web_app.session import add_goal, add_message, clear_goals, get_goals, get_user_profile
from src.utils.validators import validate_goal_inputs


def render_goals_tab() -> None:
    """Render the Goals tab UI."""
    show_disclaimer_banner("general")

    st.markdown("## Financial Goal Planner")
    st.markdown(
        "Set up a financial goal and see illustrative projections based on "
        "different return scenarios. **These are educational illustrations, not predictions.**"
    )

    col_form, col_chart = st.columns([1, 1])

    with col_form:
        st.markdown("### Define Your Goal")
        with st.form("goal_form"):
            goal_name = st.text_input("Goal Name", placeholder="Retirement, Down Payment, College…")
            target_amount = st.number_input(
                "Target Amount ($)",
                min_value=1_000.0,
                max_value=50_000_000.0,
                value=1_000_000.0,
                step=10_000.0,
                format="%.0f",
            )
            time_horizon = st.slider(
                "Time Horizon (Years)",
                min_value=1,
                max_value=50,
                value=20,
                step=1,
            )
            current_savings = st.number_input(
                "Current Savings ($)",
                min_value=0.0,
                max_value=10_000_000.0,
                value=0.0,
                step=1_000.0,
                format="%.0f",
            )
            monthly_contribution = st.number_input(
                "Monthly Contribution ($)",
                min_value=0.0,
                max_value=100_000.0,
                value=500.0,
                step=50.0,
                format="%.0f",
            )
            risk_level = st.select_slider(
                "Return Assumption",
                options=["Conservative (5%)", "Moderate (7%)", "Optimistic (9%)"],
                value="Moderate (7%)",
            )
            submitted = st.form_submit_button("Calculate Projections", type="primary")

        if submitted:
            ok, error_msg = validate_goal_inputs(
                target_amount, time_horizon, current_savings, monthly_contribution
            )
            if not ok:
                st.error(error_msg)
            else:
                goal_data = {
                    "name": goal_name or "My Goal",
                    "target_amount": target_amount,
                    "time_horizon_years": time_horizon,
                    "current_savings": current_savings,
                    "monthly_contribution": monthly_contribution,
                    "risk_assumption": risk_level,
                }
                # Store in session
                clear_goals()
                add_goal(goal_data)
                st.success("Goal saved! See projections on the right.")
                st.rerun()

    goals = get_goals()

    with col_chart:
        if goals:
            goal = goals[0]
            st.markdown(f"### Projections: {goal.get('name', 'Your Goal')}")

            fig = goal_projection_chart(
                current_savings=goal["current_savings"],
                monthly_contribution=goal["monthly_contribution"],
                time_horizon_years=goal["time_horizon_years"],
                target_amount=goal["target_amount"],
            )
            st.plotly_chart(fig, use_container_width=True)

            show_inline_disclaimer(
                "Projections use compound interest formulas with constant assumed returns. "
                "Actual investment returns vary and are not guaranteed."
            )

            # Scenario table
            from src.agents.goal_planning_agent import _generate_projections
            scenarios = _generate_projections(
                goal["current_savings"],
                goal["monthly_contribution"],
                goal["time_horizon_years"],
                goal["target_amount"],
            )
            st.markdown("**Scenario Summary:**")
            for label, data in scenarios.items():
                status = "✅ On track" if data["on_track"] else "⚠️ Short"
                st.markdown(
                    f"- **{label.title()}**: ${data['future_value']:,.0f} — {status}"
                )
                if not data["on_track"] and data.get("required_monthly"):
                    st.caption(
                        f"  To reach target: ~${data['required_monthly']:,.0f}/month needed"
                    )
        else:
            st.info("Enter your goal details on the left to see projections here.")

    # AI Discussion
    if goals:
        st.markdown("---")
        goal = goals[0]
        if st.button("Discuss This Goal with Finnie", type="primary", use_container_width=True):
            with st.spinner("Finnie is analyzing your goal…"):
                try:
                    from src.workflow.graph import get_compiled_graph
                    from src.web_app.session import build_graph_input
                    profile = get_user_profile()
                    profile.goals = goals
                    query = (
                        f"I want to save ${goal['target_amount']:,.0f} in {goal['time_horizon_years']} years "
                        f"starting with ${goal['current_savings']:,.0f} and contributing "
                        f"${goal['monthly_contribution']:,.0f} per month. What should I know?"
                    )
                    state_input = build_graph_input(query)
                    state_input["intent"] = "goal_planning"
                    graph = get_compiled_graph()
                    final_state = graph.invoke(state_input)
                    response = final_state.get("final_response", "")
                    sources = final_state.get("sources", [])
                    st.markdown("### Finnie's Analysis")
                    st.markdown(response)
                    if sources:
                        show_sources(sources)
                    add_message("user", query)
                    add_message("assistant", response, sources=sources)
                except Exception as exc:
                    st.error(f"Error: {exc}")
