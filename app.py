import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import copy
from datetime import datetime
import numpy as np
from full_optimizer import MultiDestinationSupplyChainOptimizer

def convert_tuple_keys_to_strings(obj):
    """Convert tuple keys to strings for JSON serialization"""
    if isinstance(obj, dict):
        new_dict = {}
        for key, value in obj.items():
            if isinstance(key, tuple):
                new_key = "_".join(str(k) for k in key)
            else:
                new_key = str(key)
            new_dict[new_key] = convert_tuple_keys_to_strings(value)
        return new_dict
    elif isinstance(obj, list):
        return [convert_tuple_keys_to_strings(item) for item in obj]
    else:
        return obj

# Page config
st.set_page_config(
    page_title="Global Supply Chain Optimizer",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 0rem;
    }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1rem;
        font-weight: 600;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'optimizer' not in st.session_state:
    st.session_state.optimizer = None
if 'solution' not in st.session_state:
    st.session_state.solution = None
if 'config' not in st.session_state:
    optimizer = MultiDestinationSupplyChainOptimizer()
    st.session_state.config = optimizer.get_default_config()

# Title and description
st.title("🌐 Global Supply Chain Network Optimizer")
st.markdown("""
This advanced optimization tool helps design optimal supply chain networks considering:
- **Multi-origin sourcing** (China, Vietnam)
- **Multi-destination distribution** (USA, Japan, EU)
- **Product-specific HS code based tariffs**
- **Comprehensive cost modeling** (production, freight, tariffs, warehousing)
- **Lead time optimization**
- **Capacity constraints**
""")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Configuration tabs
    config_tab1, config_tab2, config_tab3, config_tab4 = st.tabs(
        ["📦 Products", "🏭 Production", "💰 Costs", "📊 Demand"]
    )
    
    with config_tab1:
        st.subheader("Product Configuration")
        
        products = {}
        for product_key in ["VR_QUEST_3", "VR_QUEST_3_PRO"]:
            st.markdown(f"**{product_key.replace('_', ' ')}**")
            col1, col2 = st.columns(2)
            
            with col1:
                weight = st.number_input(
                    f"Weight (kg)", 
                    value=st.session_state.config["products"][product_key]["weight_kg"],
                    step=0.001,
                    key=f"weight_{product_key}"
                )
                volume = st.number_input(
                    f"Volume (cbm)", 
                    value=st.session_state.config["products"][product_key]["volume_cbm"],
                    step=0.001,
                    key=f"volume_{product_key}"
                )
            
            with col2:
                value = st.number_input(
                    f"Value (USD)", 
                    value=st.session_state.config["products"][product_key]["value_usd"],
                    step=0.01,
                    key=f"value_{product_key}"
                )
                hs_code = st.text_input(
                    f"HS Code", 
                    value=st.session_state.config["products"][product_key]["hs_code"],
                    key=f"hs_{product_key}"
                )
            
            products[product_key] = {
                "weight_kg": weight,
                "volume_cbm": volume,
                "value_usd": value,
                "hs_code": hs_code
            }
        
        st.session_state.config["products"] = products
        
        # Container capacity
        st.subheader("Container Capacity")
        container_capacity = {}
        for product in ["VR_QUEST_3", "VR_QUEST_3_PRO"]:
            container_capacity[product] = st.number_input(
                f"Units per container - {product.replace('_', ' ')}", 
                value=st.session_state.config["container_capacity"][product],
                step=100,
                key=f"container_{product}"
            )
        st.session_state.config["container_capacity"] = container_capacity
    
    with config_tab2:
        st.subheader("Production Capacity (units/month)")
        
        production_capacity = {}
        production_costs = {}
        
        for origin in ["China", "Vietnam"]:
            st.markdown(f"**{origin}**")
            production_capacity[origin] = {}
            production_costs[origin] = {}
            
            for product in ["VR_QUEST_3", "VR_QUEST_3_PRO"]:
                col1, col2 = st.columns(2)
                
                with col1:
                    capacity = st.number_input(
                        f"{product.replace('_', ' ')} Capacity",
                        value=st.session_state.config["production_capacity"][origin][product],
                        step=1000,
                        key=f"cap_{origin}_{product}"
                    )
                    production_capacity[origin][product] = capacity
                
                with col2:
                    cost = st.number_input(
                        f"{product.replace('_', ' ')} Cost ($/unit)",
                        value=st.session_state.config["production_costs"][origin][product],
                        step=1,
                        key=f"cost_{origin}_{product}"
                    )
                    production_costs[origin][product] = cost
        
        st.session_state.config["production_capacity"] = production_capacity
        st.session_state.config["production_costs"] = production_costs
    
    with config_tab3:
        st.subheader("Tariff Rates (%)")
        
        tariff_rates = {}
        origins = ["China", "Vietnam"]
        destinations = ["USA", "Japan", "Germany"]
        
        for origin in origins:
            for dest in destinations:
                for product_key, product_data in st.session_state.config["products"].items():
                    hs = product_data["hs_code"]
                    key = (origin, dest, hs)
                    current_rate = st.session_state.config["tariff_rates"].get(key, 0.0)
                    
                    rate = st.number_input(
                        f"{origin} → {dest} ({hs}) - {product_key.replace('_', ' ')}",
                        value=current_rate,
                        step=0.1,
                        key=f"tariff_{origin}_{dest}_{hs}_{product_key}"
                    )
                    tariff_rates[key] = rate
        
        st.session_state.config["tariff_rates"] = tariff_rates
        
        # Ocean freight multipliers
        st.subheader("Ocean Freight Adjustment")
        st.info("Adjust freight rates by percentage")
        
        freight_adjustment = st.slider(
            "Freight Rate Adjustment %",
            -50, 100, 0, 5,
            help="Adjust all ocean freight rates"
        )
        
        # Store original freight rates if not already stored
        if 'original_freight_rates' not in st.session_state:
            optimizer = MultiDestinationSupplyChainOptimizer()
            st.session_state.original_freight_rates = optimizer.get_default_config()["ocean_freight_rates"]
        
        # Apply adjustment to original rates
        if freight_adjustment != 0:
            factor = 1 + (freight_adjustment / 100)
            adjusted_rates = {}
            for route, original_rate in st.session_state.original_freight_rates.items():
                adjusted_rates[route] = original_rate * factor
            st.session_state.config["ocean_freight_rates"] = adjusted_rates
        else:
            # Reset to original rates
            st.session_state.config["ocean_freight_rates"] = st.session_state.original_freight_rates.copy()
    
    with config_tab4:
        st.subheader("Monthly Demand (units)")
        
        demand = {}
        
        # USA DCs
        st.markdown("**USA Distribution Centers**")
        usa_dcs = ["CA_DC", "TX_DC", "IL_DC", "NJ_DC"]
        for dc in usa_dcs:
            with st.expander(f"{dc.replace('_', ' ')}"):
                for product in ["VR_QUEST_3", "VR_QUEST_3_PRO"]:
                    current_demand = st.session_state.config["demand"].get((dc, product), 0)
                    new_demand = st.number_input(
                        f"{product.replace('_', ' ')}",
                        value=current_demand,
                        step=100,
                        key=f"demand_{dc}_{product}"
                    )
                    if new_demand > 0:
                        demand[(dc, product)] = new_demand
        
        # Japan DCs
        st.markdown("**Japan Distribution Centers**")
        japan_dcs = ["Tokyo_DC", "Osaka_DC"]
        for dc in japan_dcs:
            with st.expander(f"{dc.replace('_', ' ')}"):
                for product in ["VR_QUEST_3", "VR_QUEST_3_PRO"]:
                    current_demand = st.session_state.config["demand"].get((dc, product), 0)
                    new_demand = st.number_input(
                        f"{product.replace('_', ' ')}",
                        value=current_demand,
                        step=100,
                        key=f"demand_{dc}_{product}"
                    )
                    if new_demand > 0:
                        demand[(dc, product)] = new_demand
        
        # EU DCs
        st.markdown("**EU Distribution Centers**")
        eu_dcs = ["Berlin_DC", "Munich_DC", "Paris_DC"]
        for dc in eu_dcs:
            with st.expander(f"{dc.replace('_', ' ')}"):
                for product in ["VR_QUEST_3", "VR_QUEST_3_PRO"]:
                    current_demand = st.session_state.config["demand"].get((dc, product), 0)
                    new_demand = st.number_input(
                        f"{product.replace('_', ' ')}",
                        value=current_demand,
                        step=100,
                        key=f"demand_{dc}_{product}"
                    )
                    if new_demand > 0:
                        demand[(dc, product)] = new_demand
        
        st.session_state.config["demand"] = demand

# Main content area
main_tab1, main_tab2, main_tab3, main_tab4, main_tab5 = st.tabs(
    ["🚀 Optimization", "📊 Cost Analysis", "🌍 Network View", 
     "📈 Sensitivity", "📋 Reports"]
)

with main_tab1:
    st.header("🚀 Optimization Results")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔄 Run Optimization", type="primary", use_container_width=True):
            with st.spinner("Optimizing supply chain network..."):
                try:
                    # Create optimizer with current config
                    optimizer = MultiDestinationSupplyChainOptimizer(st.session_state.config)
                    st.session_state.optimizer = optimizer
                    
                    # Build and solve model
                    optimizer.build_model()
                    solution = optimizer.solve(time_limit=300, gap=0.01)
                    
                    if solution:
                        st.session_state.solution = solution
                        st.success("✅ Optimization completed successfully!")
                        st.balloons()
                    else:
                        st.error("❌ Optimization failed. Please check your configuration.")
                        
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    st.info("Make sure Gurobi is properly installed and licensed.")
    
    if st.session_state.solution:
        solution = st.session_state.solution
        
        # Key metrics
        st.subheader("📊 Key Performance Indicators")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Cost",
                f"${solution['objective_value']:,.0f}",
                help="Total landed cost including all components"
            )
        
        with col2:
            st.metric(
                "Cost per Unit",
                f"${solution['kpis']['avg_cost_per_unit']:.2f}",
                help="Average total cost per unit delivered"
            )
        
        with col3:
            st.metric(
                "Total Units",
                f"{solution['kpis']['total_units_shipped']:,}",
                help="Total units shipped across all routes"
            )
        
        with col4:
            st.metric(
                "Capacity Utilization",
                f"{solution['kpis']['avg_capacity_utilization']:.1f}%",
                help="Average production capacity utilization"
            )
        
        # Origin and destination splits
        st.subheader("📍 Sourcing and Distribution Split")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Origin pie chart
            origin_data = [(k, v) for k, v in solution['origin_split'].items()]
            if origin_data:
                origin_df = pd.DataFrame(origin_data, columns=['Origin', 'Units'])
                fig_origin = px.pie(
                    origin_df, 
                    values='Units', 
                    names='Origin',
                    title='Production by Origin',
                    color_discrete_map={'China': '#FF6B6B', 'Vietnam': '#4ECDC4'}
                )
                st.plotly_chart(fig_origin, use_container_width=True)
        
        with col2:
            # Destination pie chart
            dest_data = [(k, v) for k, v in solution['destination_split'].items()]
            if dest_data:
                dest_df = pd.DataFrame(dest_data, columns=['Destination', 'Units'])
                fig_dest = px.pie(
                    dest_df, 
                    values='Units', 
                    names='Destination',
                    title='Distribution by Destination',
                    color_discrete_map={'USA': '#45B7D1', 'Japan': '#F7DC6F', 'Germany': '#BB8FCE'}
                )
                st.plotly_chart(fig_dest, use_container_width=True)
        
        # Production details
        st.subheader("🏭 Production Plan Details")
        prod_data = []
        for (origin, product), qty in solution['production_plan'].items():
            prod_data.append({
                'Origin': origin,
                'Product': product.replace('_', ' '),
                'Quantity': f"{qty:,.0f}",
                'Capacity': f"{st.session_state.config['production_capacity'][origin][product]:,}",
                'Utilization': f"{solution['capacity_utilization'][origin][product]:.1f}%"
            })
        
        if prod_data:
            prod_df = pd.DataFrame(prod_data)
            st.dataframe(prod_df, use_container_width=True, hide_index=True)

with main_tab2:
    st.header("💰 Cost Breakdown Analysis")
    
    if st.session_state.solution:
        solution = st.session_state.solution
        
        # Cost breakdown visualization
        col1, col2 = st.columns(2)
        
        with col1:
            # Pie chart
            cost_components = []
            for component, value in solution['cost_breakdown'].items():
                if component != 'total_cost' and value > 0:
                    cost_components.append({
                        'Component': component.replace('_', ' ').title(),
                        'Cost': value
                    })
            
            if cost_components:
                cost_df = pd.DataFrame(cost_components)
                fig_cost_pie = px.pie(
                    cost_df,
                    values='Cost',
                    names='Component',
                    title='Cost Components Distribution',
                    hole=0.4
                )
                st.plotly_chart(fig_cost_pie, use_container_width=True)
        
        with col2:
            # Bar chart
            if cost_components:
                fig_cost_bar = px.bar(
                    cost_df.sort_values('Cost', ascending=True),
                    x='Cost',
                    y='Component',
                    orientation='h',
                    title='Cost Components Breakdown',
                    color='Cost',
                    color_continuous_scale='Viridis'
                )
                st.plotly_chart(fig_cost_bar, use_container_width=True)
        
        # Detailed cost metrics
        st.subheader("Cost Component Details")
        
        cols = st.columns(3)
        for i, (component, cost) in enumerate(solution['cost_breakdown'].items()):
            if component != 'total_cost':
                with cols[i % 3]:
                    percentage = (cost / solution['objective_value']) * 100
                    st.metric(
                        component.replace('_', ' ').title(),
                        f"${cost:,.0f}",
                        f"{percentage:.1f}%"
                    )
        
        # Tariff analysis
        st.subheader("🛃 Tariff Impact Analysis")
        
        tariff_data = []
        for route, analysis in solution['tariff_analysis'].items():
            tariff_data.append({
                'Route': route,
                'Units': f"{analysis['units']:,}",
                'Product Value': f"${analysis['value']:,.0f}",
                'Tariff Paid': f"${analysis['tariff_paid']:,.0f}",
                'Effective Rate': f"{analysis['effective_rate']:.1f}%"
            })
        
        if tariff_data:
            tariff_df = pd.DataFrame(tariff_data)
            st.dataframe(tariff_df, use_container_width=True, hide_index=True)
            
            # Tariff savings potential
            total_tariff = solution['cost_breakdown']['tariff_cost']
            tariff_pct = (total_tariff / solution['objective_value']) * 100
            
            if tariff_pct > 10:
                st.warning(f"⚠️ Tariffs represent {tariff_pct:.1f}% of total cost. Consider tariff optimization strategies.")
    else:
        st.info("Run optimization first to see cost analysis")

with main_tab3:
    st.header("🌐 Supply Chain Network Visualization")
    
    if st.session_state.solution:
        solution = st.session_state.solution
        
        # Create Sankey diagram
        st.subheader("Network Flow Visualization")
        
        # Prepare data for Sankey
        nodes = []
        node_dict = {}
        node_counter = 0
        
        # Add origins
        for origin in ["China", "Vietnam"]:
            nodes.append(origin)
            node_dict[origin] = node_counter
            node_counter += 1
        
        # Add destinations
        for dest in ["USA", "Japan", "Germany"]:
            nodes.append(dest)
            node_dict[dest] = node_counter
            node_counter += 1
        
        # Add DCs
        for dc in solution['active_dcs']:
            nodes.append(dc.replace('_', ' '))
            node_dict[dc] = node_counter
            node_counter += 1
        
        # Create links
        source = []
        target = []
        value = []
        link_color = []
        
        # Origin to destination flows
        for (origin, d_port, dc, product), qty in solution['distribution_plan'].items():
            if qty > 0:
                # Find destination country
                dest_country = None
                if dc in ["CA_DC", "TX_DC", "IL_DC", "NJ_DC"]:
                    dest_country = "USA"
                elif dc in ["Tokyo_DC", "Osaka_DC"]:
                    dest_country = "Japan"
                elif dc in ["Berlin_DC", "Munich_DC", "Paris_DC"]:
                    dest_country = "Germany"
                
                if dest_country and origin in node_dict and dest_country in node_dict:
                    # Origin to destination
                    if node_dict[origin] not in source or node_dict[dest_country] not in target:
                        source.append(node_dict[origin])
                        target.append(node_dict[dest_country])
                        value.append(qty)
                        link_color.append('rgba(69, 183, 209, 0.4)' if origin == "China" else 'rgba(78, 205, 196, 0.4)')
                
                if dest_country in node_dict and dc in node_dict:
                    # Destination to DC
                    source.append(node_dict[dest_country])
                    target.append(node_dict[dc])
                    value.append(qty)
                    link_color.append('rgba(187, 143, 206, 0.4)')
        
        # Create Sankey
        fig_sankey = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=nodes,
                color=["#FF6B6B", "#4ECDC4", "#45B7D1", "#F7DC6F", "#BB8FCE"] + 
                      ["#85C1E2"] * len(solution['active_dcs'])
            ),
            link=dict(
                source=source,
                target=target,
                value=value,
                color=link_color
            )
        )])
        
        fig_sankey.update_layout(
            title_text="Supply Chain Network Flow",
            font_size=10,
            height=600
        )
        st.plotly_chart(fig_sankey, use_container_width=True)
        
        # Lead time analysis
        st.subheader("⏱️ Lead Time Analysis")
        
        lead_time_data = []
        for route, times in solution['lead_time_analysis'].items():
            lead_time_data.append({
                'Route': route,
                'Production': f"{times['production']} days",
                'Ocean Transit': f"{times.get('ocean_transit_avg', 0):.1f} days",
                'Customs': f"{times['customs']} days",
                'Total': f"{times['total']:.1f} days"
            })
        
        if lead_time_data:
            lead_df = pd.DataFrame(lead_time_data)
            st.dataframe(lead_df, use_container_width=True, hide_index=True)
        
        # Active DCs
        st.subheader("🏢 Active Distribution Centers")
        
        dc_cols = st.columns(3)
        for i, dc in enumerate(solution['active_dcs']):
            with dc_cols[i % 3]:
                st.info(f"✅ {dc.replace('_', ' ')}")
    else:
        st.info("Run optimization first to see network visualization")

with main_tab4:
    st.header("📊 Sensitivity Analysis")
    
    if st.session_state.solution and st.session_state.optimizer:
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.subheader("Analysis Parameters")
            
            parameter = st.selectbox(
                "Parameter to Analyze",
                ["demand", "capacity", "tariff", "cost"],
                format_func=lambda x: x.title()
            )
            
            variations = st.multiselect(
                "Variations (%)",
                [-30, -20, -10, 0, 10, 20, 30],
                default=[-20, -10, 0, 10, 20]
            )
            
            run_sensitivity = st.button("Run Analysis", type="primary")
        
        with col2:
            if run_sensitivity and variations:
                with st.spinner(f"Running sensitivity analysis on {parameter}..."):
                    # Run sensitivity analysis
                    results = st.session_state.optimizer.sensitivity_analysis(
                        parameter,
                        variations,
                        st.session_state.solution
                    )
                    
                    # Display results
                    st.subheader(f"Sensitivity to {parameter.title()} Changes")
                    
                    # Create line chart
                    sens_data = pd.DataFrame(results['results'])
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=sens_data['variation'],
                        y=sens_data['total_cost'],
                        mode='lines+markers',
                        name='Total Cost',
                        line=dict(color='#3498DB', width=3),
                        marker=dict(size=10)
                    ))
                    
                    fig.update_layout(
                        title=f'Total Cost Sensitivity to {parameter.title()}',
                        xaxis_title='Variation (%)',
                        yaxis_title='Total Cost ($)',
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display table
                    st.subheader("Detailed Results")
                    display_df = sens_data.copy()
                    display_df['total_cost'] = display_df['total_cost'].apply(lambda x: f"${x:,.0f}")
                    display_df['cost_change'] = display_df['cost_change'].apply(lambda x: f"{x:+.1f}%")
                    display_df['variation'] = display_df['variation'].apply(lambda x: f"{x:+d}%")
                    
                    st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        # What-if scenarios
        st.subheader("🎯 What-If Scenarios")
        
        scenario = st.selectbox(
            "Select Scenario",
            [
                "Current Configuration",
                "100% China Sourcing",
                "100% Vietnam Sourcing", 
                "No Tariffs",
                "Double USA Demand",
                "50% Capacity Reduction"
            ]
        )
        
        if st.button("Run Scenario"):
            if scenario != "Current Configuration":
                with st.spinner(f"Running {scenario} scenario..."):
                    # Copy current config
                    scenario_config = copy.deepcopy(st.session_state.config)
                    
                    # Modify based on scenario
                    if scenario == "100% China Sourcing":
                        for product in scenario_config['production_capacity']['Vietnam']:
                            scenario_config['production_capacity']['Vietnam'][product] = 0
                            scenario_config['production_capacity']['China'][product] *= 2
                    
                    elif scenario == "100% Vietnam Sourcing":
                        for product in scenario_config['production_capacity']['China']:
                            scenario_config['production_capacity']['China'][product] = 0
                            scenario_config['production_capacity']['Vietnam'][product] *= 2
                    
                    elif scenario == "No Tariffs":
                        for key in scenario_config['tariff_rates']:
                            scenario_config['tariff_rates'][key] = 0
                    
                    elif scenario == "Double USA Demand":
                        for key in list(scenario_config['demand'].keys()):
                            if key[0] in ["CA_DC", "TX_DC", "IL_DC", "NJ_DC"]:
                                scenario_config['demand'][key] *= 2
                    
                    elif scenario == "50% Capacity Reduction":
                        for origin in scenario_config['production_capacity']:
                            for product in scenario_config['production_capacity'][origin]:
                                scenario_config['production_capacity'][origin][product] *= 0.5
                    
                    # Run scenario optimization
                    scenario_optimizer = MultiDestinationSupplyChainOptimizer(scenario_config)
                    scenario_optimizer.build_model()
                    scenario_solution = scenario_optimizer.solve()
                    
                    if scenario_solution:
                        # Compare results
                        st.subheader("Scenario Comparison")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**Current Solution**")
                            current_cost = st.session_state.solution['objective_value']
                            st.metric("Total Cost", f"${current_cost:,.0f}")
                            st.metric("Cost/Unit", f"${st.session_state.solution['kpis']['avg_cost_per_unit']:.2f}")
                            st.metric("Total Units", f"{st.session_state.solution['kpis']['total_units_shipped']:,}")
                            
                            # Origin split
                            st.markdown("**Origin Split:**")
                            for origin, units in st.session_state.solution['origin_split'].items():
                                st.write(f"- {origin}: {units:,.0f} units")
                        
                        with col2:
                            st.markdown(f"**{scenario}**")
                            scenario_cost = scenario_solution['objective_value']
                            cost_delta = ((scenario_cost - current_cost) / current_cost) * 100
                            
                            st.metric(
                                "Total Cost", 
                                f"${scenario_cost:,.0f}",
                                f"{cost_delta:+.1f}%"
                            )
                            st.metric(
                                "Cost/Unit", 
                                f"${scenario_solution['kpis']['avg_cost_per_unit']:.2f}",
                                f"${scenario_solution['kpis']['avg_cost_per_unit'] - st.session_state.solution['kpis']['avg_cost_per_unit']:+.2f}"
                            )
                            st.metric(
                                "Total Units", 
                                f"{scenario_solution['kpis']['total_units_shipped']:,}",
                                f"{scenario_solution['kpis']['total_units_shipped'] - st.session_state.solution['kpis']['total_units_shipped']:+,}"
                            )
                            
                            # Origin split
                            st.markdown("**Origin Split:**")
                            for origin, units in scenario_solution['origin_split'].items():
                                st.write(f"- {origin}: {units:,.0f} units")
                        
                        # Cost breakdown comparison
                        st.subheader("Cost Component Comparison")
                        
                        comparison_data = []
                        for component in ['production_cost', 'ocean_freight_cost', 'tariff_cost', 'inland_transport_cost']:
                            current_val = st.session_state.solution['cost_breakdown'][component]
                            scenario_val = scenario_solution['cost_breakdown'][component]
                            comparison_data.append({
                                'Component': component.replace('_', ' ').title(),
                                'Current': current_val,
                                'Scenario': scenario_val,
                                'Change': scenario_val - current_val
                            })
                        
                        comp_df = pd.DataFrame(comparison_data)
                        
                        # Create grouped bar chart
                        fig = go.Figure()
                        
                        fig.add_trace(go.Bar(
                            x=comp_df['Component'],
                            y=comp_df['Current'],
                            name='Current',
                            marker_color='#3498DB'
                        ))
                        
                        fig.add_trace(go.Bar(
                            x=comp_df['Component'],
                            y=comp_df['Scenario'],
                            name=scenario,
                            marker_color='#E74C3C'
                        ))
                        
                        fig.update_layout(
                            title='Cost Component Comparison',
                            barmode='group',
                            yaxis_title='Cost ($)'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Key insights
                        st.subheader("Key Insights")
                        
                        if cost_delta > 10:
                            st.warning(f"⚠️ This scenario increases costs by {cost_delta:.1f}%")
                        elif cost_delta < -10:
                            st.success(f"✅ This scenario reduces costs by {abs(cost_delta):.1f}%")
                        else:
                            st.info(f"ℹ️ This scenario has minimal impact on costs ({cost_delta:+.1f}%)")
                        
                        # Tariff impact
                        current_tariff = st.session_state.solution['cost_breakdown']['tariff_cost']
                        scenario_tariff = scenario_solution['cost_breakdown']['tariff_cost']
                        tariff_change = scenario_tariff - current_tariff
                        
                        if abs(tariff_change) > 1000:
                            st.write(f"📊 Tariff impact: ${tariff_change:+,.0f} ({(tariff_change/current_tariff*100):+.1f}%)")
    else:
        st.info("Run optimization first to perform sensitivity analysis")

with main_tab5:
    st.header("📑 Reports & Export")
    
    if st.session_state.solution:
        solution = st.session_state.solution
        
        # Report type selection
        report_type = st.selectbox(
            "Select Report Type",
            ["Executive Summary", "Detailed Cost Analysis", "Network Flow Report", 
             "Tariff Impact Analysis", "Capacity Utilization Report"]
        )
        
        # Generate report content
        report_content = ""
        
        if report_type == "Executive Summary":
            report_content = f"""
SUPPLY CHAIN OPTIMIZATION EXECUTIVE SUMMARY
==========================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

OPTIMIZATION STATUS
------------------
Status: {solution['status']}
Optimality Gap: {solution['gap']:.2%}

KEY RESULTS
-----------
Total Landed Cost: ${solution['objective_value']:,.2f}
Average Cost per Unit: ${solution['kpis']['avg_cost_per_unit']:.2f}
Total Units Shipped: {solution['kpis']['total_units_shipped']:,}
Active Distribution Centers: {solution['kpis']['active_dcs_count']}

COST BREAKDOWN
--------------
Production: ${solution['cost_breakdown']['production_cost']:,.2f} ({solution['kpis'].get('production_cost_percentage', 0):.1f}%)
Ocean Freight: ${solution['cost_breakdown']['ocean_freight_cost']:,.2f} ({solution['kpis'].get('ocean_freight_cost_percentage', 0):.1f}%)
Tariffs: ${solution['cost_breakdown']['tariff_cost']:,.2f} ({solution['kpis'].get('tariff_cost_percentage', 0):.1f}%)
Inland Transport: ${solution['cost_breakdown']['inland_transport_cost']:,.2f} ({solution['kpis'].get('inland_transport_cost_percentage', 0):.1f}%)
DC Fixed Costs: ${solution['cost_breakdown']['dc_fixed_cost']:,.2f} ({solution['kpis'].get('dc_fixed_cost_percentage', 0):.1f}%)
Inventory Holding: ${solution['cost_breakdown']['inventory_holding_cost']:,.2f} ({solution['kpis'].get('inventory_holding_cost_percentage', 0):.1f}%)

SOURCING STRATEGY
-----------------
"""
            for origin, units in solution['origin_split'].items():
                pct = (units / solution['kpis']['total_units_shipped'] * 100) if solution['kpis']['total_units_shipped'] > 0 else 0
                report_content += f"{origin}: {units:,.0f} units ({pct:.1f}%)\n"
            
            report_content += "\nMARKET DISTRIBUTION\n-------------------\n"
            for dest, units in solution['destination_split'].items():
                pct = (units / solution['kpis']['total_units_shipped'] * 100) if solution['kpis']['total_units_shipped'] > 0 else 0
                report_content += f"{dest}: {units:,.0f} units ({pct:.1f}%)\n"
            
            report_content += "\nKEY RECOMMENDATIONS\n-------------------\n"
            
            # Add recommendations based on results
            if solution['kpis'].get('tariff_cost_percentage', 0) > 20:
                report_content += "1. High tariff exposure (>20%) - Consider increasing Vietnam production\n"
            
            china_pct = (solution['origin_split'].get('China', 0) / solution['kpis']['total_units_shipped'] * 100) if solution['kpis']['total_units_shipped'] > 0 else 0
            if china_pct > 70:
                report_content += "2. High concentration in China (>70%) - Diversify supply base for risk mitigation\n"
            
            if solution['kpis']['avg_capacity_utilization'] < 60:
                report_content += "3. Low capacity utilization (<60%) - Opportunity to consolidate facilities\n"
            elif solution['kpis']['avg_capacity_utilization'] > 90:
                report_content += "4. High capacity utilization (>90%) - Consider capacity expansion\n"
        
        elif report_type == "Detailed Cost Analysis":
            report_content = f"""
DETAILED COST ANALYSIS REPORT
============================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

TOTAL COST COMPONENTS
--------------------
"""
            total = solution['objective_value']
            for component, value in solution['cost_breakdown'].items():
                if component != 'total_cost':
                    pct = (value / total * 100) if total > 0 else 0
                    report_content += f"{component.replace('_', ' ').title()}: ${value:,.2f} ({pct:.1f}%)\n"
            
            report_content += f"\nTOTAL: ${total:,.2f}\n"
            
            report_content += "\nTARIFF ANALYSIS BY ROUTE\n------------------------\n"
            for route, data in solution['tariff_analysis'].items():
                report_content += f"\n{route}:\n"
                report_content += f"  Units: {data['units']:,}\n"
                report_content += f"  Product Value: ${data['value']:,.2f}\n"
                report_content += f"  Tariff Paid: ${data['tariff_paid']:,.2f}\n"
                report_content += f"  Effective Rate: {data['effective_rate']:.1f}%\n"
        
        elif report_type == "Network Flow Report":
            report_content = f"""
SUPPLY CHAIN NETWORK FLOW REPORT
================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

PRODUCTION BY ORIGIN
-------------------
"""
            for (origin, product), qty in solution['production_plan'].items():
                report_content += f"{origin} - {product}: {qty:,.0f} units\n"
            
            report_content += "\nACTIVE DISTRIBUTION CENTERS\n---------------------------\n"
            for dc in solution['active_dcs']:
                report_content += f"- {dc.replace('_', ' ')}\n"
            
            report_content += "\nLEAD TIME ANALYSIS\n------------------\n"
            for route, times in solution['lead_time_analysis'].items():
                report_content += f"\n{route}:\n"
                report_content += f"  Production: {times['production']} days\n"
                report_content += f"  Ocean Transit: {times.get('ocean_transit_avg', 0):.1f} days\n"
                report_content += f"  Customs: {times['customs']} days\n"
                report_content += f"  Total: {times['total']:.1f} days\n"
        
        # Display report
        st.text_area("Report Preview", report_content, height=400)
        
        # Export options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Text export
            st.download_button(
                label="📄 Download as Text",
                data=report_content,
                file_name=f"{report_type.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
        
        with col2:
            # JSON export
            export_data = {
                'metadata': {
                    'report_type': report_type,
                    'generated': datetime.now().isoformat(),
                    'optimizer_version': '1.0'
                },
                'configuration': convert_tuple_keys_to_strings(st.session_state.config),
                'solution': convert_tuple_keys_to_strings(solution)
            }
            
            st.download_button(
                label="📊 Download as JSON",
                data=json.dumps(export_data, indent=2),
                file_name=f"optimization_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        with col3:
            # CSV export for flows
            flow_data = []
            for (origin, d_port, dc, product), qty in solution.get('distribution_plan', {}).items():
                # Find destination country
                dest_country = None
                if dc in ["CA_DC", "TX_DC", "IL_DC", "NJ_DC"]:
                    dest_country = "USA"
                elif dc in ["Tokyo_DC", "Osaka_DC"]:
                    dest_country = "Japan"
                elif dc in ["Berlin_DC", "Munich_DC", "Paris_DC"]:
                    dest_country = "Germany"
                
                flow_data.append({
                    'Origin': origin,
                    'Port': d_port,
                    'Destination': dest_country,
                    'DC': dc,
                    'Product': product,
                    'Quantity': qty
                })
            
            if flow_data:
                flow_df = pd.DataFrame(flow_data)
                csv = flow_df.to_csv(index=False)
                
                st.download_button(
                    label="📈 Download Flows CSV",
                    data=csv,
                    file_name=f"network_flows_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
    else:
        st.info("Run optimization first to generate reports")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "Supply Chain Optimizer v1.0 | Powered by BlueNorth AI "
    "</div>",
    unsafe_allow_html=True
)