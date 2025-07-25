import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import copy

class MultiDestinationSupplyChainOptimizer:
    """
    Full-featured multi-destination supply chain network optimization
    with origin-destination-product-HSCode level tariff modeling
    """
    
    def __init__(self, config=None):
        """
        Initialize the optimizer with configurable parameters
        
        Args:
            config: Dictionary containing all configurable parameters
        """
        if config is None:
            config = self.get_default_config()
        
        self.config = config
        self._setup_network()
        
    def get_default_config(self):
        """Return default configuration"""
        return {
            # Products with HS Codes
            "products": {
                "VR_QUEST_3": {
                    "weight_kg": 0.515,
                    "volume_cbm": 0.003,
                    "value_usd": 499.99,
                    "hs_code": "9504.50.00"
                },
                "VR_QUEST_3_PRO": {
                    "weight_kg": 0.722,
                    "volume_cbm": 0.004,
                    "value_usd": 999.99,
                    "hs_code": "9504.50.00"
                }
            },
            
            # Production capacities (units per month)
            "production_capacity": {
                "China": {
                    "VR_QUEST_3": 50000,
                    "VR_QUEST_3_PRO": 20000
                },
                "Vietnam": {
                    "VR_QUEST_3": 30000,
                    "VR_QUEST_3_PRO": 15000
                }
            },
            
            # Production costs (USD per unit)
            "production_costs": {
                "China": {
                    "VR_QUEST_3": 180,
                    "VR_QUEST_3_PRO": 380
                },
                "Vietnam": {
                    "VR_QUEST_3": 195,
                    "VR_QUEST_3_PRO": 400
                }
            },
            
            # Tariff rates by origin-destination-HS code (percentage)
            "tariff_rates": {
                ("China", "USA", "9504.50.00"): 50.0,
                ("Vietnam", "USA", "9504.50.00"): 40.0,
                ("China", "Japan", "9504.50.00"): 0.0,
                ("Vietnam", "Japan", "9504.50.00"): 0.0,
                ("China", "Germany", "9504.50.00"): 6.0,
                ("Vietnam", "Germany", "9504.50.00"): 0.0
            },
            
            # Ocean freight rates (USD per container)
            "ocean_freight_rates": {
                # China to destinations
                ("Shanghai", "LosAngeles"): 2800,
                ("Shanghai", "LongBeach"): 2850,
                ("Shanghai", "Tokyo"): 1200,
                ("Shanghai", "Yokohama"): 1250,
                ("Shanghai", "Hamburg"): 3500,
                ("Shanghai", "Rotterdam"): 3400,
                ("Shenzhen", "LosAngeles"): 2750,
                ("Shenzhen", "LongBeach"): 2800,
                ("Shenzhen", "Tokyo"): 1150,
                ("Shenzhen", "Yokohama"): 1200,
                ("Shenzhen", "Hamburg"): 3450,
                ("Shenzhen", "Rotterdam"): 3350,
                
                # Vietnam to destinations
                ("HoChiMinh", "LosAngeles"): 3200,
                ("HoChiMinh", "LongBeach"): 3250,
                ("HoChiMinh", "Tokyo"): 1800,
                ("HoChiMinh", "Yokohama"): 1850,
                ("HoChiMinh", "Hamburg"): 3900,
                ("HoChiMinh", "Rotterdam"): 3800,
                ("Haiphong", "LosAngeles"): 3400,
                ("Haiphong", "LongBeach"): 3450,
                ("Haiphong", "Tokyo"): 1600,
                ("Haiphong", "Yokohama"): 1650,
                ("Haiphong", "Hamburg"): 3700,
                ("Haiphong", "Rotterdam"): 3600
            },
            
            # Lead times (days)
            "lead_times": {
                # Ocean transit times
                "ocean_transit": {
                    ("Shanghai", "LosAngeles"): 14,
                    ("Shanghai", "LongBeach"): 14,
                    ("Shanghai", "Tokyo"): 3,
                    ("Shanghai", "Yokohama"): 3,
                    ("Shanghai", "Hamburg"): 30,
                    ("Shanghai", "Rotterdam"): 28,
                    ("Shenzhen", "LosAngeles"): 13,
                    ("Shenzhen", "LongBeach"): 13,
                    ("Shenzhen", "Tokyo"): 4,
                    ("Shenzhen", "Yokohama"): 4,
                    ("Shenzhen", "Hamburg"): 29,
                    ("Shenzhen", "Rotterdam"): 27,
                    ("HoChiMinh", "LosAngeles"): 18,
                    ("HoChiMinh", "LongBeach"): 18,
                    ("HoChiMinh", "Tokyo"): 7,
                    ("HoChiMinh", "Yokohama"): 7,
                    ("HoChiMinh", "Hamburg"): 32,
                    ("HoChiMinh", "Rotterdam"): 30,
                    ("Haiphong", "LosAngeles"): 20,
                    ("Haiphong", "LongBeach"): 20,
                    ("Haiphong", "Tokyo"): 6,
                    ("Haiphong", "Yokohama"): 6,
                    ("Haiphong", "Hamburg"): 31,
                    ("Haiphong", "Rotterdam"): 29
                },
                # Production lead time
                "production": {
                    "China": 7,
                    "Vietnam": 10
                },
                # Customs clearance
                "customs": {
                    "USA": 3,
                    "Japan": 2,
                    "Germany": 2
                }
            },
            
            # Inland transportation costs (USD per unit from port to DC)
            "inland_transport_costs": {
                # USA
                ("LosAngeles", "CA_DC"): 2.5,
                ("LosAngeles", "TX_DC"): 8.0,
                ("LosAngeles", "IL_DC"): 12.0,
                ("LosAngeles", "NJ_DC"): 15.0,
                ("LongBeach", "CA_DC"): 2.5,
                ("LongBeach", "TX_DC"): 8.0,
                ("LongBeach", "IL_DC"): 12.0,
                ("LongBeach", "NJ_DC"): 15.0,
                
                # Japan
                ("Tokyo", "Tokyo_DC"): 3.0,
                ("Tokyo", "Osaka_DC"): 8.0,
                ("Yokohama", "Tokyo_DC"): 2.5,
                ("Yokohama", "Osaka_DC"): 7.5,
                
                # Germany/EU
                ("Hamburg", "Berlin_DC"): 4.0,
                ("Hamburg", "Munich_DC"): 8.0,
                ("Hamburg", "Paris_DC"): 10.0,
                ("Rotterdam", "Berlin_DC"): 6.0,
                ("Rotterdam", "Munich_DC"): 7.0,
                ("Rotterdam", "Paris_DC"): 8.0
            },
            
            # Demand by DC and product (units per month)
            "demand": {
                # USA DCs
                ("CA_DC", "VR_QUEST_3"): 15000,
                ("CA_DC", "VR_QUEST_3_PRO"): 5000,
                ("TX_DC", "VR_QUEST_3"): 10000,
                ("TX_DC", "VR_QUEST_3_PRO"): 3000,
                ("IL_DC", "VR_QUEST_3"): 8000,
                ("IL_DC", "VR_QUEST_3_PRO"): 2500,
                ("NJ_DC", "VR_QUEST_3"): 12000,
                ("NJ_DC", "VR_QUEST_3_PRO"): 4000,
                
                # Japan DCs
                ("Tokyo_DC", "VR_QUEST_3"): 8000,
                ("Tokyo_DC", "VR_QUEST_3_PRO"): 3000,
                ("Osaka_DC", "VR_QUEST_3"): 5000,
                ("Osaka_DC", "VR_QUEST_3_PRO"): 2000,
                
                # EU DCs
                ("Berlin_DC", "VR_QUEST_3"): 6000,
                ("Berlin_DC", "VR_QUEST_3_PRO"): 2000,
                ("Munich_DC", "VR_QUEST_3"): 4000,
                ("Munich_DC", "VR_QUEST_3_PRO"): 1500,
                ("Paris_DC", "VR_QUEST_3"): 7000,
                ("Paris_DC", "VR_QUEST_3_PRO"): 2500
            },
            
            # Container capacity
            "container_capacity": {
                "VR_QUEST_3": 2000,
                "VR_QUEST_3_PRO": 1500
            },
            
            # Fixed costs
            "fixed_costs": {
                "warehouse_monthly": {
                    "CA_DC": 50000,
                    "TX_DC": 40000,
                    "IL_DC": 45000,
                    "NJ_DC": 55000,
                    "Tokyo_DC": 60000,
                    "Osaka_DC": 45000,
                    "Berlin_DC": 40000,
                    "Munich_DC": 35000,
                    "Paris_DC": 45000
                }
            },
            
            # Inventory holding cost (% of product value per month)
            "inventory_holding_cost_rate": 0.02,
            
            # Safety stock days
            "safety_stock_days": 14
        }
    
    def _setup_network(self):
        """Setup network structure from config"""
        # Origins
        self.origins = ["China", "Vietnam"]
        
        # Products
        self.products = list(self.config["products"].keys())
        
        # Origin ports
        self.origin_ports = {
            "China": ["Shanghai", "Shenzhen"],
            "Vietnam": ["HoChiMinh", "Haiphong"]
        }
        
        # Destinations and their ports
        self.destinations = {
            "USA": ["LosAngeles", "LongBeach"],
            "Japan": ["Tokyo", "Yokohama"],
            "Germany": ["Hamburg", "Rotterdam"]
        }
        
        # Distribution centers by destination
        self.dcs_by_destination = {
            "USA": ["CA_DC", "TX_DC", "IL_DC", "NJ_DC"],
            "Japan": ["Tokyo_DC", "Osaka_DC"],
            "Germany": ["Berlin_DC", "Munich_DC", "Paris_DC"]
        }
        
        # All ports and DCs
        self.dest_ports = [port for ports in self.destinations.values() for port in ports]
        self.dcs = [dc for dcs in self.dcs_by_destination.values() for dc in dcs]
        
    def build_model(self):
        """Build the MILP optimization model"""
        
        # Create model
        self.model = gp.Model("Multi_Destination_Supply_Chain")
        
        # Decision variables
        
        # Production quantities
        self.x_prod = {}
        for origin in self.origins:
            for product in self.products:
                capacity = self.config["production_capacity"][origin].get(product, 0)
                self.x_prod[origin, product] = self.model.addVar(
                    vtype=GRB.INTEGER,
                    lb=0,
                    ub=capacity,
                    name=f"prod_{origin}_{product}"
                )
        
        # Flow from origin to destination port
        self.x_flow = {}
        for origin in self.origins:
            for o_port in self.origin_ports[origin]:
                for d_port in self.dest_ports:
                    for product in self.products:
                        self.x_flow[origin, o_port, d_port, product] = self.model.addVar(
                            vtype=GRB.INTEGER,
                            lb=0,
                            name=f"flow_{origin}_{o_port}_{d_port}_{product}"
                        )
        
        # Container shipments
        self.y_containers = {}
        for origin in self.origins:
            for o_port in self.origin_ports[origin]:
                for d_port in self.dest_ports:
                    for product in self.products:
                        self.y_containers[origin, o_port, d_port, product] = self.model.addVar(
                            vtype=GRB.INTEGER,
                            lb=0,
                            name=f"containers_{origin}_{o_port}_{d_port}_{product}"
                        )
        
        # Flow from destination port to DC
        self.z_port_dc = {}
        for origin in self.origins:
            for d_port in self.dest_ports:
                for dc in self.dcs:
                    for product in self.products:
                        self.z_port_dc[origin, d_port, dc, product] = self.model.addVar(
                            vtype=GRB.INTEGER,
                            lb=0,
                            name=f"port_dc_{origin}_{d_port}_{dc}_{product}"
                        )
        
        # Binary variable for DC activation
        self.dc_active = {}
        for dc in self.dcs:
            self.dc_active[dc] = self.model.addVar(
                vtype=GRB.BINARY,
                name=f"dc_active_{dc}"
            )
        
        # Update model
        self.model.update()
        
        # Build objective function
        self._build_objective()
        
        # Add constraints
        self._add_constraints()
        
    def _build_objective(self):
        """Build the objective function"""
        obj = gp.LinExpr()
        
        # 1. Production costs
        for origin in self.origins:
            for product in self.products:
                cost = self.config["production_costs"][origin].get(product, 0)
                obj += cost * self.x_prod[origin, product]
        
        # 2. Ocean freight costs
        for origin in self.origins:
            for o_port in self.origin_ports[origin]:
                for d_port in self.dest_ports:
                    for product in self.products:
                        freight_rate = self.config["ocean_freight_rates"].get((o_port, d_port), 999999)
                        obj += freight_rate * self.y_containers[origin, o_port, d_port, product]
        
        # 3. Tariff costs - based on origin-destination-HS code
        for origin in self.origins:
            for d_port in self.dest_ports:
                # Determine destination country
                destination = None
                for dest, ports in self.destinations.items():
                    if d_port in ports:
                        destination = dest
                        break
                
                if destination:
                    for dc in self.dcs:
                        for product in self.products:
                            hs_code = self.config["products"][product]["hs_code"]
                            product_value = self.config["products"][product]["value_usd"]
                            
                            tariff_key = (origin, destination, hs_code)
                            tariff_rate = self.config["tariff_rates"].get(tariff_key, 0) / 100.0
                            tariff_cost_per_unit = product_value * tariff_rate
                            
                            obj += tariff_cost_per_unit * self.z_port_dc[origin, d_port, dc, product]
        
        # 4. Inland transportation costs
        for origin in self.origins:
            for d_port in self.dest_ports:
                for dc in self.dcs:
                    for product in self.products:
                        inland_cost = self.config["inland_transport_costs"].get((d_port, dc), 999999)
                        obj += inland_cost * self.z_port_dc[origin, d_port, dc, product]
        
        # 5. DC fixed costs
        for dc in self.dcs:
            monthly_cost = self.config["fixed_costs"]["warehouse_monthly"].get(dc, 0)
            obj += monthly_cost * self.dc_active[dc]
        
        # 6. Inventory holding costs (simplified)
        holding_rate = self.config["inventory_holding_cost_rate"]
        safety_days = self.config["safety_stock_days"]
        
        for dc in self.dcs:
            for product in self.products:
                product_value = self.config["products"][product]["value_usd"]
                demand = self.config["demand"].get((dc, product), 0)
                
                # Safety stock cost
                safety_stock_units = (demand / 30) * safety_days
                safety_stock_value = safety_stock_units * product_value
                monthly_holding_cost = safety_stock_value * holding_rate
                
                obj += monthly_holding_cost * self.dc_active[dc]
        
        self.model.setObjective(obj, GRB.MINIMIZE)
    
    def _add_constraints(self):
        """Add all constraints to the model"""
        
        # 1. Production = Total outbound flow
        for origin in self.origins:
            for product in self.products:
                self.model.addConstr(
                    self.x_prod[origin, product] == gp.quicksum(
                        self.x_flow[origin, o_port, d_port, product]
                        for o_port in self.origin_ports[origin]
                        for d_port in self.dest_ports
                    ),
                    name=f"prod_balance_{origin}_{product}"
                )
        
        # 2. Container capacity constraints
        for origin in self.origins:
            for o_port in self.origin_ports[origin]:
                for d_port in self.dest_ports:
                    for product in self.products:
                        capacity = self.config["container_capacity"][product]
                        self.model.addConstr(
                            self.x_flow[origin, o_port, d_port, product] <= 
                            self.y_containers[origin, o_port, d_port, product] * capacity,
                            name=f"container_cap_{origin}_{o_port}_{d_port}_{product}"
                        )
        
        # 3. Port flow balance
        for origin in self.origins:
            for d_port in self.dest_ports:
                for product in self.products:
                    inflow = gp.quicksum(
                        self.x_flow[origin, o_port, d_port, product]
                        for o_port in self.origin_ports[origin]
                    )
                    outflow = gp.quicksum(
                        self.z_port_dc[origin, d_port, dc, product]
                        for dc in self.dcs
                    )
                    self.model.addConstr(
                        inflow == outflow,
                        name=f"port_balance_{origin}_{d_port}_{product}"
                    )
        
        # 4. Demand satisfaction
        for dc in self.dcs:
            for product in self.products:
                demand = self.config["demand"].get((dc, product), 0)
                self.model.addConstr(
                    gp.quicksum(
                        self.z_port_dc[origin, d_port, dc, product]
                        for origin in self.origins
                        for d_port in self.dest_ports
                    ) >= demand,
                    name=f"demand_{dc}_{product}"
                )
        
        # 5. DC activation constraints
        for dc in self.dcs:
            # DC must be active if it receives any flow
            for origin in self.origins:
                for d_port in self.dest_ports:
                    for product in self.products:
                        # Big-M constraint
                        M = 100000  # Large number
                        self.model.addConstr(
                            self.z_port_dc[origin, d_port, dc, product] <= M * self.dc_active[dc],
                            name=f"dc_activation_{origin}_{d_port}_{dc}_{product}"
                        )
        
        # 6. Valid port-DC connections (avoid illogical routes)
        for origin in self.origins:
            for d_port in self.dest_ports:
                for dc in self.dcs:
                    # Check if this is a valid connection
                    inland_cost = self.config["inland_transport_costs"].get((d_port, dc), None)
                    if inland_cost is None:
                        # No valid route
                        for product in self.products:
                            self.model.addConstr(
                                self.z_port_dc[origin, d_port, dc, product] == 0,
                                name=f"invalid_route_{origin}_{d_port}_{dc}_{product}"
                            )
    
    def solve(self, time_limit=300, gap=0.01):
        """Solve the optimization model"""
        
        # Set Gurobi parameters
        self.model.setParam('TimeLimit', time_limit)
        self.model.setParam('MIPGap', gap)
        self.model.setParam('OutputFlag', 0)  # Suppress output
        
        # Optimize
        self.model.optimize()
        
        if self.model.status == GRB.OPTIMAL or self.model.status == GRB.TIME_LIMIT:
            return self._extract_solution()
        else:
            return None
    
    def _extract_solution(self):
        """Extract and format the solution"""
        
        solution = {
            'objective_value': self.model.objVal,
            'status': 'Optimal' if self.model.status == GRB.OPTIMAL else 'Time Limit',
            'gap': self.model.MIPGap if hasattr(self.model, 'MIPGap') else 0,
            'production_plan': {},
            'flow_plan': {},
            'container_plan': {},
            'distribution_plan': {},
            'active_dcs': [],
            'cost_breakdown': {
                'production_cost': 0,
                'ocean_freight_cost': 0,
                'tariff_cost': 0,
                'inland_transport_cost': 0,
                'dc_fixed_cost': 0,
                'inventory_holding_cost': 0,
                'total_cost': 0
            },
            'origin_split': {},
            'destination_split': {},
            'lead_time_analysis': {},
            'tariff_analysis': {},
            'capacity_utilization': {},
            'kpis': {}
        }
        
        # Extract production plan
        for origin in self.origins:
            solution['origin_split'][origin] = 0
            solution['capacity_utilization'][origin] = {}
            
            for product in self.products:
                qty = self.x_prod[origin, product].x
                if qty > 0.5:
                    solution['production_plan'][origin, product] = qty
                    solution['cost_breakdown']['production_cost'] += qty * self.config["production_costs"][origin][product]
                    solution['origin_split'][origin] += qty
                    
                # Capacity utilization
                capacity = self.config["production_capacity"][origin].get(product, 0)
                if capacity > 0:
                    solution['capacity_utilization'][origin][product] = (qty / capacity) * 100
        
        # Extract container plan and freight costs
        total_containers = 0
        for origin in self.origins:
            for o_port in self.origin_ports[origin]:
                for d_port in self.dest_ports:
                    for product in self.products:
                        containers = self.y_containers[origin, o_port, d_port, product].x
                        if containers > 0.5:
                            solution['container_plan'][origin, o_port, d_port, product] = containers
                            freight_cost = containers * self.config["ocean_freight_rates"].get((o_port, d_port), 0)
                            solution['cost_breakdown']['ocean_freight_cost'] += freight_cost
                            total_containers += containers
        
        # Extract distribution plan and calculate tariffs
        for origin in self.origins:
            for d_port in self.dest_ports:
                # Determine destination country
                destination = None
                for dest, ports in self.destinations.items():
                    if d_port in ports:
                        destination = dest
                        break
                
                if destination:
                    if destination not in solution['destination_split']:
                        solution['destination_split'][destination] = 0
                    
                    for dc in self.dcs:
                        for product in self.products:
                            qty = self.z_port_dc[origin, d_port, dc, product].x
                            if qty > 0.5:
                                solution['distribution_plan'][origin, d_port, dc, product] = qty
                                solution['destination_split'][destination] += qty
                                
                                # Calculate tariff
                                hs_code = self.config["products"][product]["hs_code"]
                                product_value = self.config["products"][product]["value_usd"]
                                tariff_key = (origin, destination, hs_code)
                                tariff_rate = self.config["tariff_rates"].get(tariff_key, 0) / 100.0
                                tariff_cost = qty * product_value * tariff_rate
                                solution['cost_breakdown']['tariff_cost'] += tariff_cost
                                
                                # Track tariff by origin-destination
                                od_key = f"{origin}-{destination}"
                                if od_key not in solution['tariff_analysis']:
                                    solution['tariff_analysis'][od_key] = {
                                        'units': 0,
                                        'value': 0,
                                        'tariff_paid': 0,
                                        'effective_rate': 0
                                    }
                                solution['tariff_analysis'][od_key]['units'] += qty
                                solution['tariff_analysis'][od_key]['value'] += qty * product_value
                                solution['tariff_analysis'][od_key]['tariff_paid'] += tariff_cost
                                
                                # Inland transport
                                inland_cost = qty * self.config["inland_transport_costs"].get((d_port, dc), 0)
                                solution['cost_breakdown']['inland_transport_cost'] += inland_cost
        
        # Calculate effective tariff rates
        for od_key in solution['tariff_analysis']:
            analysis = solution['tariff_analysis'][od_key]
            if analysis['value'] > 0:
                analysis['effective_rate'] = (analysis['tariff_paid'] / analysis['value']) * 100
        
        # Active DCs and fixed costs
        for dc in self.dcs:
            if self.dc_active[dc].x > 0.5:
                solution['active_dcs'].append(dc)
                solution['cost_breakdown']['dc_fixed_cost'] += self.config["fixed_costs"]["warehouse_monthly"][dc]
                
                # Add inventory holding cost
                holding_rate = self.config["inventory_holding_cost_rate"]
                safety_days = self.config["safety_stock_days"]
                
                for product in self.products:
                    product_value = self.config["products"][product]["value_usd"]
                    demand = self.config["demand"].get((dc, product), 0)
                    safety_stock_units = (demand / 30) * safety_days
                    safety_stock_value = safety_stock_units * product_value
                    monthly_holding_cost = safety_stock_value * holding_rate
                    solution['cost_breakdown']['inventory_holding_cost'] += monthly_holding_cost
        
        # Lead time analysis
        solution['lead_time_analysis'] = self._calculate_lead_times(solution)
        
        # Calculate KPIs
        solution['kpis'] = self._calculate_kpis(solution, total_containers)
        
        solution['cost_breakdown']['total_cost'] = self.model.objVal
        
        return solution
    
    def _calculate_lead_times(self, solution):
        """Calculate average lead times by route"""
        lead_times = {}
        
        for (origin, o_port, d_port, product), containers in solution['container_plan'].items():
            # Find destination
            destination = None
            for dest, ports in self.destinations.items():
                if d_port in ports:
                    destination = dest
                    break
            
            if destination:
                route_key = f"{origin}-{destination}"
                
                if route_key not in lead_times:
                    lead_times[route_key] = {
                        'production': self.config["lead_times"]["production"][origin],
                        'ocean_transit': [],
                        'customs': self.config["lead_times"]["customs"][destination],
                        'total': 0
                    }
                
                ocean_time = self.config["lead_times"]["ocean_transit"].get((o_port, d_port), 0)
                lead_times[route_key]['ocean_transit'].append(ocean_time)
        
        # Calculate averages
        for route in lead_times:
            if lead_times[route]['ocean_transit']:
                avg_ocean = sum(lead_times[route]['ocean_transit']) / len(lead_times[route]['ocean_transit'])
                lead_times[route]['ocean_transit_avg'] = avg_ocean
                lead_times[route]['total'] = (
                    lead_times[route]['production'] +
                    avg_ocean +
                    lead_times[route]['customs']
                )
        
        return lead_times
    
    def _calculate_kpis(self, solution, total_containers):
        """Calculate key performance indicators"""
        kpis = {}
        
        # Total units shipped
        total_units = sum(solution['production_plan'].values())
        kpis['total_units_shipped'] = total_units
        
        # Average cost per unit
        if total_units > 0:
            kpis['avg_cost_per_unit'] = solution['objective_value'] / total_units
        
            # Cost breakdown percentages
            for cost_type, value in solution['cost_breakdown'].items():
                if cost_type != 'total_cost':
                    kpis[f'{cost_type}_percentage'] = (value / solution['objective_value']) * 100
        
        # Number of active DCs
        kpis['active_dcs_count'] = len(solution['active_dcs'])
        
        # Total containers
        kpis['total_containers'] = total_containers
        
        # Average capacity utilization
        total_utilization = 0
        count = 0
        for origin in solution['capacity_utilization']:
            for product, util in solution['capacity_utilization'][origin].items():
                total_utilization += util
                count += 1
        
        if count > 0:
            kpis['avg_capacity_utilization'] = total_utilization / count
        
        return kpis
    
    def sensitivity_analysis(self, parameter_type, variations, base_solution=None):
        """
        Perform sensitivity analysis on specified parameters
        
        Args:
            parameter_type: 'demand', 'capacity', 'tariff', or 'cost'
            variations: List of percentage variations (e.g., [-20, -10, 0, 10, 20])
            base_solution: Base solution to compare against
        
        Returns:
            Dictionary with sensitivity results
        """
        results = {
            'parameter': parameter_type,
            'variations': variations,
            'results': []
        }
        
        # Save original config using deep copy (avoids JSON serialization issues with tuple keys)
        original_config = copy.deepcopy(self.config)
        
        for variation in variations:
            # Apply variation
            self._apply_variation(parameter_type, variation)
            
            # Rebuild and solve model
            self.build_model()
            solution = self.solve()
            
            if solution:
                result = {
                    'variation': variation,
                    'total_cost': solution['objective_value'],
                    'cost_change': 0,
                    'origin_split': solution['origin_split'],
                    'destination_split': solution['destination_split']
                }
                
                if base_solution:
                    result['cost_change'] = (
                        (solution['objective_value'] - base_solution['objective_value']) / 
                        base_solution['objective_value'] * 100
                    )
                
                results['results'].append(result)
            
            # Restore original config
            self.config = copy.deepcopy(original_config)
        
        return results
    
    def _apply_variation(self, parameter_type, variation):
        """Apply variation to specified parameter type"""
        factor = 1 + (variation / 100)
        
        if parameter_type == 'demand':
            for key in self.config['demand']:
                self.config['demand'][key] = int(self.config['demand'][key] * factor)
        
        elif parameter_type == 'capacity':
            for origin in self.config['production_capacity']:
                for product in self.config['production_capacity'][origin]:
                    self.config['production_capacity'][origin][product] = int(
                        self.config['production_capacity'][origin][product] * factor
                    )
        
        elif parameter_type == 'tariff':
            for key in self.config['tariff_rates']:
                self.config['tariff_rates'][key] = self.config['tariff_rates'][key] * factor
        
        elif parameter_type == 'cost':
            # Vary production costs
            for origin in self.config['production_costs']:
                for product in self.config['production_costs'][origin]:
                    self.config['production_costs'][origin][product] *= factor
            
            # Vary freight rates
            for route in self.config['ocean_freight_rates']:
                self.config['ocean_freight_rates'][route] *= factor