"""
Core warehouse inventory environment logic.
Simulates multi-product warehouse with stochastic demand.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class TaskConfig:
    """Configuration for each task difficulty level."""
    num_products: int
    num_steps: int
    holding_cost_per_unit: float
    stockout_penalty_per_unit: float
    demand_mean: List[int]
    demand_std: List[int]
    reorder_lead_time: int
    initial_inventory: List[int]
    max_inventory: List[int]


class WarehouseEnvironment:
    """Warehouse inventory management environment."""
    
    # Task configurations
    TASK_CONFIGS = {
        "easy": TaskConfig(
            num_products=1,
            num_steps=50,
            holding_cost_per_unit=0.5,
            stockout_penalty_per_unit=5.0,
            demand_mean=[50],
            demand_std=[5],
            reorder_lead_time=2,
            initial_inventory=[100],
            max_inventory=[500]
        ),
        "medium": TaskConfig(
            num_products=3,
            num_steps=100,
            holding_cost_per_unit=0.75,
            stockout_penalty_per_unit=8.0,
            demand_mean=[40, 30, 20],
            demand_std=[8, 6, 4],
            reorder_lead_time=2,
            initial_inventory=[120, 90, 60],
            max_inventory=[400, 300, 200]
        ),
        "hard": TaskConfig(
            num_products=5,
            num_steps=150,
            holding_cost_per_unit=1.0,
            stockout_penalty_per_unit=10.0,
            demand_mean=[45, 35, 25, 20, 30],
            demand_std=[12, 10, 8, 6, 9],
            reorder_lead_time=3,
            initial_inventory=[150, 120, 100, 80, 110],
            max_inventory=[500, 400, 350, 300, 400]
        )
    }
    
    def __init__(self, task_id: str = "easy", seed: Optional[int] = None):
        """Initialize environment for given task."""
        if task_id not in self.TASK_CONFIGS:
            raise ValueError(f"Unknown task_id: {task_id}")
        
        self.task_id = task_id
        self.config = self.TASK_CONFIGS[task_id]
        self.seed_value = seed
        
        if seed is not None:
            np.random.seed(seed)
        
        # Initialize state
        self.reset_to_initial_state()
    
    def reset_to_initial_state(self):
        """Reset environment to initial state."""
        self.current_step = 0
        self.inventory_levels = list(self.config.initial_inventory)
        self.pending_orders = [0] * self.config.num_products
        
        # Track costs for reward calculation
        self.holding_cost_accumulated = 0.0
        self.stockout_penalty_accumulated = 0.0
        self.step_count = 0
        
        # Generate demand sequence
        self.generate_demand_sequence()
        
        # Order pipeline: [step t, step t+1, ..., step t+lead_time]
        self.order_pipeline = [
            [0] * self.config.num_products 
            for _ in range(self.config.reorder_lead_time + 1)
        ]
    
    def generate_demand_sequence(self):
        """Generate stochastic demand sequence for episode."""
        self.demand_sequence = []
        for _ in range(self.config.num_steps + self.config.reorder_lead_time):
            step_demand = []
            for product_idx in range(self.config.num_products):
                # Generate demand with seasonal pattern for medium/hard
                seasonality = 1.0
                if self.task_id in ["medium", "hard"]:
                    # Add seasonal pattern
                    step_in_season = (_ % 20) / 20.0
                    seasonality = 0.8 + 0.4 * np.sin(2 * np.pi * step_in_season)
                
                demand = max(
                    0,
                    int(np.random.normal(
                        self.config.demand_mean[product_idx] * seasonality,
                        self.config.demand_std[product_idx]
                    ))
                )
                step_demand.append(demand)
            self.demand_sequence.append(step_demand)
    
    def get_current_demand(self) -> List[int]:
        """Get demand for current step."""
        if self.current_step >= len(self.demand_sequence):
            return [0] * self.config.num_products
        return self.demand_sequence[self.current_step]
    
    def get_demand_forecast(self, steps_ahead: int = 1) -> List[int]:
        """Get forecasted demand for steps_ahead."""
        forecast_step = min(
            self.current_step + steps_ahead,
            len(self.demand_sequence) - 1
        )
        if forecast_step >= len(self.demand_sequence):
            return list(self.config.demand_mean)
        return self.demand_sequence[forecast_step]
    
    def step(self, order_quantities: List[int]) -> Tuple[Dict, float, bool, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            order_quantities: List of order quantities for each product
            
        Returns:
            (observation, reward, done, info)
        """
        # Validate action
        if len(order_quantities) != self.config.num_products:
            raise ValueError(
                f"Expected {self.config.num_products} order quantities, "
                f"got {len(order_quantities)}"
            )
        
        # Process orders: move items through pipeline
        received_orders = self.order_pipeline.pop(0)
        self.order_pipeline.append(order_quantities)
        
        # Get current demand
        current_demand = self.get_current_demand()
        
        # Add received orders to inventory
        for product_idx in range(self.config.num_products):
            self.inventory_levels[product_idx] += received_orders[product_idx]
            self.inventory_levels[product_idx] = min(
                self.inventory_levels[product_idx],
                self.config.max_inventory[product_idx]
            )
        
        # Calculate costs and demand satisfaction
        holding_cost = 0.0
        stockout_penalty = 0.0
        demand_satisfied = []
        
        for product_idx in range(self.config.num_products):
            # Handle demand
            demand = current_demand[product_idx]
            if self.inventory_levels[product_idx] >= demand:
                self.inventory_levels[product_idx] -= demand
                demand_satisfied.append(True)
            else:
                # Stockout
                demand_satisfied.append(False)
                shortage = demand - self.inventory_levels[product_idx]
                stockout_penalty += shortage * self.config.stockout_penalty_per_unit
                self.inventory_levels[product_idx] = 0
            
            # Calculate holding cost
            holding_cost += (
                self.inventory_levels[product_idx] * 
                self.config.holding_cost_per_unit
            )
        
        # Accumulate costs
        self.holding_cost_accumulated += holding_cost
        self.stockout_penalty_accumulated += stockout_penalty
        
        # Calculate step reward (negative costs / max possible costs)
        step_cost = holding_cost + stockout_penalty
        max_cost = (
            sum(self.config.max_inventory) * self.config.holding_cost_per_unit +
            sum(current_demand) * self.config.stockout_penalty_per_unit
        )
        step_reward = 1.0 - (step_cost / max_cost if max_cost > 0 else 0.0)
        
        # Calculate cumulative normalized reward
        total_cost = self.holding_cost_accumulated + self.stockout_penalty_accumulated
        max_accumulated_cost = max_cost * (self.current_step + 1)
        cumulative_reward = 1.0 - (
            total_cost / max_accumulated_cost if max_accumulated_cost > 0 else 0.0
        )
        cumulative_reward = max(0.0, min(1.0, cumulative_reward))
        
        self.current_step += 1
        self.step_count += 1
        
        done = self.current_step >= self.config.num_steps
        
        # Build observation
        observation = {
            "inventory_levels": list(self.inventory_levels),
            "pending_orders": list(self.order_pipeline[-1]) if self.order_pipeline else [0] * self.config.num_products,
            "demand_forecast": self.get_demand_forecast(steps_ahead=1),
            "current_step": self.current_step,
            "holding_cost_accumulated": self.holding_cost_accumulated,
            "stockout_penalty_accumulated": self.stockout_penalty_accumulated,
        }
        
        info = {
            "step_reward": step_reward,
            "costs": {
                "holding_cost": holding_cost,
                "stockout_penalty": stockout_penalty,
            },
            "demand_satisfied": demand_satisfied,
        }
        
        return observation, cumulative_reward, done, info
    
    def reset(self) -> Dict:
        """Reset environment and return initial state."""
        self.reset_to_initial_state()
        
        return {
            "inventory_levels": list(self.inventory_levels),
            "pending_orders": [0] * self.config.num_products,
            "demand_forecast": self.get_demand_forecast(steps_ahead=1),
            "current_step": 0,
            "holding_cost_accumulated": 0.0,
            "stockout_penalty_accumulated": 0.0,
        }
    
    def get_state(self) -> Dict:
        """Get current state without stepping."""
        return {
            "inventory_levels": list(self.inventory_levels),
            "pending_orders": list(self.order_pipeline[-1]) if self.order_pipeline else [0] * self.config.num_products,
            "demand_forecast": self.get_demand_forecast(steps_ahead=1),
            "current_step": self.current_step,
            "holding_cost_accumulated": self.holding_cost_accumulated,
            "stockout_penalty_accumulated": self.stockout_penalty_accumulated,
        }
