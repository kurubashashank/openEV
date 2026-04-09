"""
Pydantic models for OpenEnv warehouse environment.
Provides type safety and request/response validation.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class Action(BaseModel):
    """Action space: order quantities for each product."""
    order_quantities: List[int] = Field(
        ..., 
        description="Order quantity for each product (must be non-negative)"
    )


class State(BaseModel):
    """Observation space: current state of the environment."""
    inventory_levels: List[int] = Field(
        ..., 
        description="Current inventory level for each product"
    )
    pending_orders: List[int] = Field(
        ..., 
        description="Pending order quantities for each product"
    )
    demand_forecast: List[int] = Field(
        ..., 
        description="Forecasted demand for next period for each product"
    )
    current_step: int = Field(
        ..., 
        description="Current time step"
    )
    holding_cost_accumulated: float = Field(
        ..., 
        description="Accumulated holding costs"
    )
    stockout_penalty_accumulated: float = Field(
        ..., 
        description="Accumulated stockout penalties"
    )


class ResetRequest(BaseModel):
    """Request to reset the environment."""
    task_id: str = Field(
        default="easy",
        description="Task difficulty: 'easy', 'medium', or 'hard'"
    )
    seed: Optional[int] = Field(
        default=None,
        description="Random seed for reproducibility"
    )


class ResetResponse(BaseModel):
    """Response from reset endpoint."""
    state: State
    task_id: str


class StepRequest(BaseModel):
    """Request for one environment step."""
    action: Action = Field(
        ..., 
        description="Agent action (order quantities)"
    )


class CostInfo(BaseModel):
    """Cost breakdown for a step."""
    holding_cost: float = Field(
        ...,
        description="Cost of holding inventory"
    )
    stockout_penalty: float = Field(
        ...,
        description="Penalty for not meeting demand"
    )


class StepInfo(BaseModel):
    """Additional information from a step."""
    step_reward: float = Field(
        ...,
        description="Reward for this step"
    )
    costs: CostInfo
    demand_satisfied: List[bool] = Field(
        ...,
        description="Whether demand was satisfied for each product"
    )


class StepResponse(BaseModel):
    """Response from step endpoint."""
    state: State
    reward: float = Field(
        ...,
        gt=0.0,
        lt=1.0,
        description="Cumulative normalized reward strictly between 0.0 and 1.0"
    )
    done: bool = Field(
        ...,
        description="Whether episode is finished"
    )
    info: StepInfo = Field(
        ...,
        description="Additional step information"
    )


class TaskGradeRequest(BaseModel):
    """Request to run a task grader."""
    task_id: str
    num_episodes: int = Field(
        default=1,
        ge=1,
        le=10,
        description="Number of episodes to run for grading"
    )
    seed: Optional[int] = None


class TaskGradeResponse(BaseModel):
    """Response from task grader."""
    task_id: str
    num_episodes: int
    average_reward: float = Field(
        ...,
        gt=0.0,
        lt=1.0,
        description="Average reward across episodes, strictly between 0.0 and 1.0"
    )
    episode_rewards: List[float] = Field(
        ...,
        description="Reward for each episode"
    )
    grade: str = Field(
        ...,
        description="Grade: PASS or FAIL"
    )
