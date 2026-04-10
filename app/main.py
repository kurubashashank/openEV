"""
FastAPI application for OpenEnv warehouse environment.
Implements reset(), step(), and state() endpoints.
"""

from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from typing import Dict, Optional
import logging

from app.models import (
    ResetRequest, ResetResponse, StepRequest, StepResponse,
    State, Action, StepInfo, CostInfo, TaskGradeRequest, TaskGradeResponse,
    TaskInfo, TaskListResponse
)
from app.environment import WarehouseEnvironment
from app.graders import TaskGrader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Active environment instances keyed by task, plus the most recently reset task.
environments: Dict[str, WarehouseEnvironment] = {}
current_task: Optional[str] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize environments on startup."""
    global current_task
    logger.info("Starting warehouse environment API")
    environments.clear()
    current_task = None
    yield
    
    logger.info("Shutting down warehouse environment API")
    environments.clear()
    current_task = None


app = FastAPI(
    title="Warehouse Inventory Environment",
    description="OpenEnv environment for warehouse inventory optimization",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "Warehouse Inventory Environment",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/reset", response_model=ResetResponse)
async def reset(request: Optional[ResetRequest] = None):
    """
    Reset environment to initial state.
    
    OpenEnv Spec: Resets the environment and returns initial observation/state.
    """
    try:
        global current_task
        reset_request = request or ResetRequest()
        task_id = reset_request.task_id

        if task_id not in WarehouseEnvironment.TASK_CONFIGS:
            raise ValueError(f"Unknown task_id: {task_id}")

        # Create a fresh environment and make it the active task.
        env = WarehouseEnvironment(task_id=task_id, seed=reset_request.seed)
        environments[task_id] = env
        current_task = task_id

        initial_state = env.reset()
        
        logger.info(f"Environment reset for task: {task_id}")
        
        return ResetResponse(
            state=State(**initial_state),
            task_id=task_id
        )
    
    except Exception as e:
        logger.error(f"Error in reset: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step", response_model=StepResponse)
async def step(request: StepRequest):
    """
    Execute one step in the environment.
    
    OpenEnv Spec: Takes action, returns (observation, reward, done, info).
    """
    try:
        if current_task is None:
            raise ValueError("Environment not initialized. Call /reset first.")

        env = environments.get(current_task)
        if env is None:
            raise ValueError("Environment not initialized. Call /reset first.")
        
        # Validate action
        action = request.action
        if len(action.order_quantities) != env.config.num_products:
            raise ValueError(
                f"Expected {env.config.num_products} order quantities, "
                f"got {len(action.order_quantities)}"
            )
        
        # Step environment
        observation, reward, done, info = env.step(action.order_quantities)
        
        logger.debug(
            f"Step {env.current_step}: reward={reward:.4f}, "
            f"done={done}, step_reward={info['step_reward']:.4f}"
        )
        
        return StepResponse(
            state=State(**observation),
            reward=reward,
            done=done,
            info=StepInfo(
                step_reward=info["step_reward"],
                costs=CostInfo(**info["costs"]),
                demand_satisfied=info["demand_satisfied"]
            )
        )
    
    except Exception as e:
        logger.error(f"Error in step: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state", response_model=State)
async def get_state():
    """
    Get current environment state.
    
    OpenEnv Spec: Returns current observation/state without stepping.
    """
    try:
        if current_task is None:
            raise ValueError("Environment not initialized. Call /reset first.")

        env = environments.get(current_task)
        if env is None:
            raise ValueError("Environment not initialized. Call /reset first.")
        
        state = env.get_state()
        return State(**state)
    
    except Exception as e:
        logger.error(f"Error in get_state: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/tasks", response_model=TaskListResponse)
async def list_tasks():
    """Return task metadata in a validator-friendly format."""
    return TaskListResponse(
        tasks=[
            TaskInfo(
                id="easy",
                difficulty="easy",
                description="Single warehouse, single product, stable demand with low variance",
                reset_params={"task_id": "easy"},
            ),
            TaskInfo(
                id="medium",
                difficulty="medium",
                description="Single warehouse, three products, seasonal demand patterns",
                reset_params={"task_id": "medium"},
            ),
            TaskInfo(
                id="hard",
                difficulty="hard",
                description="Multiple locations, multiple products, volatile demand with supply constraints",
                reset_params={"task_id": "hard"},
            ),
        ]
    )


@app.post("/grade", response_model=TaskGradeResponse)
async def grade(request: TaskGradeRequest):
    """
    Grade agent performance on a task using baseline policy.
    
    This endpoint evaluates task difficulty by running episodes
    with a forecast-based baseline policy.
    """
    try:
        if request.task_id not in ["easy", "medium", "hard"]:
            raise ValueError(f"Unknown task_id: {request.task_id}")
        
        # Grade with forecast policy baseline
        grading_result = TaskGrader.grade_task(
            task_id=request.task_id,
            num_episodes=request.num_episodes,
            policy_fn=TaskGrader.forecast_policy,
            seed=request.seed
        )
        
        logger.info(
            f"Graded task {request.task_id}: "
            f"avg_reward={grading_result['average_reward']:.4f}, "
            f"grade={grading_result['grade']}"
        )
        
        return TaskGradeResponse(**grading_result)
    
    except Exception as e:
        logger.error(f"Error in grade: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/info")
async def info():
    """Get environment information."""
    return {
        "name": "Warehouse Inventory Environment",
        "description": "Multi-product warehouse inventory optimization",
        "tasks": ["easy", "medium", "hard"],
        "observation_space": {
            "inventory_levels": "array of integers",
            "pending_orders": "array of integers",
            "demand_forecast": "array of integers",
            "current_step": "integer",
            "holding_cost_accumulated": "float",
            "stockout_penalty_accumulated": "float",
        },
        "action_space": {
            "order_quantities": "array of non-negative integers"
        },
        "reward": "float strictly between 0.0 and 1.0"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
