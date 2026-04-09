#!/usr/bin/env python3
"""
Baseline inference script for Warehouse Inventory Environment.

This script demonstrates how an AI agent can interact with the
OpenEnv warehouse environment using the OpenAI Client API.

Output follows the strict [START], [STEP], and [END] format for evaluation.
"""

import json
import os
from typing import Dict, List
import logging
from datetime import datetime
import sys

import requests
from openai import OpenAI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class WarehouseAgent:
    """Agent that interacts with warehouse environment via OpenEnv API."""
    
    def __init__(self, env_api_base_url: str, llm_api_base_url: str, model_name: str, api_key: str):
        """Initialize agent with environment and LLM configuration."""
        self.api_base_url = env_api_base_url.rstrip("/")
        self.model_name = model_name
        
        # Route all model calls through the injected LiteLLM/OpenAI-compatible proxy.
        self.client = OpenAI(
            api_key=api_key,
            base_url=llm_api_base_url.rstrip("/")
        )
        self._proxy_verified = False
    
    def reset_environment(self, task_id: str) -> Dict:
        """Reset environment and get initial state."""
        response = requests.post(
            f"{self.api_base_url}/reset",
            json={"task_id": task_id, "seed": None}
        )
        response.raise_for_status()
        return response.json()
    
    def step_environment(self, action: Dict) -> Dict:
        """Execute one step in environment."""
        response = requests.post(
            f"{self.api_base_url}/step",
            json=action
        )
        response.raise_for_status()
        return response.json()
    
    def get_action_from_llm(self, state: Dict, task_id: str) -> List[int]:
        """
        Use LLM to decide ordering action based on current state.
        
        Demonstrates how an AI agent can use the state observation
        to make inventory decisions.
        """
        # Format state for LLM
        state_description = f"""
Current Warehouse State (Task: {task_id}):
- Inventory Levels: {state['inventory_levels']}
- Pending Orders: {state['pending_orders']}
- Demand Forecast (next period): {state['demand_forecast']}
- Current Step: {state['current_step']}
- Accumulated Holding Cost: ${state['holding_cost_accumulated']:.2f}
- Accumulated Stockout Penalty: ${state['stockout_penalty_accumulated']:.2f}

Based on the current state, decide how much to order for each product.
Return ONLY a JSON object with the format: {{"order_quantities": [qty1, qty2, ...]}}
Focus on balancing inventory costs with preventing stockouts.
Orders will arrive in {self.reorder_lead_time} steps.
"""
        
        prompt = state_description
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3,
            max_tokens=200,
            top_p=0.9
        )
        self._proxy_verified = True
        
        response_text = response.choices[0].message.content.strip()
        
        # Parse JSON response
        # Try to extract JSON from response
        import json
        if "```json" in response_text:
            json_str = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            json_str = response_text.split("```")[1].split("```")[0].strip()
        else:
            json_str = response_text
        
        try:
            action_data = json.loads(json_str)
            order_quantities = action_data.get("order_quantities", [0] * len(state['demand_forecast']))
            
            # Ensure non-negative integers
            order_quantities = [max(0, int(q)) for q in order_quantities]
            return order_quantities
        except Exception as e:
            logger.warning(f"LLM output was not valid JSON: {e}. Using forecast policy.")
            return self.forecast_policy(state)

    def verify_llm_proxy(self) -> None:
        """Fail fast unless we can complete at least one proxy-backed LLM call."""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Reply with exactly this JSON and nothing else: "
                        "{\"order_quantities\": [0]}"
                    ),
                }
            ],
            temperature=0.0,
            max_tokens=32,
        )
        self._proxy_verified = True
        logger.info(
            "Verified LLM proxy call succeeded using model response id=%s",
            getattr(response, "id", "unknown"),
        )
    
    def forecast_policy(self, state: Dict) -> List[int]:
        """
        Fallback policy: order based on demand forecast.
        Simple greedy approach to ensure valid actions.
        """
        forecast = state['demand_forecast']
        current_inv = state['inventory_levels']
        
        order_quantities = []
        for inv, demand in zip(current_inv, forecast):
            # Target 2x forecast demand
            target = demand * 2
            order_qty = max(0, target - inv)
            order_quantities.append(order_qty)
        
        return order_quantities
    
    def run_episode(self, task_id: str, episode_num: int) -> Dict:
        """
        Run one complete episode and return results.
        
        Output strictly follows [START], [STEP], [END] format.
        """
        # [START] block
        print(f"[START]")
        print(json.dumps({
            "episode": episode_num,
            "task_id": task_id,
            "timestamp": datetime.now().isoformat(),
            "model": self.model_name,
            "phase": "initialization"
        }))
        print(f"[/START]")
        
        # Reset environment
        reset_response = self.reset_environment(task_id)
        state = reset_response['state']
        self.reorder_lead_time = 2 if task_id == "easy" else (
            2 if task_id == "medium" else 3
        )
        
        total_reward = 0.0
        step_num = 0
        episode_results = []
        
        done = False
        max_steps = 200  # Safety limit
        
        while not done and step_num < max_steps:
            # Get action from LLM-based agent
            try:
                action = self.get_action_from_llm(state, task_id)
            except Exception as e:
                logger.warning(f"Failed to get LLM action: {e}")
                action = self.forecast_policy(state)
            
            # [STEP] block
            print(f"[STEP]")
            print(json.dumps({
                "step": step_num,
                "episode": episode_num,
                "task_id": task_id,
                "action": {
                    "order_quantities": action
                },
                "observation": state
            }))
            print(f"[/STEP]")
            
            # Execute step
            step_response = self.step_environment({"action": {"order_quantities": action}})
            
            state = step_response['state']
            reward = step_response['reward']
            done = step_response['done']
            info = step_response['info']
            
            total_reward = reward
            step_num += 1
            
            episode_results.append({
                "step": step_num,
                "reward": reward,
                "info": info
            })
            
            logger.info(
                f"Episode {episode_num}, Step {step_num}: "
                f"reward={reward:.4f}, done={done}"
            )
        
        # [END] block
        print(f"[END]")
        print(json.dumps({
            "episode": episode_num,
            "task_id": task_id,
            "total_steps": step_num,
            "total_reward": total_reward,
            "average_step_reward": total_reward / max(1, step_num),
            "timestamp": datetime.now().isoformat(),
            "status": "completed"
        }))
        print(f"[/END]")
        
        return {
            "episode": episode_num,
            "task_id": task_id,
            "total_reward": total_reward,
            "steps": step_num,
            "results": episode_results
        }
    
    def evaluate_tasks(self, episodes_per_task: int = 2) -> Dict:
        """
        Evaluate agent performance on all tasks.
        """
        all_results = {}
        
        for task_id in ["easy", "medium", "hard"]:
            logger.info(f"="*60)
            logger.info(f"Evaluating task: {task_id}")
            logger.info(f"="*60)
            
            task_rewards = []
            task_episodes = []
            
            for episode_num in range(episodes_per_task):
                logger.info(f"Running episode {episode_num+1}/{episodes_per_task}")
                
                try:
                    result = self.run_episode(task_id, episode_num)
                    task_rewards.append(result['total_reward'])
                    task_episodes.append(result)
                except Exception as e:
                    logger.error(f"Episode failed: {e}")
                    task_rewards.append(0.0)
            
            avg_reward = sum(task_rewards) / len(task_rewards) if task_rewards else 0.0
            
            all_results[task_id] = {
                "task_id": task_id,
                "episodes": episodes_per_task,
                "rewards": task_rewards,
                "average_reward": avg_reward,
                "episodes_data": task_episodes
            }
            
            logger.info(f"Task {task_id} complete. Average reward: {avg_reward:.4f}")
        
        return all_results


def main():
    """Main entry point."""
    # Model requests must use the evaluator-provided proxy settings.
    llm_api_base_url = os.environ["API_BASE_URL"]
    api_key = os.environ["API_KEY"]

    # The warehouse environment URL is configured separately to avoid
    # accidentally sending environment HTTP traffic to the LLM proxy.
    env_api_base_url = os.getenv(
        "ENV_API_BASE_URL",
        os.getenv("WAREHOUSE_API_BASE_URL", "http://localhost:8000")
    )
    model_name = os.getenv("MODEL_NAME", "meta-llama/Llama-2-7b")
    
    logger.info(f"Starting inference script")
    logger.info(f"LLM API_BASE_URL: {llm_api_base_url}")
    logger.info(f"ENV_API_BASE_URL: {env_api_base_url}")
    logger.info(f"MODEL_NAME: {model_name}")
    
    # Create agent
    agent = WarehouseAgent(env_api_base_url, llm_api_base_url, model_name, api_key)
    agent.verify_llm_proxy()
    
    # Evaluate tasks
    try:
        results = agent.evaluate_tasks(episodes_per_task=2)
        
        # Print summary
        print("\n" + "="*60)
        print("FINAL RESULTS SUMMARY")
        print("="*60)
        for task_id, task_results in results.items():
            print(f"\n{task_id.upper()}")
            print(f"  Episodes: {task_results['episodes']}")
            print(f"  Rewards: {[f'{r:.4f}' for r in task_results['rewards']]}")
            print(f"  Average Reward: {task_results['average_reward']:.4f}")
        
        return 0
    
    except Exception as e:
        logger.error(f"Inference failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
