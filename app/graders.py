"""
Task graders for evaluating agent performance.
Implements baseline strategies and scoring rubrics.
"""

from typing import List, Dict
from app.environment import WarehouseEnvironment


class TaskGrader:
    """Grader for warehouse environment tasks."""
    
    @staticmethod
    def run_episode(
        task_id: str,
        policy_fn=None,
        seed: int = None
    ) -> float:
        """
        Run one episode with given policy.
        
        Args:
            task_id: Task identifier ('easy', 'medium', 'hard')
            policy_fn: Policy function that takes state and returns action.
                      If None, uses random baseline.
            seed: Random seed
            
        Returns:
            Episode reward (0.0 to 1.0)
        """
        env = WarehouseEnvironment(task_id=task_id, seed=seed)
        state = env.reset()
        
        if policy_fn is None:
            policy_fn = TaskGrader.random_policy
        
        total_reward = 0.0
        episode_length = 0
        
        done = False
        while not done and episode_length < 500:
            # Get action from policy
            action = policy_fn(state, env.config)
            
            # Step environment
            state, reward, done, info = env.step(action["order_quantities"])
            total_reward = reward
            episode_length += 1
        
        return total_reward
    
    @staticmethod
    def random_policy(state: Dict, config) -> Dict:
        """Random baseline policy - orders mean demand."""
        import numpy as np
        
        mean_demand = config.demand_mean
        noise = np.random.normal(0, 0.2, len(mean_demand))
        order_quantities = [
            max(0, int(mean * (1 + noise_val)))
            for mean, noise_val in zip(mean_demand, noise)
        ]
        
        return {"order_quantities": order_quantities}
    
    @staticmethod
    def conservative_policy(state: Dict, config) -> Dict:
        """Conservative policy - maintain target inventory."""
        target_inventory = [int(mean * 2) for mean in config.demand_mean]
        order_quantities = [
            max(0, target - inv)
            for target, inv in zip(target_inventory, state["inventory_levels"])
        ]
        
        return {"order_quantities": order_quantities}
    
    @staticmethod
    def forecast_policy(state: Dict, config) -> Dict:
        """Forecast-based policy - order based on demand forecast."""
        forecast = state["demand_forecast"]
        current_inv = state["inventory_levels"]
        
        order_quantities = []
        for inv, demand in zip(current_inv, forecast):
            # Order to maintain 2x forecast demand
            target = demand * 2
            order_qty = max(0, target - inv)
            order_quantities.append(order_qty)
        
        return {"order_quantities": order_quantities}
    
    @staticmethod
    def grade_task(
        task_id: str,
        num_episodes: int = 3,
        policy_fn=None,
        seed: int = None
    ) -> Dict:
        """
        Grade agent performance on a task.
        
        Args:
            task_id: Task identifier
            num_episodes: Number of episodes to evaluate
            policy_fn: Policy function (uses random baseline if None)
            seed: Random seed
            
        Returns:
            Grading results with average reward and PASS/FAIL
        """
        episode_rewards = []
        
        for episode_idx in range(num_episodes):
            episode_seed = (
                seed + episode_idx if seed is not None else None
            )
            reward = TaskGrader.run_episode(
                task_id=task_id,
                policy_fn=policy_fn,
                seed=episode_seed
            )
            episode_rewards.append(reward)
        
        average_reward = sum(episode_rewards) / len(episode_rewards)
        
        # Grading rubric: pass if average reward > 0.5
        grade = "PASS" if average_reward > 0.5 else "FAIL"
        
        return {
            "task_id": task_id,
            "num_episodes": num_episodes,
            "average_reward": average_reward,
            "episode_rewards": episode_rewards,
            "grade": grade,
        }
    
    @staticmethod
    def baseline_scores() -> Dict:
        """
        Get baseline scores using all policies.
        Returns comparison of different baseline policies.
        """
        results = {}
        
        for task_id in ["easy", "medium", "hard"]:
            results[task_id] = {
                "random": TaskGrader.grade_task(
                    task_id, num_episodes=3, policy_fn=TaskGrader.random_policy
                ),
                "conservative": TaskGrader.grade_task(
                    task_id, num_episodes=3, policy_fn=TaskGrader.conservative_policy
                ),
                "forecast": TaskGrader.grade_task(
                    task_id, num_episodes=3, policy_fn=TaskGrader.forecast_policy
                ),
            }
        
        return results
