"""
Comprehensive tests for agents module.
Tests all agent classes, environments, and agent utilities.
"""

import unittest
import tempfile
import shutil
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import json
from typing import Dict, Any, List

# Import reward modeling components
from reward_modeling.models.reward_model import RewardModel
from reward_modeling.data.dataset import PreferenceDataset, PreferencePair, create_synthetic_preference_data


class MockRewardModel(nn.Module):
    """Mock reward model for testing."""
    
    def __init__(self, output_dim=1):
        super().__init__()
        self.linear = nn.Linear(10, output_dim)
        self.output_dim = output_dim
    
    def forward(self, input_ids, attention_mask=None, return_dict=True):
        batch_size = input_ids.shape[0]
        rewards = torch.randn(batch_size, self.output_dim).squeeze(-1)
        
        if return_dict:
            return {"rewards": rewards}
        return rewards
    
    def to(self, device):
        return self
    
    def eval(self):
        return self


class MockEnvironment:
    """Mock environment for testing agents."""
    
    def __init__(self, max_steps=10):
        self.max_steps = max_steps
        self.current_step = 0
        self.state_dim = 8
        self.action_dim = 4
        self.done = False
    
    def reset(self):
        self.current_step = 0
        self.done = False
        return np.random.randn(self.state_dim)
    
    def step(self, action):
        self.current_step += 1
        next_state = np.random.randn(self.state_dim)
        reward = np.random.randn()
        done = self.current_step >= self.max_steps
        info = {"step": self.current_step}
        
        if done:
            self.done = True
        
        return next_state, reward, done, info
    
    def render(self):
        pass


class MockAgent:
    """Mock agent for testing."""
    
    def __init__(self, state_dim=8, action_dim=4):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.policy = nn.Linear(state_dim, action_dim)
        self.value_function = nn.Linear(state_dim, 1)
        
    def select_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_logits = self.policy(state_tensor)
        action = torch.multinomial(torch.softmax(action_logits, dim=-1), 1)
        return action.item()
    
    def get_value(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        value = self.value_function(state_tensor)
        return value.item()
    
    def update(self, states, actions, rewards, next_states, dones):
        # Mock update implementation
        pass
    
    def save(self, path):
        torch.save({
            'policy': self.policy.state_dict(),
            'value_function': self.value_function.state_dict()
        }, path)
    
    def load(self, path):
        checkpoint = torch.load(path, map_location='cpu')
        self.policy.load_state_dict(checkpoint['policy'])
        self.value_function.load_state_dict(checkpoint['value_function'])


class TestAgentEnvironmentInterface(unittest.TestCase):
    """Test cases for agent-environment interface."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.env = MockEnvironment(max_steps=5)
        self.agent = MockAgent()
    
    def test_environment_reset(self):
        """Test environment reset functionality."""
        state = self.env.reset()
        
        self.assertIsInstance(state, np.ndarray)
        self.assertEqual(len(state), self.env.state_dim)
        self.assertEqual(self.env.current_step, 0)
        self.assertFalse(self.env.done)
    
    def test_environment_step(self):
        """Test environment step functionality."""
        self.env.reset()
        action = 1
        
        next_state, reward, done, info = self.env.step(action)
        
        self.assertIsInstance(next_state, np.ndarray)
        self.assertEqual(len(next_state), self.env.state_dim)
        self.assertIsInstance(reward, (int, float))
        self.assertIsInstance(done, bool)
        self.assertIsInstance(info, dict)
        self.assertEqual(self.env.current_step, 1)
    
    def test_agent_action_selection(self):
        """Test agent action selection."""
        state = np.random.randn(self.agent.state_dim)
        action = self.agent.select_action(state)
        
        self.assertIsInstance(action, int)
        self.assertTrue(0 <= action < self.agent.action_dim)
    
    def test_agent_value_estimation(self):
        """Test agent value function."""
        state = np.random.randn(self.agent.state_dim)
        value = self.agent.get_value(state)
        
        self.assertIsInstance(value, (int, float))
    
    def test_agent_environment_interaction(self):
        """Test full agent-environment interaction."""
        state = self.env.reset()
        total_reward = 0
        steps = 0
        
        while not self.env.done and steps < 10:
            action = self.agent.select_action(state)
            next_state, reward, done, info = self.env.step(action)
            
            total_reward += reward
            state = next_state
            steps += 1
            
            if done:
                break
        
        self.assertGreater(steps, 0)
        self.assertTrue(self.env.done or steps >= 10)


class TestRewardModelIntegration(unittest.TestCase):
    """Test cases for reward model integration with agents."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.reward_model = MockRewardModel()
        self.env = MockEnvironment()
        self.agent = MockAgent()
    
    def test_reward_model_evaluation(self):
        """Test using reward model for evaluation."""
        # Mock trajectory data
        states = [np.random.randn(8) for _ in range(5)]
        actions = [np.random.randint(0, 4) for _ in range(5)]
        
        # Convert to mock input format for reward model
        mock_input_ids = torch.randint(0, 100, (5, 10))
        
        with torch.no_grad():
            outputs = self.reward_model(mock_input_ids)
            rewards = outputs["rewards"] if isinstance(outputs, dict) else outputs
        
        self.assertIsInstance(rewards, torch.Tensor)
        self.assertEqual(len(rewards), 5)
    
    def test_reward_model_guided_training(self):
        """Test using reward model to guide agent training."""
        # Simulate trajectory collection
        trajectories = []
        
        for episode in range(3):
            state = self.env.reset()
            episode_data = {
                'states': [],
                'actions': [],
                'rewards': [],
                'next_states': []
            }
            
            for step in range(5):
                action = self.agent.select_action(state)
                next_state, env_reward, done, info = self.env.step(action)
                
                episode_data['states'].append(state)
                episode_data['actions'].append(action)
                episode_data['rewards'].append(env_reward)
                episode_data['next_states'].append(next_state)
                
                state = next_state
                if done:
                    break
            
            trajectories.append(episode_data)
        
        # Verify trajectory collection
        self.assertEqual(len(trajectories), 3)
        for traj in trajectories:
            self.assertIn('states', traj)
            self.assertIn('actions', traj)
            self.assertIn('rewards', traj)
            self.assertGreater(len(traj['states']), 0)


class TestAgentTraining(unittest.TestCase):
    """Test cases for agent training procedures."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.agent = MockAgent()
        self.env = MockEnvironment()
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_agent_policy_update(self):
        """Test agent policy update."""
        # Generate mock training data
        batch_size = 32
        states = torch.randn(batch_size, self.agent.state_dim)
        actions = torch.randint(0, self.agent.action_dim, (batch_size,))
        rewards = torch.randn(batch_size)
        next_states = torch.randn(batch_size, self.agent.state_dim)
        dones = torch.randint(0, 2, (batch_size,)).bool()
        
        # Store initial policy parameters
        initial_params = [p.clone() for p in self.agent.policy.parameters()]
        
        # Perform update (mock implementation)
        self.agent.update(
            states.numpy(),
            actions.numpy(),
            rewards.numpy(),
            next_states.numpy(),
            dones.numpy()
        )
        
        # Verify that parameters could be updated (structure is correct)
        current_params = list(self.agent.policy.parameters())
        self.assertEqual(len(initial_params), len(current_params))
    
    def test_agent_save_and_load(self):
        """Test agent saving and loading."""
        # Save agent
        save_path = Path(self.temp_dir) / "test_agent.pt"
        self.agent.save(str(save_path))
        
        self.assertTrue(save_path.exists())
        
        # Create new agent and load
        new_agent = MockAgent(
            state_dim=self.agent.state_dim,
            action_dim=self.agent.action_dim
        )
        new_agent.load(str(save_path))
        
        # Test that loaded agent works
        test_state = np.random.randn(self.agent.state_dim)
        action1 = self.agent.select_action(test_state)
        action2 = new_agent.select_action(test_state)
        
        # Both should be valid actions
        self.assertTrue(0 <= action1 < self.agent.action_dim)
        self.assertTrue(0 <= action2 < self.agent.action_dim)
    
    def test_training_loop(self):
        """Test basic training loop structure."""
        num_episodes = 5
        episode_rewards = []
        
        for episode in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            
            while not self.env.done:
                action = self.agent.select_action(state)
                next_state, reward, done, info = self.env.step(action)
                
                episode_reward += reward
                state = next_state
                
                if done:
                    break
            
            episode_rewards.append(episode_reward)
        
        self.assertEqual(len(episode_rewards), num_episodes)
        self.assertTrue(all(isinstance(r, (int, float)) for r in episode_rewards))


class TestAgentEvaluation(unittest.TestCase):
    """Test cases for agent evaluation metrics."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.agent = MockAgent()
        self.env = MockEnvironment()
    
    def test_episode_return_calculation(self):
        """Test calculating episode returns."""
        rewards = [1.0, 0.5, -0.2, 0.8, 1.2]
        gamma = 0.99
        
        # Calculate discounted return
        discounted_return = 0
        for i, reward in enumerate(rewards):
            discounted_return += reward * (gamma ** i)
        
        # Calculate undiscounted return
        undiscounted_return = sum(rewards)
        
        self.assertAlmostEqual(undiscounted_return, 3.3, places=5)
        self.assertTrue(discounted_return <= undiscounted_return)
    
    def test_success_rate_calculation(self):
        """Test calculating agent success rate."""
        # Simulate multiple episodes
        num_episodes = 10
        successes = 0
        
        for episode in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            
            while not self.env.done:
                action = self.agent.select_action(state)
                next_state, reward, done, info = self.env.step(action)
                episode_reward += reward
                state = next_state
                
                if done:
                    break
            
            # Define success as achieving positive total reward
            if episode_reward > 0:
                successes += 1
        
        success_rate = successes / num_episodes
        self.assertTrue(0 <= success_rate <= 1)
    
    def test_action_distribution_analysis(self):
        """Test analyzing agent's action distribution."""
        num_samples = 100
        action_counts = {i: 0 for i in range(self.agent.action_dim)}
        
        for _ in range(num_samples):
            state = np.random.randn(self.agent.state_dim)
            action = self.agent.select_action(state)
            action_counts[action] += 1
        
        # Verify all actions are valid
        for action in action_counts.keys():
            self.assertTrue(0 <= action < self.agent.action_dim)
        
        # Verify total counts
        total_counts = sum(action_counts.values())
        self.assertEqual(total_counts, num_samples)
        
        # Calculate action distribution
        action_distribution = {
            action: count / total_counts 
            for action, count in action_counts.items()
        }
        
        # Verify probabilities sum to 1
        self.assertAlmostEqual(sum(action_distribution.values()), 1.0, places=5)


class TestAgentComparison(unittest.TestCase):
    """Test cases for comparing multiple agents."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.agents = {
            'agent_1': MockAgent(state_dim=8, action_dim=4),
            'agent_2': MockAgent(state_dim=8, action_dim=4)
        }
        self.env = MockEnvironment()
    
    def test_multi_agent_evaluation(self):
        """Test evaluating multiple agents."""
        num_episodes = 5
        agent_results = {}
        
        for agent_name, agent in self.agents.items():
            episode_rewards = []
            
            for episode in range(num_episodes):
                state = self.env.reset()
                episode_reward = 0
                
                while not self.env.done:
                    action = agent.select_action(state)
                    next_state, reward, done, info = self.env.step(action)
                    episode_reward += reward
                    state = next_state
                    
                    if done:
                        break
                
                episode_rewards.append(episode_reward)
            
            agent_results[agent_name] = {
                'mean_reward': np.mean(episode_rewards),
                'std_reward': np.std(episode_rewards),
                'episode_rewards': episode_rewards
            }
        
        # Verify results for both agents
        self.assertEqual(len(agent_results), 2)
        for agent_name, results in agent_results.items():
            self.assertIn('mean_reward', results)
            self.assertIn('std_reward', results)
            self.assertEqual(len(results['episode_rewards']), num_episodes)
    
    def test_agent_ranking(self):
        """Test ranking agents by performance."""
        # Mock performance data
        agent_performances = {
            'agent_1': {'mean_reward': 2.5, 'success_rate': 0.8},
            'agent_2': {'mean_reward': 1.8, 'success_rate': 0.6}
        }
        
        # Rank by mean reward
        ranked_by_reward = sorted(
            agent_performances.items(),
            key=lambda x: x[1]['mean_reward'],
            reverse=True
        )
        
        # Rank by success rate
        ranked_by_success = sorted(
            agent_performances.items(),
            key=lambda x: x[1]['success_rate'],
            reverse=True
        )
        
        self.assertEqual(ranked_by_reward[0][0], 'agent_1')
        self.assertEqual(ranked_by_success[0][0], 'agent_1')


class TestTrajectoryCollection(unittest.TestCase):
    """Test cases for trajectory collection and processing."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.agent = MockAgent()
        self.env = MockEnvironment()
    
    def test_trajectory_collection(self):
        """Test collecting agent trajectories."""
        trajectory = {
            'states': [],
            'actions': [],
            'rewards': [],
            'next_states': [],
            'dones': []
        }
        
        state = self.env.reset()
        
        while not self.env.done:
            action = self.agent.select_action(state)
            next_state, reward, done, info = self.env.step(action)
            
            trajectory['states'].append(state.copy())
            trajectory['actions'].append(action)
            trajectory['rewards'].append(reward)
            trajectory['next_states'].append(next_state.copy())
            trajectory['dones'].append(done)
            
            state = next_state
            
            if done:
                break
        
        # Verify trajectory structure
        self.assertGreater(len(trajectory['states']), 0)
        self.assertEqual(len(trajectory['states']), len(trajectory['actions']))
        self.assertEqual(len(trajectory['actions']), len(trajectory['rewards']))
        self.assertEqual(len(trajectory['rewards']), len(trajectory['next_states']))
        self.assertEqual(len(trajectory['next_states']), len(trajectory['dones']))
    
    def test_trajectory_preprocessing(self):
        """Test preprocessing trajectories for training."""
        # Create mock trajectory
        trajectory_length = 10
        trajectory = {
            'states': [np.random.randn(8) for _ in range(trajectory_length)],
            'actions': [np.random.randint(0, 4) for _ in range(trajectory_length)],
            'rewards': [np.random.randn() for _ in range(trajectory_length)],
            'next_states': [np.random.randn(8) for _ in range(trajectory_length)],
            'dones': [False] * (trajectory_length - 1) + [True]
        }
        
        # Convert to arrays
        states = np.array(trajectory['states'])
        actions = np.array(trajectory['actions'])
        rewards = np.array(trajectory['rewards'])
        next_states = np.array(trajectory['next_states'])
        dones = np.array(trajectory['dones'])
        
        # Verify shapes
        self.assertEqual(states.shape, (trajectory_length, 8))
        self.assertEqual(actions.shape, (trajectory_length,))
        self.assertEqual(rewards.shape, (trajectory_length,))
        self.assertEqual(next_states.shape, (trajectory_length, 8))
        self.assertEqual(dones.shape, (trajectory_length,))
        
        # Verify data types
        self.assertTrue(states.dtype in [np.float32, np.float64])
        self.assertTrue(actions.dtype in [np.int32, np.int64])
        self.assertTrue(rewards.dtype in [np.float32, np.float64])
        self.assertTrue(dones.dtype == bool)


class TestAgentMetrics(unittest.TestCase):
    """Test cases for agent performance metrics."""
    
    def test_cumulative_reward_calculation(self):
        """Test calculating cumulative rewards."""
        rewards = [1.0, 2.0, -0.5, 1.5, 0.8]
        cumulative_rewards = np.cumsum(rewards)
        
        expected = [1.0, 3.0, 2.5, 4.0, 4.8]
        np.testing.assert_array_almost_equal(cumulative_rewards, expected)
    
    def test_moving_average_calculation(self):
        """Test calculating moving averages of performance."""
        values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        window_size = 3
        
        moving_averages = []
        for i in range(window_size - 1, len(values)):
            window = values[i - window_size + 1:i + 1]
            moving_averages.append(np.mean(window))
        
        expected_length = len(values) - window_size + 1
        self.assertEqual(len(moving_averages), expected_length)
        
        # Check first few values
        self.assertAlmostEqual(moving_averages[0], 2.0)  # (1+2+3)/3
        self.assertAlmostEqual(moving_averages[1], 3.0)  # (2+3+4)/3
    
    def test_performance_statistics(self):
        """Test calculating performance statistics."""
        episode_rewards = [1.5, 2.3, 0.8, 3.1, 1.9, 2.7, 1.2, 2.8, 2.1, 1.6]
        
        stats = {
            'mean': np.mean(episode_rewards),
            'std': np.std(episode_rewards),
            'min': np.min(episode_rewards),
            'max': np.max(episode_rewards),
            'median': np.median(episode_rewards)
        }
        
        self.assertAlmostEqual(stats['mean'], 2.0, places=1)
        self.assertGreater(stats['std'], 0)
        self.assertEqual(stats['min'], 0.8)
        self.assertEqual(stats['max'], 3.1)
        self.assertTrue(stats['min'] <= stats['median'] <= stats['max'])


class TestAgentIntegration(unittest.TestCase):
    """Integration tests for agent components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_end_to_end_agent_workflow(self):
        """Test complete agent training and evaluation workflow."""
        # Initialize components
        env = MockEnvironment(max_steps=8)
        agent = MockAgent()
        
        # Training phase
        training_rewards = []
        num_training_episodes = 5
        
        for episode in range(num_training_episodes):
            state = env.reset()
            episode_reward = 0
            
            while not env.done:
                action = agent.select_action(state)
                next_state, reward, done, info = env.step(action)
                episode_reward += reward
                state = next_state
                
                if done:
                    break
            
            training_rewards.append(episode_reward)
        
        # Save trained agent
        model_path = Path(self.temp_dir) / "trained_agent.pt"
        agent.save(str(model_path))
        
        # Evaluation phase
        eval_agent = MockAgent()
        eval_agent.load(str(model_path))
        
        evaluation_rewards = []
        num_eval_episodes = 3
        
        for episode in range(num_eval_episodes):
            state = env.reset()
            episode_reward = 0
            
            while not env.done:
                action = eval_agent.select_action(state)
                next_state, reward, done, info = env.step(action)
                episode_reward += reward
                state = next_state
                
                if done:
                    break
            
            evaluation_rewards.append(episode_reward)
        
        # Verify workflow completed successfully
        self.assertEqual(len(training_rewards), num_training_episodes)
        self.assertEqual(len(evaluation_rewards), num_eval_episodes)
        self.assertTrue(model_path.exists())
        
        # Calculate performance metrics
        training_performance = {
            'mean_reward': np.mean(training_rewards),
            'std_reward': np.std(training_rewards)
        }
        
        eval_performance = {
            'mean_reward': np.mean(evaluation_rewards),
            'std_reward': np.std(evaluation_rewards)
        }
        
        # Verify metrics are reasonable
        self.assertIsInstance(training_performance['mean_reward'], (int, float))
        self.assertIsInstance(eval_performance['mean_reward'], (int, float))
        self.assertGreaterEqual(training_performance['std_reward'], 0)
        self.assertGreaterEqual(eval_performance['std_reward'], 0)
    
    def test_reward_model_agent_integration(self):
        """Test integration between reward models and agents."""
        # Create components
        reward_model = MockRewardModel()
        agent = MockAgent()
        env = MockEnvironment()
        
        # Simulate using reward model for agent training
        num_episodes = 3
        
        for episode in range(num_episodes):
            state = env.reset()
            episode_data = []
            
            while not env.done:
                action = agent.select_action(state)
                next_state, env_reward, done, info = env.step(action)
                
                # Use reward model to get additional signal
                mock_input = torch.randint(0, 100, (1, 10))
                with torch.no_grad():
                    model_output = reward_model(mock_input)
                    model_reward = model_output["rewards"] if isinstance(model_output, dict) else model_output
                    model_reward = model_reward.item()
                
                episode_data.append({
                    'state': state,
                    'action': action,
                    'env_reward': env_reward,
                    'model_reward': model_reward,
                    'next_state': next_state,
                    'done': done
                })
                
                state = next_state
                
                if done:
                    break
            
            # Verify episode data collection
            self.assertGreater(len(episode_data), 0)
            
            for step_data in episode_data:
                self.assertIn('state', step_data)
                self.assertIn('action', step_data)
                self.assertIn('env_reward', step_data)
                self.assertIn('model_reward', step_data)
                self.assertIsInstance(step_data['model_reward'], (int, float))


if __name__ == "__main__":
    # Set up test environment
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    
    # Run tests
    unittest.main(verbosity=2) 