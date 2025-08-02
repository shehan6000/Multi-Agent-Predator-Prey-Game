# Multi-Agent Predator-Prey MARL System

A comprehensive implementation of Multi-Agent Deep Deterministic Policy Gradient (MADDPG) for a predator-prey environment, featuring optional inter-agent communication and comparative analysis.

## 🎯 Overview

This project implements a multi-agent reinforcement learning system where predator agents learn to cooperatively hunt prey in a grid-based environment. The system compares the effectiveness of MADDPG with and without communication between agents.

### Key Features

- **MADDPG Implementation**: Centralized training with decentralized execution
- **Communication Module**: Optional message passing between agents
- **Custom Environment**: Grid-based predator-prey simulation
- **Comparative Analysis**: Performance comparison with/without communication
- **Visualization**: Real-time environment rendering and training metrics

## 🏗️ Architecture

### Environment (`PredatorPreyEnv`)
- **Grid Size**: Configurable NxN grid world
- **Agents**: Multiple predators and prey
- **Objectives**: Predators cooperate to catch all prey
- **Rewards**: +10 for catching prey, -0.1 per timestep
- **Observations**: Agent positions relative to all other entities

### MADDPG Agent Components

#### Actor Network
- **Input**: Agent state + optional communication vector
- **Output**: Action probabilities + communication message
- **Architecture**: 2-layer MLP with ReLU activations
- **Communication**: 4-dimensional message vectors

#### Critic Network
- **Input**: All agent states + all agent actions (centralized)
- **Output**: Q-value estimation
- **Architecture**: 2-layer MLP with ReLU activations
- **Purpose**: Centralized value function for stable multi-agent learning

## 📋 Requirements

```python
numpy>=1.21.0
torch>=1.9.0
gymnasium>=0.26.0
matplotlib>=3.5.0
```

## 🚀 Quick Start

### Basic Usage

```python
from maddpg_predator_prey import main

# Run complete training and comparison
main()
```

### Custom Configuration

```python
from maddpg_predator_prey import PredatorPreyEnv, MADDPGTrainer

# Create custom environment
env = PredatorPreyEnv(
    grid_size=10,          # 10x10 grid
    num_predators=3,       # 3 predator agents
    num_prey=4,            # 4 prey entities
    max_steps=100          # Maximum episode length
)

# Train without communication
trainer = MADDPGTrainer(env, communication=False)
trainer.train(num_episodes=1000)

# Train with communication
trainer_comm = MADDPGTrainer(env, communication=True)
trainer_comm.train(num_episodes=1000)
```

## 🔧 Configuration Options

### Environment Parameters
- `grid_size`: Size of the square grid (default: 8)
- `num_predators`: Number of predator agents (default: 2)
- `num_prey`: Number of prey entities (default: 2)
- `max_steps`: Maximum steps per episode (default: 50)

### Training Parameters
- `num_episodes`: Total training episodes (default: 500)
- `learning_rate`: Adam optimizer learning rate (default: 0.001)
- `gamma`: Discount factor (default: 0.95)
- `tau`: Soft update coefficient (default: 0.01)
- `buffer_size`: Experience replay buffer size (default: 10000)

### Network Architecture
- `hidden_dim`: Hidden layer dimensions (default: 64)
- `communication_dim`: Communication vector size (default: 4)
- `batch_size`: Training batch size (default: 32)

## 📊 Performance Metrics

The system tracks and visualizes several key metrics:

### Training Metrics
- **Episode Rewards**: Average reward per episode
- **Prey Caught**: Number of prey successfully caught
- **Episode Length**: Steps taken to complete episode
- **Success Rate**: Percentage of episodes with all prey caught

### Communication Analysis
- **Coordination Improvement**: Quantified benefit of inter-agent communication
- **Convergence Speed**: Episodes required to reach stable performance
- **Emergent Behaviors**: Observed cooperative strategies

## 🎮 Environment Details

### State Space
Each predator observes:
- Own position (x, y)
- All prey positions (x, y) or (-1, -1) if caught
- Other predator positions (x, y)

### Action Space
5 discrete actions per agent:
- 0: Stay in place
- 1: Move up
- 2: Move down
- 3: Move left
- 4: Move right

### Reward Structure
- **Catch Reward**: +10 points for catching prey
- **Time Penalty**: -0.1 points per timestep
- **Cooperative Bonus**: Shared success encourages cooperation

## 🧠 MADDPG Algorithm

### Key Concepts

**Centralized Training, Decentralized Execution**
- Training: Critics access global state information
- Execution: Actors use only local observations

**Experience Replay**
- Shared memory buffer across all agents
- Batch sampling for stable learning
- Temporal credit assignment

**Target Networks**
- Soft updates for stable Q-learning
- Separate target networks for actors and critics

## 📈 Results Analysis

### Expected Outcomes

1. **Communication Advantage**: Agents with communication typically achieve:
   - Higher success rates (60-80% vs 40-60%)
   - Faster convergence (200-300 episodes vs 400-500)
   - More efficient hunting strategies

2. **Emergent Behaviors**:
   - **Coordination**: Agents learn to surround prey
   - **Role Specialization**: Different hunting strategies
   - **Information Sharing**: Communication about prey locations

3. **Learning Curves**:
   - Initial exploration phase (100-200 episodes)
   - Rapid improvement phase (200-400 episodes)
   - Convergence to optimal policy (400+ episodes)
