# Intelligent Automotive Agent for Test Case Generation and Cooperative Autonomous Driving

## Overview

This project develops an AI-driven automotive agent combining Large Language Models (LLMs), Retrieval-Augmented Generation (RAG), Reinforcement Learning (RL), and C-V2X communication for automotive testing and autonomous driving.

The first component is a RAG-based automotive test case generation system. A knowledge base containing driving rules and guidelines for vehicles operating in school zones is used. Given a user-defined automotive requirement, the system retrieves relevant rules using semantic retrieval and provides the context to an LLM, which generates corresponding test cases. These test cases are then simulated in the CARLA simulator to evaluate vehicle behavior under the specified conditions.

The second component focuses on developing a reinforcement learning-based autonomous driving agent for cooperative driving. Two autonomous vehicles are simulated in CARLA and exchange vehicle-state information through a simulated C-V2X communication layer. MAPPO (Multi-Agent Proximal Policy Optimization) with Centralized Training and Decentralized Execution (CTDE) is used to learn cooperative driving decisions. LSTM-based actor and critic networks are employed to capture temporal information, while the reward function considers factors such as safety, collision avoidance, progress, waiting behavior, and driving comfort.

## Technologies

Python, PyTorch, CARLA 0.9.15, RAG, LLMs, Semantic Embeddings, FAISS, Reinforcement Learning, MAPPO, LSTM, C-V2X

## Current Status

- RAG-based automotive requirement retrieval
- LLM-based automotive test case generation
- CARLA-based test case simulation
- C-V2X communication simulation
- Two-vehicle cooperative driving environment
- MAPPO-based multi-agent reinforcement learning
- LSTM-based actor and critic networks
- RL-based cooperative driving policy currently under development
