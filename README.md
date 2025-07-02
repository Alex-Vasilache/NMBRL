# Neuromorphic Model-Based Reinforcement Learning (NMBRL) Implementation Plan

## 1. Overview & Goals

This document outlines a plan to implement a neuromorphic model-based reinforcement learning (MBRL) framework within this repository. The primary motivation is to leverage the energy efficiency of neuromorphic computing for the computationally intensive task of MBRL, enabling real-world deployment for fast (sample-efficient) and mobile on-device (energy-efficient) learning.

### Core Goals:
- **Sample and Energy-Efficient Learning:** The primary goal is to achieve both sample-efficient and energy-efficient learning. This will be accomplished by:
    - Utilizing Spiking Neural Networks (SNNs) with sparse activations and connectivity to reduce the computational cost of the world model and the agent during both **training and inference**.
    - Retaining the sample efficiency of MBRL by training the agent on a learned world model.
- **Fast Embedded Learning:** A key research question to investigate is: *How fast can the agent learn CartPole from scratch with computation suitable for embedded systems?*
- **Modularity:** Design the system with swappable components (world model, agent, environment) to encourage experimentation and extension to new environments beyond CartPole.
- **Parallel Development:** Structure the project to allow multiple developers to work on different modules concurrently.
- **On-Chip Learning Potential:** Employ learning rules like sparse Real-Time Recurrent Learning (RTRL) that are amenable to future on-chip hardware implementation.

## 2. System Architecture

The system will be architected into two primary modules that operate in a decoupled, parallel manner:
1.  **World Model (WM) Module:** Responsible for learning a predictive model of the environment's dynamics.
2.  **Actor-Critic (AC) Module:** Responsible for learning a policy (actor) and a value function (critic) using the world model.

These two modules interact through a well-defined `WorldModel` interface, allowing the AC module to be agnostic to whether it's interacting with the real environment or a learned SNN model.

```mermaid
graph TD
    subgraph "Real Environment"
        A["A: Physics Engine /<br/> CartPole"]
    end

    subgraph "Module 1: World Model"
        direction LR
        B("B: Data Collector") -- "Collects (s, a, s', r)" --> A
        C{"C: World Model Interface"}
        B -- "Provides data" --> D["D: WM Trainer"]
        D -- "Updates" --> E["E: SNN World Model"]
        
        subgraph "WM Implementations"
            direction TB
            F["F: Environment Wrapper"] -- "Implements" --> C
            E -- "Implements" --> C
        end
    end
    
    subgraph "Module 2: Actor-Critic"
        direction LR
        G["G: AC Trainer"] -- "Uses" --> C
        G -- "Updates" --> H{"H: Actor-Critic Agent"}
        H -- "Provides actions" --> C
        subgraph "Agent Networks"
            I["I: SNN Actor"]
            J["J: SNN Critic"]
        end
        H -- "Contains" --> I
        H -- "Contains" --> J
    end

    A -- "Provides initial state" --> G
```

### Architecture Component Mapping

The components in the architecture diagram above map to the following files and development tasks:

| Diagram ID | Component Name        | Implementation File(s)                 | Development Task(s) |
| :--------: | :-------------------- | :------------------------------------- | :------------------ |
|     A      | Physics Engine        | `environments/CartPoleSimulation/`     | Pre-existing        |
|     B      | Data Collector        | `learning/world_model_trainer.py`      | T3.2                |
|     C      | World Model Interface | `world_models/base_world_model.py`     | T1.2                |
|     D      | WM Trainer            | `learning/world_model_trainer.py`      | T3.2                |
|     E      | SNN World Model       | `world_models/snn_world_model.py`      | T3.1                |
|     F      | Environment Wrapper   | `world_models/ini_cartpole_wrapper.py` | T1.3                |
|     G      | AC Trainer            | `learning/actor_critic_trainer.py`     | T2.2, T2.4          |
|     H      | Actor-Critic Agent    | `agents/snn_actor_critic_agent.py`     | T2.1                |
|    I, J    | SNN Actor/Critic      | `agents/snn_actor_critic_agent.py`     | T2.1                |

## 3. Project Structure

To maintain modularity the project will have following structure:

```
Neuromorphic_MBRL/
├── agents/
│   ├── __init__.py
│   ├── snn_actor_critic_agent.py # Defines the SNN-based actor-critic agent
│   └── base_agent.py           # Abstract base class for agents
├── environments/
│   └── CartPoleSimulation/     # The CartPole simulation environment (submodule)
├── learning/
│   ├── __init__.py
│   ├── actor_critic_trainer.py # Training loop for the AC module
│   └── world_model_trainer.py  # Training loop for the WM module
├── world_models/
│   ├── __init__.py
│   ├── ini_cartpole_wrapper.py # Wraps the existing environment to fit the WM interface
│   ├── snn_world_model.py      # The SNN-based world model implementation
│   └── base_world_model.py     # Abstract base class for world models
├── utils/
│   └── __init__.py
├── README.md                   # This readme file
└── run_mbrl.py                 # Main script to configure and run an experiment
```

## 4. Development Roadmap & Task Breakdown

The project can be broken down into the following phases and tasks. This structure allows for parallel work on the World Model and Actor-Critic components after the initial foundation is laid.

| Phase                      | Task ID | Status | Task Description                                                                                                                                                                                        | Key Components/Files                     | Dependencies                    |
| :------------------------- | :------ | :----: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | :--------------------------------------- | :------------------------------ |
| **1. Foundation**          | T1.1    |   ✅    | **Create Project Structure:** Set up the directory and `__init__.py` files as outlined above.                                                                                                           | `Neuromorphic_MBRL/`                     | -                               |
|                            | T1.2    |   ✅    | **Define Base Interfaces:** Create the abstract base classes `BaseWorldModel` and `BaseAgent`.                                                                                                          | `base_world_model.py`, `base_agent.py`   | T1.1                            |
|                            | T1.3    |   ✅    | **Implement Environment Wrapper:** Create a concrete `WorldModel` by wrapping the existing `CartPole` simulation. This allows agent development to begin immediately.                                   | `ini_cartpole_wrapper.py`                | T1.2                            |
| **2. Actor-Critic Module** | T2.1    |   -    | **Initial SNN Actor-Critic Agent:** Implement an `SnnActorCriticAgent` class with a basic SNN structure.                                                                                                | `snn_actor_critic_agent.py`              | T1.2                            |
|                            | T2.2    |   -    | **Actor-Critic Trainer:** Create the training loop that has the agent interact with the `WorldModel` interface (using the `EnvironmentWrapper` for now).                                                | `actor_critic_trainer.py`, `run_mbrl.py` | T1.3, T2.1                      |
|                            | T2.3    |   -    | **Implement Sparse RTRL:** Adapt the `ActorCriticTrainer` to use a sparse RTRL algorithm for updating the SNN agent.                                                                                    | `actor_critic_trainer.py`                | T2.2                            |
| **3. World Model Module**  | T3.1    |   -    | **SNN World Model:** Implement the `SNNWorldModel` class. Initially, this can be a simple recurrent SNN architecture.                                                                                   | `snn_world_model.py`                     | T1.2                            |
|                            | T3.2    |   -    | **World Model Trainer:** Implement the training loop for the `SNNWorldModel`. It should sample data from the real environment and train the SNN to predict `(s', r) = f(s, a)`.                         | `world_model_trainer.py`                 | `ini_cartpole_wrapper.py`, T3.1 |
| **4. Integration**         | T4.1    |   -    | **Full Pipeline Integration:** Update `run_mbrl.py` to run both training modules. The `ActorCriticTrainer` should be configured to use the trained `SNNWorldModel` instead of the `EnvironmentWrapper`. | `run_mbrl.py`                            | T2.3, T3.2                      |
|                            | T4.2    |   -    | **Parallel Execution:** Refactor the training loops to run concurrently, with the agent training on the latest version of the world model.                                                              | `run_mbrl.py`, `*_trainer.py`            | T4.1                            |

## 5. Task Dependencies

The following graph visualizes the dependencies between the tasks outlined in the roadmap. This highlights the opportunities for parallel development after the initial foundation is complete.

```mermaid
graph TD;
    subgraph "Phase 1: Foundation"
        T1_1["T1.1<br/>Create Project Structure"];
        T1_2["T1.2<br/>Define Base Interfaces"];
        T1_3["T1.3<br/>Implement Environment Wrapper"];
        T1_1 --> T1_2 --> T1_3;
    end

    subgraph "Phase 2: Actor-Critic Module (Parallel)"
        T2_1["T2.1<br/>Initial SNN Actor-Critic Agent"];
        T2_2["T2.2<br/>Actor-Critic Trainer"];
        T2_3["T2.3<br/>Implement Sparse RTRL"];
        T2_1 --> T2_2;
        T2_2 --> T2_3;
    end
    
    subgraph "Phase 3: World Model Module (Parallel)"
        T3_1["T3.1<br/>SNN World Model"];
        T3_2["T3.2<br/>World Model Trainer"];
        T3_1 --> T3_2;
    end
    
    subgraph "Phase 4: Integration"
        T4_1["T4.1<br/>Full Pipeline Integration"];
        T4_2["T4.2<br/>Parallel Execution"];
        T4_1 --> T4_2;
    end

    T1_3 --> T2_2;
    T1_2 --> T2_1;
    
    T1_2 --> T3_1;
    T1_3 --> T3_2;
    
    T2_3 --> T4_1;
    T3_2 --> T4_1;
```

## 6. Tutorial: Model-Based Reinforcement Learning with an Actor-Critic Agent

This section provides a conceptual overview of the Model-Based Reinforcement Learning (MBRL) approach with an Actor-Critic agent, as implemented in this project.

### 6.1 What is Model-Based Reinforcement Learning?

In contrast to model-free RL (where an agent learns a policy through trial-and-error interacting directly with the environment), MBRL involves two distinct stages:

1.  **Learn a Model:** First, the agent learns a *model* of the environment. This "world model" is a function that predicts the consequences of actions. Given a current state `s` and an action `a`, the model predicts the next state `s'` and the immediate reward `r`.
2.  **Train an Agent:** Once the world model is learned, the agent is trained on this model instead of the real environment. The agent can "imagine" or "dream" of interacting with the environment by using the world model to simulate trajectories.

The primary advantage of MBRL is **sample efficiency**. Since the agent can generate a vast amount of simulated experience from the learned model, it requires significantly fewer interactions with the real environment, which can be expensive, slow, or dangerous.

### 6.2 The Actor-Critic Method

Actor-Critic is a popular model-free RL algorithm that we use to train our agent *within* the learned world model. It combines the strengths of two other types of algorithms:

-   **The Actor:** This is the policy. It takes the current state `s` as input and decides which action `a` to take (`a = π(s)`). The actor's goal is to learn the optimal policy.
-   **The Critic:** This is a value function. It evaluates the "goodness" of a state or a state-action pair by predicting the expected future reward (the "value"). For instance, `V(s)` estimates the total reward an agent can expect to receive starting from state `s`.

The Actor and Critic work together: The Actor selects an action, and the Critic evaluates how good that action was. The Actor then updates its policy based on the Critic's feedback, making it more likely to choose actions that the Critic deems good. Simultaneously, the Critic improves its own estimations by observing the rewards received.

### 6.3 Integrating MBRL with Actor-Critic: The "Dreamer" Loop

This project combines MBRL and Actor-Critic in a powerful loop, inspired by algorithms like Dreamer. Here's how it works:

1.  **Data Collection (Interaction with Reality):** Initially, the agent interacts with the real environment (the `INICartPoleWrapper`) using a preliminary (e.g., random) policy to collect a set of experiences `(s, a, s', r)`.

2.  **World Model Training (Learning the Dynamics):** The `WorldModelTrainer` uses this collected data to train the `SNNWorldModel`. This is a supervised learning problem where the model learns to predict `(s', r)` given `(s, a)`.

3.  **Behavior Learning (Training in "Imagination"):** This is the core of the sample efficiency gain. The `ActorCriticTrainer` trains the `SnnActorCriticAgent` *entirely within the learned `SNNWorldModel`*.
    - The trainer simulates long trajectories of experience using the world model without ever touching the real environment.
    - For each step in the imagination, the **Actor** proposes an action. The **World Model** predicts the next state and reward. The **Critic** evaluates the imagined state.
    - This large volume of imagined data is used to update the Actor and Critic networks. Because this all happens in simulation, it is fast and generates no wear-and-tear on real hardware.

4.  **Repeat:** After a period of "dreaming," the improved Actor is used to interact with the real environment again (Step 1), collecting higher-quality data. This new data is used to further refine the world model (Step 2), which in turn allows for even better agent training (Step 3).

This cycle allows the agent to build a robust understanding of the world and a high-performance policy with minimal real-world interaction. The use of SNNs for both the world model and the agent further aims to make the computationally-heavy "dreaming" phase energy-efficient.

### 6.4 Mathematical Formulation

To understand the learning process more deeply, let's look at the objectives for each component. Let $\theta$ be the parameters of the world model, $\phi$ be the parameters of the actor, and $\psi$ be the parameters of the critic.

#### World Model Loss

The world model, $p_\theta(s_{t+1}, r_{t+1} | s_t, a_t)$, is trained to predict the next state and reward given the current state and action. It is trained by minimizing a reconstruction loss on real data collected from the environment. A common choice for this loss is the Mean Squared Error (MSE):

$$
L_{WM}(\theta) = \mathbb{E}_{(s_t, a_t, r_{t+1}, s_{t+1}) \sim \mathcal{D}} \left[ \|s_{t+1} - \hat{s}_{t+1}\|^2 + \|r_{t+1} - \hat{r}_{t+1}\|^2 \right]
$$

where $(\hat{s}_{t+1}, \hat{r}_{t+1})$ are the predictions from the world model $p_\theta$, and $\mathcal{D}$ is the dataset of real-world experiences.

#### Critic (Value) Loss

The critic, $V_\psi(s_t)$, learns to estimate the expected future rewards from a given state. It is trained on trajectories imagined by the world model. For each state $s_\tau$ in an imagined trajectory, the critic's goal is to predict the sum of future rewards over a certain horizon $H$, known as the value target $v_\tau$.

The value target is the discounted sum of rewards in the imagined trajectory:
$$
v_{\tau} = \sum_{k=\tau}^{\tau+H} \gamma^{k-\tau} r_k
$$
where $r_k$ are the rewards predicted by the world model and $\gamma$ is a discount factor.

The critic is then updated by minimizing the MSE between its predictions $V_\psi(s_\tau)$ and these calculated value targets:

$$
L_{Critic}(\psi) = \mathbb{E}_{s_\tau \sim p_\theta} \left[ \| V_{\psi}(s_\tau) - \mathrm{stop\_grad}(v_\tau) \|^2 \right]
$$

The expectation $\mathbb{E}_{s_\tau \sim p_\theta}$ means we average this loss over all states in many imagined trajectories. The `stop_grad` function prevents the value targets (which depend on the world model) from propagating gradients back into the world model during the agent's training phase.

#### Actor (Policy) Loss

The actor (policy), $\pi_\phi(a_t | s_t)$, is updated to choose actions that lead to states with higher values as estimated by the critic. The actor's objective is to maximize the expected value of states it visits within the imagination.

This is achieved by performing gradient ascent on the value estimates. Equivalently, we can define the loss as the negative of the critic's value predictions, encouraging the actor to move towards states the critic deems more valuable:
$$
L_{Actor}(\phi) = - \mathbb{E}_{s_\tau \sim p_\theta} \left[ V_{\psi}(s_\tau) \right]
$$
This loss encourages the policy $\pi_\phi$ to select action sequences that produce trajectories with high cumulative reward, as judged by the learned world model and value function. The actor is not trained on the value targets $v_\tau$ directly, but rather on the critic's learned, smooth approximation of them, $V_\psi$.
