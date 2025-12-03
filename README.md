# Relational Empowerment, ToM, Empathy, and Emergent Alignment

## Active Inference Multi-Agent Experimental Framework

### 1. Overview

This repository implements a modular framework for studying emergent alignment in multi-agent active inference systems.

The core idea is that **alignment does not need to be encoded directly in agents' preference vectors**.
Instead, alignment emerges (or fails to emerge) depending on how agents:

- Predict each other's behaviour using **Theory of Mind (ToM)**.
- Weigh each other's expected free energy using **Empathy (�)**.
- Preserve or collapse each other's future option sets via **Empowerment**, our operational measure of path flexibility.

The goal is to investigate how these three ingredients interact inside a minimal, transparent, controlled world where we can literally see how belief dynamics give rise to coordination or brittleness.

 

### 2. Key Concepts

#### 2.1 Path Flexibility (Operationalised as Empowerment)

**Empowerment** is the subjective mutual information between an agent's actions and its future observations.
It measures how many distinct, controllable futures an agent can induce from a given state.

We use empowerment as a direct, model-based proxy for **path flexibility**:

- **High empowerment** � wide corridor of possible futures; robust, multi-path behaviour.
- **Low empowerment** � bottlenecks, traps, deadlocks; small perturbations can cause catastrophes.

In multi-agent settings, empowerment becomes explicitly **relational**:

- If Agent A's actions constrain Agent B's movements, B's empowerment drops.
- If A preserves B's space of possible futures, B's empowerment stays high.

This **relational collapse of empowerment** is the operational definition of interaction-induced brittleness.

#### 2.2 Empathy (�)

Each agent weighs the other's expected free energy (EFE) and empowerment using an empathy parameter **�  [0,1]**:

- **� = 0**: purely selfish planning
- **� > 0**: partially or strongly weighting the other agent's risks and controllability

This provides a clean way to study:

- Selfish ToM
- Empathic ToM
- Empowerment-aware ToM
- Fully relational alignment (both empathy and empowerment sensitivity)

#### 2.3 Theory of Mind (ToM)

Agents maintain nested generative models of each other's beliefs, actions, and expected free energies.

Using the ToM tree-search (`si_tom`, `Tree`, `rollout_tom`):

- Each agent simulates its own future
- And simulates the other's future
- Including their predicted actions, belief updates, and empowerment

This allows the agent to evaluate:

- its own trajectory EFE
- the other agent's trajectory EFE
- its own empowerment
- the other agent's empowerment

under each candidate policy.

 

### 3. Experiment 1

#### Goal

Demonstrate that even when agents have:

- identical, purely selfish preferences
- bounded ToM
- imperfect predictions
- stochastic transitions

the inclusion of **relational empowerment awareness** and/or **empathy** changes behavioural equilibria:

- avoiding brittle bottlenecks
- staggering through high-risk areas
- taking flexible detours
- achieving mutually non-destructive outcomes

#### Environment

A tiny, transparent, multi-agent gridworld:

- narrow, risky bottleneck corridor
- wide, safe detour
- stochastic slips/drifts
- collision = catastrophe
- no shared rewards

#### Metrics

Three classes:

**System-Level Performance**

- joint success rate
- collisions/deadlocks
- robustness to noise

**Relational Empowerment / Path Flexibility**

- empowerment over time
- "option-set collapse" events
- solo vs multi-agent empowerment gaps

**Emergent Alignment**

- mutually non-destructive outcomes
- sensitivity to preference mismatches
- stability of coordination

 

### 4. Roadmap

1. **Define Environment**
   - Gridworld with bottleneck & detour
   - Transition matrix with stochastic slips
   - Catastrophic collisions
   - Individual goals and rewards

2. **Implement Agents**
   - Active Inference Agent (pymdp.Agent)
   - ToM wrappers (ToMAgent)
   - Empathy integration
   - Empowerment computation

3. **Policy Evaluation**
   - Generate candidate policies
   - For each:
     - compute self/other EFE
     - compute self/other empowerment
     - combine under empathy-weighted objective

4. **Simulation Loop**
   - Roll each episode
   - Each agent: ToM � evaluate policies � sample action
   - Update environment
   - Log empowerment, EFE, actions, collisions

5. **Analysis**
   - Plot empowerment trajectories
   - Compare bottleneck usage
   - Compare success rates
   - Study dependence on � (empathy) and empowerment-weight �

6. **Generalization**
   - shocks (layout changes)
   - preference plasticity
   - precision asymmetries (power)
   - horizon mismatches (temporal power)


