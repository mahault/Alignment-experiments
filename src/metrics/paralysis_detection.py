"""
Paralysis detection utilities for multi-agent coordination.

Detects various forms of coordination failure:
- Cycle detection: Agents stuck in repeating state patterns
- Mutual stay: Both agents repeatedly choose STAY
- Oscillation: Alternating between a small set of states
- Timeout: Episode ends without success
"""

from typing import Dict, List, Tuple


def detect_paralysis(
    trajectory_i: List[Tuple[int, int]],
    trajectory_j: List[Tuple[int, int]],
    actions_i: List[int],
    actions_j: List[int],
    goal_reached_i: bool,
    goal_reached_j: bool,
    max_timesteps: int,
    cycle_threshold: int = 3,
    stay_threshold: int = 3
) -> Dict:
    """
    Detect various forms of paralysis in an episode.

    Parameters
    ----------
    trajectory_i, trajectory_j : List[Tuple[int, int]]
        Position trajectories for each agent
    actions_i, actions_j : List[int]
        Action sequences for each agent (4 = STAY)
    goal_reached_i, goal_reached_j : bool
        Whether each agent reached their goal
    max_timesteps : int
        Maximum episode length
    cycle_threshold : int
        Number of times same joint state must repeat to count as cycle
    stay_threshold : int
        Number of consecutive mutual stays to count as paralysis

    Returns
    -------
    result : dict
        Paralysis detection results with keys:
        - paralysis: bool
        - paralysis_type: str or None
        - cycle_length: int or None
        - stay_streak: int
    """
    result = {
        "paralysis": False,
        "paralysis_type": None,
        "cycle_length": None,
        "stay_streak": 0
    }

    # If both succeeded, no paralysis
    if goal_reached_i and goal_reached_j:
        return result

    # Timeout without success (but no collision/lava) is potential paralysis
    if len(trajectory_i) >= max_timesteps:
        # Check for cycle detection
        joint_states = [(trajectory_i[t], trajectory_j[t]) for t in range(len(trajectory_i))]
        state_counts = {}
        for state in joint_states:
            state_counts[state] = state_counts.get(state, 0) + 1

        max_repeats = max(state_counts.values()) if state_counts else 0
        if max_repeats >= cycle_threshold:
            result["paralysis"] = True
            result["paralysis_type"] = "cycle"
            # Find cycle length (approximate)
            for state, count in state_counts.items():
                if count == max_repeats:
                    # Find indices where this state occurs
                    indices = [t for t, s in enumerate(joint_states) if s == state]
                    if len(indices) >= 2:
                        result["cycle_length"] = indices[1] - indices[0]
                    break
            return result

    # Check for mutual stay paralysis
    if len(actions_i) > 0 and len(actions_j) > 0:
        stay_streak = 0
        max_stay_streak = 0
        for t in range(len(actions_i)):
            if actions_i[t] == 4 and actions_j[t] == 4:  # Both STAY
                stay_streak += 1
                max_stay_streak = max(max_stay_streak, stay_streak)
            else:
                stay_streak = 0

        result["stay_streak"] = max_stay_streak

        if max_stay_streak >= stay_threshold:
            result["paralysis"] = True
            result["paralysis_type"] = "mutual_stay"
            return result

    # Check for oscillation (alternating between 2-3 states)
    if len(trajectory_i) >= 6:
        # Look at last 6 positions
        recent_joint = [(trajectory_i[t], trajectory_j[t]) for t in range(-6, 0)]
        unique_states = set(recent_joint)
        if len(unique_states) <= 3:
            # Check if oscillating
            result["paralysis"] = True
            result["paralysis_type"] = "oscillation"
            return result

    # Timeout without clear pattern
    if len(trajectory_i) >= max_timesteps and not (goal_reached_i and goal_reached_j):
        result["paralysis"] = True
        result["paralysis_type"] = "timeout"

    return result
