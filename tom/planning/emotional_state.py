"""
Emotional state tracking using the Circumplex Model (Pattisapu et al. 2024).

In this active inference formulation, emotions map to a 2D arousal-valence space:

- Arousal = H[Q(s|o)] = entropy of posterior beliefs
  High entropy = high uncertainty = high arousal (alert, anxious)
  Low entropy = high certainty = low arousal (calm, relaxed)

- Valence = Utility - Expected Utility = log P(o|C) - E[log P(o|C)]
  Positive = "better than expected" outcome (happy, excited)
  Negative = "worse than expected" outcome (sad, angry)

The Circumplex Model organizes emotions in a circle:
- 0deg = Happy (high valence, neutral arousal)
- 45deg = Excited (high valence, high arousal)
- 90deg = Alert (neutral valence, high arousal)
- 135deg = Angry (low valence, high arousal)
- 180deg = Sad (low valence, neutral arousal)
- 225deg = Depressed (low valence, low arousal)
- 270deg = Calm (neutral valence, low arousal)
- 315deg = Relaxed (high valence, low arousal)

Reference: Pattisapu, Verbelen, Pitliya, Kiefer & Albarracin (2024)
"Free Energy in a Circumplex Model of Emotion"
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


def compute_belief_entropy(qs: np.ndarray, eps: float = 1e-16) -> float:
    """
    Compute entropy of posterior beliefs H[Q(s|o)].

    This is the arousal signal in the Circumplex Model:
    - High entropy = high uncertainty = high arousal
    - Low entropy = high certainty = low arousal

    Parameters
    ----------
    qs : np.ndarray
        Posterior belief distribution over states
    eps : float
        Numerical floor to avoid log(0)

    Returns
    -------
    entropy : float
        Entropy of the belief distribution (arousal)
    """
    qs_safe = np.clip(qs, eps, 1.0)
    qs_safe = qs_safe / qs_safe.sum()
    entropy = -np.sum(qs_safe * np.log(qs_safe))
    return float(entropy)


def compute_utility(obs: int, C: np.ndarray, eps: float = 1e-16) -> float:
    """
    Compute utility of an observation given preferences.

    U = log P(o|C)

    Parameters
    ----------
    obs : int
        Observed state index
    C : np.ndarray
        Preference distribution (log preferences or raw preferences)
    eps : float
        Numerical floor

    Returns
    -------
    utility : float
        Utility of the observation
    """
    # C is typically log preferences, so just index
    # If C contains raw preferences, we take log
    c_val = C[obs]
    if np.all(C <= 0):
        # Already log preferences
        return float(c_val)
    else:
        # Raw preferences, take log
        return float(np.log(max(c_val, eps)))


def compute_expected_utility(
    qs: np.ndarray,
    A: np.ndarray,
    C: np.ndarray,
    eps: float = 1e-16
) -> float:
    """
    Compute expected utility before seeing observation.

    EU = E_Q(o|s) [log P(o|C)] = sum_o P(o|qs) * log P(o|C)

    Parameters
    ----------
    qs : np.ndarray
        Prior belief over states (before observation)
    A : np.ndarray
        Observation model P(o|s), shape (num_obs, num_states)
    C : np.ndarray
        Preference distribution (log preferences)
    eps : float
        Numerical floor

    Returns
    -------
    expected_utility : float
        Expected utility
    """
    # Predicted observation distribution: P(o) = A @ qs
    obs_dist = A @ qs
    obs_dist = np.clip(obs_dist, eps, 1.0)
    obs_dist = obs_dist / obs_dist.sum()

    # If C is log preferences, use directly; otherwise take log
    if np.all(C <= 0):
        log_C = C
    else:
        log_C = np.log(np.clip(C, eps, 1.0))

    # Expected utility
    eu = np.sum(obs_dist * log_C)
    return float(eu)


def compute_valence(
    obs: int,
    qs_prior: np.ndarray,
    A: np.ndarray,
    C: np.ndarray,
    eps: float = 1e-16
) -> float:
    """
    Compute valence as reward prediction error.

    Valence = Utility - Expected Utility
            = log P(o|C) - E[log P(o|C)]

    Positive valence = "better than expected"
    Negative valence = "worse than expected"

    Parameters
    ----------
    obs : int
        Observed state index
    qs_prior : np.ndarray
        Prior belief before observation
    A : np.ndarray
        Observation model
    C : np.ndarray
        Preference distribution
    eps : float
        Numerical floor

    Returns
    -------
    valence : float
        Valence (reward prediction error)
    """
    u = compute_utility(obs, C, eps)
    eu = compute_expected_utility(qs_prior, A, C, eps)
    return u - eu


@dataclass
class EmotionalState:
    """
    Single emotional state measurement in the Circumplex Model.

    Attributes
    ----------
    arousal : float
        Arousal level = entropy of posterior beliefs H[Q(s|o)]
        Higher = more uncertain/alert, Lower = more certain/calm

    valence : float
        Valence = Utility - Expected Utility (reward prediction error)
        Positive = better than expected, Negative = worse than expected

    intensity : float
        Emotional intensity = sqrt(arousal^2 + valence^2)
        Distance from neutral in the circumplex

    angle : float
        Angle in the circumplex (degrees)
        Maps to specific emotion labels

    timestep : int
        Timestep when this state was recorded
    """
    arousal: float
    valence: float
    intensity: float = 0.0
    angle: float = 0.0
    timestep: int = 0

    def __post_init__(self):
        # Compute polar coordinates
        self.intensity = float(np.sqrt(self.arousal**2 + self.valence**2))
        # atan2 gives angle in radians, convert to degrees
        # Note: atan2(y, x) where y=arousal, x=valence
        self.angle = float(np.degrees(np.arctan2(self.arousal, self.valence)))
        # Normalize to 0-360
        if self.angle < 0:
            self.angle += 360

    def emotion_label(self) -> str:
        """
        Get discrete emotion label based on angle in circumplex.

        Returns emotion based on 8 sectors of the circumplex.
        """
        # Normalize angle to 0-360
        angle = self.angle % 360

        # 8 emotion sectors (45 degrees each)
        if 337.5 <= angle or angle < 22.5:
            return "happy"
        elif 22.5 <= angle < 67.5:
            return "excited"
        elif 67.5 <= angle < 112.5:
            return "alert"
        elif 112.5 <= angle < 157.5:
            return "angry"
        elif 157.5 <= angle < 202.5:
            return "sad"
        elif 202.5 <= angle < 247.5:
            return "depressed"
        elif 247.5 <= angle < 292.5:
            return "calm"
        else:  # 292.5 <= angle < 337.5
            return "relaxed"

    def quadrant(self) -> str:
        """Get quadrant description."""
        if self.valence > 0 and self.arousal > 0:
            return "high-arousal-positive"
        elif self.valence <= 0 and self.arousal > 0:
            return "high-arousal-negative"
        elif self.valence > 0 and self.arousal <= 0:
            return "low-arousal-positive"
        else:
            return "low-arousal-negative"


@dataclass
class EmotionalStateTracker:
    """
    Track emotional states over time using the Circumplex Model.

    Computes arousal from belief entropy and valence from reward prediction error,
    following Pattisapu et al. (2024).

    Attributes
    ----------
    history : List[EmotionalState]
        History of emotional states

    arousal_scale : float
        Scale factor for normalizing arousal (max expected entropy)

    valence_scale : float
        Scale factor for normalizing valence
    """
    history: List[EmotionalState] = field(default_factory=list)
    arousal_scale: float = 3.0  # ~log(20) for 20 states
    valence_scale: float = 5.0  # Typical utility range

    def record_from_beliefs(
        self,
        qs_posterior: np.ndarray,
        obs: int,
        qs_prior: np.ndarray,
        A: np.ndarray,
        C: np.ndarray,
        timestep: int = -1,
        normalize: bool = True,
    ) -> EmotionalState:
        """
        Record emotional state from belief update.

        Parameters
        ----------
        qs_posterior : np.ndarray
            Posterior belief after observation
        obs : int
            Observed state index
        qs_prior : np.ndarray
            Prior belief before observation
        A : np.ndarray
            Observation model
        C : np.ndarray
            Preference distribution
        timestep : int
            Current timestep (-1 to auto-increment)
        normalize : bool
            Whether to normalize arousal/valence to [-1, 1]

        Returns
        -------
        state : EmotionalState
            Recorded emotional state
        """
        if timestep == -1:
            timestep = len(self.history)

        # Arousal = entropy of posterior beliefs
        arousal = compute_belief_entropy(qs_posterior)

        # Valence = utility - expected utility
        valence = compute_valence(obs, qs_prior, A, C)

        if normalize:
            # Normalize to roughly [-1, 1]
            arousal = arousal / self.arousal_scale
            valence = np.tanh(valence / self.valence_scale)

        state = EmotionalState(
            arousal=float(arousal),
            valence=float(valence),
            timestep=timestep,
        )
        self.history.append(state)
        return state

    def record_raw(
        self,
        arousal: float,
        valence: float,
        timestep: int = -1,
    ) -> EmotionalState:
        """
        Record emotional state from raw arousal/valence values.

        Parameters
        ----------
        arousal : float
            Raw arousal value
        valence : float
            Raw valence value
        timestep : int
            Current timestep

        Returns
        -------
        state : EmotionalState
            Recorded emotional state
        """
        if timestep == -1:
            timestep = len(self.history)

        state = EmotionalState(
            arousal=float(arousal),
            valence=float(valence),
            timestep=timestep,
        )
        self.history.append(state)
        return state

    def get_average(self) -> Optional[EmotionalState]:
        """Get average emotional state over history."""
        if not self.history:
            return None

        avg_arousal = np.mean([s.arousal for s in self.history])
        avg_valence = np.mean([s.valence for s in self.history])

        return EmotionalState(
            arousal=float(avg_arousal),
            valence=float(avg_valence),
            timestep=-1,
        )

    def get_trajectory(self) -> Tuple[List[float], List[float]]:
        """Get arousal and valence trajectories."""
        arousals = [s.arousal for s in self.history]
        valences = [s.valence for s in self.history]
        return arousals, valences

    def get_emotions(self) -> List[str]:
        """Get list of emotion labels over time."""
        return [s.emotion_label() for s in self.history]

    def reset(self):
        """Clear history."""
        self.history = []

    def summary(self) -> str:
        """Get human-readable summary."""
        if not self.history:
            return "No emotional states recorded."

        avg = self.get_average()
        start = self.history[0]
        end = self.history[-1]

        lines = [
            f"Emotional trajectory ({len(self.history)} states):",
            f"  Start: {start.emotion_label()} (A={start.arousal:.2f}, V={start.valence:.2f})",
            f"  End:   {end.emotion_label()} (A={end.arousal:.2f}, V={end.valence:.2f})",
            f"  Avg:   {avg.emotion_label()} (A={avg.arousal:.2f}, V={avg.valence:.2f})",
        ]

        # Count emotions
        emotions = self.get_emotions()
        unique_emotions = set(emotions)
        emotion_counts = {e: emotions.count(e) for e in unique_emotions}
        lines.append(f"  Emotions: {emotion_counts}")

        return "\n".join(lines)


def compute_empathic_emotional_state(
    own_arousal: float,
    own_valence: float,
    other_arousal: float,
    other_valence: float,
    empathy_weight: float,
) -> Tuple[float, float]:
    """
    Compute empathy-weighted emotional state.

    An empathic agent's emotional state is influenced by the other's state.

    Parameters
    ----------
    own_arousal, own_valence : float
        Own arousal and valence
    other_arousal, other_valence : float
        Other agent's arousal and valence
    empathy_weight : float
        Weight on other's state (alpha in [0, 1])

    Returns
    -------
    social_arousal, social_valence : Tuple[float, float]
        Empathy-weighted arousal and valence
    """
    alpha = empathy_weight
    social_arousal = (1 - alpha) * own_arousal + alpha * other_arousal
    social_valence = (1 - alpha) * own_valence + alpha * other_valence
    return float(social_arousal), float(social_valence)
