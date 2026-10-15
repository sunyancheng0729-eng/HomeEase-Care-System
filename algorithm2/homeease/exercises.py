EXERCISES = {
    "chair_sit_to_stand": {"movement": "m05", "phases": ["sitting", "rising", "standing", "lowering"]},
    "squat": {"movement": "m01", "phases": ["standing", "lowering", "squatting", "rising"]},
    "standing_leg_raise": {"movement": "m06", "phases": ["standing", "lifting", "raised", "lowering"]},
    "hurdle_step": {"movement": "m02", "phases": ["standing", "stepping", "raised", "returning"]},
}
EXERCISE_NAMES = list(EXERCISES)
MOVEMENT_TO_INDEX = {v["movement"]: i for i, v in enumerate(EXERCISES.values())}
NUM_CLASSES = 17


def decode(probabilities, exercise):
    import numpy as np
    p = np.asarray(probabilities)
    action_probabilities = np.array([p[i * 4:i * 4 + 4].sum() for i in range(4)] + [p[16]])
    action = int(action_probabilities.argmax())
    target = EXERCISE_NAMES.index(exercise)
    phase = int(p[target * 4:target * 4 + 4].argmax())
    # Rest poses overlap across exercises. Condition phases on the user-selected
    # exercise; require supporting action evidence during the moving phases.
    if phase != 0 and action_probabilities[target] < .15:
        phase = 4
    canonical = np.zeros(5)
    if phase < 4:
        canonical[:4] = p[target * 4:target * 4 + 4]
        canonical[4] = 1 - canonical[:4].sum()
        canonical[:4] /= max(canonical[:4].sum(), 1e-8)
        canonical[4] = 0
    else:
        canonical[4] = 1
    return phase, canonical, action_probabilities
