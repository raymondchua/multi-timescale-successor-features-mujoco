DOMAINS = [
    "walker",
    "walkerbigleg",
    "quadruped",
    "jaco",
    "cheetah",
    "cheetahbigleg",
    "cheetahstiffbackleg",
    "cheetahstifffrontleg",
    "cheetahfast",
    "humanoid",
    "humanoidsmallleg",
    "humanoidbigleg",
    "cartpole",
    "cartpoleheavy",
    "fish",
    "dog",
]

CHEETAH_TASKS = [
    "cheetah_run",
    "cheetah_run_backward",
    "cheetah_flip",
    "cheetah_flip_backward",
]

CHEETAHBIGLEG_TASKS = [
    "cheetahbigleg_run",
    "cheetahbigleg_run_backward",
    "cheetahbigleg_flip",
    "cheetahbigleg_flip_backward",
]

CHEETAHSTIFFBACKLEG_TASKS = [
    "cheetahstiffbackleg_run",
    "cheetahstiffbackleg_run_backward",
    "cheetahstiffbackleg_flip",
    "cheetahstiffbackleg_flip_backward",
]

CHEETAHSTIFFFRONTLEG_TASKS = [
    "cheetahstifffrontleg_run",
    "cheetahstifffrontleg_run_backward",
    "cheetahstifffrontleg_flip",
    "cheetahstifffrontleg_flip_backward",
]

WALKER_TASKS = [
    "walker_stand",
    "walker_walk",
    "walker_run",
    "walker_run_backward",
    "walker_flip",
]

WALKERBIGLEG_TASKS = [
    "walkerbigleg_stand",
    "walkerbigleg_walk",
    "walkerbigleg_run",
    "walkerbigleg_flip",
]

QUADRUPED_TASKS = [
    "quadruped_walk",
    "quadruped_run",
    "quadruped_stand",
    "quadruped_jump",
    "quadruped_run_backward",
]

JACO_TASKS = [
    "jaco_reach_top_left",
    "jaco_reach_top_right",
    "jaco_reach_bottom_left",
    "jaco_reach_bottom_right",
]

HUMANOID_TASKS = [
    "humanoid_stand",
    "humanoid_walk",
    "humanoid_run",
    "humanoid_run_backward",
]

HUMANOIDSMALLLEG_TASKS = [
    "humanoidsmallleg_stand",
    "humanoidsmallleg_walk",
    "humanoidsmallleg_run",
]

HUMANOIDBIGLEG_TASKS = [
    "humanoidbigleg_stand",
    "humanoidbigleg_walk",
    "humanoidbigleg_run",
]

CARTPOLE_TASKS = [
    "cartpole_balance",
    "cartpole_balance_sparse",
    "cartpole_swingup",
    "cartpole_swingup_sparse",
]

CARTPOLE_HEAVY_TASKS = [
    "cartpoleheavy_balance",
    "cartpoleheavy_balance_sparse",
    "cartpoleheavy_swingup",
    "cartpoleheavy_swingup_sparse",
]

TASKS = (
    WALKER_TASKS
    + QUADRUPED_TASKS
    + JACO_TASKS
    + CHEETAH_TASKS
    + CHEETAHBIGLEG_TASKS
    + CHEETAHSTIFFBACKLEG_TASKS
    + CHEETAHSTIFFFRONTLEG_TASKS
    + WALKERBIGLEG_TASKS
    + HUMANOID_TASKS
    + HUMANOIDSMALLLEG_TASKS
    + HUMANOIDBIGLEG_TASKS
    + CARTPOLE_TASKS
    + CARTPOLE_HEAVY_TASKS
)

PRIMAL_TASKS = {
    "cheetah": "cheetah_run",
    "walker": "walker_run",
    "quadruped": "quadruped_roll",
    "humanoid": "humanoid_walk",
    "fish": "fish_swim",
}

PRIMAL_TASKS_WALK = {
    "humanoid": "humanoid_walk",
    "quadruped": "quadruped_roll",
}

PRIMAL_TASKS_FAST_RUN = {
    "cheetah": "cheetahfast_run",
    "walker": "walkerfast_run",
}

PRIMAL_TASKS_RUN_BACKWARD = {
    "cheetah": "cheetah_run_backward",
    "walker": "walker_run_backward",
}

"""Walker, Cheetah and Cartpole tasks for CRL experiments."""


# =============================== Walker ===============================
CRL_WALKER_TASKS = [
    "walker_run",
    "walkerlongtorso_run",
]

CRL_WALKER_DIFF_REWARD_TASKS = [
    "walker_run",
    "walker_run_backward",
]

CRL_WALKER_DIFF_DYNAMICS_DIFF_REWARD_TASKS = [
    "walker_run",
    "walkerlongtorso_run_backward",
]

CRL_WALKER_WALK_RUN_TASKS = [
    "walker_walk",
    "walker_run",
]

CRL_WALKER_STAND_RUN_TASKS = [
    "walker_stand",
    "walker_run",
]

# =============================== Cartpole ===============================
CRL_CARTPOLE_BALANCE_TASKS = [
    "cartpole_balance",
    "cartpole_balance",
]

# =============================== Cheetah ===============================

CRL_CHEETAH_TASKS = [
    "cheetah_run",
    "cheetah_run",
]

CRL_CHEETAH_DIFF_REWARD_TASKS = [
    "cheetah_run",
    "cheetah_run_backward",
]

CRL_CHEETAH_DIFF_RUN_SPEED_REWARD_TASKS = [
    "cheetah_run",
    "cheetahfast_run",
]

CRL_CHEETAH_DIFF_DYNAMICS_DIFF_REWARD_TASKS = [
    "cheetah_run",
    "cheetahlongtorso_run_backward",
]

# =============================== Humanoid ===============================
CRL_HUMANOID_DIFF_REWARD_TASKS = [
    "humanoid_run",
    "humanoid_run_backward",
]


# =============================== Fish ===============================
CRL_FISH_DIFF_REWARD_TASKS = [
    "fish_swim",
    "fish_swim_backward",
]


# =============================== Quadruped ===============================
CRL_QUADRUPED_RUN_JUMP_TASKS = [
    "quadruped_run",
    "quadruped_jump",
]

# Did not have drastic transition changes which cause the model to generalise well
# CRL_QUADRUPED_DIFF_REWARD_TASKS = [
#     "quadruped_escape",
#     "quadruped_fetch",
# ]

CRL_QUADRUPED_DIFF_REWARD_TASKS = [
    "quadruped_run",
    "quadruped_run_backward",
]

# =============================== Dog ===============================
CRL_DOG_DIFF_REWARD_TASKS = [
    "dog_walk",
    "dog_walk_backward",
]



# ================================= CRL Different Domain ======================
CRL_DIFF_DOMAINS_SAME_REWARD = [
    "cheetah_run",
    "walker_run",
]


# ================================= CRL Different Domain ======================
CRL_DIFF_DOMAINS_DIFF_REWARD = [
    "quadruped_run",
    "fish_swim",
    "cheetah_run_backward",
    "finger_turn_hard",
]

CRL_DIFF_DOMAINS_DIFF_REWARD_CHEETAH_FISH = [
    "cheetah_run",
    "fish_swim",
]

# =============================== CRL Task Sets ===============================

CRL_TASKS_SAME_REWARD = {
    'walker': CRL_WALKER_TASKS,
    'cheetah': CRL_CHEETAH_TASKS,
    'cartpole': CRL_CARTPOLE_BALANCE_TASKS,
}

CRL_TASKS_DIFF_REWARD = {
    'walker': CRL_WALKER_DIFF_REWARD_TASKS,
    'cheetah': CRL_CHEETAH_DIFF_REWARD_TASKS,
    'quadruped': CRL_QUADRUPED_DIFF_REWARD_TASKS,
    'humanoid': CRL_HUMANOID_DIFF_REWARD_TASKS,
    'fish': CRL_FISH_DIFF_REWARD_TASKS,
    'dog': CRL_DOG_DIFF_REWARD_TASKS,

}

CRL_DIFF_DYNAMICS_DIFF_REWARD = {
    'cheetah': CRL_CHEETAH_DIFF_DYNAMICS_DIFF_REWARD_TASKS,
    'walker': CRL_WALKER_DIFF_DYNAMICS_DIFF_REWARD_TASKS,
}

CRL_TASKS_DIFF_RUN_SPEED_REWARD = {
    'cheetah': CRL_CHEETAH_DIFF_RUN_SPEED_REWARD_TASKS,
}

# needs to be rename to remove the walker
CRL_WALKER_WALK_RUN_TASKS = {
    'walker': CRL_WALKER_WALK_RUN_TASKS,
}

# needs to be rename to remove the walker
CRL_WALKER_STAND_RUN_TASKS = {
    'walker': CRL_WALKER_STAND_RUN_TASKS,
}

CRL_RUN_JUMP_TASKS = {
    'quadruped': CRL_QUADRUPED_RUN_JUMP_TASKS,
}






