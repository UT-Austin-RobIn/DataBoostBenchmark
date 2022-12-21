import os

from databoost.utils.general import AttrDict

from databoost.envs.calvin.custom import CalvinEnv


'''General configs'''
env_root = "/data/jullian-yapeter/DataBoostBenchmark/calvin"


'''Tasks configs'''
tasks = {
    # rotate
    "rotate_red_block_right": AttrDict({
        "task_name": "rotate_red_block_right",
        "env": CalvinEnv,
        "seed_dataset": os.path.join(env_root, "data/seed/rotate_red_block_right"),
    }),
    "rotate_red_block_left": AttrDict({
        "task_name": "rotate_red_block_left",
        "env": CalvinEnv,
        "seed_dataset": os.path.join(env_root, "data/seed/rotate_red_block_left"),
    }),
    "rotate_blue_block_right": AttrDict({
        "task_name": "rotate_blue_block_right",
        "env": CalvinEnv,
        "seed_dataset": os.path.join(env_root, "data/seed/rotate_blue_block_right"),
    }),
    "rotate_blue_block_left": AttrDict({
        "task_name": "rotate_blue_block_left",
        "env": CalvinEnv,
        "seed_dataset": os.path.join(env_root, "data/seed/rotate_blue_block_left"),
    }),
    "rotate_pink_block_right": AttrDict({
        "task_name": "rotate_pink_block_right",
        "env": CalvinEnv,
        "seed_dataset": os.path.join(env_root, "data/seed/rotate_pink_block_right"),
    }),
    "rotate_pink_block_left": AttrDict({
        "task_name": "rotate_pink_block_left",
        "env": CalvinEnv,
        "seed_dataset": os.path.join(env_root, "data/seed/rotate_pink_block_left"),
    }),

    # pushing
    "push_red_block_right": AttrDict({
        "task_name": "push_red_block_right",
        "env": CalvinEnv,
        "seed_dataset": os.path.join(env_root, "data/seed/push_red_block_right"),
    }),
    "push_red_block_left": AttrDict({
        "task_name": "push_red_block_left",
        "env": CalvinEnv,
        "seed_dataset": os.path.join(env_root, "data/seed/push_red_block_left"),
    }),
    "push_blue_block_right": AttrDict({
        "task_name": "push_blue_block_right",
        "env": CalvinEnv,
        "seed_dataset": os.path.join(env_root, "data/seed/push_blue_block_right"),
    }),
    "push_blue_block_left": AttrDict({
        "task_name": "push_blue_block_left",
        "env": CalvinEnv,
        "seed_dataset": os.path.join(env_root, "data/seed/push_blue_block_left"),
    }),
    "push_pink_block_right": AttrDict({
        "task_name": "push_pink_block_right",
        "env": CalvinEnv,
        "seed_dataset": os.path.join(env_root, "data/seed/push_pink_block_right"),
    }),
    "push_pink_block_left": AttrDict({
        "task_name": "push_pink_block_left",
        "env": CalvinEnv,
        "seed_dataset": os.path.join(env_root, "data/seed/push_pink_block_left"),
    }),

    # open/close
    "move_slider_left": AttrDict({
        "task_name": "move_slider_left",
        "env": CalvinEnv,
        "seed_dataset": os.path.join(env_root, "data/seed/move_slider_left"),
    }),
    "move_slider_right": AttrDict({
        "task_name": "move_slider_right",
        "env": CalvinEnv,
        "seed_dataset": os.path.join(env_root, "data/seed/move_slider_right"),
    }),
    "open_drawer": AttrDict({
        "task_name": "open_drawer",
        "env": CalvinEnv,
        "seed_dataset": os.path.join(env_root, "data/seed/open_drawer"),
    }),
    "close_drawer": AttrDict({
        "task_name": "close_drawer",
        "env": CalvinEnv,
        "seed_dataset": os.path.join(env_root, "data/seed/close_drawer"),
    }),

    # lifting
    "lift_red_block_table": AttrDict({
        "task_name": "lift_red_block_table",
        "env": CalvinEnv,
        "seed_dataset": os.path.join(env_root, "data/seed/lift_red_block_table"),
    }),
    "lift_red_block_slider": AttrDict({
        "task_name": "lift_red_block_slider",
        "env": CalvinEnv,
        "seed_dataset": os.path.join(env_root, "data/seed/lift_red_block_slider"),
    }),
    "lift_red_block_drawer": AttrDict({
        "task_name": "lift_red_block_drawer",
        "env": CalvinEnv,
        "seed_dataset": os.path.join(env_root, "data/seed/lift_red_block_drawer"),
    }),
    "lift_blue_block_table": AttrDict({
        "task_name": "lift_blue_block_table",
        "env": CalvinEnv,
        "seed_dataset": os.path.join(env_root, "data/seed/lift_blue_block_table"),
    }),
    "lift_blue_block_slider": AttrDict({
        "task_name": "lift_blue_block_slider",
        "env": CalvinEnv,
        "seed_dataset": os.path.join(env_root, "data/seed/lift_blue_block_slider"),
    }),
    "lift_blue_block_drawer": AttrDict({
        "task_name": "lift_blue_block_drawer",
        "env": CalvinEnv,
        "seed_dataset": os.path.join(env_root, "data/seed/lift_blue_block_drawer"),
    }),
    "lift_pink_block_table": AttrDict({
        "task_name": "lift_pink_block_table",
        "env": CalvinEnv,
        "seed_dataset": os.path.join(env_root, "data/seed/lift_pink_block_table"),
    }),
    "lift_pink_block_slider": AttrDict({
        "task_name": "lift_pink_block_slider",
        "env": CalvinEnv,
        "seed_dataset": os.path.join(env_root, "data/seed/lift_pink_block_slider"),
    }),
    "lift_pink_block_drawer": AttrDict({
        "task_name": "lift_pink_block_drawer",
        "env": CalvinEnv,
        "seed_dataset": os.path.join(env_root, "data/seed/lift_pink_block_drawer"),
    }),

    # placing
    "place_in_slider": AttrDict({
        "task_name": "place_in_slider",
        "env": CalvinEnv,
        "seed_dataset": os.path.join(env_root, "data/seed/place_in_slider"),
    }),
    "place_in_drawer": AttrDict({
        "task_name": "place_in_drawer",
        "env": CalvinEnv,
        "seed_dataset": os.path.join(env_root, "data/seed/place_in_drawer"),
    }),

    # stacking
    "stack_block": AttrDict({
        "task_name": "stack_block",
        "env": CalvinEnv,
        "seed_dataset": os.path.join(env_root, "data/seed/stack_block"),
    }),
    "unstack_block": AttrDict({
        "task_name": "unstack_block",
        "env": CalvinEnv,
        "seed_dataset": os.path.join(env_root, "data/seed/unstack_block"),
    }),

    # lights
    "turn_on_lightbulb": AttrDict({
        "task_name": "turn_on_lightbulb",
        "env": CalvinEnv,
        "seed_dataset": os.path.join(env_root, "data/seed/turn_on_lightbulb"),
    }),
    "turn_off_lightbulb": AttrDict({
        "task_name": "turn_off_lightbulb",
        "env": CalvinEnv,
        "seed_dataset": os.path.join(env_root, "data/seed/turn_off_lightbulb"),
    }),
    "turn_on_led": AttrDict({
        "task_name": "turn_on_led",
        "env": CalvinEnv,
        "seed_dataset": os.path.join(env_root, "data/seed/turn_on_led"),
    }),
    "turn_off_led": AttrDict({
        "task_name": "turn_off_led",
        "env": CalvinEnv,
        "seed_dataset": os.path.join(env_root, "data/seed/turn_off_led"),
    }),

    # pushing into drawer
    "push_into_drawer": AttrDict({
        "task_name": "push_into_drawer",
        "env": CalvinEnv,
        "seed_dataset": os.path.join(env_root, "data/seed/push_into_drawer"),
    })
}


'''Seed tasks configs'''
seed_tasks_list = [
    "move_slider_left"
]
seed_dataset_dir = os.path.join(env_root, "data/seed")
seed_n_demos = 10
seed_do_render = True
seed_dataset_kwargs = AttrDict({
    "act_noise_pct": 0.1,
    "resolution": (224, 224)
})


'''Prior tasks configs'''
prior_tasks_list = list(tasks.keys())
prior_dataset_dir = os.path.join(env_root, "data/prior")
prior_n_demos = 10
prior_do_render = True
prior_dataset_kwargs = AttrDict({
    "act_noise_pct": 0.1,
    "resolution": (224, 224),
})
