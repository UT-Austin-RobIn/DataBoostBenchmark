import os

from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE as ALL_V2_ENVS

from databoost.utils.general import AttrDict

import metaworld.policies as policies


'''General configs'''
env_root = "/data/jullian-yapeter/DataBoostBenchmark/metaworld"


'''Tasks configs'''
tasks = {
    "door-open": AttrDict({
        "task_name": "door-open",
        "env": ALL_V2_ENVS["door-open-v2-goal-observable"],
        "seed_dataset": os.path.join(env_root, "data/seed/door-open"),
        "expert_policy": policies.SawyerDoorOpenV2Policy,
    }),
    "door-close": AttrDict({
        "task_name": "door-close",
        "env": ALL_V2_ENVS["door-close-v2-goal-observable"],
        "seed_dataset": os.path.join(env_root, "data/seed/door-close"),
        "expert_policy": policies.SawyerDoorCloseV2Policy,
    }),
    "door-lock": AttrDict({
        "task_name": "door-lock",
        "env": ALL_V2_ENVS["door-lock-v2-goal-observable"],
        "seed_dataset": os.path.join(env_root, "data/seed/door-lock"),
        "expert_policy": policies.SawyerDoorLockV2Policy,
    }),
    "door-unlock": AttrDict({
        "task_name": "door-unlock",
        "env": ALL_V2_ENVS["door-unlock-v2-goal-observable"],
        "seed_dataset": os.path.join(env_root, "data/seed/door-unlock"),
        "expert_policy": policies.SawyerDoorUnlockV2Policy,
    }),
    "assembly": AttrDict({
        "task_name": "assembly",
        "env": ALL_V2_ENVS["assembly-v2-goal-observable"],
        "seed_dataset": os.path.join(env_root, "data/seed/assembly"),
        "expert_policy": policies.SawyerAssemblyV2Policy,
    }),
    "basketball": AttrDict({
        "task_name": "basketball",
        "env": ALL_V2_ENVS["basketball-v2-goal-observable"],
        "seed_dataset": os.path.join(env_root, "data/seed/basketball"),
        "expert_policy": policies.SawyerBasketballV2Policy,
    }),
    "bin-picking": AttrDict({
        "task_name": "bin-picking",
        "env": ALL_V2_ENVS["bin-picking-v2-goal-observable"],
        "seed_dataset": os.path.join(env_root, "data/seed/bin-picking"),
        "expert_policy": policies.SawyerBinPickingV2Policy,
    }),
    "box-close": AttrDict({
        "task_name": "box-close",
        "env": ALL_V2_ENVS["box-close-v2-goal-observable"],
        "seed_dataset": os.path.join(env_root, "data/seed/box-close"),
        "expert_policy": policies.SawyerBoxCloseV2Policy,
    }),
    "button-press-topdown": AttrDict({
        "task_name": "button-press-topdown",
        "env": ALL_V2_ENVS["button-press-topdown-v2-goal-observable"],
        "seed_dataset": os.path.join(env_root, "data/seed/button-press-topdown"),
        "expert_policy": policies.SawyerButtonPressTopdownV2Policy,
    }),
    "button-press-topdown-wall": AttrDict({
        "task_name": "button-press-topdown-wall",
        "env": ALL_V2_ENVS["button-press-topdown-wall-v2-goal-observable"],
        "seed_dataset": os.path.join(env_root, "data/seed/button-press-topdown-wall"),
        "expert_policy": policies.SawyerButtonPressTopdownWallV2Policy,
    }),
    "button-press": AttrDict({
        "task_name": "button-press",
        "env": ALL_V2_ENVS["button-press-v2-goal-observable"],
        "seed_dataset": os.path.join(env_root, "data/seed/button-press"),
        "expert_policy": policies.SawyerButtonPressV2Policy,
    }),
    "button-press-wall": AttrDict({
        "task_name": "button-press-wall",
        "env": ALL_V2_ENVS["button-press-wall-v2-goal-observable"],
        "seed_dataset": os.path.join(env_root, "data/seed/button-press-wall"),
        "expert_policy": policies.SawyerButtonPressWallV2Policy,
    }),
    "coffee-button": AttrDict({
        "task_name": "coffee-button",
        "env": ALL_V2_ENVS["coffee-button-v2-goal-observable"],
        "seed_dataset": os.path.join(env_root, "data/seed/coffee-button"),
        "expert_policy": policies.SawyerCoffeeButtonV2Policy,
    }),
    "coffee-pull": AttrDict({
        "task_name": "coffee-pull",
        "env": ALL_V2_ENVS["coffee-pull-v2-goal-observable"],
        "seed_dataset": os.path.join(env_root, "data/seed/coffee-pull"),
        "expert_policy": policies.SawyerCoffeePullV2Policy,
    }),
    "coffee-push": AttrDict({
        "task_name": "coffee-push",
        "env": ALL_V2_ENVS["coffee-push-v2-goal-observable"],
        "seed_dataset": os.path.join(env_root, "data/seed/coffee-push"),
        "expert_policy": policies.SawyerCoffeePushV2Policy,
    }),
    "dial-turn": AttrDict({
        "task_name": "dial-turn",
        "env": ALL_V2_ENVS["dial-turn-v2-goal-observable"],
        "seed_dataset": os.path.join(env_root, "data/seed/dial-turn"),
        "expert_policy": policies.SawyerDialTurnV2Policy,
    }),
    "disassemble": AttrDict({
        "task_name": "disassemble",
        "env": ALL_V2_ENVS["disassemble-v2-goal-observable"],
        "seed_dataset": os.path.join(env_root, "data/seed/disassemble"),
        "expert_policy": policies.SawyerDisassembleV2Policy,
    }),
    "hand-insert": AttrDict({
        "task_name": "hand-insert",
        "env": ALL_V2_ENVS["hand-insert-v2-goal-observable"],
        "seed_dataset": os.path.join(env_root, "data/seed/hand-insert"),
        "expert_policy": policies.SawyerHandInsertV2Policy,
    }),
    "drawer-close": AttrDict({
        "task_name": "drawer-close",
        "env": ALL_V2_ENVS["drawer-close-v2-goal-observable"],
        "seed_dataset": os.path.join(env_root, "data/seed/drawer-close"),
        "expert_policy": policies.SawyerDrawerCloseV2Policy,
    }),
    "drawer-open": AttrDict({
        "task_name": "drawer-open",
        "env": ALL_V2_ENVS["drawer-open-v2-goal-observable"],
        "seed_dataset": os.path.join(env_root, "data/seed/drawer-open"),
        "expert_policy": policies.SawyerDrawerOpenV2Policy,
    }),
    "faucet-open": AttrDict({
        "task_name": "faucet-open",
        "env": ALL_V2_ENVS["faucet-open-v2-goal-observable"],
        "seed_dataset": os.path.join(env_root, "data/seed/faucet-open"),
        "expert_policy": policies.SawyerFaucetOpenV2Policy,
    }),
    "faucet-close": AttrDict({
        "task_name": "faucet-close",
        "env": ALL_V2_ENVS["faucet-close-v2-goal-observable"],
        "seed_dataset": os.path.join(env_root, "data/seed/faucet-close"),
        "expert_policy": policies.SawyerFaucetCloseV2Policy,
    }),
    "hammer": AttrDict({
        "task_name": "hammer",
        "env": ALL_V2_ENVS["hammer-v2-goal-observable"],
        "seed_dataset": os.path.join(env_root, "data/seed/hammer"),
        "expert_policy": policies.SawyerHammerV2Policy,
    }),
    "handle-press-side": AttrDict({
        "task_name": "handle-press-side",
        "env": ALL_V2_ENVS["handle-press-side-v2-goal-observable"],
        "seed_dataset": os.path.join(env_root, "data/seed/handle-press-side"),
        "expert_policy": policies.SawyerHandlePressSideV2Policy,
    }),
    "handle-press": AttrDict({
        "task_name": "handle-press",
        "env": ALL_V2_ENVS["handle-press-v2-goal-observable"],
        "seed_dataset": os.path.join(env_root, "data/seed/handle-press"),
        "expert_policy": policies.SawyerHandlePressV2Policy,
    }),
    "handle-pull-side": AttrDict({
        "task_name": "handle-pull-side",
        "env": ALL_V2_ENVS["handle-pull-side-v2-goal-observable"],
        "seed_dataset": os.path.join(env_root, "data/seed/handle-pull-side"),
        "expert_policy": policies.SawyerHandlePullSideV2Policy,
    }),
    "handle-pull": AttrDict({
        "task_name": "handle-pull",
        "env": ALL_V2_ENVS["handle-pull-v2-goal-observable"],
        "seed_dataset": os.path.join(env_root, "data/seed/handle-pull"),
        "expert_policy": policies.SawyerHandlePullV2Policy,
    }),
    "lever-pull": AttrDict({
        "task_name": "lever-pull",
        "env": ALL_V2_ENVS["lever-pull-v2-goal-observable"],
        "seed_dataset": os.path.join(env_root, "data/seed/lever-pull"),
        "expert_policy": policies.SawyerLeverPullV2Policy,
    }),
    "peg-insert-side": AttrDict({
        "task_name": "peg-insert-side",
        "env": ALL_V2_ENVS["peg-insert-side-v2-goal-observable"],
        "seed_dataset": os.path.join(env_root, "data/seed/peg-insert-side"),
        "expert_policy": policies.SawyerPegInsertionSideV2Policy,
    }),
    "pick-place-wall": AttrDict({
        "task_name": "pick-place-wall",
        "env": ALL_V2_ENVS["pick-place-wall-v2-goal-observable"],
        "seed_dataset": os.path.join(env_root, "data/seed/pick-place-wall"),
        "expert_policy": policies.SawyerPickPlaceWallV2Policy,
    }),
    "pick-out-of-hole": AttrDict({
        "task_name": "pick-out-of-hole",
        "env": ALL_V2_ENVS["pick-out-of-hole-v2-goal-observable"],
        "seed_dataset": os.path.join(env_root, "data/seed/pick-out-of-hole"),
        "expert_policy": policies.SawyerPickOutOfHoleV2Policy,
    }),
    "reach": AttrDict({
        "task_name": "reach",
        "env": ALL_V2_ENVS["reach-v2-goal-observable"],
        "seed_dataset": os.path.join(env_root, "data/seed/reach"),
        "expert_policy": policies.SawyerReachV2Policy,
    }),
    "push-back": AttrDict({
        "task_name": "push-back",
        "env": ALL_V2_ENVS["push-back-v2-goal-observable"],
        "seed_dataset": os.path.join(env_root, "data/seed/push-back"),
        "expert_policy": policies.SawyerPushBackV2Policy,
    }),
    "push": AttrDict({
        "task_name": "push",
        "env": ALL_V2_ENVS["push-v2-goal-observable"],
        "seed_dataset": os.path.join(env_root, "data/seed/push-back"),
        "expert_policy": policies.SawyerPushV2Policy,
    }),
    "pick-place": AttrDict({
        "task_name": "pick-place",
        "env": ALL_V2_ENVS["pick-place-v2-goal-observable"],
        "seed_dataset": os.path.join(env_root, "data/seed/pick-place"),
        "expert_policy": policies.SawyerPickPlaceV2Policy,
    }),
    "plate-slide": AttrDict({
        "task_name": "plate-slide",
        "env": ALL_V2_ENVS["plate-slide-v2-goal-observable"],
        "seed_dataset": os.path.join(env_root, "data/seed/plate-slide"),
        "expert_policy": policies.SawyerPlateSlideV2Policy,
    }),
    "plate-slide-side": AttrDict({
        "task_name": "plate-slide-side",
        "env": ALL_V2_ENVS["plate-slide-side-v2-goal-observable"],
        "seed_dataset": os.path.join(env_root, "data/seed/plate-slide-side"),
        "expert_policy": policies.SawyerPlateSlideSideV2Policy,
    }),
    "plate-slide-back": AttrDict({
        "task_name": "plate-slide-back",
        "env": ALL_V2_ENVS["plate-slide-back-v2-goal-observable"],
        "seed_dataset": os.path.join(env_root, "data/seed/plate-slide-back"),
        "expert_policy": policies.SawyerPlateSlideBackV2Policy,
    }),
    "plate-slide-back-side": AttrDict({
        "task_name": "plate-slide-back-side",
        "env": ALL_V2_ENVS["plate-slide-back-side-v2-goal-observable"],
        "seed_dataset": os.path.join(env_root, "data/seed/plate-slide-back-side"),
        "expert_policy": policies.SawyerPlateSlideBackSideV2Policy,
    }),
    "peg-unplug-side": AttrDict({
        "task_name": "peg-unplug-side",
        "env": ALL_V2_ENVS["peg-unplug-side-v2-goal-observable"],
        "seed_dataset": os.path.join(env_root, "data/seed/peg-unplug-side"),
        "expert_policy": policies.SawyerPegUnplugSideV2Policy,
    }),
    "soccer": AttrDict({
        "task_name": "soccer",
        "env": ALL_V2_ENVS["soccer-v2-goal-observable"],
        "seed_dataset": os.path.join(env_root, "data/seed/soccer"),
        "expert_policy": policies.SawyerSoccerV2Policy,
    }),
    "stick-push": AttrDict({
        "task_name": "stick-push",
        "env": ALL_V2_ENVS["stick-push-v2-goal-observable"],
        "seed_dataset": os.path.join(env_root, "data/seed/stick-push"),
        "expert_policy": policies.SawyerStickPushV2Policy,
    }),
    "stick-pull": AttrDict({
        "task_name": "stick-pull",
        "env": ALL_V2_ENVS["stick-pull-v2-goal-observable"],
        "seed_dataset": os.path.join(env_root, "data/seed/stick-pull"),
        "expert_policy": policies.SawyerStickPullV2Policy,
    }),
    "push-wall": AttrDict({
        "task_name": "push-wall",
        "env": ALL_V2_ENVS["push-wall-v2-goal-observable"],
        "seed_dataset": os.path.join(env_root, "data/seed/push-wall"),
        "expert_policy": policies.SawyerPushWallV2Policy,
    }),
    "reach-wall": AttrDict({
        "task_name": "reach-wall",
        "env": ALL_V2_ENVS["reach-wall-v2-goal-observable"],
        "seed_dataset": os.path.join(env_root, "data/seed/reach-wall"),
        "expert_policy": policies.SawyerReachWallV2Policy,
    }),
    "shelf-place": AttrDict({
        "task_name": "shelf-place",
        "env": ALL_V2_ENVS["shelf-place-v2-goal-observable"],
        "seed_dataset": os.path.join(env_root, "data/seed/shelf-place"),
        "expert_policy": policies.SawyerShelfPlaceV2Policy,
    }),
    "sweep-into": AttrDict({
        "task_name": "sweep-into",
        "env": ALL_V2_ENVS["sweep-into-v2-goal-observable"],
        "seed_dataset": os.path.join(env_root, "data/seed/sweep-into"),
        "expert_policy": policies.SawyerSweepIntoV2Policy,
    }),
    "sweep": AttrDict({
        "task_name": "sweep",
        "env": ALL_V2_ENVS["sweep-v2-goal-observable"],
        "seed_dataset": os.path.join(env_root, "data/seed/sweep"),
        "expert_policy": policies.SawyerSweepV2Policy,
    }),
    "window-open": AttrDict({
        "task_name": "window-open",
        "env": ALL_V2_ENVS["window-open-v2-goal-observable"],
        "seed_dataset": os.path.join(env_root, "data/seed/window-open"),
        "expert_policy": policies.SawyerWindowOpenV2Policy,
    }),
    "window-close": AttrDict({
        "task_name": "window-close",
        "env": ALL_V2_ENVS["window-close-v2-goal-observable"],
        "seed_dataset": os.path.join(env_root, "data/seed/window-close"),
        "expert_policy": policies.SawyerWindowCloseV2Policy,
    })
}


'''Seed tasks configs'''
seed_tasks_list = [
    "assembly",
    "pick-place-wall",
    "door-open",
    "plate-slide-back-side"
]
seed_dataset_dir = os.path.join(env_root, "data/seed")
seed_n_demos = 10
seed_do_render = True
seed_dataset_kwargs = AttrDict({
    "act_noise_pct": 0.1,
    "resolution": (224, 224),
    "camera": "corner"
})


'''Prior tasks configs'''
prior_tasks_list = list(tasks.keys())
prior_dataset_dir = os.path.join(env_root, "data/prior")
prior_n_demos = 10
prior_do_render = True
prior_dataset_kwargs = AttrDict({
    "act_noise_pct": 0.1,
    "resolution": (224, 224),
    "camera": "corner"
})
