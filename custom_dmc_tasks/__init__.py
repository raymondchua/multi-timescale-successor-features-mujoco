from custom_dmc_tasks import (
    cartpole,
    cartpoleheavy,
    cheetah,
    cheetahbigleg,
    cheetahfast,
    cheetahlongtorso,
    cheetahstiffbackleg,
    cheetahstifffrontleg,
    dog,
    hopper,
    humanoid,
    humanoidbigleg,
    humanoidsmallleg,
    jaco,
    quadruped,
    walker,
    walkerbigleg,
    walkerfast,
    walkerlongtorso,
)


def make(
    domain, task, task_kwargs=None, environment_kwargs=None, visualize_reward=False
):

    if domain == "cheetah":
        return cheetah.make(
            task,
            task_kwargs=task_kwargs,
            environment_kwargs=environment_kwargs,
            visualize_reward=visualize_reward,
        )

    elif domain == "cheetahfast":
        return cheetahfast.make(
            task,
            task_kwargs=task_kwargs,
            environment_kwargs=environment_kwargs,
            visualize_reward=visualize_reward,
        )

    elif domain == "cheetahbigleg":
        return cheetahbigleg.make(
            task,
            task_kwargs=task_kwargs,
            environment_kwargs=environment_kwargs,
            visualize_reward=visualize_reward,
        )

    elif domain == "cheetahstiffbackleg":
        return cheetahstiffbackleg.make(
            task,
            task_kwargs=task_kwargs,
            environment_kwargs=environment_kwargs,
            visualize_reward=visualize_reward,
        )

    elif domain == "cheetahstifffrontleg":
        return cheetahstifffrontleg.make(
            task,
            task_kwargs=task_kwargs,
            environment_kwargs=environment_kwargs,
            visualize_reward=visualize_reward,
        )

    elif domain == "dog":
        return dog.make(
            task,
            task_kwargs=task_kwargs,
            environment_kwargs=environment_kwargs,
            visualize_reward=visualize_reward,
        )

    elif domain == "walker":
        return walker.make(
            task,
            task_kwargs=task_kwargs,
            environment_kwargs=environment_kwargs,
            visualize_reward=visualize_reward,
        )

    elif domain == "walkerbigleg":
        return walkerbigleg.make(
            task,
            task_kwargs=task_kwargs,
            environment_kwargs=environment_kwargs,
            visualize_reward=visualize_reward,
        )

    elif domain == "walkerfast":
        return walkerfast.make(
            task,
            task_kwargs=task_kwargs,
            environment_kwargs=environment_kwargs,
            visualize_reward=visualize_reward,
        )

    elif domain == "hopper":
        return hopper.make(
            task,
            task_kwargs=task_kwargs,
            environment_kwargs=environment_kwargs,
            visualize_reward=visualize_reward,
        )
    elif domain == "quadruped":
        return quadruped.make(
            task,
            task_kwargs=task_kwargs,
            environment_kwargs=environment_kwargs,
            visualize_reward=visualize_reward,
        )

    elif domain == "humanoid":
        return humanoid.make(
            task,
            task_kwargs=task_kwargs,
            environment_kwargs=environment_kwargs,
            visualize_reward=visualize_reward,
        )

    elif domain == "humanoidsmallleg":
        return humanoidsmallleg.make(
            task,
            task_kwargs=task_kwargs,
            environment_kwargs=environment_kwargs,
            visualize_reward=visualize_reward,
        )

    elif domain == "humanoidbigleg":
        return humanoidbigleg.make(
            task,
            task_kwargs=task_kwargs,
            environment_kwargs=environment_kwargs,
            visualize_reward=visualize_reward,
        )

    elif domain == "cheetahlongtorso":
        return cheetahlongtorso.make(
            task,
            task_kwargs=task_kwargs,
            environment_kwargs=environment_kwargs,
            visualize_reward=visualize_reward,
        )

    elif domain == "walkerlongtorso":
        return walkerlongtorso.make(
            task,
            task_kwargs=task_kwargs,
            environment_kwargs=environment_kwargs,
            visualize_reward=visualize_reward,
        )

    elif domain == "cartpole":
        return cartpole.make(
            task,
            task_kwargs=task_kwargs,
            environment_kwargs=environment_kwargs,
            visualize_reward=visualize_reward,
        )

    elif domain == "cartpoleheavy":
        return cartpoleheavy.make(
            task,
            task_kwargs=task_kwargs,
            environment_kwargs=environment_kwargs,
            visualize_reward=visualize_reward,
        )

    else:
        raise f"{task} not found"

    assert None


def make_jaco(task, obs_type, seed):
    return jaco.make(task, obs_type, seed)
