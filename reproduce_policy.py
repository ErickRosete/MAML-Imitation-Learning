from metaworld.policies.sawyer_door_lock_v1_policy import SawyerDoorLockV1Policy
import metaworld
import random
from utils import test_policy

ml45 = metaworld.ML45()

name = "door-lock-v1"
env_cls = ml45.test_classes[name]
policy = SawyerDoorLockV1Policy()

all_tasks = [task for task in ml45.test_tasks if task.env_name == name]

env = env_cls()
query_task = random.choice(all_tasks[25:])
env.set_task(query_task)
env.max_path_length = 200
test_policy(env, policy, render=True, stop=False)
