from metaworld.policies.sawyer_bin_picking_v2_policy import SawyerBinPickingV2Policy
from metaworld.policies.sawyer_box_close_v1_policy import SawyerBoxCloseV1Policy
from metaworld.policies.sawyer_hand_insert_policy import SawyerHandInsertPolicy
from metaworld.policies.sawyer_door_lock_v1_policy import SawyerDoorLockV1Policy
from metaworld.policies.sawyer_door_unlock_v1_policy import SawyerDoorUnlockV1Policy
import metaworld
import time
import glfw
import numpy as np
import random
from PIL import Image
from torchmeta.utils.gradient_based import gradient_update_parameters
from network import MIL
import argparse
import os
import torch
import torch.nn.functional as F
from utils import get_accuracy, get_data, close, resize_img, load_model, test_policy, save_model
from torch.utils.tensorboard import SummaryWriter
import time
from pyautogui import press


def test_model(env, model):
    model.eval()

    env.max_path_length = 200
    with torch.no_grad():
        obs = env.reset()    
        for _ in range(env.max_path_length):
            #Prepare tensors
            img = env._get_viewer('human')._read_pixels_as_in_window()
            img = resize_img(img)
            img = np.expand_dims(img, axis=0)
            t_img = torch.from_numpy(img).to(dtype=torch.float, device=args.device)
            config = np.expand_dims(obs[:3], axis=0)
            t_config = torch.from_numpy(config).to(dtype=torch.float, device=args.device)

            #Execute action
            action = model(t_img, t_config)
            a = action.squeeze().cpu().numpy()
            obs, reward, done, info = env.step(a)
            env.render()

            # Save video
            if(_ < 1):
                time.sleep(0.25)
        close(env)



if __name__ == "__main__":
    print("Start test program")

    #Parameters
    parser = argparse.ArgumentParser('Meta-Imitation Learning (MIL)')

    parser.add_argument('--step-size', type=float, default=0.4,
        help='Step-size for the gradient step for adaptation (default: 0.4).')
    parser.add_argument('--output-folder', type=str, default='./models',
        help='Path to the output folder for saving the model (optional).')
    parser.add_argument('--batch-size', type=int, default=16,
        help='Number of tasks in a mini-batch of tasks (default: 16).')
    parser.add_argument('--use-cuda', action='store_true',
        help='Use CUDA if available.')
    args = parser.parse_args()
    args.device = torch.device('cuda' if args.use_cuda
        and torch.cuda.is_available() else 'cpu')


    ml45 = metaworld.ML45() # Construct the benchmark, sampling tasks

    # Test tasks
    # custom_tasks = ["bin-picking-v1", "box-close-v1", "hand-insert-v1", "door-lock-v1", "door-unlock-v1"]
    # policies = {"bin-picking-v1": SawyerBinPickingV2Policy(),
    #             "box-close-v1": SawyerBoxCloseV1Policy(),
    #             "hand-insert-v1":SawyerHandInsertPolicy(),
    #             "door-lock-v1": SawyerDoorLockV1Policy(),
    #             "door-unlock-v1": SawyerDoorUnlockV1Policy()}
    # ml_custom = {name: ml45.test_classes[name] for name in custom_tasks if name in ml45.test_classes}
    

    # Define model
    model = MIL()
    model.to(device=args.device)
    load_model(model, "./models/mil_499.th")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.step_size)

    name = "door-lock-v1"
    env_cls = ml45.test_classes[name]
    policy = SawyerDoorLockV1Policy()
    
    all_tasks = [task for task in ml45.test_tasks if task.env_name == name]

    #Fine tune in support task
    print("Fine tuning ...")
    env = env_cls()
    support_task = random.choice(all_tasks[:25])
    env.set_task(support_task)

    batches_imgs, batches_configs, batches_actions = get_data(env, policy, args)
    loss = torch.tensor(0., device=args.device)
    number_batches = len(batches_imgs)
    while(len(batches_imgs) > 0):
        pred_actions = model(batches_imgs.pop().to(device=args.device), 
                                batches_configs.pop().to(device=args.device))
        loss += F.mse_loss(pred_actions, batches_actions.pop().to(device=args.device))
    loss.div_(number_batches)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # save_model(model, args.output_folder, "fine_tuned.th")

    print("Testing ...")
    for i in range(3):
        #Test in query task
        env = env_cls()
        query_task = random.choice(all_tasks[25:])
        env.set_task(query_task)
        test_model(env, model)