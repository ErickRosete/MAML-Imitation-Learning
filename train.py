from metaworld.policies.sawyer_reach_v2_policy import SawyerReachV2Policy
from metaworld.policies.sawyer_push_v2_policy import  SawyerPushV2Policy
from metaworld.policies.sawyer_pick_place_v2_policy import SawyerPickPlaceV2Policy
from metaworld.policies.sawyer_door_open_v1_policy import SawyerDoorOpenV1Policy
from metaworld.policies.sawyer_drawer_open_v1_policy import  SawyerDrawerOpenV1Policy
from metaworld.policies.sawyer_drawer_close_v1_policy import  SawyerDrawerCloseV1Policy
from metaworld.policies.sawyer_button_press_topdown_v1_policy import SawyerButtonPressTopdownV1Policy
from metaworld.policies.sawyer_peg_insertion_side_v2_policy import SawyerPegInsertionSideV2Policy
from metaworld.policies.sawyer_window_open_v2_policy import SawyerWindowOpenV2Policy
from metaworld.policies.sawyer_window_close_v2_policy import SawyerWindowCloseV2Policy
from metaworld.policies.sawyer_door_close_v1_policy import SawyerDoorCloseV1Policy
from metaworld.policies.sawyer_reach_wall_v2_policy import SawyerReachWallV2Policy
from metaworld.policies.sawyer_pick_place_wall_v2_policy import SawyerPickPlaceWallV2Policy
from metaworld.policies.sawyer_push_wall_v2_policy import SawyerPushWallV2Policy
from metaworld.policies.sawyer_button_press_v1_policy import SawyerButtonPressV1Policy
from metaworld.policies.sawyer_button_press_topdown_wall_v1_policy import SawyerButtonPressTopdownWallV1Policy
from metaworld.policies.sawyer_button_press_wall_v1_policy import SawyerButtonPressWallV1Policy
from metaworld.policies.sawyer_peg_unplug_side_v1_policy import SawyerPegUnplugSideV1Policy
from metaworld.policies.sawyer_disassemble_v1_policy import SawyerDisassembleV1Policy
from metaworld.policies.sawyer_hammer_v1_policy import SawyerHammerV1Policy
from metaworld.policies.sawyer_plate_slide_v1_policy import SawyerPlateSlideV1Policy
from metaworld.policies.sawyer_plate_slide_side_v1_policy import SawyerPlateSlideSideV1Policy
from metaworld.policies.sawyer_plate_slide_back_v1_policy import SawyerPlateSlideBackV1Policy
from metaworld.policies.sawyer_plate_slide_back_side_v2_policy import SawyerPlateSlideBackSideV2Policy
from metaworld.policies.sawyer_handle_press_v1_policy import SawyerHandlePressV1Policy
from metaworld.policies.sawyer_handle_pull_v1_policy import SawyerHandlePullV1Policy
from metaworld.policies.sawyer_handle_press_side_v2_policy import SawyerHandlePressSideV2Policy
from metaworld.policies.sawyer_handle_pull_side_v1_policy import SawyerHandlePullSideV1Policy
from metaworld.policies.sawyer_stick_push_policy import SawyerStickPushV1Policy
from metaworld.policies.sawyer_stick_pull_policy import SawyerStickPullV1Policy
from metaworld.policies.sawyer_basketball_v1_policy import  SawyerBasketballV1Policy
from metaworld.policies.sawyer_soccer_v1_policy import SawyerSoccerV1Policy
from metaworld.policies.sawyer_faucet_open_v1_policy import SawyerFaucetOpenV1Policy
from metaworld.policies.sawyer_faucet_close_v1_policy import SawyerFaucetCloseV1Policy
from metaworld.policies.sawyer_coffee_push_v1_policy import SawyerCoffeePushV1Policy
from metaworld.policies.sawyer_coffee_pull_v1_policy import SawyerCoffeePullV1Policy
from metaworld.policies.sawyer_coffee_button_v1_policy import SawyerCoffeeButtonV1Policy
from metaworld.policies.sawyer_sweep_v1_policy import SawyerSweepV1Policy
from metaworld.policies.sawyer_sweep_into_v1_policy import SawyerSweepIntoV1Policy
from metaworld.policies.sawyer_pick_out_of_hole_v1_policy import SawyerPickOutOfHoleV1Policy
from metaworld.policies.sawyer_assembly_v1_policy import SawyerAssemblyV1Policy
from metaworld.policies.sawyer_shelf_place_v1_policy import SawyerShelfPlaceV1Policy
from metaworld.policies.sawyer_push_back_v1_policy import SawyerPushBackV1Policy
from metaworld.policies.sawyer_lever_pull_v2_policy import SawyerLeverPullV2Policy
from metaworld.policies.sawyer_dial_turn_v1_policy import SawyerDialTurnV1Policy
import metaworld
import time
import numpy as np
import random
from torchmeta.utils.gradient_based import gradient_update_parameters
from network import MIL
import argparse
import os
import torch
import torch.nn.functional as F
from utils import get_accuracy, get_data, close, resize_img, load_model, save_model
from torch.utils.tensorboard import SummaryWriter

def train(ml_custom, policies, args):
    # Prepare to log info
    writer = SummaryWriter()

    # Define model
    model = MIL()
    model.to(device=args.device)
    # load_model(model, "./models/mil_499.th")
    model.train()
    meta_optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    print("Start training")
    for batch in range(args.num_batches):
        outer_loss = torch.tensor(0., device=args.device)
        accuracy = torch.tensor(0., device=args.device)
        
        for name in np.random.choice(list(ml_custom.keys()), 3, replace=False):
            env_cls = ml_custom[name]
            print("Task: %s" % name)

            policy = policies[name]
            all_tasks = [task for task in ml45.train_tasks
                                    if task.env_name == name]

            # Adapt in support task
            env = env_cls()
            support_task = random.choice(all_tasks[:25])
            env.set_task(support_task)
            batches_imgs, batches_configs, batches_actions = get_data(env, policy, args)
            inner_loss = torch.tensor(0., device=args.device)
            number_batches = len(batches_imgs)
            while(len(batches_imgs) > 0):
                pred_actions = model(batches_imgs.pop().to(device=args.device), 
                                     batches_configs.pop().to(device=args.device))
                inner_loss += F.mse_loss(pred_actions, batches_actions.pop().to(device=args.device))
            inner_loss.div_(number_batches)
            model.zero_grad()
            params = gradient_update_parameters(model, inner_loss, step_size=args.step_size,
                                                first_order=args.first_order)

            # Evaluate in query task
            env = env_cls()
            query_task = random.choice(all_tasks[25:])
            env.set_task(support_task)
            batches_imgs, batches_configs, batches_actions = get_data(env, policy, args)
            aux_loss = torch.tensor(0., device=args.device)
            aux_accuracy = torch.tensor(0., device=args.device)
            number_batches = len(batches_imgs)

            while(len(batches_imgs) > 0):
                pred_actions = model(batches_imgs.pop().to(device=args.device), 
                                     batches_configs.pop().to(device=args.device))
                batch_actions = batches_actions.pop().to(device=args.device)
                aux_loss += F.mse_loss(pred_actions, batch_actions)
                with torch.no_grad():
                    aux_accuracy += get_accuracy(pred_actions, batch_actions)

            aux_loss.div_(number_batches)
            aux_accuracy.div_(number_batches)
            outer_loss += aux_loss
            accuracy += aux_accuracy
        
        outer_loss.div_(3)
        accuracy.div_(3)
        meta_optimizer.zero_grad()
        outer_loss.backward()
        meta_optimizer.step()

        #Log info
        writer.add_scalar('meta_train/loss', outer_loss.item(), batch)
        writer.add_scalar('meta_train/accuracy', accuracy.item(), batch)
        print("batch: %d loss: %.4f accuracy: %.4f" % (batch, outer_loss.item(), accuracy.item()))

        # Save model
        save_model(model, args.output_folder, 'mil_%d.th' % batch)

if __name__ == "__main__":
    ml45 = metaworld.ML45() # Construct the benchmark, sampling tasks

    custom_tasks = ["door-open-v1", "drawer-open-v1", "drawer-close-v1", "button-press-topdown-wall-v1",
                "button-press-wall-v1", "hammer-v1", "plate-slide-side-v1", "plate-slide-back-side-v1",
                "handle-press-v1", "handle-pull-v1", "handle-pull-side-v1", "faucet-open-v1", 
                "faucet-close-v1", "sweep-v1"]

    #All the following policies work, the ones selected require less time.
    policies = {"door-open-v1": SawyerDoorOpenV1Policy(),
                "drawer-open-v1": SawyerDrawerOpenV1Policy(),
                "drawer-close-v1": SawyerDrawerCloseV1Policy(),
                "button-press-topdown-wall-v1": SawyerButtonPressTopdownWallV1Policy(),
                "button-press-wall-v1": SawyerButtonPressWallV1Policy(),
                "hammer-v1": SawyerHammerV1Policy(),
                "plate-slide-side-v1": SawyerPlateSlideSideV1Policy(),
                "plate-slide-back-side-v1": SawyerPlateSlideBackSideV2Policy(),
                "handle-press-v1": SawyerHandlePressV1Policy(),
                "handle-pull-v1": SawyerHandlePullV1Policy(),
                "handle-pull-side-v1": SawyerHandlePullSideV1Policy(),
                "faucet-open-v1": SawyerFaucetOpenV1Policy(),
                "faucet-close-v1": SawyerFaucetCloseV1Policy(),
                "sweep-v1": SawyerSweepV1Policy(),
                # "peg-unplug-side-v1": SawyerPegUnplugSideV1Policy(),
                # "plate-slide-back-v1": SawyerPlateSlideBackV1Policy(),
                # "handle-press-side-v1": SawyerHandlePressSideV2Policy(),
                # "coffee-pull-v1": SawyerCoffeePullV1Policy(),
                # "coffee-button-v1": SawyerCoffeeButtonV1Policy(),
                # "sweep-into-v1": SawyerSweepIntoV1Policy(),
                # "dial-turn-v1": SawyerDialTurnV1Policy(),
                # "button-press-topdown-v1": SawyerButtonPressTopdownV1Policy(),
                # "window-close-v1": SawyerWindowCloseV2Policy(),
                # "door-close-v1": SawyerDoorCloseV1Policy(),
                # "button-press-v1": SawyerButtonPressV1Policy(),
                }

    ml_custom = {name: ml45.train_classes[name] for name in custom_tasks if name in ml45.train_classes}

    parser = argparse.ArgumentParser('Meta-Imitation Learning (MIL)')

    parser.add_argument('--first-order', action='store_true',
        help='Use the first-order approximation of MAML.')
    parser.add_argument('--step-size', type=float, default=0.4,
        help='Step-size for the gradient step for adaptation (default: 0.4).')
    parser.add_argument('--output-folder', type=str, default='./models',
        help='Path to the output folder for saving the model (optional).')
    parser.add_argument('--batch-size', type=int, default=16,
        help='Number of tasks in a mini-batch of tasks (default: 16).')
    parser.add_argument('--num-batches', type=int, default=100,
        help='Number of batches the model is trained over (default: 100).')
    parser.add_argument('--use-cuda', action='store_true',
        help='Use CUDA if available.')

    args = parser.parse_args()
    args.device = torch.device('cuda' if args.use_cuda
        and torch.cuda.is_available() else 'cpu')

    train(ml_custom, policies, args)