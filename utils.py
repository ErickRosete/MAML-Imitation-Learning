import torch
import os
from PIL import Image
import numpy as np
import glfw
import matplotlib.pyplot as plt
import time

def close(env):
    if env.viewer is not None:
        # self.viewer.finish()
        glfw.destroy_window(env.viewer.window)
    env.viewer = None

def get_accuracy(predictions, targets):
    accuracy = torch.isclose(predictions, targets, atol=0.2)
    accuracy = torch.mean(torch.all(accuracy, axis=-1).to(dtype=torch.float))
    return accuracy

def test_policy(env, policy, render=False, stop=True):
    obs = env.reset()    
    for step in range(env.max_path_length):
        a = policy.get_action(obs)
        #Perform action
        obs, reward, done, info = env.step(a)
        if(render):
            time.sleep(0.1)
            env.render()
        if(stop and info['success']):
            break
        if(step < 1):
            time.sleep(1)
    close(env)
    return step

def get_data(env, policy, args):
    imgs, configs, actions = [], [], []
    obs = env.reset()    
    img = env._get_viewer('human')._read_pixels_as_in_window()
    img = resize_img(img)
    for _ in range(env.max_path_length):
        a = policy.get_action(obs)

        #Save img, obs, action
        imgs.append(img)
        configs.append(obs[:3])
        actions.append(a)

        #Perform action
        obs, reward, done, info = env.step(a)
        img = env._get_viewer('human')._read_pixels_as_in_window()
        img = resize_img(img)
    close(env)

    #Stack as sequence
    imgs = np.stack(imgs, axis=0)
    configs = np.stack(configs, axis=0)
    actions = np.stack(actions, axis=0)
    
    #Shuffle sequence
    inds = np.arange(configs.shape[0])
    np.random.shuffle(inds)
    imgs = imgs[inds]
    configs = configs[inds]
    actions = actions[inds]

    #Split to batches
    total = (imgs.shape[0]//args.batch_size) * args.batch_size
    imgs = np.split(imgs[:total], args.batch_size)
    configs = np.split(configs[:total], args.batch_size)
    actions = np.split(actions[:total], args.batch_size)

    #Transform each batch to tensor
    for i in range(len(imgs)):
        imgs[i] = torch.from_numpy(imgs[i]).to(dtype=torch.float)
        configs[i] = torch.from_numpy(configs[i]).to(dtype=torch.float)
        actions[i] = torch.from_numpy(actions[i]).to(dtype=torch.float)

    return imgs, configs, actions

def resize_img(img):
    im_pil = Image.fromarray(img)
    
    # Crop the center of the image
    new_width, new_height = 900, 900
    width, height = im_pil.size   # Get dimensions
    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2
    im_pil = im_pil.crop((left, top, right, bottom))

    # Resize the img
    im_pil = im_pil.resize((200, 200), Image.ANTIALIAS)
    img = np.array(im_pil)
    # plt.imshow(img)
    # plt.show()
    return np.transpose(img, (2, 0, 1))

def load_model(model, model_name):
    if os.path.isfile(model_name):
        print("=> loading checkpoint... ")
        checkpoint = torch.load(model_name)
        model.load_state_dict(checkpoint)
        print("done !")
    else:
        print("no checkpoint found...")

def save_model(model, output_folder, name):
    if output_folder is not None:
        filename = os.path.join(output_folder, name)
        with open(filename, 'wb') as f:
            state_dict = model.state_dict()
            torch.save(state_dict, f)
        print("Model saved")