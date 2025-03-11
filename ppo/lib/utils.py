import cv2
import gym
import numpy as np
import retro
import torch
from gym.wrappers import FrameStack
from torch.distributions import Categorical

SKIP_FRAME = 8


def log_video(env, agent, device, video_path, fps=30):
    """
    Log a video of one episode of the agent playing in the environment.
    :param env: a test environment which supports video recording and doesn't conflict with the other environments.
    :param agent: the agent to record.
    :param device: the device to run the agent on.
    :param video_path: the path to save the video.
    :param fps: the frames per second of the video.
    """
    frames = []
    next_obs = torch.tensor(np.array([env.reset()]), dtype=torch.float32, device=device)
    done = False
    frame_counter = 0  # To skip frames
    while not done and len(frames) < 512 * SKIP_FRAME:
        # Render the current frame
        frame = env.render(mode='rgb_array').copy()
        frames.append(frame)

        if frame_counter % SKIP_FRAME == 0:
            obs = next_obs
            # Get action from the agent
            with torch.no_grad():
                action_logits, _ = agent(obs)
                dist = Categorical(logits=action_logits)
                action = dist.sample().item()

        # Step the environment
        next_obs, _, done, _ = env.step(action)
        next_obs = torch.tensor(np.array([next_obs]), dtype=torch.float32, device=device)
        frame_counter += 1

    # Save the video
    height, width, _ = frames[0].shape
    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    for frame in frames:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    out.release()


def make_mario_kart_env(render_env=False):
    env = retro.make("SuperMarioKart-Snes")
    env = ActionWrapper(env)
    env = FrameStack(env, num_stack=3)
    if not render_env:
        env = SkipEnv(env, skip=SKIP_FRAME)
        # env = TimeLimit(env, max_episode_steps=256)
    env.reset()
    return env


class SkipEnv(gym.Wrapper):
    """
    This is useful because processing every frame with an NN is quite expensive.
    The number of frames to skip is usually 4 or 3.
    """

    def __init__(self, env=None, skip=4):
        super(SkipEnv, self).__init__(env)
        self._skip = skip

    def step(self, action):
        obs = None
        total_reward = 0.0
        done = None
        info = None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return obs


class ActionWrapper(gym.ActionWrapper):
    """
    Gym ActionWrapper to map a discrete action (0-5) into a 12-length binary action list.

    Discrete Actions:
        0: No-op (no button pressed)
        1: Forward
        2: Left
        3: Right
        4: Forward and Left
        5: Forward and Right
    """

    def __init__(self, env):
        super(ActionWrapper, self).__init__(env)
        # Define the new discrete action space with 6 actions.
        self.action_space = gym.spaces.Discrete(6)

    def action(self, act):
        # Create a default action list (all buttons off)
        mapped_action = [0] * 12

        if act == 1:  # Forward
            mapped_action[0] = 1
        elif act == 2:  # Left
            mapped_action[6] = 1
        elif act == 3:  # Right
            mapped_action[7] = 1
        elif act == 4:  # Forward and Left
            mapped_action[0] = 1
            mapped_action[6] = 1
        elif act == 5:  # Forward and Right
            mapped_action[0] = 1
            mapped_action[7] = 1
        # Action 0 is implicitly a no-op (all zeros)

        return mapped_action
