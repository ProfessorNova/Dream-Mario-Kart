import argparse
import datetime
import os
import time

import gym.vector
import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.distributions import Categorical
from tqdm import tqdm

from lib.agent import Agent
from lib.buffer import Buffer
from lib.utils import make_mario_kart_env, log_video


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=True if torch.cuda.is_available() else False, action="store_true",
                        help="Use CUDA")
    parser.add_argument("--n-envs", type=int, default=16, help="Number of environments")
    parser.add_argument("--n-epochs", type=int, default=100, help="Number of epochs to run")
    parser.add_argument("--n-steps", type=int, default=512, help="Number of steps per epoch per environment")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size")
    parser.add_argument("--train-iters", type=int, default=20, help="Number of training iterations")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="Lambda for GAE")
    parser.add_argument("--clip-ratio", type=float, default=0.1, help="PPO clip ratio")
    parser.add_argument("--ent-coef", type=float, default=0.01, help="Entropy coefficient")
    parser.add_argument("--vf-coef", type=float, default=0.5, help="Value function coefficient")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--checkpoint-model", type=str, default=None,
                        help="Path to a model checkpoint")
    parser.add_argument("--recording-start", type=int, default=0,
                        help="Start recording after this epoch")
    parser.add_argument("--recording-end", type=int, default=100,
                        help="End recording after this epoch")
    assert 0 <= parser.parse_args().recording_start <= parser.parse_args().recording_end
    assert parser.parse_args().recording_start <= parser.parse_args().n_epochs
    return parser.parse_args()


def ppo_update(agent, optimizer, batch_obs, batch_actions, batch_returns, batch_old_log_probs, batch_adv, clip_epsilon,
               vf_coef, ent_coef):
    agent.train()
    action_logits, values = agent(batch_obs)
    dist = Categorical(logits=action_logits)
    new_log_probs = dist.log_prob(batch_actions)
    ratio = torch.exp(new_log_probs - batch_old_log_probs)
    # Normalize advantages
    batch_adv = (batch_adv - batch_adv.mean()) / (batch_adv.std() + 1e-8)

    surr1 = ratio * batch_adv
    surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * batch_adv
    policy_loss = -torch.min(surr1, surr2).mean()
    value_loss = nn.MSELoss()(values.squeeze(1), batch_returns)
    entropy = dist.entropy().mean()
    loss = policy_loss + vf_coef * value_loss - ent_coef * entropy
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
    optimizer.step()
    return loss.item(), policy_loss.item(), value_loss.item(), entropy.item()


def main():
    args = parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")
    print(f"Using device {device}")

    # create folders for logging
    current_dir = os.path.dirname(__file__)
    folder_name = f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    videos_dir = os.path.join(current_dir, "videos", folder_name)
    os.makedirs(videos_dir, exist_ok=True)
    checkpoint_dir = os.path.join(current_dir, "checkpoints", folder_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # create tensorboard writer
    log_dir = os.path.join(current_dir, "logs", folder_name)
    writer = SummaryWriter(log_dir)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # create environments
    envs = gym.vector.AsyncVectorEnv([lambda: make_mario_kart_env() for _ in range(args.n_envs)])
    obs_dim = envs.single_observation_space.shape
    num_actions = envs.single_action_space.n

    # create independent environments for rendering
    render_env = make_mario_kart_env(render_env=True)

    # create agent
    # We stack frames, so we need to multiply the number of channels by the number of frames
    agent = Agent(input_shape=(obs_dim[1], obs_dim[2], obs_dim[3] * obs_dim[0]), num_actions=num_actions).to(device)
    if args.checkpoint_model is not None:
        agent.load_state_dict(torch.load(args.checkpoint_model, map_location=device, weights_only=True))
        print(f"Model loaded from {args.checkpoint_model}")
    optimizer = torch.optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    print(agent)

    # create the buffer
    buffer = Buffer(obs_dim, args.n_steps, args.n_envs, device, args.gamma, args.gae_lambda)

    # training loop
    best_mean_reward = -np.inf
    global_step_idx = 0
    start_time = time.time()
    obs = None
    next_obs = torch.tensor(np.array(envs.reset()), dtype=torch.uint8, device=device)
    next_terminateds = torch.zeros(args.n_envs, dtype=torch.float32, device=device)
    next_truncateds = torch.zeros(args.n_envs, dtype=torch.float32, device=device)
    reward_list = []

    # # gather trajectories for the later diffusion model
    saved_trajectories = (np.array([]), np.array([]))  # (frames, actions)

    for epoch in range(1, args.n_epochs + 1):
        for _ in tqdm(range(args.n_steps), desc="Gathering trajectories"):
            global_step_idx += args.n_envs
            obs = next_obs
            terminateds = next_terminateds
            truncateds = next_truncateds

            # Sample actions
            with torch.no_grad():
                action_logits, values = agent(obs)
                dist = Categorical(logits=action_logits)
                actions = dist.sample()
                log_probs = dist.log_prob(actions)
                values = values.reshape(-1)

            # Step the environments
            next_obs, rewards, next_terminateds, next_infos = envs.step(actions.cpu().numpy())
            # Parse everything to tensors
            next_obs = torch.tensor(np.array(next_obs, dtype=np.uint8), device=device)
            reward_list.extend(rewards)
            rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
            # Scale the rewards
            rewards /= 100.0
            # next_truncateds will be determined by TimeLimit.truncated in the info
            next_truncateds = torch.as_tensor([info.get("TimeLimit.truncated", False) for info in next_infos],
                                              dtype=torch.float32, device=device)
            next_terminateds = torch.as_tensor(next_terminateds, dtype=torch.float32, device=device)
            next_terminateds = next_terminateds * (1.0 - next_truncateds)

            # Store the transition
            buffer.store(obs, actions, rewards, values, terminateds, truncateds, log_probs)

        # After the trajectories are gathered, calculate the advantages
        with torch.no_grad():
            next_values = agent(obs)[1].reshape(1, -1)
            next_terminateds = next_terminateds.reshape(1, -1)
            next_truncateds = next_truncateds.reshape(1, -1)
            traj_adv, traj_ret = buffer.calculate_advantages(next_values, next_terminateds, next_truncateds)

        # Get the stored trajectories from the buffer
        traj_obs, traj_act, traj_val, traj_log_probs, _, traj_trunc = buffer.get()

        # Save trajectories
        if args.recording_start < epoch <= args.recording_end:
            for environment in range(args.n_envs):
                trajectory_obs_save = traj_obs[:, environment, -1, :, :, :].permute(0, 3, 1, 2)

                # Save the first trajectory
                if saved_trajectories[0].size == 0:
                    saved_trajectories = (
                        np.array(trajectory_obs_save.unsqueeze(0).cpu().numpy(), dtype=np.uint8),
                        np.array(traj_act[:, environment].unsqueeze(0).cpu().numpy(), dtype=np.int64),
                    )
                else:
                    saved_trajectories = (
                        np.append(saved_trajectories[0], trajectory_obs_save.unsqueeze(0).cpu().numpy(), axis=0),
                        np.append(saved_trajectories[1], traj_act[:, environment].unsqueeze(0).cpu().numpy(), axis=0),
                    )

                # Take only the trajectory of the first environment to have a wide variety of play styles
                break

        # Flatten the trajectories
        traj_obs = traj_obs.view(-1, *obs_dim)
        traj_act = traj_act.view(-1)
        traj_log_probs = traj_log_probs.view(-1)
        traj_adv = traj_adv.view(-1)
        traj_ret = traj_ret.view(-1)

        # Train the agent
        # Shuffle the trajectories
        dataset_size = traj_obs.size(0)
        traj_indices = np.arange(dataset_size)

        sum_loss_policy = 0.0
        sum_loss_value = 0.0
        sum_loss_entropy = 0.0
        sum_loss_total = 0.0
        for _ in tqdm(range(args.train_iters), desc="PPO training"):
            np.random.shuffle(traj_indices)
            for start_idx in range(0, dataset_size, args.batch_size):
                end_idx = start_idx + args.batch_size
                batch_indices = traj_indices[start_idx:end_idx]

                batch_obs = traj_obs[batch_indices]
                batch_actions = traj_act[batch_indices]
                batch_returns = traj_ret[batch_indices]
                batch_old_log_probs = traj_log_probs[batch_indices]
                batch_adv = traj_adv[batch_indices]

                loss, loss_policy, loss_value, loss_entropy = ppo_update(agent, optimizer, batch_obs, batch_actions,
                                                                         batch_returns, batch_old_log_probs, batch_adv,
                                                                         args.clip_ratio, args.vf_coef, args.ent_coef)
                sum_loss_policy += loss_policy
                sum_loss_value += loss_value
                sum_loss_entropy += loss_entropy
                sum_loss_total += loss

        writer.add_scalar("loss/total", sum_loss_total / args.train_iters / (dataset_size / args.batch_size), epoch)
        writer.add_scalar("loss/policy", sum_loss_policy / args.train_iters / (dataset_size / args.batch_size), epoch)
        writer.add_scalar("loss/value", sum_loss_value / args.train_iters / (dataset_size / args.batch_size), epoch)
        writer.add_scalar("loss/entropy", sum_loss_entropy / args.train_iters / (dataset_size / args.batch_size), epoch)

        # Log the rewards
        mean_reward = float(np.mean(reward_list))
        writer.add_scalar("reward/mean", mean_reward, epoch)
        reward_list = []
        print(f"Epoch {epoch} done in {time.time() - start_time:.2f}s, mean reward: {mean_reward:.2f}")
        start_time = time.time()

        # Save the model if the mean reward is better
        if mean_reward > best_mean_reward:
            best_mean_reward = mean_reward
            torch.save(agent.state_dict(), f"{checkpoint_dir}/best.pth")
            print(f"New best model saved with mean reward: {mean_reward:.2f}")
        # Save the model
        torch.save(agent.state_dict(), f"{checkpoint_dir}/last.pth")

        # Log a video of the agent playing every 10 epochs
        if epoch % 10 == 0:
            agent.eval()
            video_path = f"{videos_dir}/epoch_{epoch}.mp4"
            log_video(render_env, agent, device, video_path)

    # Save the trajectories for the diffusion model
    file_path = f"{checkpoint_dir}/trajectories.npz"
    np.savez(file_path, frames=saved_trajectories[0], actions=saved_trajectories[1])

    # Close the environments
    envs.close()
    render_env.close()
    writer.close()


if __name__ == "__main__":
    main()
