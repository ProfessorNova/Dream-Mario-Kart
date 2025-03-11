import torch


def cosine_diffusion_schedule(diffusion_times, device):
    min_signal_rate = 0.02
    max_signal_rate = 0.95
    start_angle = torch.acos(torch.tensor(max_signal_rate, device=device))
    end_angle = torch.acos(torch.tensor(min_signal_rate, device=device))

    diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)

    signal_rates = torch.cos(diffusion_angles)
    noise_rates = torch.sin(diffusion_angles)

    return noise_rates, signal_rates


def denoise(model, noisy_images, noise_rates, signal_rates, previous_frames, previous_actions):
    predicted_noises = model(noisy_images, noise_rates ** 2, previous_frames, previous_actions)
    predicted_images = (noisy_images - noise_rates * predicted_noises) / signal_rates
    return predicted_noises, predicted_images


def reverse_diffusion(model, initial_noise, diffusion_steps, previous_frames, previous_actions, device):
    pred_images = None
    model.eval()
    num_images = initial_noise.shape[0]
    step_size = 1.0 / diffusion_steps

    # At the first sampling step, the "noisy image" is pure noise
    # but its signal rate is assumed to be nonzero (min_signal_rate)
    next_noisy_images = initial_noise
    for step in range(diffusion_steps):
        noisy_images = next_noisy_images

        # Separate the current noisy image into its components
        diffusion_times = torch.ones((num_images, 1, 1, 1), device=device) - step * step_size
        noise_rates, signal_rates = cosine_diffusion_schedule(diffusion_times, device)
        pred_noises, pred_images = denoise(model, noisy_images, noise_rates, signal_rates, previous_frames,
                                           previous_actions)

        # Remix the predicted components using the next signal and noise rates
        next_diffusion_time = diffusion_times - step_size
        next_noise_rates, next_signal_rates = cosine_diffusion_schedule(next_diffusion_time, device)
        next_noisy_images = next_signal_rates * pred_images + next_noise_rates * pred_noises

    return pred_images


@torch.no_grad()
def generate(model, num_images, diffusion_steps, image_shape, previous_frames, previous_actions, device):
    # Noise -> images -> denormalized images
    initial_noise = torch.randn(num_images, *image_shape, device=device)
    generated_images = reverse_diffusion(model, initial_noise, diffusion_steps, previous_frames, previous_actions,
                                         device)
    return generated_images
