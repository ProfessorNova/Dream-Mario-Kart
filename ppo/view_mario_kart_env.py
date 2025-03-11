import time

import cv2

from lib.utils import make_mario_kart_env


def main():
    env = make_mario_kart_env()
    done = True
    frame_rate = 10
    time_per_frame = 1.0 / frame_rate

    try:
        for step in range(5000):
            start_time = time.time()

            if done:
                env.reset()

            action = env.action_space.sample()
            _, _, done, _ = env.step(action)
            frame = env.render(mode="rgb_array")
            # Convert the frame to BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            # Transform the frame to a larger size
            frame = cv2.resize(frame, (256 * 4, 224 * 4), interpolation=cv2.INTER_NEAREST)
            cv2.imshow("Environment", frame)
            cv2.waitKey(1)

            # Calculate the time taken for the frame and sleep for the remaining time
            elapsed_time = time.time() - start_time
            time.sleep(max(0, time_per_frame - elapsed_time))

    finally:
        env.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
