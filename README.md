# Deep Reinforcement Learning method for Cassie
This project is about training the bipedal robot Cassie to walk around using the DRL method. The training is deployed on MJX (GPU Accelerated platform of MuJoCo) using BRAX as the training package.

The insight of this project is using a GPU-accelerated training method which significantly improved the training efficiency and reduced the hardware requirements. The initial training environment (hardware) is the RTX4060 Laptop, and it will only cost 8 hours of training for 100 million time steps.

# Installation and Training
Install [MJX-MuJoCo](https://github.com/google-deepmind/mujoco) and any associate packages as well as [BRAX](https://github.com/google/brax) (Suggest visiting [this page](https://github.com/jihan1218/brax) to add a load feature for BRAX), then start your training!

The initial simulation model of the Cassie robot is edited from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) by removing unnecessary contacts between the robot component and ground. But you are welcome to use your own!

For the training process, the `train.py` is completely commented on and easy to improve.

# Result
Currently, after 150 million timesteps of training, Cassie could do the following:
## Slow Gait


https://github.com/user-attachments/assets/4c68b231-fd3c-45af-8e79-b6a4d0635bcb


## Quick Gait


https://github.com/user-attachments/assets/72106bc4-0ab1-4839-9044-911cfdb13983


## Lateral Velocity Gait


https://github.com/user-attachments/assets/f2e4a691-c571-4c20-8a1b-f37f9c370cd5


## Multiple Velocities Gait


https://github.com/user-attachments/assets/e8f95806-9539-4252-a62c-478f79b1b39d


# Future

 - Gait stability at low velocity
 - More gaits (running, jumping)
 - Code structure (more professional)
 - Efficient reward terms (use of contact force or more complex reward terms)

