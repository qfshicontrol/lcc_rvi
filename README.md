# lcc_rvi
Supplementary animations for the manuscript "Delay-Aware Dynamics Modeling and Data-Driven Optimal Control of Mixed Vehicle Platoons" submitted to IEEE Internet of Things Journal. Visualizes platoon top-down dynamics and real-time velocity profiles across various algorithms, traffic scenarios, and communication scales.


# Supplementary Material for "Delay-Aware Dynamics Modeling and Data-Driven Optimal Control of Mixed Vehicle Platoons"

## Overview
This repository contains supplementary animation videos demonstrating the car-following behavior and velocity perturbation dissipation of mixed vehicle platoons. 

The animations illustrate a top-down view for positional profiles ($p(t)$) paired with real-time velocity profiles ($v(t)$). The files are organized below exactly following the sequence of the simulation experiments presented in the manuscript.

---

## Simulation Part 1: Performance Comparison Among Different Control Methods (Fixed Scale)
In this section, the communication topology is fixed to **5 connected vehicles** ($m=n=2$). The control performance of different algorithms is evaluated and compared across three diverse traffic scenarios: Emergency Braking (EB), Finite-duration Sinusoid (FS), and NGSIM Trajectory.

### 1. Emergency Braking (EB) Scenario
* `animation_acc_eb.mp4` (ACC baseline)  https://github.com/user-attachments/assets/24f4e462-84ec-4bf6-b7f8-ca2b201518dc
* `animation_lcc_original_5connected_eb.mp4` (Original LCC method [Wang et al., 2021])  https://github.com/user-attachments/assets/ff61a788-bc21-4873-b648-0f3cb92d0fb6
* `animation_lcc_pi_zsg_5connected_eb.mp4` (Zero-sum game policy iteration-based LCC [Li et al., 2023])  https://github.com/user-attachments/assets/0377913d-20ce-452e-b579-5c5cc7ccc163
* `animation_lcc_ql_zsg_5connected_eb.mp4` (Zero-sum game Q-learning-based LCC [Liu et al., 2023])  https://github.com/user-attachments/assets/ef7f7c27-ff03-4dec-9ad2-93e94ed6f511
* `animation_lcc_rvi_5connected_eb.mp4` (**Proposed** RVI-based LCC)  https://github.com/user-attachments/assets/a37192be-94ea-48db-b5a3-7a8bb9d28343

### 2. Finite-duration Sinusoid (FS) Scenario
* `animation_acc_fs.mp4` (ACC baseline)  https://github.com/user-attachments/assets/800f9ecd-1db7-45f4-871f-4a043c0fe16a
* `animation_lcc_original_5connected_fs.mp4` (Original LCC method)  https://github.com/user-attachments/assets/680b4e5b-f57b-4824-ad33-2bc3a6b652bc
* `animation_lcc_pi_zsg_5connected_fs.mp4` (Zero-sum game policy iteration-based LCC)  https://github.com/user-attachments/assets/bbd4f17d-ad7e-4469-a212-2b4ae7a96d69
* `animation_lcc_ql_zsg_5connected_fs.mp4` (Zero-sum game Q-learning-based LCC)  https://github.com/user-attachments/assets/61798c9a-aa12-48f9-9054-7b824ef571e1
* `animation_lcc_rvi_5connected_fs.mp4` (**Proposed** RVI-based LCC)  https://github.com/user-attachments/assets/97975879-0f86-45ab-9943-4b559f919408

### 3. NGSIM Scenario
* `animation_acc_ngsim.mp4` (ACC baseline)  https://github.com/user-attachments/assets/cb2f2011-4616-4ca4-82b4-8b81a151873f
* `animation_lcc_original_5connected_ngsim.mp4` (Original LCC method)  https://github.com/user-attachments/assets/c97a598d-fc2d-49d7-97df-8ee6b495bc43
* `animation_lcc_pi_zsg_5connected_ngsim.mp4` (Zero-sum game policy iteration-based LCC)  https://github.com/user-attachments/assets/bb29a2c4-2d3f-4f69-a865-78032eb815b7
* `animation_lcc_ql_zsg_5connected_ngsim.mp4` (Zero-sum game Q-learning-based LCC)  https://github.com/user-attachments/assets/9def4186-4b73-43dc-a540-ad2403d68054
* `animation_lcc_rvi_5connected_ngsim.mp4` (**Proposed** RVI-based LCC)  https://github.com/user-attachments/assets/7769dbd1-5ed1-42df-b437-4bed5a8a0ddd

---

## Simulation Part 2: Impact of Communication Scales (Proposed RVI-based LCC)
In this section, the control strategy is fixed to the **Proposed RVI-based LCC** and the testing environment is fixed to the **Emergency Braking (EB)** scenario. This simulation investigates the platoon control performance under four different V2V communication scales ($m=n \in \{1, 2, 3, 4\}$):

* `animation_lcc_rvi_3connected_eb.mp4`: Scale $m=n=1$ (3 connected vehicles total)  https://github.com/user-attachments/assets/8840eb77-c6aa-43a0-8994-065ea490270d
* `animation_lcc_rvi_5connected_eb.mp4`: Scale $m=n=2$ (5 connected vehicles total) *(Cross-referenced from Part 1)*  https://github.com/user-attachments/assets/a37192be-94ea-48db-b5a3-7a8bb9d28343
* `animation_lcc_rvi_7connected_eb.mp4`: Scale $m=n=3$ (7 connected vehicles total)  https://github.com/user-attachments/assets/56272855-8188-444c-bfa8-4786b520375a
* `animation_lcc_rvi_9connected_eb.mp4`: Scale $m=n=4$ (9 connected vehicles total)  https://github.com/user-attachments/assets/24d291f7-dc66-4185-a938-5184a3a44640

---

## Visual Guide (Legend)
Vehicles in the top-down platoon animation are color-coded to denote their respective sensing and communication roles:
* **CAV (Blue):** The central Connected and Automated Vehicle executing the control algorithm.
* **CHDV (Green):** Connected Human-Driven Vehicles that actively transmit data to the CAV.
* **UHDV (Grey):** Unconnected Human-Driven Vehicles (states are implicitly handled or observed by the LCC system).
* **HV (Purple):** The Head Vehicle leading the platoon and initiating traffic perturbations.

## System Requirements
All files are encoded in standard H.264 `.mp4` format. They can be played natively on standard media players (e.g., VLC, Windows Media Player, QuickTime) or within modern web browsers.

