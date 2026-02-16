# 🏎️ Autonomous Racing Agent (PPO + CNN)

An autonomous driving agent trained to navigate the OpenAI Gym `CarRacing-v3` environment using Deep Reinforcement Learning. The agent processes raw pixel data (Computer Vision) to make steering, acceleration, and braking decisions in real-time.

## 🧠 Tech Stack
* **Language:** Python 3.10+
* **Frameworks:** Stable-Baselines3, Gymnasium (Box2D)
* **Algorithm:** Proximal Policy Optimization (PPO) with a CnnPolicy
* **Environment:** Google Colab (Training), Local Python (Inference)

## 🎥 Demo
*(Upload your "Good" video here later! For now, you can leave this blank)*

## 🚀 How It Works
1.  **Observation:** The agent receives a 96x96px RGB image of the track (top-down view).
2.  **Processing:** A Convolutional Neural Network (CNN) extracts features (road borders, curves, car position).
3.  **Action:** The PPO policy outputs a continuous action vector: `[Steering, Gas, Brake]`.
4.  **Reward:** The agent is rewarded for visiting new track tiles and penalized for driving off-track.

## 🛠️ Installation & Usage
To run the pre-trained model:

```bash
# 1. Clone the repository
git clone [https://github.com/Kushagra2503-del/autonomous-racing-agent.git](https://github.com/Kushagra2503-del/autonomous-racing-agent.git)
cd autonomous-racing-agent

# 2. Install dependencies
pip install gymnasium[box2d] stable-baselines3 shimmy moviepy

# 3. Run the agent
python test_driver.py
