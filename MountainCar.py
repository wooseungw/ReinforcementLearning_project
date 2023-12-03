import gym
import numpy as np
import matplotlib.pyplot as plt

# 환경 생성
env = gym.make("MountainCar-v0")

# Q-value 초기화
num_states = 40  # 위치와 속도에 대한 상태를 이산화
num_actions = env.action_space.n
q_table = np.zeros((num_states, num_states, num_actions))

# 하이퍼파라미터
learning_rate = 0.1
discount_factor = 0.99
epsilon = 0.1
num_episodes = 10000

# 이산화 함수
def discretize_state(state):
    position, velocity = state
    discrete_position = int((position + 1.2) / 1.8 * num_states)
    discrete_velocity = int((velocity + 0.07) / 0.14 * num_states)
    return discrete_position, discrete_velocity

# 탐험 또는 활용을 기반으로 행동 선택
def select_action(state):
    if np.random.rand() < epsilon:
        return np.random.choice(num_actions)  # 무작위 탐험
    else:
        return np.argmax(q_table[state])

# Q-learning 알고리즘
for episode in range(num_episodes):
    state = discretize_state(env.reset()[0])  # 수정된 부분

    total_reward = 0

    while True:
        # 행동 선택
        action = select_action(state)

        # 선택한 행동을 실행하고 다음 상태, 보상, 종료 여부 얻기
        result = env.step(action)
        next_state, reward, done, _ , _2= result
        next_state = discretize_state(next_state)

        # Q-value 업데이트
        q_table[state][action] += learning_rate * (
            reward + discount_factor * np.max(q_table[next_state]) - q_table[state][action]
        )

        total_reward += reward
        state = next_state

        if done:
            break

    if episode % 100 == 0:
        print(f"Episode: {episode}, Total Reward: {total_reward}")

# 학습된 정책 평가
total_rewards = []
for _ in range(10):
    state = discretize_state(env.reset()[0])  # 수정된 부분
    total_reward = 0


    while True:
        action = np.argmax(q_table[state])
        next_state, reward, done, _, _ = env.step(action)
        next_state = discretize_state(next_state)

        total_reward += reward
        state = next_state

        if done:
            break

    total_rewards.append(total_reward)

# 결과 시각화
plt.plot(total_rewards)
plt.title('Total rewards per episode (Evaluation)')
plt.xlabel('Episode')
plt.ylabel('Total reward')
plt.show()

# 환경 종료
env.close()
