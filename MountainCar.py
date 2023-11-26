import gym
from tqdm import tqdm
# 환경 생성
env = gym.make("MountainCar-v0")

## 환경 초기화
observation = env.reset()

# 에피소드당 총 보상을 저장할 리스트
total_rewards = []

# 행동 공간 출력
print("Action Space:", env.action_space)
# 상태 공간 출력
print("Observation Space:", env.observation_space)
epoch = 500
iter = 3000
for i_episode in tqdm(range(epoch)):
    # 환경 초기화
    observation = env.reset()

    # 에피소드당 총 보상 초기화
    total_reward = 0

    for t in range(iter):
        
        # 환경을 시각화
        env.render()

        # 무작위로 행동을 선택
        action = env.action_space.sample() 

        # 행동을 실행하고 다음 상태, 보상, 종료 여부, 추가 정보를 얻음
        observation, reward, done, _, _ = env.step(action)

        # 총 보상 업데이트
        total_reward += reward

        # 에피소드가 종료되었다면 환경을 다시 초기화
        if done:
            break
    
    # 에피소드당 총 보상 저장
    total_rewards.append(total_reward)


import matplotlib.pyplot as plt

plt.plot(total_rewards)
plt.title('Total rewards per episode')
plt.xlabel('Episode')
plt.ylabel('Total reward')
plt.show()

# 환경 종료
env.close()
