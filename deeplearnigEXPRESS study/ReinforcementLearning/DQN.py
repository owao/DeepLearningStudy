import gym
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense

#게임 생성
env = gym.make('FrozenLake-v1', is_slippery=False)

discount_factor = 0.95
epsilon = 0.5
epsilon_decay_factor = 0.999
num_episodes = 500

#모델 구축(회귀)
model = Sequential()
model.add(InputLayer(batch_input_shape=(1, env.observation_space.n)))
model.add(Dense(20, activation='relu'))
model.add(Dense(env.action_space.n, activation='linear'))
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

#원-핫 벡터 반환 함수
def one_hot(state):
    state_m = np.zeros((1, env.observation_space.n))
    state_m[0][state] = 1
    return state_m

#DQN 구현
for i in range(num_episodes):
    state = env.reset()
    epsilon *= epsilon_decay_factor
    done = False

    while not done:
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(model.predict(one_hot(state)))
        
        new_state, reward, done, _ = env.step(action)
        target = reward + discount_factor * np.max(model.predict(one_hot(new_state)))
        target_vector = model.predict(one_hot(state))[0]
        target_vector[action] = target

        model.fit(one_hot(state), target_vector.reshape(-1, env.action_space.n), epochs=1, verbose=0)

        state = new_state
        print(i)

        if i==(num_episodes-1):
            env.render()
