import random
from collections import deque, defaultdict

import numpy as np

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPooling2D, Input

import matplotlib.pyplot as plt

import gym
from gym import spaces

'''
순차적으로 들어오는 데이터 학습시 correlation 문제와 데이터의 재사용을 위해 replay buffer를 사용함
'''
class ReplayBuffer:
    # 초기화
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=self.buffer_size)
    
    # 샘플링
    def sample(self, batch_size):
        size = batch_size if len(self.buffer) > batch_size else len(self.buffer)
        return random.sample(self.buffer, size)

    # reset
    def clear(self):
        self.buffer.clear()
    
    # 추가
    def append(self, transition):
        self.buffer.append(transition)

    # 길이 체크
    def __len__(self):
        return len(self.buffer)

# q net layer
class QNetworkDense(tf.keras.Model):
    def __init__(self, obs_dim, acs_dim):
        super(QNetworkDense, self).__init__()
        self.layer1 = Dense(24, activation='relu', kernel_initializer='he_uniform')
        self.layer2 = Dense(24, activation='relu', kernel_initializer='he_uniform')
        self.output_layer = Dense(acs_dim, activation='linear', kernel_initializer='he_uniform')
        self.build(input_shape=(None,) + obs_dim)

    def call(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return self.output_layer(x)
        
# agent : dqn agent, env, hyper parameter를 받아서 Agent가 동작함
class Agent(object):
    def __init__(self, env, obs_dim, acs_dim, steps, 
                gamma=0.99, epsilon=1.0, epsilon_decay=0.999, 
                buffer_size=2000, batch_size=64, target_update_step=100):
        self.env = env
        self.obs_dim = obs_dim
        self.acs_dim = acs_dim
        self.steps = steps
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.buffer_size = buffer_size
        self.target_update_step = target_update_step
        self.batch_size = batch_size

        self.replay_buffer = ReplayBuffer(self.buffer_size)
        
        self.q_fn = QNetworkDense(self.obs_dim, self.acs_dim)
        self.q_fn_target = QNetworkDense(self.obs_dim, self.acs_dim)

        self.q_fn.compile(optimizer='adam', loss='mse')
        self.target_update()
        
        self.loss_fn = tf.keras.losses.MeanSquaredError()
        self.optimizer = tf.keras.optimizers.Adam()

    # transition 값을 받아서 러닝 시작
    @tf.function
    def learn(self, obs, acs, next_obs, rewards, dones):
        q_target = rewards + (1 - dones) * self.gamma * tf.reduce_max(self.q_fn_target(next_obs), axis=1, keepdims=True)

        with tf.GradientTape() as tape:
            qt = self.q_fn(obs)
            acs_onehot = tf.one_hot(tf.cast(tf.reshape(acs, [-1]), tf.int32), self.acs_dim)
            qt = tf.reduce_sum(qt * acs_onehot, axis=1, keepdims=True)
            loss = self.loss_fn(q_target, qt)
        grads = tape.gradient(loss, self.q_fn.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.q_fn.trainable_weights))

    # 미리 설정한 스텝만큼만 학습진행하고 gym env interface에 따라 매 스텝마다 ob, ac, next_ob, reward, done의 transition을 버퍼에 넣어주고 replay buffer에 어느정도 쌓이면 학습을 시작한다.
    def train(self):
        epochs = 0
        global_step = 0
        reward_list = []
        while global_step < self.steps:
            ob = env.reset()
            rewards = 0
            while True:
                ac = self.select_action(ob)
                next_ob, reward, done, _ = env.step(ac)

                transition = (ob, ac, next_ob, reward, done)
                self.replay_buffer.append(transition)

                rewards += reward
                ob = next_ob

                if done:
                    reward_list.append(rewards)
                    self.target_update()
                    print(f"epochs #{epochs+1} end, score is {rewards}")
                    break

                self.env.render()

                if global_step > 1000:
                    transitions = self.replay_buffer.sample(batch_size=self.batch_size)
                    self.learn(*map(lambda x: np.vstack(x).astype('float32'), np.transpose(transitions)))

                global_step += 1
            epochs += 1
        # 결과 그래프 표시
        plt.title('Cartpole-v1')
        plt.xlabel('epoch')
        plt.ylabel('rewards')
        plt.plot(reward_list)
        plt.show()


    def test(self):
        pass

    # td error 계산시 학습하는 q값이 같으면 td error가 많이 차이 나지 않아서 학습하는데 unstable해지므로 target network를 특정 주기마다 target network param을 업데이트 함.
    def target_update(self):
        self.q_fn_target.set_weights(self.q_fn.get_weights())

    # epsilon 확률 만큼 exploit, exploration 해주고 가장 큰 값을 선택하는 greedy 처리 / e-greedy
    def select_action(self, ob):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon, 0.01)
        if np.random.rand() <= self.epsilon: # exploration
            return np.random.randint(self.acs_dim)
        else:
            action = self.q_fn(ob[np.newaxis])
            return np.argmax(action[0])

# env 설정 후 실행
if __name__ == '__main__':
    tf.keras.backend.set_floatx('float32')
    env = gym.make('CartPole-v0')

    obs_dim = env.observation_space.shape
    acs_dim = None
    
    if isinstance(env.action_space, spaces.Box):
        acs_type = 'continuous'
        acs_dim = env.action_space.shape
    elif isinstance(env.action_space, spaces.Discrete):
        acs_type = 'discrete'
        acs_dim = env.action_space.n
    else:
        raise NotImplementedError('Not implemented ㅎㅎ')

    agent = Agent(env, obs_dim, acs_dim, 10000)
    agent.train()