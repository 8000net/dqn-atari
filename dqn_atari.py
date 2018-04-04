import math
import random

from keras.layers import Input, Conv2D, Dense, Flatten
from keras.models import Model
import numpy as np
import gym
from scipy.misc import imresize

NUM_EPISODES = 20
RENDER = True

class DQNAgent:
    def __init__(self, input_shape, n_outputs,
                       max_epsilon=1.0, min_epsilon=0.1, gamma=0.99,
                       lambda_=0.001, mem_size=5e5,
                       batch_size=32, update_target_freq=1000):

        self.input_shape = input_shape
        self.n_outputs = n_outputs

        # network we are training
        self.nn = self._create_nn()

        # network we are using to predict targets in Q calculation
        self.nn_ = self._create_nn()

        self.memory = []

        self.mem_size = mem_size
        self.epsilon = max_epsilon
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.gamma = gamma
        self.lambda_ = lambda_
        self.batch_size = batch_size
        self.update_target_freq = update_target_freq

        self.episode = 0
        self.steps = 0

    def _create_nn(self):
        inputs = Input(shape=self.input_shape)
        net = Conv2D(16, 8, strides=4, activation='relu')(inputs)
        net = Conv2D(32, 4, strides=2, activation='relu')(net)
        net = Flatten()(net)
        net = Dense(256, activation='relu')(net)
        outputs = Dense(self.n_outputs, activation='linear')(net)

        model = Model(inputs, outputs)
        model.compile('rmsprop', 'mse')
        return model

    def _get_batch(self):
        n = min(len(self.memory), self.batch_size)
        return random.sample(self.memory, n)

    def _decrease_epsilon(self):
        self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon)\
                       * math.exp(-self.lambda_ * self.steps)

    def _update_target_nn(self):
        self.nn_.set_weights(self.nn.get_weights())

    def action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.n_outputs - 1)
        else:
            if state.ndim == 3:
                state = np.expand_dims(state, axis=0)
            return np.argmax(self.nn.predict(state))

    def observe(self, sample):
        """
        sample should consist of state, action, reward, state_
        """
        if len(self.memory) > self.mem_size:
            self.memory.pop(0)
        self.memory.append(sample)

        self.steps += 1
        self._decrease_epsilon()

        if self.steps % self.update_target_freq == 0:
            self._update_target_nn()


    def replay(self):
        batch = self._get_batch()

        X = []
        y = []

        for s, a, r, s_ in batch:
            X.append(s)
            y.append(self.Q(s).ravel())

            if s_ is None:
                y[-1][a] = r
            else:
                y[-1][a] = r + self.gamma*np.amax(self.Q(s_, target=True))

        X = np.array(X)
        y = np.array(y)
        self.nn.fit(X, y, epochs=1, verbose=0)


    def Q(self, state, target=False):
        if state.ndim == 3:
            state = np.expand_dims(state, axis=0)

        if target:
            return self.nn_.predict(state)
        else:
            return self.nn.predict(state)


def to_gray(rgb):
    gray = np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
    return gray


def process(frame):
    frame = imresize(frame, (110, 84, 3))
    frame = to_gray(frame)
    frame = frame[20:104]
    return frame



env = gym.make('Breakout-v0')
"""
Actions in Breakout:
    0 - Noop
    1 - Fire (start game)
    2 - Right
    3 - Left

State: 210x160x3 Image
"""
dqn_agent = DQNAgent((84, 84, 4), 4)

frames = []
for ep_i in range(NUM_EPISODES):
    frame = env.reset()
    frame = process(frame)

    # Fill frame history with initial state
    frames += [frame, frame, frame, frame]
    state = np.stack(frames, axis=2)

    done = False
    while not done:
        action = dqn_agent.action(state)
        frame, reward, done, info = env.step(action)

        if RENDER:
            env.render()

        frame = process(frame)
        frames.pop()
        frames.append(frame)
        state_ = np.stack(frames, axis=2)

        dqn_agent.observe((state, action, reward, state_))
        dqn_agent.replay()

        state = state_

    print('Episode %d finished' % ep_i)
