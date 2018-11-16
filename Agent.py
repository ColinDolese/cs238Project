
import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout, Activation
from keras.optimizers import Adam
from keras import regularizers

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque()
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.001
        # self.epsilon_decay = 0.995
        self.epsilon_decay = 0.9
        self.learning_rate = 0.0001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        l2reg = .0001
        model = Sequential()
        model.add(Dense(50, kernel_regularizer=regularizers.l2(l2reg), input_dim=self.state_size))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dense(50, kernel_regularizer=regularizers.l2(l2reg)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dense(50, kernel_regularizer=regularizers.l2(l2reg)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dense(20, kernel_regularizer=regularizers.l2(l2reg)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        # model.add(Dense(100, input_dim=self.state_size, activation='relu'))
        # model.add(Dropout(.15))
        # model.add(Dense(50, activation='relu'))
        # model.add(Dropout(.15))
        # model.add(Dense(25, activation='relu'))
        # model.add(Dropout(.05))
        model.add(Dense(self.action_size, activation='softmax'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def short_replay(self, state, action, reward, next_state, done):
    	target = reward
    	if not done:
    		target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
    	target_f = self.model.predict(state)
    	target_f[0][action] = target
    	self.model.fit(state, target_f, epochs=1, verbose=0)

    def replay(self, batch_size):
    	if len(self.memory) > batch_size:
        	minibatch = random.sample(self.memory, batch_size)
        else:
        	minibatch = self.memory
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        # if self.epsilon > self.epsilon_min:
        #     self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


