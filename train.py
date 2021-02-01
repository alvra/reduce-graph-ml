import os
import random
import logging
import numpy as np
from pathlib import Path
from collections import deque

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential

from envs.reduce_graph import ReduceGraphEnv

log = tf.get_logger()
log.setLevel("INFO")
log.handlers = []
logging.basicConfig(level=logging.INFO,
                    format="[%(levelname)s] - %(asctime)s - %(name)s - %(pathname)s:%(lineno)d: %(message)s")


class DQN:
    def __init__(self, env):
        self.env = env
        self.memory = deque(maxlen=2000)

        self.gamma = 0.85
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.998
        self.learning_rate = 0.01
        self.optimizer = Adam(lr=self.learning_rate)
        self.learn_batch_size = 8
        self.tau = .125  # part of the learned parameters to transfer from active to final model

        self.model = self.create_model(from_saved=True)
        self.target_model = self.create_model(from_saved=True)

    @staticmethod
    def get_newest_model(filepath: [str, Path] = None) -> keras.Model:
        """
        filepath: Should be a path to the model file/folder
        returns: loaded keras model
        """
        if filepath:
            return keras.models.load_model(filepath)
        else:
            models_dir = Path(__file__).parent / "models"
            models = os.listdir(models_dir)
            models.remove(".keep")  # remove ".keep" file from list

            if not models:
                return
            newest_model = models_dir / sorted(models)[-1]
            log.info(f"Loaded model from : {newest_model}")
            return keras.models.load_model(newest_model)

    def create_model(self, from_saved: [bool, str, Path] = None) -> keras.Model:
        """
        from_saved: Either a pathlike to load an intermediate step, or a True to load the newest
        returns: loaded keras model
        """
        if from_saved is True:
            model = self.get_newest_model()
            if model:
                return model
        elif isinstance(str, from_saved) or isinstance(Path, from_saved):
            model = self.get_newest_model(from_saved)
            if model:
                return model

        model = Sequential()

        # TODO get a not "None" shape out of self.env.observation_space
        state_shape = self.env.observation_space['graph'].shape[0]
        model.add(Dense(512, input_dim=state_shape, activation="elu"))
        model.add(Dense(1024, activation="elu"))

        # 20 x 20 x 3 output, location dependent
        model.add(Dense(1200))  # TODO dynamic output based on output size
        model.compile(loss="mse",
                      optimizer=self.optimizer)
        return model

    @staticmethod
    def parse_action(raw_action: int) -> dict:
        """
        We assume that an action is of the shape (20, 20, 3) but flattened to an int of range [0, 1199]
        Now we pull it apart into the real action
            {node_1: [0-19]
             node_2: [0-19]
             action_type: [0-2]}
        """
        remainder, action_type = divmod(raw_action, 3)
        block_1, block_2 = divmod(remainder, 20)
        return {"action_type": action_type,
                "block_1": block_1,
                "block_2": block_2}

    @staticmethod
    def unparse_action(parsed_action: dict) -> int:
        """ Undo the parsing of the action. Returns the integer location of the value
        """
        return ((parsed_action["block_1"] * 20) + parsed_action["block_2"]) * 3 + parsed_action["action_type"]

    def act(self, state):
        """ Act out a single step of the simulation with a randomly sampled action of the model
            Sampling a random action makes sure that we cover output for the whole spectrum
                - Otherwise we might find a mode-collapse like state
        """
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        next_state = self.parse_action(np.argmax(self.model.predict(np.expand_dims(state, axis=0))[0]))
        return next_state

    def remember(self, state, action, reward, new_state, done):
        """ Add game state and model actions to learn from later.
        """
        self.memory.append([state, action, reward, new_state, done])

    def replay(self):
        """ Replay an action from the memory of actions.
            Train on output of this ~recent simulation step
        """
        if len(self.memory) < self.learn_batch_size:
            return

        # get random samples from memory to learn from, this reduces the change for catastrophic forgetting
        samples = random.sample(self.memory, self.learn_batch_size)
        for sample in samples:
            state, parsed_action, reward, new_state, done = sample
            target = self.target_model.predict(np.expand_dims(state, axis=0))
            if done:
                target[0][self.unparse_action(parsed_action)] = reward
            else:
                Q_future = max(self.target_model.predict(np.expand_dims(new_state, axis=0))[0])
                target[0][self.unparse_action(parsed_action)] = reward + Q_future * self.gamma
            self.model.fit(np.expand_dims(state, axis=0), target, epochs=1, verbose=0)

    def target_train(self):
        """ We train by fitting the active model "model"
            And update a part (tau) of the weights fo the final model "target model" after a training batch
        """
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.target_model.set_weights(target_weights)

    def save_model(self, filepath: [str, Path]):
        self.model.save(filepath)


def main():
    """
    Run training loop
    """
    env = ReduceGraphEnv(max_graph_size=20)
    dqn_agent = DQN(env=env)
    trials = 10000  # we probably need many trials to get a good model
    steps_per_trial = 50  # our simulation doesn't really run for that many steps

    for trial in range(trials):
        cur_state = env.reset()[()]['graph']
        for step in range(steps_per_trial):
            action = dqn_agent.act(cur_state)

            new_state, reward, done, _ = env.step(action)

            # reward = reward if not done else -20
            new_state = new_state[()]['graph']
            dqn_agent.remember(cur_state, action, reward, new_state, done)
            dqn_agent.replay()
            dqn_agent.target_train()

            cur_state = new_state
            env.render()  # can render but batches take so long it seems inactive
            if done:
                # current version is never done
                break

            if step % 10 == 0:
                log.info(f"Running at trial: {trial} step: {step}")
        if trial % 10 == 0:
            save_name = f"models/trial-{trial}.model"
            log.info(f"Saving model: {save_name}")
            dqn_agent.save_model(save_name)


if __name__ == "__main__":
    main()
