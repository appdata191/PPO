import gymnasium as gym

class Environment:
    def __init__(self):
        self.env = gym.make("CartPole-v1", render_mode="human")
    def reset(self):
        state, info = self.env.reset()
        return state
    def step(self, action):
        next_state, reward, terminated, truncated, info = self.env.step(action)
        return next_state, reward, terminated, truncated, info 
    def render(self):
        self.env.render()
    def close(self):
        self.env.close()