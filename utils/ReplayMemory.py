from collections import namedtuple, deque
import random
class ReplayMemory():
    def __init__(self, action_size, memory_size, rnd):
        if rnd!=-1:
            random.seed(rnd)
            
        self.action_size = action_size
        self.replay_queue = deque(maxlen=memory_size)  
        self.experience = namedtuple("Experience", field_names=["state", "action", "next_state", "reward", "done"])
    
    def get_size(self):
        return len(self.replay_queue)
    
    def add(self, state, action, next_state, reward, done):
        self.replay_queue.append(self.experience(state, action, next_state, reward, done))
        
    def get_sample(self, sample_size):
        samples = random.sample(self.replay_queue, sample_size)
        
        states = []
        action = []
        next_state = []
        reward = []
        done = []
        
        for exp in samples: 
            states.append(exp.state)
            action.append(exp.action)
            next_state.append(exp.next_state)
            reward.append(exp.reward)
            done.append(exp.done)
            
        return (states, action, next_state, reward, done)