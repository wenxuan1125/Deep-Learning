'''DLP DQN Lab'''
__author__ = 'chengscott'
__copyright__ = 'Copyright 2020, NCTU CGI Lab'
import argparse
from collections import deque
import itertools
import random
import time
from collections import deque
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from atari_wrappers import wrap_deepmind, make_atari


class ReplayMemory(object):
    ## TODO ##
    def __init__(self, capacity, state_shape, device):
        c,h,w = state_shape
        self.capacity = capacity
        self.device = device
        self.m_states = torch.zeros((capacity, c, h, w), dtype=torch.uint8)
        self.m_actions = torch.zeros((capacity, 1), dtype=torch.long)
        self.m_rewards = torch.zeros((capacity, 1), dtype=torch.int8)
        self.m_dones = torch.zeros((capacity, 1), dtype=torch.bool)
        self.position = 0
        self.size = 0        



    def push(self, state, action, reward, done):
        """Saves a transition"""
        self.m_states[self.position] = state # 5,84,84
        self.m_actions[self.position,0] = action
        self.m_rewards[self.position,0] = reward
        self.m_dones[self.position,0] = done
        self.position = (self.position + 1) % self.capacity
        self.size = max(self.size, self.position)


    def sample(self, bs):
        """Sample a batch of transitions"""
        i = torch.randint(0, high=self.size, size=(bs,))
        bs = self.m_states[i, :4] 
        bns = self.m_states[i, 1:] 
        ba = self.m_actions[i].to(self.device)
        br = self.m_rewards[i].to(self.device).float()
        bd = self.m_dones[i].to(self.device).float()
        return bs, ba, br, bns, bd

    
        

    def __len__(self):
        return self.size


class Net(nn.Module):
    def __init__(self, num_classes=4, init_weights=True):
        super(Net, self).__init__()

        self.cnn = nn.Sequential(nn.Conv2d(4, 32, kernel_size=8, stride=4),
                                        nn.ReLU(True),
                                        nn.Conv2d(32, 64, kernel_size=4, stride=2),
                                        nn.ReLU(True),
                                        nn.Conv2d(64, 64, kernel_size=3, stride=1),
                                 nn.ReLU(True)
                                        )
        self.classifier = nn.Sequential(nn.Linear(7*7*64, 512),
                                        nn.ReLU(True),
                                        nn.Linear(512, num_classes)
                                        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = x.float() / 255.
        x = self.cnn(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0.0)


class DQN:
    def __init__(self, args):
        self._behavior_net = Net().to(args.device)
        self._target_net = Net().to(args.device)
        # initialize target network
        self._target_net.load_state_dict(self._behavior_net.state_dict())
        self._target_net.eval()
        self._optimizer = torch.optim.Adam(self._behavior_net.parameters(), lr=args.lr, eps=1.5e-4)

        ## TODO ##
        """Initialize replay buffer"""
        #self._memory = ReplayMemory(...)
        self._memory = ReplayMemory(args.capacity, [5, 84, 84], args.device)

        ## config ##
        self.device = args.device
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.freq = args.freq
        self.target_freq = args.target_freq

    def select_action(self, state, epsilon, action_space):
        '''epsilon-greedy based on behavior network'''
        ## TODO ##

        if random.random() <= epsilon:
            ## Exploration
            action = action_space.sample()
        else:
			## Exploitation
            with torch.no_grad():
                # input_state = torch.from_numpy(state).to(self.device)
                input_state = state.to(self.device)
                # print(input_state.shape)
                actions_vec = self._behavior_net(input_state)
                action = torch.argmax(actions_vec).item()
        return action
        

    def append(self, state, action, reward, done):
        ## TODO ##
        """Push a transition into replay buffer"""
        #self._memory.push(...)
        self._memory.push(state, action, reward, done)
        
    def update(self, total_steps):
        if total_steps % self.freq == 0:
            self._update_behavior_network(self.gamma)
        if total_steps % self.target_freq == 0:
            self._update_target_network()

    def _update_behavior_network(self, gamma):
        # sample a minibatch of transitions
        state_batch, action_batch, reward_batch, n_state_batch, done_batch = self._memory.sample(self.batch_size)
        # state, action, reward, next_state, done = self._memory.sample(self.batch_size)
        ## TODO ##
        state_batch = state_batch.to(self.device)
        action_batch = action_batch.to(self.device)
        reward_batch = reward_batch.to(self.device)
        n_state_batch = n_state_batch.to(self.device)
        done_batch = done_batch.to(self.device)

        q = self._behavior_net(state_batch).gather(1, action_batch).to(self.device)
        nq = self._target_net(n_state_batch).max(1)[0].detach().to(self.device)

        # Compute the expected Q values
        expected_state_action_values = (nq * gamma)*(1.-done_batch[:,0]) + reward_batch[:,0]

        # Compute Huber loss
        loss = F.smooth_l1_loss(q, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self._optimizer.zero_grad()
        loss.backward()
        for param in self._behavior_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self._optimizer.step()
        

    def _update_target_network(self):
        '''update target network by copying from behavior network'''
        ## TODO ##
        self._target_net.load_state_dict(self._behavior_net.state_dict())
        

    def save(self, model_path, checkpoint=False):
        if checkpoint:
            torch.save({
                'behavior_net': self._behavior_net.state_dict(),
                'target_net': self._target_net.state_dict(),
                'optimizer': self._optimizer.state_dict(),
            }, model_path)
        else:
            torch.save({
                'behavior_net': self._behavior_net.state_dict(),
            }, model_path)

    def load(self, model_path, checkpoint=False):
        model = torch.load(model_path)
        self._behavior_net.load_state_dict(model['behavior_net'])
        if checkpoint:
            self._target_net.load_state_dict(model['target_net'])
            self._optimizer.load_state_dict(model['optimizer'])
def fp(n_frame):
    n_frame = torch.from_numpy(n_frame)
    h = n_frame.shape[-2]
    return n_frame.view(1,h,h)

def train(args, agent, writer):
    print('Start Training')
    env_raw = make_atari('BreakoutNoFrameskip-v4')
    env = wrap_deepmind(env_raw, episode_life=True, clip_rewards=True)
    action_space = env.action_space
    total_steps, epsilon = 0, 1.
    ewma_reward = 0
    q = deque(maxlen=5) 

    for i in range(5):
        empty_frame = torch.from_numpy(np.zeros((1, 84, 84)))
        q.append(empty_frame)


    for episode in range(args.episode):
        total_reward = 0
        state = env.reset()
        new_state, reward, done, _ = env.step(1) # fire first !!!
        frame = fp(new_state)
        q.append(frame)

        for t in itertools.count(start=1):
            if total_steps < args.warmup:
                action = action_space.sample()
            else:
                # select action
                current_state = torch.cat(list(q)[1:]).unsqueeze(0)
                action = agent.select_action(current_state, epsilon, action_space)
                # decay epsilon
                epsilon -= (1 - args.eps_min) / args.eps_decay
                epsilon = max(epsilon, args.eps_min)

            # execute action
            new_state, reward, done, _ = env.step(action)       # state.shape = (84, 84, 1), numpy.ndarray
            
            frame = fp(new_state)
            q.append(frame)

            # print(done)
            # print('Frame: ', frame.shape)
            # print(type(state))
            ## TODO ##
            # store transition
            #agent.append(...)
            agent.append(torch.cat(list(q)).unsqueeze(0), action, reward, done)

            if total_steps >= args.warmup:
                agent.update(total_steps)

            total_reward += reward

            if total_steps % args.eval_freq == 0:
                """You can write another evaluate function, or just call the test function."""
                test(args, agent, writer)
                agent.save(args.model + "dqn_breakout_" + str(total_steps) + ".pt")

            total_steps += 1
            if done:
                ewma_reward = 0.05 * total_reward + (1 - 0.05) * ewma_reward
                writer.add_scalar('Train/Episode Reward', total_reward, episode)
                writer.add_scalar('Train/Ewma Reward', ewma_reward, episode)
                print('Step: {}\tEpisode: {}\tLength: {:3d}\tTotal reward: {:.2f}\tEwma reward: {:.2f}\tEpsilon: {:.3f}'
                        .format(total_steps, episode, t, total_reward, ewma_reward, epsilon))
                break

    env.close()


def test(args, agent, writer):
    print('Start Testing')
    env_raw = make_atari('BreakoutNoFrameskip-v4')
    env = wrap_deepmind(env_raw)
    action_space = env.action_space
    e_rewards = []
    q = deque(maxlen=5) 

    for i in range(5):
        empty_frame = torch.from_numpy(np.zeros((1, 84, 84)))
        q.append(empty_frame)
    
    for i in range(args.test_episode):
        state = env.reset()
        e_reward = 0
        done = False

        while not done:
            time.sleep(0.01)
            #env.render()

            current_state = torch.cat(list(q)[1:]).unsqueeze(0)
            action = agent.select_action(current_state, args.test_epsilon, action_space)
            new_state, reward, done, _ = env.step(action)
            e_reward += reward
            frame = fp(new_state)
            q.append(frame)

        print('episode {}: {:.2f}'.format(i+1, e_reward))
        e_rewards.append(e_reward)

    env.close()
    print('Average Reward: {:.2f}'.format(float(sum(e_rewards)) / float(args.test_episode)))


def main():
    ## arguments ##
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-d', '--device', default='cuda')
    parser.add_argument('-m', '--model', default='ckpt/')
    parser.add_argument('--logdir', default='log/dqn_breakout')
    # train
    parser.add_argument('--warmup', default=20000, type=int)
    parser.add_argument('--episode', default=20000, type=int)
    parser.add_argument('--capacity', default=100000, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--lr', default=0.0000625, type=float)
    parser.add_argument('--eps_decay', default=1000000, type=float)
    parser.add_argument('--eps_min', default=0.1, type=float)
    parser.add_argument('--gamma', default=.99, type=float)
    parser.add_argument('--freq', default=4, type=int)
    parser.add_argument('--target_freq', default=10000, type=int)
    parser.add_argument('--eval_freq', default=200000, type=int)
    # test
    parser.add_argument('--test_only', action='store_true')
    parser.add_argument('-tmp', '--test_model_path', default='ckpt/dqn_1000000.pt')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--test_episode', default=10, type=int)
    parser.add_argument('--seed', default=20230422, type=int)
    parser.add_argument('--test_epsilon', default=0.01, type=float)
    args = parser.parse_args()

    ## main ##
    agent = DQN(args)
    writer = SummaryWriter(args.logdir)
    if args.test_only:
        agent.load(args.test_model_path)
        test(args, agent, writer)
    else:
        train(args, agent, writer)
        


if __name__ == '__main__':
    main()
