from leducholdem import *
from envs import *
import random
import math
import numpy as np
from collections import defaultdict
import os
import pickle


""" action类型
    check
	跟注 (Call)：匹配对手的下注金额。
	弃牌 (Fold)：放弃当前牌局。
	加注 (Raise)：增加下注金额。"""

class agent(object):
    def __init__(self, env: Env):
        self.use_raw = False
        self.env = env

    def step(self, state: dict) -> int:
        '''训练时调用这个函数获得下一个action'''
        return NotImplemented
    
    def eval_step(self, state: dict)-> tuple[int, list]:
        return self.step(state), []
    
    def get_legal_actions(self, state: dict) -> list[int]:
        return list(state['legal_actions'].keys())

    def train(self):
        return NotImplemented
    
    def get_state(self, player_id):
        state = self.env.get_state(player_id)
        return state['obs'].tostring(), list(state['legal_actions'].keys())
    
    # 定义别的需要的method

class randomAgent(agent):
    def __init__(self, env):
        super().__init__(env)
        self.num_actions = env.num_actions

    def step(self, state: dict):
        action_ind = np.random.choice(self.get_legal_actions(state))
        return action_ind

class bayesAgent(agent):
    def __init__(self, env, model_path = "./data/bayes_data"):
        super().__init__(env)
        self.model_path = model_path

    def step(self, state, EV):
        ## 根据computerEV的结果根据EVs对action进行softmax，然后采样并返回一个action
        actions = list(EV.keys())
        values = np.array([EV[action] for action in actions])
    
        # 防止数值溢出，进行标准化
        exp_values = np.exp(values - np.max(values))
        probabilities = exp_values / np.sum(exp_values)

        # 按照概率分布采样一个动作
        selected_action = np.random.choice(actions, p=probabilities)
        # print('step OK')
        return selected_action

    def computeEV(self, probs:list, state, player_id):
        ## 根据赢，平局，输的分布来计算EV
        EV = {}
        # state = state
        for action in self.get_legal_actions(state):
            if state['legal_actions'][action] == 'check' or state['legal_actions'][action] == 'fold':
                EV[action] = 0
            else:
                print(state)
                previous_chip = state['raw_obs']['all_chips'][player_id]
                next_state, next_player_id = self.env.step(action, self.env.agents[player_id].use_raw) # NOTE
                current_chip = next_state['raw_obs']['all_chips'][player_id]
                chip_in = current_chip - previous_chip
                pot = 0
                if state['legal_actions'] == 'call':
                    pot = state['all_chips']['raw_obs'][next_player_id] + state['all_chips']['raw_obs'][player_id]
                elif state['legal_actions'] == 'raise':
                    pot = state['all_chips']['raw_obs'][player_id] + next_state['all_chips']['raw_obs'][next_player_id]
                EV[action] = pot * probs[0] + pot / 2 * probs[1] - chip_in * probs[2]
        return EV

    def computeProb(self, state, opponent_hand:list=None):
        ## opponent_hand = [P(J), P(Q), P(K)]
        ## 根据我的手牌，对手的手牌和公共牌计算赢，平局和输的分布
        prob = [0, 0, 0] # [P(win), P(tie), P(lose)]
        state = state['raw_obs']
        hand = state['hand']
        public_card = state['public_card']
        if opponent_hand is None:
            # print(hand)
            if hand == 'J':
                opponent_hand = [0.2, 0.4, 0.4]
            elif hand == 'Q':
                opponent_hand = [0.4, 0.2, 0.4]
            elif hand == 'K':
                opponent_hand = [0.4, 0.4, 0.2]
        if public_card==None:
            # print(hand)  
            if hand == 'J':
                prob[1] = opponent_hand[0] ## tie
                prob[0] = (1 - prob[1]) / 4 ## win 
                prob[2] = 1 - prob[0] - prob[1] ## lose
            if hand == 'Q':
                prob[1] = opponent_hand[1]
                prob[0] = 3 * opponent_hand[0] / 4 + opponent_hand[2] / 4
                prob[2] = 1 - prob[0] - prob[1]
            if hand == 'K':
                prob[1] = opponent_hand[2]
                prob[2] = opponent_hand[0] / 4 + opponent_hand[1] / 4
                prob[0] = 1 - prob[1] - prob[2]
        else:
            if public_card[1] == 'J':
                if hand[1] == 'J':
                    prob = [1, 0, 0]
                if hand[1] == 'Q':
                    prob[1] = opponent_hand[1]
                    prob[0] = 0
                    prob[2] = 1 - opponent_hand[1]
                if hand[1] == 'K':
                    prob[1] = opponent_hand[2]
                    prob[2] = opponent_hand[0]
                    prob[0] = 1 - prob[1] - prob[2]
            elif public_card[1] == 'Q':
                if hand[1] == 'J':
                    prob[1] = opponent_hand[0]  # tie
                    prob[2] = 1 - opponent_hand[0]  # lose
                    prob[0] = 0  # win
                elif hand[1] == 'Q':
                    prob = [1, 0, 0]  # 我是 Q，对手没有赢牌机会
                elif hand[1] == 'K':
                    prob[1] = opponent_hand[2]  # tie
                    prob[0] = opponent_hand[0]  # win
                    prob[2] = 1 - prob[0] - prob[1]  # lose
            elif public_card[1] == 'K':
                if hand[1] == 'J':
                    prob[1] = opponent_hand[0]  # tie
                    prob[0] = 0  # win
                    prob[2] = 1 - prob[0] - prob[1]  # lose
                elif hand[1] == 'Q':
                    prob[1] = opponent_hand[1]  # tie
                    prob[2] = opponent_hand[2]  # lose
                    prob[0] = 1 - prob[1] - prob[2]  # win
                elif hand[1] == 'K':
                    prob = [1, 0, 0]  # 我是 K，对手没有赢牌机会

        return prob
    def infer_opponent_hand(self, state, opponent_action, opponent_hand_prior, next_player_id) -> list:
        if opponent_hand_prior is None:
            return []
        # 获取当前状态信息
        public_card = state['raw_obs']['public_card']
        legal_actions = state['legal_actions']
        
        # 定义可能的手牌
        possible_hands = ['J', 'Q', 'K']

        # 存储每种可能手牌的似然值
        likelihoods = []

        for i, hand in enumerate(possible_hands):
            # 模拟对手手牌为当前假设手牌
            simulated_state = {
                'raw_obs': {
                    'hand': hand,
                    'public_card': public_card
                },
                'legal_actions': legal_actions
            }
            # 计算该假设手牌下的动作分布
            probs = self.computeProb(simulated_state)
            evs = self.computeEV(probs, state, player_id=next_player_id)
            action_probs = self._action_distribution(evs)

            # 取对手实际动作的概率作为似然值
            likelihood = action_probs.get(opponent_action, 0)
            likelihoods.append(likelihood)

        # 使用贝叶斯公式计算后验概率
        posterior = []
        total_likelihood = 0
        for i, hand_prior in enumerate(opponent_hand_prior):
            posterior_val = hand_prior * likelihoods[i]
            posterior.append(posterior_val)
            total_likelihood += posterior_val

        # 归一化
        if total_likelihood > 0:
            posterior = [p / total_likelihood for p in posterior]
        else:
            posterior = [1 / len(possible_hands) for _ in possible_hands]  # 均匀分布
        
        return posterior

    def _action_distribution(self, evs):
        """
        根据EV计算动作的概率分布 (Softmax)
        :param evs: 每个动作的EV值
        :return: 动作的概率分布 (dict)
        """
        actions = list(evs.keys())
        values = np.array(list(evs.values()))

        # 防止溢出
        exp_values = np.exp(values - np.max(values))
        probabilities = exp_values / np.sum(exp_values)

        return dict(zip(actions, probabilities))
class heuristicAgent(agent):
    pass

class modelbasedAgent(agent):
    """
    一个示例性的 Model-based Agent，实现思路参考 rlcard 中 cfr_agent 的设计结构。
    """

    def __init__(self,
                 env: Env,
                 model_path: str = "./data/mb_data",
                 #is_training: bool = True,
                 #gamma: float = 1.0,**kwargs
                 ):
        """
        Args:
            env (Env): RLCard 环境对象
            model_path (str): 如果需要加载/保存模型时的文件路径
            is_training (bool): 是否处于训练模式
            num_simulations (int): 在决策时进行的模拟次数（示例中暂用不到多步模拟，可自行扩展）
            gamma (float): 折扣因子，用于多步价值评估（本示例中只做演示，可视需求使用）
            **kwargs: 其他可能用得到的参数
        """
        self.use_raw = False
        self.env = env
        self.model_path = model_path
        self.iteration = 0
        # self.is_training = is_training 

        #self.gamma = gamma

        # 状态转移模型: transition_model[(s_key, a)][s_next_key] = 出现次数
        self.transition_model = defaultdict(lambda: defaultdict(int))

        # 奖励模型: reward_model[(s_key, a)] = [奖励1, 奖励2, ...]
        self.reward_model = defaultdict(list)
        self.opponent_policy = {} ## 简单声明一下
        # 如果有更多关于对手的策略或权重，可以在此添加，如：
        # self.opponent_policy = {...}
        # self.opponent_models = {...}


    def train(self): ##我们如何调用env里的trajectory来更新数据
        ## Do one iteration
        self.iteration += 1
        trajectories, payoff  = self.env.run(is_training=True)
        
        all_transitions = []
        num_players = self.env.num_players

        for pid in range(num_players):
            # 该玩家的完整序列
            t_list = trajectories[pid]
            final_reward = payoff[pid]
            tie = (abs(final_reward) < 0.1)
            sign = 1
            if final_reward < 0:
                sign = -1

            # 注意：t_list 的结构通常是 [state0, action0, state1, action1, ..., stateN]
            # 我们依次遍历 (S, A, S', A', S''...) 直到最后一个 state
            # 索引走两步拿 (state, action, next_state)
            # 中间的奖励是 0，最后一次把 final_reward 记到最后一个动作中
            pre_chip = 0.0
            current_chip = 0.0
            for i in range(0, len(t_list) - 1, 2):
                s = t_list[i]      # state
                a = t_list[i + 1]  # action
                # print(self.get_legal_actions(s))
                chip_list = s['raw_obs']['all_chips']
                current_chip = chip_list[pid]
                delta_chip = (current_chip - pre_chip) * sign # 现在的状态与上一个状态的筹码之差，如果最终赢了为正，反之为负
                pre_chip = current_chip # update pre_chip

                # 可能还有下一个 state
                if i + 2 < len(t_list):
                    s_next = t_list[i + 2]
                else:
                    s_next = None  # 万一越界则说明是终局
                

                r = delta_chip
                if tie:
                    r = 0
                # 如果这是玩家该局的**最后一次**动作，则把最后的 payoffs[pid] 当作reward
                # if i + 2 == len(t_list) - 1:
                #     r = final_reward

                # 收集成 (s, pid, a, r, s_next)
                all_transitions.append((s, pid, a, r, s_next))

        # 3) 调用 _update_model 来更新转移与奖励统计
        self._update_model(all_transitions)

    def _update_model(self, trajectory): ##如何更新数据
        for (s, pid, a, r, s_next) in trajectory:
            # 如果只想跟踪“自己所在位置”时的状态转换，可加一个判断：
            # if pid != self.env.get_player_id(): continue

            s_key = self._state_to_key(s)
            s_next_key = self._state_to_key(s_next)

            # 更新转移计数
            self.transition_model[(s_key, a)][s_next_key] += 1
            # 更新奖励列表
            self.reward_model[(s_key, a)].append(r)

    def step(self, state):
        """
        训练模式下的决策函数。参考 cfr_agent 的 `_sample_action()`。
        内部会调用 `_compute_action_probs()` 取得对各动作的概率分布，然后进行抽样。
        
        Args:
            state (dict): 来自环境的状态信息，如 {'obs': ..., 'legal_actions': {...}, ...}
        Returns:
            action (int): 最终选择的动作
        """
        # if not self.is_training:
        #     # 如果不是训练模式，直接用 eval_step() 的逻辑
        #     return self.eval_step(state)

        legal_actions = state['legal_actions']  # dict: {action_idx: None, ...}
        action_probs = self._compute_action_probs(state, legal_actions, eval_mode=False)

        # 按概率分布随机抽样一个动作
        actions_list = list(legal_actions.keys())
        chosen_action = np.random.choice(actions_list, p=action_probs)
        return chosen_action

    # def eval_step(self, state):
    #     """
    #     评估/测试模式下的决策函数。与 cfr_agent 中的 `eval_step()` 类似。
    #     这里一般直接选取当前估计最优动作（或概率最大的动作）。
        
    #     Args:
    #         state (dict): 环境状态
    #     Returns:
    #         action (int): 选定的动作
    #     """
        
    #     # legal_actions = state['legal_actions']
    #     # action_probs = self._compute_action_probs(state, legal_actions, eval_mode=True)

    #     # # 选取概率最大的动作
    #     # best_idx = np.argmax(action_probs)
    #     # actions_list = list(legal_actions.keys())
    #     # chosen_action = actions_list[best_idx]
    #     # return chosen_action, []
    #     return self.step(state), []

    def _compute_action_probs(self, state, legal_actions, eval_mode=False):
        """
        简化版:只用平均即时奖励做softmax。若要多步推断或对手建模可进一步扩展。
        """
        actions_list = list(legal_actions.keys())
        s_key = self._state_to_key(state)
        action_values = []
        for a in actions_list:
            # 若没记录过该 (state, action)，给个默认值 0
            if (s_key, a) not in self.reward_model or len(self.reward_model[(s_key, a)]) == 0:
                est_reward = 0.0
            else:
                est_reward = np.mean(self.reward_model[(s_key, a)])
            action_values.append(est_reward)

        # softmax
        values_array = np.array(action_values)
        exp_values = np.exp(values_array - np.max(values_array))
        probs = exp_values / np.sum(exp_values)

        if eval_mode:
            # 在评估模式，你也可以只取 argmax。此处保留softmax做示例
            pass
        return probs

    def _state_to_key(self, state):
        """
        将 RLCard 返回的状态 dict 转为可hash的 key，以用于字典存储。
        这里是最简示例：取 `obs` 的 tuple 形式；你也可加更多信息（公共牌、筹码等）。
        """
        # state['obs'] 默认是 numpy array，也可先转成 tuple。
        # 注意：若要区分玩家ID等信息，也需加到 key 里。
        obs_tuple = tuple(state['obs']) if 'obs' in state else ()
        return obs_tuple
    
    def save(self):
        """
        保存模型，包括 transition_model 和 reward_model
        """
        if not self.model_path:
            print("Model path is not specified.")
            return

        # 确保保存路径存在
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        # 保存 transition_model
        transition_file = os.path.join(self.model_path, 'transition_model.pkl')
        with open(transition_file, 'wb') as f:
            pickle.dump(dict(self.transition_model), f)

        # 保存 reward_model
        reward_file = os.path.join(self.model_path, 'reward_model.pkl')
        with open(reward_file, 'wb') as f:
            pickle.dump(dict(self.reward_model), f)

        # 保存当前迭代次数
        iteration_file = os.path.join(self.model_path, 'iteration.pkl')
        with open(iteration_file, 'wb') as f:
            pickle.dump(self.iteration, f)

        print(f"Model saved to {self.model_path}")

    def load(self):
        """
        加载模型，包括 transition_model 和 reward_model
        """
        if not self.model_path or not os.path.exists(self.model_path):
            print("Model path does not exist or is not specified.")
            return

        # 加载 transition_model
        transition_file = os.path.join(self.model_path, 'transition_model.pkl')
        if os.path.exists(transition_file):
            with open(transition_file, 'rb') as f:
                self.transition_model = defaultdict(lambda: defaultdict(int), pickle.load(f))

        # 加载 reward_model
        reward_file = os.path.join(self.model_path, 'reward_model.pkl')
        if os.path.exists(reward_file):
            with open(reward_file, 'rb') as f:
                self.reward_model = defaultdict(list, pickle.load(f))

        # 加载当前迭代次数
        iteration_file = os.path.join(self.model_path, 'iteration.pkl')
        if os.path.exists(iteration_file):
            with open(iteration_file, 'rb') as f:
                self.iteration = pickle.load(f)

        print(f"Model loaded from {self.model_path}")

class QLearningAgent(agent):
    def __init__(self, env: Env, alpha=0.01, gamma=0.8, epsilon=0.1, model_path="./data/ql_data"):
        super().__init__(env)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.model_path = model_path
        self.Q = defaultdict(float)
        self.iteration = 0
    
    def step(self, state: dict) -> int:
        legal_actions = self.get_legal_actions(state)
        # epsilon-greedy
        if random.random() < self.epsilon:
            return np.random.choice(legal_actions)
        else:
            return self._get_best_action(state, legal_actions)
    
    def train(self):
        self.iteration += 1
        
        # collect trajectories
        trajectories, payoff = self.env.run(is_training=True)
        num_players = self.env.num_players
        
        # traverse (s, a, r, s'), update Q
        for pid in range(num_players):
            t_list = trajectories[pid]
            final_reward = payoff[pid]
            tie = (abs(final_reward) < 0.1)
            sign = 1 if final_reward >= 0 else -1
            
            # use delta_chip, same as modelbasedAgent
            pre_chip = 0.0
            current_chip = 0.0
            
            for i in range(0, len(t_list) - 1, 2):
                s = t_list[i]   # state
                a = t_list[i + 1]   # action
                
                # reward
                chip_list = s['raw_obs']['all_chips']
                current_chip = chip_list[pid]
                delta_chip = (current_chip - pre_chip) * sign
                pre_chip = current_chip
                r = 0 if tie else delta_chip
                
                # next state
                if i + 2 < len(t_list):
                    s_next = t_list[i + 2]
                else:
                    s_next = None
                
                # Q update:
                s_key = self._state_to_key(s)
                if s_next is not None:
                    s_next_key = self._state_to_key(s_next)
                    # next legal
                    next_legal_acts = list(s_next['legal_actions'].keys())
                    best_q_next = max(
                        self.Q[(s_next_key, a_next)] for a_next in next_legal_acts
                    ) if len(next_legal_acts) > 0 else 0.0
                else:
                    # final
                    best_q_next = 0.0
                
                old_q = self.Q[(s_key, a)]
                td_target = r + self.gamma * best_q_next
                self.Q[(s_key, a)] = old_q + self.alpha * (td_target - old_q)
    
    def eval_step(self, state: dict) -> tuple[int, list]:
        legal_actions = self.get_legal_actions(state)
        best_action = self._get_best_action(state, legal_actions)
        return best_action, []
    
    def _get_best_action(self, state: dict, legal_actions: list[int]) -> int:
        s_key = self._state_to_key(state)
        best_act, best_q = None, float('-inf')
        candidates = []
        
        for act in legal_actions:
            q_val = self.Q[(s_key, act)]
            if q_val > best_q:
                best_q = q_val
                best_act = act
                candidates = [act]
            elif q_val == best_q:
                candidates.append(act)
        
        # multiple Q, choose one randomly
        if len(candidates) > 1:
            return np.random.choice(candidates)
        else:
            return best_act
    
    def _state_to_key(self, state: dict):
        if state is None:
            return 'terminal'
        
        obs_tuple = tuple(state['obs']) if 'obs' in state else ()
        return obs_tuple
    
    def save(self):
        if not self.model_path:
            print("Model path is not specified.")
            return
        
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        
        q_file = os.path.join(self.model_path, 'q_table.pkl')
        with open(q_file, 'wb') as f:
            pickle.dump(dict(self.Q), f)
        
        iteration_file = os.path.join(self.model_path, 'iteration.pkl')
        with open(iteration_file, 'wb') as f:
            pickle.dump(self.iteration, f)
        
        print(f"Q table saved to {self.model_path}")
    
    def load(self):
        if not self.model_path or not os.path.exists(self.model_path):
            print("Model path does not exist or is not specified.")
            return
        
        q_file = os.path.join(self.model_path, 'q_table.pkl')
        if os.path.exists(q_file):
            with open(q_file, 'rb') as f:
                raw_q = pickle.load(f)
                self.Q = defaultdict(float, raw_q)
        
        iteration_file = os.path.join(self.model_path, 'iteration.pkl')
        if os.path.exists(iteration_file):
            with open(iteration_file, 'rb') as f:
                self.iteration = pickle.load(f)
        
        print(f"Q table loaded from {self.model_path}")

class ruleAgent(agent):
    def __init__(self, env):
        super().__init__(env)
        self.use_raw = True

    def step(self, state):
        legal_actions = state['raw_legal_actions']
        state = state['raw_obs']
        hand = state['hand']
        # print(hand)
        public_card = state['public_card']
        action = 'fold'
        # When having only 2 hand cards at the game start, choose fold to drop terrible cards:
        # Acceptable hand cards:
        # Pairs
        # K
        # Fold all hand types except those mentioned above to save money
        # actions: fold, call, raise, fold
        if public_card:
            if public_card[1] == hand[1]:
                action = 'raise'
            else:
                if hand[1] == 'K':
                    rand = random.uniform(0, 1)
                    if rand < 0.5:
                        action = 'check'
                    else:
                        action = 'call'
                else: 
                    action = 'check'
        else:
            if hand[1] == 'K':
                action = 'raise'
            else:
                if 'call' in legal_actions:
                    action = 'call'
                else:
                    action = 'check'

        #return action
        if action in legal_actions:
            return action
        else:
            if action == 'raise':
                return 'call'
            if action == 'check':
                return 'fold'
            if action == 'call':
                return 'raise'
            else:
                return action