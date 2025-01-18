import json
import os
from collections import OrderedDict

# import 
from envs import Env
from leducholdem import Game
from utils import *

DEFAULT_GAME_CONFIG = {
        'game_num_players': 2,
        }

class LeducholdemEnv(Env):
    ''' Leduc Hold'em Environment
    '''

    def __init__(self, config):
        ''' Initialize the Limitholdem environment
        '''
        self.name = 'leduc-holdem' 
        self.default_game_config = DEFAULT_GAME_CONFIG
        self.game = Game()
        super().__init__(config)
        self.actions = ['call', 'raise', 'fold', 'check']
        self.state_shape = [[36] for _ in range(self.num_players)]
        self.action_shape = [None for _ in range(self.num_players)]

        with open(os.path.join('leducholdem/card2index.json'), 'r') as file:
            self.card2index = json.load(file)

    def _get_legal_actions(self):
        return self.game.get_legal_actions()

    def _extract_state(self, state):
        if state is None:
            return []
        
        extracted_state = {}

        legal_actions = OrderedDict({self.actions.index(a): None for a in state['legal_actions']})
        extracted_state['legal_actions'] = legal_actions

        public_card = state['public_card']
        hand = state['hand']
        obs = np.zeros(36)
        obs[self.card2index[hand]] = 1
        if public_card:
            obs[self.card2index[public_card]+3] = 1
        obs[state['my_chips']+6] = 1
        obs[sum(state['all_chips'])-state['my_chips']+21] = 1
        extracted_state['obs'] = obs

        extracted_state['raw_obs'] = state
        extracted_state['raw_legal_actions'] = [a for a in state['legal_actions']]
        extracted_state['action_record'] = self.action_recorder

        return extracted_state

    def get_payoffs(self):
        return self.game.get_payoffs()

    def _decode_action(self, action_id):
        legal_actions = self.game.get_legal_actions()
        if self.actions[action_id] not in legal_actions:
            if 'check' in legal_actions:
                return 'check'
            else:
                return 'fold'
        return self.actions[action_id]

    def get_perfect_information(self):
        state = {}
        state['chips'] = [self.game.players[i].in_chips for i in range(self.num_players)]
        state['public_card'] = self.game.public_card.get_index() if self.game.public_card else None
        state['hand_cards'] = [self.game.players[i].hand.get_index() for i in range(self.num_players)]
        state['current_round'] = self.game.round_counter
        state['current_player'] = self.game.game_pointer
        state['legal_actions'] = self.game.get_legal_actions()
        return state
    
class bayesLeducEnv(LeducholdemEnv):
    def __init__(self, config):
        super().__init__(config)
        self.name = 'bayes-leduc'

    def run(self, my_player:int, is_training=False):
        state, player_id = self.reset()
        opponent_hand = None
        while not self.is_over():
            if player_id is None:
                return 0
            if player_id == my_player:
                # Use bayesAgent's infer_opponent_hand to update opponent modeling
                prob = self.agents[my_player].computeProb(state, opponent_hand)
                EV = self.agents[my_player].computeEV(prob, state, my_player)
                # Call step with the updated opponent_hand
                action = self.agents[my_player].step(state, EV)
            else:
                # For other players, execute standard step or eval_step
                if not is_training:
                    action, _ = self.agents[player_id].eval_step(state)
                else:
                    action = self.agents[player_id].step(state)

            # Environment steps
            next_state, next_player_id = self.step(action, self.agents[player_id].use_raw)
            if self.is_over():
                break
            if next_player_id == my_player:
                n_p_id = 0 if next_player_id == 1 else 1
                opponent_hand = self.agents[my_player].infer_opponent_hand(state, action, opponent_hand, next_player_id=n_p_id)
                
            # Set the state and player
            state = next_state
            player_id = next_player_id

        # Payoffs
        payoffs = self.get_payoffs()
        print("over")

        return payoffs



