from agents import agents
from envs import *
from utils import utils
import pprint
from agents import cfr_agent
from agents import leduc_holdem_human_agent
import argparse

def main(seed):
    env = make('leduc-holdem', config={'seed':seed})
    utils.set_seed(seed)
    # agent = agents.randomAgent(env)
    agent = agents.randomAgent(env)
    # agent.load()
    agent2 = agents.ruleAgent(env)
    # agent2.load()
    env.set_agents([agent,agent2])
    win_average = 0.0
    for i in range(5000):
        _ , player_wins = env.run(is_training=False)
        win_average += player_wins[1] 
    win_average /= 5000
    print(win_average)
    # env.set_agents([agent for _ in range(env.num_players)])
    # env.run(False)
    # trajectories, player_wins = env.run(is_training=False)
    # # Print out the trajectories
    # print('\nTrajectories:')
    # print(trajectories)
    # print('\nSample raw observation:')
    # pprint.pprint(trajectories[0][0]['raw_obs'])
    # print('\nSample raw legal_actions:')
    # pprint.pprint(trajectories[0][0]['raw_legal_actions'])


def mbRun(seed, cfr: bool, rand: bool ):
    env = make('leduc-holdem', config={'seed':seed})
    utils.set_seed(seed)
    agent1 = agents.ruleAgent(env)
    if rand:
        agent1 = agents.randomAgent(env)
    elif cfr:
        agent1 = cfr_agent.CFRAgent(env)
        agent1.load()
    else:
        agent1 = agents.ruleAgent(env)
    agent2 = agents.modelbasedAgent(env, "./data/mb_data")
    env.set_agents([agent1, agent2])

    agent2.load()
    for i in range(3):
        agent2.train()
    agent2.save()

    # agent2.load()

    # win_average = 0.0
    # for i in range(500):
    #     _ , player_wins = env.run(is_training=False)
    #     win_average += player_wins[1] 
    # win_average /= 500
    # print(win_average)

    # # Print out the trajectories
    # print('\nTrajectories:')
    # # print(trajectories)
    # for i in range(len(trajectories)):
    #     print('\n')
    #     for j in range(len(trajectories[i])):
    #         print(trajectories[i][j])
    # print('\nSample raw observation:')
    # pprint.pprint(trajectories[0][0]['raw_obs'])
    # print('\nSample raw legal_actions:')
    # pprint.pprint(trajectories[0][0]['raw_legal_actions'])
    # print('action record')
    # pprint.pprint(trajectories[0][0]['action_record'])
    # print('payoff')
    # pprint.pprint(trajectories[0][1])


def qlRun(seed, cfr: bool, rand: bool, train_iters=1000):
    env = make('leduc-holdem', config={'seed': seed})
    utils.set_seed(seed)
    
    agent1 = agents.ruleAgent(env)
    if rand:
        agent1 = agents.randomAgent(env)
    if cfr:
        agent1 = cfr_agent.CFRAgent(env)
        agent1.load()
    
    q_agent = agents.QLearningAgent(env, alpha=0.01, gamma=0.8, epsilon=0.1,
                                    model_path="./data/ql_data")
    q_agent.load()
    
    ## evaluation ##
    eval_episodes = 500
    win_sum = 0.0
    total_payoff = 0.0
    env.set_agents([agent1, q_agent])

    for _ in range(eval_episodes):
        _, payoffs = env.run(is_training=False)
        if payoffs[1] > 0:  # QLearningAgent收益 > 0
            win_sum += 1
        total_payoff += payoffs[1]
    
    win_rate = win_sum / eval_episodes
    avg_chips = total_payoff / eval_episodes
    
    print(f"QLearningAgent win rate over {eval_episodes} episodes = {win_rate:.2f}")
    print(f"QLearningAgent average chips gained = {avg_chips:.2f}")


def play_with_human(seed, ql: bool, cfr: bool):
    '''默认和model-based agent打'''
    env = make('leduc-holdem', config={'seed':seed})
    utils.set_seed(seed)
    human_agent = leduc_holdem_human_agent.HumanAgent(env.num_actions)
    
    agent = agents.modelbasedAgent(env, "./data/mb_data") # 和你对打的agent
    if ql:
        print(">> Play with Q learning agent")
        agent = agents.QLearningAgent(env, model_path="./data/ql_data")
    elif cfr:
        agent = cfr_agent.CFRAgent(env)
    else:
        print(">> Play with model-based agent")
    agent.load()
    env.set_agents([
        human_agent,
        agent,
    ])

    print(">> Leduc Hold'em pre-trained model")

    while True:
        print(">> Start a new game")

        trajectories, payoffs = env.run(is_training=False)
        # If the human does not take the final action, we need to
        # print other players action
        final_state = trajectories[0][-1]
        action_record = final_state['action_record']
        state = final_state['raw_obs']
        _action_list = []
        for i in range(1, len(action_record)+1):
            if action_record[-i][0] == state['current_player']:
                break
            _action_list.insert(0, action_record[-i])
        for pair in _action_list:
            print('>> Player', pair[0], 'chooses', pair[1])

        # Let's take a look at what the agent card is
        print('===============     OUR Agent    ===============')
        utils.print_card(env.get_perfect_information()['hand_cards'][1])

        print('===============     Result     ===============')
        if payoffs[0] > 0:
            print('You win {} chips!'.format(payoffs[0]))
        elif payoffs[0] == 0:
            print('It is a tie.')
        else:
            print('You lose {} chips!'.format(-payoffs[0]))
        print('')

        input("Press ENTER to continue...")

def runBayes(seed=42):
    env = make('bayes-leduc', config={'seed':seed})
    bayesAgent = agents.bayesAgent(env)
    randomAgent = agents.randomAgent(env)
    env.set_agents([bayesAgent, randomAgent])
    payoff = env.run(my_player=0)
    print(payoff)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-mb', action='store_true')
    parser.add_argument('-ql', action='store_true')
    parser.add_argument('-human', action='store_true')
    parser.add_argument('-seed', type=int, default=42)
    parser.add_argument('-cfr', action='store_true')
    parser.add_argument('-r', action='store_true')
    args = parser.parse_args()

    # if args.mb:
    #     mbRun(args.seed, args.cfr, args.r)
    # elif args.human:
    #     play_with_human(args.seed, args.ql, args.cfr)
    # elif args.ql:
    #     qlRun(args.seed, args.cfr, args.r, train_iters=2000)
    # else:
    #     main(args.seed)
    runBayes()