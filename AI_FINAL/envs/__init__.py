''' Register new environments
'''
from envs.env import Env
from envs.registration import register, make


# register(
#     env_id='limit-holdem',
#     entry_point='rlcard.envs.limitholdem:LimitholdemEnv',
# )

# register(
#     env_id='no-limit-holdem',
#     entry_point='rlcard.envs.nolimitholdem:NolimitholdemEnv',
# )

register(
    env_id='leduc-holdem',
    entry_point='envs.leducholdem:LeducholdemEnv' 
)

register(
    env_id='bayes-leduc',
    entry_point='envs.leducholdem:bayesLeducEnv' 
)