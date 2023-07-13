from ptnn.trainers.PPO import PPO
# from ptnn.trainers.DDPG import start
from ptnn.utils.parallel import run
from TicTacToe import start

# env_name = "LunarLander-v2"
# run(env_name, PPO)
# PPO(env_name)
# start(env_name)

if __name__ == '__main__':
    start()
