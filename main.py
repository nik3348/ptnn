from ptnn.models.ppo2 import train, ppo2

if __name__ == '__main__':
    # train()
    env_name = "CartPole-v1"
    ppo2(env_name)

