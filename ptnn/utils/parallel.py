import torch.multiprocessing as mp

def run(env_name, trainer, num_processes = 4):
    mp.set_start_method('spawn')
    processes = []
    for rank in range(num_processes):
        p = mp.Process(target=trainer, args=(env_name))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
