import multiprocessing as mp

num_cores = max(1, mp.cpu_count())
print(num_cores)