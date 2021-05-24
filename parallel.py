from simulation import *
import multiprocessing as mp
from joblib import Parallel, delayed
num_cores = mp.cpu_count()

dB = 2
dt = .1
T = 3
chi = 2
M = 5
std = .1

L=12

HX = [0, .5,]
HZ = [0, .5, 1, 5]

magneet_paren= []

for hx in HX:
    for hz in HZ:
        magneet_paren.append([hx, hz])

if __name__ == "__main__":
    def parallel_simulatie_mageneetveld_paren(paar):
        sim = SimulationStochasticIC(L, dB, chi, std, paar[0], paar[1], dt, T)
        sim.make_noise_chi_greater_than_2(M)
        sim.plot(True)


    start = time.time()
    Parallel(n_jobs=num_cores)(delayed(parallel_simulatie_mageneetveld_paren)(i_th_parameter) for i_th_parameter in magneet_paren)
    end = time.time()
    print(end-start)