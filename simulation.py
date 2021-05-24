from functies import *
import matplotlib.pyplot as plt
import time
# Visit https://pypi.org/project/notify-run/ to register your phone to get notifications
from notify_run import Notify

import chime
chime.theme('zelda')
notify=Notify()

plt.style.use(['science','notebook'])
plt.rcParams['figure.figsize'] = [13, 6]

#configure your own path to which figures will be saved
path = 'C:\\Users\\alexa\\OneDrive\\1MAthesis\\Figuren\\PyCharm Figuren'

class SimulationStochasticIC(object):
    ''' Class for the simulation of de expecation value of sigma_x of various stochastic intitial MPS
    ==========
    dB: max bond dimension of ground state, when 1 it's just de spins pointend in up direction on the
        Bloch sphere
    chi: max bond dimension of the time evolution
    L: length of the spin-chain
    iteration: numbers of iterations to average over
    std: the standard deviation used as noise in the tangent MPS
    hx: extern h_x field
    hz: extern h_z field
    dt: time step
    T: max time interval
    Methods
    =======
    ground_state:
        calculates the ground state for a given length L
    stochastic_MPS:
        returns 'psip', a stoachastic ground state-like MPS
    plot:
        plots the graph and can save the graph
    '''

    def __init__(self, L, dB, chi, std, hx, hz, dt, T):
        self.dB = dB    #max bond dimension for ground state
        self.L = L
        self.chi = chi  #max bond dimension for time evolution
        self.iterations = None
        self.std = std
        self.dt = dt
        self.T = T
        self.time_axis = None
        self.hx = hx
        self.hz = hz
        self.M = ZZXZChain(self.L, 1, self.hx, self.hz)
        self.psi0 = ground_state(self.L, self.dB)
        self.noise = None

        self.tdvp_params = {
                            'N_steps': 1,
                            'dt': self.dt,
                            'trunc_params': {'chi_max': self.chi, 'svd_min': 1.e-12},
                            'verbose':.1
                        }
        self.tebd_params = {
                        'N_steps': 1,
                        'dt': self.dt,
                        'order': 4,
                        'verbose': .1,
                        'trunc_params': {'chi_max': self.chi, 'svd_min': 1.e-12}
                             }

        self.sigma_chi = None
        self.energy_chi = None

        self.sigma_correct = None
        self.energy_correct = None

        self.noise_tdvp = None
        self.noise_energy = None

        self.time_axis_noise = None

    def measurement(self, eng, data,psi_):
        keys = ['t', 'Sigmax','energy']
        if data is None:
            data = dict([(k, []) for k in keys])
        data['t'].append(eng.evolved_time)
        data['energy'].append(np.real(self.M.calc_H_MPO().expectation_value(psi_)))
        data['Sigmax'].append(eng.psi.expectation_value('Sigmax', self.L//2)[0])
        return data

    def time_evolution_tdvp(self, chi):
        psi_corr=self.psi0.copy()
        tdvp_params = {
            'N_steps': 1,
            'dt': self.dt,
            'trunc_params': {'chi_max': chi, 'svd_min': 1.e-12},
            'verbose': 0.1
        }
        eng = tdvp.Engine(psi_corr, self.M, tdvp_params)
        data = self.measurement(eng, None, psi_corr)
        start = time.time()
        while eng.evolved_time < self.T:
            eng.run()
            self.measurement(eng, data,psi_corr)
        end = time.time()
        print(end-start)
        self.energy_correct = data['energy']
        self.sigma_correct = data['Sigmax']
        self.time_axis = data['t']

    def make_noise_tdvp(self, reps):
        '''
        :param reps: amount of iterations to make stochastic trajectories
        :return: a dictionary of stochastic trajectories of sigma_x using TDVP
        '''
        self.iterations = reps
        self.noise_tdvp = dict([(it, []) for it in range(reps)])
        self.noise_energy_tdvp  = dict([(it, []) for it in range(reps)])

        for i in tqdm(range(self.iterations), position=0, leave=True, desc='Noise'):
            psip= stochastic_MPS(self.psi0, self.std)
            eng = tdvp.Engine(psip, self.M, self.tdvp_params)
            data = self.measurement(eng, None,psip)
            while eng.evolved_time < self.T:
                eng.run()
                self.measurement(eng, data,psip)
            self.noise_tdvp[i] = data['Sigmax']
            self.noise_energy_tdvp[i] = data['energy']

    def plot_energy_tdvp(self):
        noise_matrix = dict_to_matrix(self.noise_energy_tdvp)
        confidence_interval1 = 95
        confidence_interval2 = 80
        confidence_interval3 = 50
        for ci in [confidence_interval1, confidence_interval2, confidence_interval3]:
            low = np.percentile(noise_matrix, 50 - ci / 2, axis=0)
            high = np.percentile(noise_matrix, 50 + ci / 2, axis=0)
            plt.fill_between(self.time_axis, low, high, color='blue', alpha=0.4)
        # with plt.style.context(['science', 'notebook']):
        plt.plot(self.time_axis, np.mean(noise_matrix, axis=0), color='red', label=r'$|\psi\rangle_{\bar{r}}$')
        plt.plot(self.time_axis, self.energy_correct, color='black', label=r'$|\psi \rangle$')
        plt.ylabel('$E$')
        plt.xlabel('t')
        plt.xlim(0, max(self.time_axis))
        plt.legend(loc='best', frameon=True, fancybox='True', shadow=True)

    def plot_energy_tebd(self):
        noise_matrix = dict_to_matrix(self.noise_energy)
        confidence_interval1 = 95
        confidence_interval2 = 80
        confidence_interval3 = 50
        for ci in [confidence_interval1, confidence_interval2, confidence_interval3]:
            low = np.percentile(noise_matrix, 50 - ci / 2, axis=0)
            high = np.percentile(noise_matrix, 50 + ci / 2, axis=0)
            plt.fill_between(self.time_axis, low, high, color='blue', alpha=0.4)
        # with plt.style.context(['science', 'notebook']):
        plt.plot(self.time_axis, np.mean(noise_matrix, axis=0), color='red', label=r'$|\psi\rangle_{\bar{r}}$')
        plt.plot(self.time_axis, self.energy_correct, color='black', label=r'$|\psi \rangle$')
        plt.ylabel('$E$')
        plt.xlabel('t')
        plt.xlim(0, max(self.time_axis))
        plt.legend(loc='upper right', frameon=True, fancybox='True', shadow=True)

    def time_evolution_sigma_correct(self):
        '''
        :return: time evolution of the time evolution for the correct time evolution, chi=L/2
        '''
        psi_corr= self.psi0.copy()
        tebd_params = {
                        'N_steps': 1,
                        'dt': self.dt,
                        'order': 4,
                        'verbose': .1,
                        'trunc_params': {'chi_max': (self.L)//2, 'svd_min': 1.e-12}
        }
        eng = tebd.Engine(psi_corr, self.M, tebd_params)
        data = self.measurement(eng, None,psi_corr)
        start = time.time()
        while eng.evolved_time < self.T:
            eng.run()
            self.measurement(eng, data,psi_corr)
        end = time.time()
        # print(end-start)
        self.sigma_correct = data['Sigmax']
        self.energy_correct = data['energy']
        self.time_axis = data['t']

    def time_evolution_sigma_chi(self):
        '''
        :return: time evolution of sigma for a given bond dimension, self.chi
        '''
        psi_corr= self.psi0.copy()
        eng = tebd.Engine(psi_corr, self.M, self.tebd_params)
        data = self.measurement(eng, None, psi_corr)
        start = time.time()
        while eng.evolved_time < self.T:
            eng.run()
            self.measurement(eng, data, psi_corr)
        end = time.time()
        # print(end-start)
        self.sigma_chi = data['Sigmax']
        self.energy_chi = data['energy']
        # self.time_axis = data['t']

    def make_noise(self, reps):
        '''
        :param reps: amount of iterations to make stochastic trajectories
        :return: a dictionary of stochastic trajectories of sigma_x
        '''
        self.iterations = reps
        self.noise = dict([(it, []) for it in range(reps)])
        self.noise_energy = dict([(it, []) for it in range(reps)])

        for i in tqdm(range(self.iterations), position=0, leave=True, desc='Noise'):
            psip = stochastic_MPS(self.psi0, self.std)
            eng = tebd.Engine(psip, self.M, self.tebd_params)
            data = self.measurement(eng, None,psip)
            while eng.evolved_time < self.T:
                eng.run()
                self.measurement(eng, data,psip)
            self.noise[i] = data['Sigmax']
            self.noise_energy[i] = data['energy']

        self.time_axis_noise = data['t']

    def plot(self, savefig):
        '''
        :param savefig: Bool, will save the graph or not
        :return: plot
        '''
        self.time_evolution_sigma_correct()
        self.time_evolution_sigma_chi()
        noise_matrix = dict_to_matrix(self.noise)

        confidence_interval1 = 95
        confidence_interval2 = 80
        confidence_interval3 = 50
        for ci in [confidence_interval1, confidence_interval2, confidence_interval3]:
            low = np.percentile(noise_matrix, 50 - ci / 2, axis=0)
            high = np.percentile(noise_matrix, 50 + ci / 2, axis=0)
            plt.fill_between(self.time_axis_noise, low, high, color='blue', alpha=0.4)
        # with plt.style.context(['science','notebook']):
        plt.plot(self.time_axis_noise, np.mean(noise_matrix, axis=0), color='red', linestyle='-.', linewidth=2, label=r'$|\psi\rangle_{\bar{r}}$')
        plt.plot(self.time_axis, self.sigma_chi, color='m', linewidth=2, linestyle='dashed', label=r'$|\psi\rangle_{\chi}$')
        plt.ylabel('$\sigma_x$')
        plt.xlabel('t')
        plt.xlim(0, max(self.time_axis))
        plt.plot(self.time_axis, self.sigma_correct, color='black', linewidth=2, label=r'$|\psi\rangle$')
        plt.legend(loc='lower left', frameon=True, fancybox='True', fontsize='large', shadow=True)
        if savefig:
            plt.savefig(path + '/L%i_chi%i_std%.2f_dB%i_hx%.1f_hz%.1f_dt%.1f_T%i.eps' % (self.L, self.chi, self.std, self.dB, \
                                                                                self.hx, self.hz, self.dt, self.T), format='eps', bbox_inches='tight')
            plt.title('L%i_chi%i_std%.2f_dB%i_hx%.1f_hz%.1f_dt%.1f_T%i.png' % (self.L, self.chi, self.std, self.dB,\
                                                                                         self.hx, self.hz, self.dt, self.T))
            plt.savefig(path + '/L%i_chi%i_std%.2f_dB%i_hx%.1f_hz%.1f_dt%.1f_T%i.png' % (self.L, self.chi, self.std, self.dB,\
                                                                                         self.hx, self.hz, self.dt, self.T), bbox_inches='tight')
            plt.close()
        else:
            plt.title('L%i_chi%i_std%.2f_dB%i_hx%.1f_hz%.1f_dt%.1f_T%i' % (self.L, self.chi, self.std, self.dB, \
                                                                                self.hx, self.hz, self.dt, self.T))
            plt.plot()
            plt.show()

    def make_noise_chi_greater_than_2(self, reps):
        '''
        :param reps: amount of iterations to make stochastic trajectories
        Will create a stochastic psi_p-time evolution once the time evolution reaches the self.chi,
        this chi cannot be achieved by means of the ground state method
        '''
        self.iterations = reps
        self.noise = dict([(it, []) for it in range(reps)])
        self.noise_energy = dict([(it, []) for it in range(reps)])

        cut = int(np.log(self.chi) / np.log(2))     #for lets say a chi=8, the chi-list will look like this [2,4,8,8,8,4,2]
                                                    #so we can only look at [-3:3] ,3=log(8)/log(2)
        psi_corr = self.psi0.copy()
        eng = tebd.Engine(psi_corr, self.M, self.tebd_params)
        data = self.measurement(eng, None, psi_corr)
        while eng.evolved_time < self.T:
            eng.run()
            if all(i == self.chi for i in psi_corr.chi[cut:-cut]):
                break

        evolved_time_part1 = eng.evolved_time
        self.iterations = reps
        self.noise = dict([(it, []) for it in range(reps)])
        self.noise_energy = dict([(it, []) for it in range(reps)])


        for i in tqdm(range(self.iterations), position=0, leave=True, desc='Noise'):
            psip = stochastic_MPS_2(psi_corr, self.std, cut)
            print(psip.expectation_value('Sigmax')[self.L//2])
        #     eng = tebd.Engine(psip, self.M, self.tebd_params)
        #     data = self.measurement(eng, None, psip)
        #
        #     while eng.evolved_time < self.T-evolved_time_part1:
        #         eng.run()
        #         self.measurement(eng, data, psip)
        #
        #     self.noise[i] = data['Sigmax']
        #     self.noise_energy[i] = data['energy']
        #
        # self.time_axis_noise = data['t']+evolved_time_part1*np.ones(len(data['t']))

    # functions for the analytical solution
    def theta_0(self, k):
        return np.angle(10000 - np.exp(1j * k))

    def delta_theta(self, k):
        return self.theta(k) - self.theta_0(k)

    def theta(self, k):
        return np.angle(self.hx - np.exp(1j * k))

    def epsilon(self, k):
            return 2 * 1 * np.sqrt(1 + self.hx ** 2 - 2 * self.hx * np.cos(k))

    def analytisch(self,t):
        K = np.arange(self.L) * (2 * np.pi) / self.L - np.pi
        som = 0
        for k in K:
            som += 1 - np.cos(self.delta_theta(k)) * np.cos(self.theta(k)) - np.cos(2 * self.epsilon(k) * t) * np.sin(self.delta_theta(k)) * np.sin(self.theta(k))
        return 1 - 1 / self.L * som

    #
    #
    # chi = 2
    #
    #
    # SIM=sim
    #
    #
    # for std in [.15]:
    #     sim = SimulationStochasticIC(L, dB, chi, std, hx, hz, dt, T)
    #     sim.time_axis = SIM.time_axis
    #     sim.sigma_correct = SIM.sigma_correct
    #     sim.make_noise(reps=50)
    #     plt.plot(np.arange(0, T, step=.01), sim.analytisch(np.arange(0, T, step=.01)), label='analytisch', color='black')
    #     plt.ylim([-.2, 1])
    #     plt.xlim([0, T])
    #     plt.grid(linestyle='--')
    #     sim.plot(True)
    #     chime.success()

    # notify.send('Klaar')