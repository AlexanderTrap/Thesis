import numpy as np
from tenpy.networks.mps import MPSEnvironment
from scipy.linalg import sqrtm
from scipy.linalg import null_space
from scipy.linalg import norm
from tenpy.linalg.np_conserved import Array
import tenpy.linalg.np_conserved as npc
import tenpy
from tenpy.networks.mps import MPS
from tenpy.models.lattice import Chain
from tenpy.networks.site import SpinHalfSite
from tenpy.models.model import CouplingModel, NearestNeighborModel, MPOModel
from tenpy.algorithms import tebd
from tenpy.algorithms import dmrg
from tqdm import tqdm
from tenpy.algorithms import tdvp

def tangent_MPS(psi, it, sigma):
    #dBl = psi.chi[it - 1]
    dBr = psi.chi[it]

    dB = dBr
    dH = 2
    PSI = psi.copy()
    PSI.canonical_form()

    # maak van toestand symmetrisch canonische vorm ('C', tenpy.readthe docs'), zo kunnen de singuliere
    # waardes gescaled worden met L en R, S-> S**0.5

    PSI.convert_form('C')
    S_VL = PSI.get_SL(it) ** 0.5
    S_VR = PSI.get_SR(it) ** 0.5

    # make MPS environment to calculate 'L'
    braket = MPSEnvironment(PSI, PSI)
    LP_og = braket.get_LP(it, store=False)
    RP_og = braket.get_RP(it, store=False)

    # npc form is used in tensor calculation, it keeps track of the different labels which are used in the tensorproducts
    # array form is used for calculations of null space
    #A_npc = PSI.get_B(it, 'C')

    A_dag_npc = PSI.get_B(it, 'C').conj()
    #A_dag = A_dag_npc.to_ndarray()

    # !! We have to scale the LP/RP first with the S**.5 in order to use it

    LP = LP_og.scale_axis(S_VL, axis='vR')
    LP = LP.scale_axis(S_VL.conj(), axis='vR*')

    RP = RP_og.scale_axis(S_VR, axis='vL')
    RP = RP.scale_axis(S_VR.conj(), axis='vL*')

    LP_squared = sqrtm(LP.to_ndarray())
    LP_squared_npc = Array.from_ndarray_trivial(LP_squared, labels=['vR*', 'vR'])
    LP_squared_inv = np.linalg.inv(LP_squared)
    LP_squared_inv_npc = Array.from_ndarray_trivial(LP_squared_inv, labels=['vR*', 'vR'])

    RP_squared = sqrtm(RP.to_ndarray())
    #RP_squared_npc = Array.from_ndarray_trivial(RP_squared, labels=['vL', 'vL*'])
    RP_squared_inv = np.linalg.inv(RP_squared)
    RP_squared_inv_npc = Array.from_ndarray_trivial(RP_squared_inv, labels=['vL', 'vL*'])

    # LP is niet de eenheidsmatrix, creëeren tensor 'A_VL'
    A_VL_npc = npc.tensordot(LP_squared_npc, A_dag_npc, axes=(['vR*'], ['vR*']))
    A_VL = A_VL_npc.to_ndarray()

    # null space
    V_L = null_space(A_VL.reshape(dB, dH * dB))
    dim_V_L = V_L.shape[-1]
    V_L = V_L.reshape(dB, dH, dim_V_L)
    V_L_npc = Array.from_ndarray_trivial(V_L, labels=['vL', 'p', 'vR'])

    # create X, a Gaussian random distrubted complex tensor
    X = np.random.normal(0, sigma, (dB, dB)) + np.random.normal(0, sigma, (dB, dB)) * 1j
    X_npc = Array.from_ndarray_trivial(X, labels=['vL', 'vR'])

    # calculate B
    X_R12 = npc.tensordot(X_npc, RP_squared_inv_npc, axes=(['vR'], ['vL']))
    L_VL = npc.tensordot(LP_squared_inv_npc, V_L_npc, axes=(['vR'], ['vL']))
    B_npc = npc.tensordot(L_VL, X_R12, axes=(['vR'], ['vL']))

    B_ar = B_npc.to_ndarray()
    B_site = Array.from_ndarray_trivial(B_ar, labels=['vL', 'p', 'vR'])

    # tangent vector
    psi_tan = PSI.copy()
    psi_tan.set_B(it, B_site, 'C')

    # manual check of <psi_tan,psi> , use of original LP and RP and replace A=B, contract
    #L_B = npc.tensordot(LP, B_site, axes=(['vR'], ['vL']))
    #A_R = npc.tensordot(A_dag_npc, RP, axes=(['vR*'], ['vL*']))
    #manual_contraction = npc.inner(L_B, A_R, axes=([['vR*', 'p', 'vR'], ['vL*', 'p*', 'vL']]), do_conj=False)

    # check norm, we have to scale B to size of A_dag.conj() (from which the Singular Values were used to scale LP)
    # -> norm(X) should be equal to norm(B)

    # B_site = B_site.scale_axis(S_VL.conj(), axis='vL')
    # B_site = B_site.scale_axis(S_VR.conj(), axis='vR')
    # norm_check = B_site.norm() / norm(X)

    #     psi_tan.canonical_form()
    return psi_tan, norm(X)


class ZZXZChain(CouplingModel, NearestNeighborModel, MPOModel):
    def __init__(self, L, J, hx, hz):
        # use predefined local Hilbert space and onsite operators
        site = SpinHalfSite(conserve=None)
        lat = Chain(L, site, bc="open", bc_MPS="finite")  # define geometry
        CouplingModel.__init__(self, lat)

        for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
            self.add_coupling(-J, u1, 'Sigmaz', u2, 'Sigmaz', dx)

        for u in range(len(self.lat.unit_cell)):
            self.add_onsite(-hz, u, 'Sigmaz')

        for u in range(len(self.lat.unit_cell)):
            self.add_onsite(-J * hx, u, 'Sigmax')

        MPOModel.__init__(self, lat, self.calc_H_MPO())
        NearestNeighborModel.__init__(self, lat, self.calc_H_bond())


# def ground_state(L, dB):
#     hx = 100000
#     J = 1
#     hz = 0
#     M = ZZXZChain(L, J, hx, hz)
#     # state up begin state
#     spin = tenpy.networks.site.SpinHalfSite(conserve=None)
#     theta, phi = np.pi / 2, 0
#     bloch_sphere_state = np.array([np.cos(theta / 2), np.exp(1.j * phi) * np.sin(theta / 2)])
#     p_state = [bloch_sphere_state] * L
#     if dB == 1:
#         psi0 = MPS.from_product_state([spin] * L, p_state, bc=M.lat.bc_MPS, dtype=np.complex)
#     else:
#         psi = MPS.from_product_state([spin] * L, p_state, bc=M.lat.bc_MPS, dtype=np.complex)
#         dmrg_params = {
#             'mixer': None,  # setting this to True helps to escape local minima
#             'max_E_err': 1.e-10,
#             'trunc_params': {
#                 'chi_max': dB,
#                 'svd_min': 1.e-10,
#             },
#             'max_sweeps': 10,
#             'verbose': .1,
#             'combine': True
#         }
#         eng = dmrg.TwoSiteDMRGEngine(psi, M, dmrg_params)
#         E, psi0 = eng.run()  # the main work; modifies psi in place
#     return psi0

def ground_state(L, dB):
    hx = 10000
    J = 1
    hz = 0
    M = ZZXZChain(L, J, hx, hz)
    # state up begin state
    spin = tenpy.networks.site.SpinHalfSite(conserve=None)
    theta, phi = np.pi / 2, 0
    bloch_sphere_state = np.array([np.cos(theta / 2), np.exp(1.j * phi) * np.sin(theta / 2)])
    p_state = [bloch_sphere_state] * L
    psi = MPS.from_product_state([spin] * L, p_state, bc=M.lat.bc_MPS, dtype=np.complex)

    dmrg_params = {
        'mixer': None,  # setting this to True helps to escape local minima
        'max_E_err': 1.e-10,
        'trunc_params': {
            'chi_max': dB,
            'svd_min': 1.e-10,
        },
        'max_sweeps': 100,
        'verbose': 0.1,
        'combine': True
    }
    eng = dmrg.TwoSiteDMRGEngine(psi, M, dmrg_params)
    E, psi0 = eng.run()  # the main work; modifies psi in place
    return psi0

def stochastic_MPS(psi0, std):
    psip = psi0.copy()
    L = psip.L
    for site in range(L - 2):
        psi_tan = tangent_MPS(psi0, site + 1, std)
        # tel beide mps op en herschaal, de normering is approx 2
        psip = psip.add(psi_tan[0], 1, psi_tan[1] ** 2)
        psip.canonical_form()
    return psip

def stochastic_MPS_2(psi0, std , cut):
    psip = psi0.copy()
    L=psip.L
    for site in range(L - int(cut*2)):
        psi_tan = tangent_MPS(psi0, site + cut, std)
        # tel beide mps op en herschaal, de normering is approx 2
        psip = psip.add(psi_tan[0], 1, psi_tan[1] ** 2)
        psip.canonical_form()
    return psip

def dict_to_matrix(dic):
    dictlist=[]
    for key in dic.keys():
        dictlist.append(dic[key])
    return np.array(dictlist)

def theta_0(k):
    return np.angle(10000 - np.exp(1j * k))
