import numpy as np
import scipy.linalg as la
import os

hk1 = 37.576610206242016
hk2 = 29.657914022948088
param_K = hk2 / hk1

alpha = 1.45  # area per particle
lp = np.sqrt(alpha * 4.0 / (3.0 * np.sqrt(3.0)))


def d_uxux(qx, qy):
    f1 = 1.5 * hk1 + 3.0 * hk2 - 3.0 * hk2 * \
        np.cos(1.5 * qx) * np.cos(0.5 * np.sqrt(3) * qy)
    return f1


def d_uyuy(qx, qy):
    f1 = 1.5 * hk1 + 3.0 * hk2 - hk2 * np.cos(1.5 * qx) * np.cos(0.5 * np.sqrt(3)
                                                                 * qy) - 2.0 * hk2 * np.cos(np.sqrt(3) * qy)
    return f1


def d_uxuy(qx, qy):
    f1 = hk2 * np.sqrt(3.0) * np.sin(1.5 * qx) * np.sin(0.5 * np.sqrt(3) * qy)
    return f1


def d_uxvy(qx, qy):
    f1 = hk1 * 0.5 * np.sqrt(3.0) * 1j * np.exp(-0.5 * qx * 1j) * \
        np.sin(0.5 * np.sqrt(3) * qy)
    return f1


def d_uxvx(qx, qy):
    f1 = -hk1 * np.exp(qx * 1j) - hk1 * 0.5 * np.exp(-0.5 * qx * 1j) * \
        np.cos(0.5 * np.sqrt(3) * qy)
    return f1


def d_uyvy(qx, qy):
    f1 = -hk1 * 1.5 * np.exp(-0.5 * qx * 1j) * np.cos(0.5 * np.sqrt(3.0) * qy)
    return f1

output_dir='output'
os.mkdir(output_dir)

nq = 80
nqby2 = int(nq / 2)

q0M = np.zeros((nqby2, 2), dtype=np.float64)
q0K = np.zeros((nqby2, 2), dtype=np.float64)


q0M[:, 0] = np.linspace(0, 2.0 * np.pi / (3.0 * lp), nqby2)
q0M[:, 1] = np.linspace(0, 0.0,  nqby2)

q0K[:, 0] = np.linspace(0,  np.sqrt(0.50) * (np.pi / lp), nqby2)
q0K[:, 1] = np.linspace(
    0, -np.sqrt(0.50) * (np.pi / (lp * np.sqrt(3.0))), nqby2)

Dyn = np.zeros((4, 4), dtype=np.complex)
Dyn_ac = np.zeros((2, 2), dtype=np.complex)
Dyn_op = np.zeros((2, 2), dtype=np.complex)

for iq1 in range(nq):
    if iq1 < nqby2:
        q_comp = q0M[iq1, :]
    if nqby2 <= iq1:
        q_comp = q0K[(iq1 - nqby2), :]

    qtemp = (q_comp[0]**2 + q_comp[1]**2)**0.5
    Dyn[0, 0] = d_uxux(q_comp[0], q_comp[1])
    Dyn[0, 1] = d_uxuy(q_comp[0], q_comp[1])
    Dyn[0, 2] = d_uxvx(q_comp[0], q_comp[1])
    Dyn[0, 3] = d_uxvy(q_comp[0], q_comp[1])

    Dyn[1, 0] = np.conj(Dyn[0, 1])
    Dyn[1, 1] = d_uyuy(q_comp[0], q_comp[1])
    Dyn[1, 2] = Dyn[0, 3]
    Dyn[1, 3] = d_uyvy(q_comp[0], q_comp[1])

    Dyn[2, 0] = np.conj(Dyn[0, 2])
    Dyn[2, 1] = np.conj(Dyn[1, 2])
    Dyn[2, 2] = Dyn[0, 0]
    Dyn[2, 3] = Dyn[0, 1]

    Dyn[3, 0] = np.conj(Dyn[0, 3])
    Dyn[3, 1] = np.conj(Dyn[1, 3])
    Dyn[3, 2] = np.conj(Dyn[2, 3])
    Dyn[3, 3] = Dyn[1, 1]

    Dyn_ac[0, 0] = Dyn[0, 0] + np.conj(Dyn[0, 0])
    Dyn_ac[0, 1] = Dyn[0, 1] + np.conj(Dyn[0, 1])
    Dyn_ac[1, 0] = Dyn[1, 0] + np.conj(Dyn[1, 0])
    Dyn_ac[1, 1] = Dyn[1, 1] + np.conj(Dyn[1, 1])

    Dyn_op[0, 0] = Dyn[0, 2] + np.conj(Dyn[0, 2])
    Dyn_op[0, 1] = Dyn[0, 3] + np.conj(Dyn[0, 3])
    Dyn_op[1, 0] = Dyn[1, 2] + np.conj(Dyn[1, 2])
    Dyn_op[1, 1] = Dyn[1, 3] + np.conj(Dyn[1, 3])

    results = la.eig(Dyn)
    ev = sorted(results[0])

    if iq1 < nqby2:
        with open(output_dir+"/harmonic_omega_sq_v_q_G0M.txt", "a") as f2:
            print(q_comp[0], q_comp[1], qtemp, np.real(
                ev[0]), np.real(ev[1]), np.real(ev[2]), np.real(ev[3]), file=f2)
    else:
        with open(output_dir+"/harmonic_omega_sq_v_q_G0K.txt", "a") as f2:
            print(q_comp[0], q_comp[1], qtemp, np.real(
                ev[0]), np.real(ev[1]), np.real(ev[2]), np.real(ev[3]), file=f2)

    results = la.eig(Dyn_ac)
    ev = sorted(results[0])

    if iq1 < nqby2:
        with open(output_dir+"/harmonic_omega_sq_v_q_G0M_ac.txt", "a") as f2:
            print(q_comp[0], q_comp[1], qtemp, np.real(
                ev[0]), np.real(ev[1]), file=f2)
    else:
        with open(output_dir+"/harmonic_omega_sq_v_q_G0K_ac.txt", "a") as f2:
            print(q_comp[0], q_comp[1], qtemp, np.real(
                ev[0]), np.real(ev[1]),  file=f2)

    results = la.eig(Dyn_op)
    ev = sorted(results[0])

    if iq1 < nqby2:
        with open(output_dir+"/harmonic_omega_sq_v_q_G0M_op.txt", "a") as f2:
            print(q_comp[0], q_comp[1], qtemp, np.real(
                ev[0]), np.real(ev[1]), file=f2)
    else:
        with open(output_dir+"/harmonic_omega_sq_v_q_G0K_op.txt", "a") as f2:
            print(q_comp[0], q_comp[1], qtemp, np.real(
                ev[0]), np.real(ev[1]),  file=f2)
