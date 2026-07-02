import hoomd
import gsd.hoomd
import time
# import matplotlib.pyplot as plt
import numpy as np
from hoomd.vbc_plugin import vbc_potential
# from hoomd.md import 
import argparse

#create parser
parser = argparse.ArgumentParser(description='')
parser.add_argument('--device', '-d', type=str, default='gpu', help='cpu or gpu', choices=['cpu', 'gpu'])
parser.add_argument('--potential','-p', type=str, default='custom', help='custom or hoomd', choices=['custom', 'hoomd'])
args = parser.parse_args()

# from hoomd.azplugins import pair
nlist = hoomd.md.nlist.Tree(buffer=0.0, exclusions=('bond',))

# nlist

# ph = pair.Hertz(nlist)

if args.potential == 'custom':
    vbc = vbc_potential.VBC(nlist, default_r_cut=2.0)
elif args.potential == 'hoomd':
    # HOOMD's OWN built-in square-density many-body potential (external API, kept
    # verbatim for the optional parity comparison — NOT our plugin's naming).
    vbc = hoomd.md.many_body.SquareDensity(nlist, default_r_cut=2.0)

# a = 10.0
# b = 10.0
# c = 1.0

if args.potential == 'custom':
    vbc.params[('A', 'A')] = dict(A=5.0, B=-5.0, attr_rcut=1.5, rep_rcut=1.0, density_upper_rcut=2.0, density_lower_rcut=1.5, ncut_ij=1.0, ncut_ji=1.0, use_global_dens=0)
else:
    vbc.params[('A', 'A')] = dict(A=5.0, B=5.0)
# vbc.params[('B', 'B')] = dict(A=a, B=b)
# vbc.params[('A', 'A')] = dict(A=a, B=b)



snapshot = gsd.hoomd.Frame()

# Place a polymer in the box.
N = 10
snapshot.particles.N = N
np.random.seed(3)
snapshot.particles.position = np.random.randint(-3,3,size=(N,3))# + np.random.randn(N,3)
snapshot.particles.types = ['A']
snapshot.particles.typeid = ([0] * (N//2)) + ([0] * (N//2))
snapshot.configuration.box = [10, 10, 10, 0, 0, 0]

# Connect particles with bonds.
snapshot.bonds.N = N-1
snapshot.bonds.types = ['A-A']
snapshot.bonds.typeid = [0] * (N-1)
snapshot.bonds.group = np.dstack((np.arange(0,N-1),np.arange(1,N)))

with gsd.hoomd.open(name=f'init_{args.potential}_{args.device}.gsd', mode='w') as f:
    f.append(snapshot)

# fig,ax = plt.subplots(1,2, figsize=(10,5))
# ax[0].set(aspect=1, xlim=(-10,10),ylim=(-10,10))
# ax[1].set(aspect=1, xlim=(-10,10),ylim=(-10,10))
# ax[0].scatter(snapshot.particles.position[:,0],snapshot.particles.position[:,1], marker="o",alpha=0.5, ec="k")
# ax[1].scatter(snapshot.particles.position[:,0],snapshot.particles.position[:,2], marker="o",alpha=0.5, ec="k")





# Apply the harmonic potential on the bonds.
harmonic = hoomd.md.bond.Harmonic()
harmonic.params['A-A'] = dict(k=100, r0=1.0)

# Perform the MD simulation.
# range_id = nvtx.start_range(message="Allocate Particle Positions", color="green")

if args.device == 'gpu':
    sim = hoomd.Simulation(device=hoomd.device.GPU(), seed=1, )
else:
    sim = hoomd.Simulation(device=hoomd.device.CPU(), seed=1, )
# allocate_positions()

sim.create_state_from_gsd(filename=f'init_{args.potential}_{args.device}.gsd')
langevin = hoomd.md.methods.Langevin(filter=hoomd.filter.All(), kT=1.0)
langevin.gamma["A"] = 10

integrator = hoomd.md.Integrator(dt=0.005,
                                 methods=[langevin],
                                 forces=[vbc])
gsd_writer = hoomd.write.GSD(filename=f'trajectory_{args.potential}_{args.device}.gsd',
                             trigger=hoomd.trigger.Periodic(1000),
                             mode='wb')
sim.operations.integrator = integrator
sim.operations.writers.append(gsd_writer)


# @nvtx.annotate("Run HOOMD Steps with Memory Transfers", color="red")
# def run_simulation():
start_time = time.time()
np.random.seed(3)
sim.run(20000)
end_time = time.time()
print(f"Simulation took {end_time - start_time} seconds")
# run_simulation()
gsd_writer.flush()

# list(nlist.r_cut.values())

traj = gsd.hoomd.open(f'trajectory_{args.potential}_{args.device}.gsd', 'r')
print(len(traj), "N =", len(traj[0].particles.position))

# idx = 1
# with nlist.cpu_local_nlist_arrays as arrays:
#     nbrs = np.array(arrays.nlist[arrays.head_list[idx]:arrays.head_list[idx]+arrays.n_neigh[idx]])
# print(nbrs) 
print(traj[0].particles.position[:5,:])
print(traj[-1].particles.position[:5,:])

# # new_snap = traj[0]

# new_snap = traj[-1]
# fig,ax = plt.subplots(1,2, figsize=(10,5))
# ax[0].set(aspect=1, xlim=(-10,10),ylim=(-10,10))
# ax[1].set(aspect=1, xlim=(-10,10),ylim=(-10,10))
# ax[0].scatter(new_snap.particles.position[:,0],new_snap.particles.position[:,1], marker="o",alpha=0.5, ec="k")
# ax[1].scatter(new_snap.particles.position[:,0],new_snap.particles.position[:,2], marker="o",alpha=0.5, ec="k")