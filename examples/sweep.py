"""Minimal driver: run the 6 pure (V1..V6) + 20 random-mixture VBC simulations for the mixture or the polymer system, writing GSDs to ./data.

Run mode (--mode):
  none : run every simulation locally, sequentially, on the single core.
  cpu  : submit one SLURM job per simulation on the CPU partition  (assumes SLURM is available).
  gpu  : submit one SLURM job per simulation on the GPU partition  (assumes SLURM is available).

Usage: python sweep.py [--system mixture|polymer] [--mode none|cpu|gpu] [-g 0|1] [--e_rep 20] [--e_att 4] [-n 500] [--n_steps 50000].
"""
import argparse
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data")

ENV = ("source ~/load_gcc.sh; "
       "source /software/2020/software/anaconda3/2019.10/etc/profile.d/conda.sh; "
       "conda activate hoomd_plugin_gpu_vbc")


def configs():
    pure = [(f"pure_V{k}", [k]) for k in range(1, 7)]
    mix = [(f"mix_{m:02d}", [1, 2, 3, 4, 5, 6]) for m in range(20)]
    return [(name, vals, i) for i, (name, vals) in enumerate(pure + mix)]


def command(system, name, valencies, seed, args, device):
    script = "run_mixture.py" if system == "mixture" else "run_polymer.py"
    out = os.path.join(OUT, f"{system}_{name}_g{args.use_global_dens}.gsd")
    vals = " ".join(str(v) for v in valencies)
    return (f"python {os.path.join(HERE, script)} -v {vals} --seed {seed} "
            f"--e_rep {args.e_rep} --e_att {args.e_att} -g {args.use_global_dens} "
            f"-n {args.n_particles} --n_steps {args.n_steps} -d {device} -o {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", choices=["mixture", "polymer"], default="mixture")
    ap.add_argument("--mode", choices=["none", "cpu", "gpu"], default="none")
    ap.add_argument("-g", "--use_global_dens", type=int, choices=[0, 1], default=1)
    ap.add_argument("--e_rep", type=float, default=20.0)
    ap.add_argument("--e_att", type=float, default=4.0)
    ap.add_argument("-n", "--n_particles", type=int, default=500)
    ap.add_argument("--n_steps", type=int, default=50000)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    device = "gpu" if args.mode == "gpu" else "cpu"
    for name, valencies, seed in configs():
        cmd = command(args.system, name, valencies, seed, args, device)
        if args.mode == "none":
            subprocess.run(cmd, shell=True, check=True)
        else:
            partition = "g" if args.mode == "gpu" else "c"
            gres = " --gres=gpu:1" if args.mode == "gpu" else ""
            subprocess.run(f'sbatch -p {partition}{gres} -c 2 --mem=4G --wrap="{ENV}; {cmd}"', shell=True, check=True)


if __name__ == "__main__":
    main()
