import argparse
import os


def generate_sbatch_script(job_name, script_filename, study_type, output_dir, time='24:00:00', ntasks=16, mem='64G', nodes=1, gpus=1, user_email='your_email@example.com', python_module='python/3.8', conda_env='your_env'):
    sbatch_script = f"""#!/bin/bash

#SBATCH --job-name={job_name}
#SBATCH --output={job_name}_output.txt
#SBATCH --error={job_name}_error.txt
#SBATCH --time={time}
#SBATCH --qos=cs
#SBATCH --nodes={nodes}
#SBATCH --ntasks={ntasks}
#SBATCH --gpus={gpus}
#SBATCH --mem={mem}
#SBATCH --mail-user={user_email}
#SBATCH --mail-type=BEGIN,END,FAIL

# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
export OMPI_MCA_opal_cuda_support=true
export OMPI_MCA_pml="ucx" 
export OMPI_MCA_osc="ucx"
export UCX_MEMTYPE_CACHE=n 

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.conda/envs/topo_gph/lib

# Load modules or activate environment
conda init bash
source ~/.bashrc
conda activate {conda_env}

# Run the meta-study for {study_type}
python cla.py --study '{study_type}' --output_dir '{output_dir}'
"""
    
    if not os.path.exists(script_filename):
        os.makedirs(os.path.dirname(script_filename), exist_ok=True)
    with open(script_filename, 'w') as f:
        f.write(sbatch_script)

    print(f"Generated sbatch script: {script_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate sbatch script for CLA meta-study')
    
    parser.add_argument('--job_name', type=str, required=True, help='Job name')
    parser.add_argument('--script_filename', type=str, required=True, help='Filename for the sbatch script')
    parser.add_argument('--study_type', type=str, required=True, choices=['embedding_dimension', 'sequence_length', 'reduction_rate'], help='Type of parameter being studied')
    parser.add_argument('--output_dir', type=str, default='meta_study_results', help='Where results are saved')
    parser.add_argument('--user_email', type=str, default='yws226@byu.edu')
    parser.add_argument('--python_module', type=str, default='python/3.8.18')
    parser.add_argument('--conda_env', type=str, default='dnnenv')
    parser.add_argument('--time', type=str, default='24:00:00', help='Time limit for the job')
    parser.add_argument('--ntasks', type=int, default=16, help='Number of tasks')
    parser.add_argument('--mem', type=str, default='64G', help='Memory per node')
    parser.add_argument('--nodes', type=int, default=1, help='Number of nodes')
    parser.add_argument('--gpus', type=int, default=1, help='Number of GPUs')
    
    args = parser.parse_args()

    generate_sbatch_script(
        job_name=args.job_name,
        script_filename=args.script_filename,
        study_type=args.study_type,
        output_dir=args.output_dir,
        user_email=args.user_email,
        python_module=args.python_module,
        conda_env=args.conda_env,
        time=args.time,
        ntasks=args.ntasks,
        mem=args.mem,
        nodes=args.nodes,
        gpus=args.gpus
    )