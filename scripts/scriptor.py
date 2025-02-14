import argparse
import os


def generate_sbatch_script(job_name, script_filename, output_dir, grid_samples, time='04:00:00', ntasks=16, mem='64G', nodes=1, gpus=1, user_email='your_email@example.com', conda_env='your_env'):
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

# Load modules or activate environment
conda init bash
source ~/.bashrc
conda activate {conda_env}

# Run the meta-study
python cla.py --output_dir '{output_dir}' --grid_samples {grid_samples}
"""
    
    script_dir = os.path.dirname(script_filename)
    if not os.path.exists(script_dir):
        os.makedirs(os.path.dirname(script_filename), exist_ok=True)
    with open(script_filename, 'w') as f:
        f.write(sbatch_script)

    print(f"Generated sbatch script: {script_filename}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate sbatch script for CLA meta-study')
    
    parser.add_argument('--job_name', type=str, required=True, help='Job name')
    parser.add_argument('--script_filename', type=str, required=True, help='Filename for the sbatch script')
    parser.add_argument('--output_dir', type=str, default='meta_study_results', help='Where results are saved')
    parser.add_argument('--grid_samples', type=int, default=6, help='Number of grid samples per variable')
    parser.add_argument('--user_email', type=str, default='yws226@byu.edu')
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
        output_dir=args.output_dir,
        grid_samples=args.grid_samples,
        user_email=args.user_email,
        conda_env=args.conda_env,
        time=args.time,
        ntasks=args.ntasks,
        mem=args.mem,
        nodes=args.nodes,
        gpus=args.gpus
    )

# In the terminal run the following command from ~/vit-topology to execute the sbatch script:
# sbatch scripts/sbatch_scripts/sbatch_cla_meta_study