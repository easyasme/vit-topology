import argparse

def generate_sbatch_script(job_name, script_filename, study_type, output_dir, time='24:00:00', ntasks=1, mem='16G', nodes=1, gpus=1, partition='compute', cpus_per_task=4, user_email='your_email@example.com', python_module='python/3.8', conda_env='your_env'):
    # Create sbatch script content
    sbatch_script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={job_name}_output.txt
#SBATCH --error={job_name}_error.txt
#SBATCH --time={time}
#SBATCH --partition={partition}
#SBATCH --nodes={nodes}
#SBATCH --ntasks={ntasks}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --gres=gpu:{gpus}
#SBATCH --mem={mem}
#SBATCH --mail-user={user_email}
#SBATCH --mail-type=BEGIN,END,FAIL

# Load modules or activate environment
module load {python_module}
source activate {conda_env}

# Run the meta-study for {study_type}
python cla.py --study {study_type} --output_dir {output_dir}/{study_type}
"""

    with open(script_filename, 'w') as f:
        f.write(sbatch_script)

    print(f"Generated sbatch script: {script_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate sbatch script for CLA meta-study')
    parser.add_argument('--job_name', type=str, required=True, help='Job name')
    parser.add_argument('--script_filename', type=str, required=True, help='Filename for the sbatch script')
    parser.add_argument('--study_type', type=str, required=True, choices=['embedding_dimension', 'sequence_length', 'reduction_rate'], help='Type of parameter being studied')
    parser.add_argument('--output_dir', type=str, default='meta_study_results', help='Directory where results will be saved')
    parser.add_argument('--user_email', type=str, default='yws226@byu.edu')
    parser.add_argument('--python_module', type=str, default='python/3.8.18')
    parser.add_argument('--conda_env', type=str, default='dnnenv')
    args = parser.parse_args()

    generate_sbatch_script(
        job_name=args.job_name,
        script_filename=args.script_filename,
        study_type=args.study_type,
        output_dir=args.output_dir,
        user_email=args.user_email,
        python_module=args.python_module,
        conda_env=args.conda_env
    )