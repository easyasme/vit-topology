import os
import argparse

def main(args):
    job_name = f'cla_meta_study'
    script_dir = os.path.join('./scripts', args.script_dir)
    script_filename = os.path.join(script_dir, f'sbatch_{job_name}_gs_{args.grid_samples}')

    print(f'\n ==> Job name: {job_name}')
    print(f'\n ==> Script filename: {script_filename}')
    print(f'\n ==> Save directory: {args.output_dir} \n')

    cmd = f"python scripts/scriptor.py "
    cmd += f"--job_name '{job_name}' "
    cmd += f"--script_filename {script_filename} "
    cmd += f"--output_dir '{args.output_dir}' "
    cmd += f"--grid_samples {args.grid_samples} "
    cmd += f"--user_email {args.user_email} "
    cmd += f"--conda_env {args.conda_env} "
    cmd += f"--time {args.time} "
    cmd += f"--ntasks {args.ntasks} "
    cmd += f"--mem {args.mem} "
    cmd += f"--nodes {args.nodes} "
    cmd += f"--gpus {args.gpus}"
    
    os.system(cmd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate and submit sbatch scripts for CLA meta-study')
    
    parser.add_argument('--script_dir', type=str, default='sbatch_scripts', help='Directory to save sbatch scripts')
    parser.add_argument('--output_dir', type=str, default='meta_study_results')
    parser.add_argument('--grid_samples', type=int, default=6, help='Number of grid samples per variable')
    parser.add_argument('--user_email', type=str, default='yws226@byu.edu')
    parser.add_argument('--conda_env', type=str, default='dnnenv')
    parser.add_argument('--time', type=str, default='04:00:00', help='Time limit for the job')
    parser.add_argument('--ntasks', type=int, default=16, help='Number of tasks')
    parser.add_argument('--mem', type=str, default='64G', help='Memory per node')
    parser.add_argument('--nodes', type=int, default=1, help='Number of nodes')
    parser.add_argument('--gpus', type=int, default=1, help='Number of GPUs')
    
    args = parser.parse_args()
    
    main(args)

# In the terminal run the following command from ~/vit-topology to generate the sbatch script:
# python scripts/make_scripts.py --user_email jdoe@byu.edu --grid_samples 6 --conda_env your_env --time 04:00:00 --ntasks 16 --mem 64G --nodes 1 --gpus 2
#
# Then run the following command to submit the job:
# sbatch scripts/sbatch_scripts/sbatch_cla_meta_study
#
# To check the status of the job, run:
# squeue -u your_net_id
#
# To cancel the job, run:
# scancel -u your_net_id
#
# Note that the job will be automatically cancelled after the time limit specified in the sbatch script.
# Also, the job run-time is currently dependent on the number of grid samples and the number of variables in the 
# meta-study. To reduce the run-time, reduce the number of grid samples.
