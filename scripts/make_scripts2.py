import os
import argparse

def main(args):
    script_dir= os.path.join('./scripts', args.script_dir)
    studies = ['embedding_dimension', 'sequence_length', 'reduction_rate']
    for study in studies:
        job_name = f'cla_{study}'
        script_filename = os.path.join(script_dir, f'sbatch_{job_name}')
        SAVE_DIR = os.path.join(args.output_dir, study)

        print(f'\n ==> Job name: {job_name} \n')
        print(f'\n ==> Script filename: {script_filename} \n')
        print(f'\n ==> Save directory: {SAVE_DIR} \n')

        cmd = f"python scripts/scriptor2.py "
        cmd += f"--job_name '{job_name}' "
        cmd += f"--script_filename {script_filename} "
        cmd += f"--study_type {study} "
        cmd += f"--output_dir '{SAVE_DIR}' "
        cmd += f"--user_email {args.user_email} "
        cmd += f"--python_module {args.python_module} "
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
    parser.add_argument('--user_email', type=str, default='yws226@byu.edu')
    parser.add_argument('--python_module', type=str, default="3.8.18")
    parser.add_argument('--conda_env', type=str, default='dnnenv')
    parser.add_argument('--time', type=str, default='24:00:00', help='Time limit for the job')
    parser.add_argument('--ntasks', type=int, default=16, help='Number of tasks')
    parser.add_argument('--mem', type=str, default='64G', help='Memory per node')
    parser.add_argument('--nodes', type=int, default=1, help='Number of nodes')
    parser.add_argument('--gpus', type=int, default=1, help='Number of GPUs')
    
    args = parser.parse_args()
    
    main(args)