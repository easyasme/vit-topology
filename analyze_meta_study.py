import os
import pickle
import matplotlib.pyplot as plt

def analyze_meta_study(study, results_dir = 'meta_study_results'):
    study_dir = os.path.join(results_dir, study)
    files = [f for f in os.listdir(study_dir) if f.endswith('.pkl')]

    data = [] # store data loaded from each file

    for file in files:
        with open(os.path.join(study_dir, file), 'rb') as f:
            params = pickle.load(f)
            data.append(params)

    # data ex) {'distribution': 'uniform', 'embedding_dimension': 100, 'delta': 0.2}
    distributions = set([d['distribution'] for d in data])
    varying_param_name = study
    varying_param_values = sorted(set([d[varying_param_name] for d in data]))

    plt.figure(figsize=(10, 6))

    for distribution in distributions:
        deltas = []
        varying_values = []

        # Filter data to gather value for specific distribution
        for d in data:
            if d['distribution'] == distribution:
                varying_values.append(d[varying_param_name])
                deltas.append(d['delta'])

        # ascending order
        sorted_data = sorted(zip(varying_values, deltas))
        varying_values_sorted, deltas_sorted = zip(*sorted_data)

        plt.plot(varying_values_sorted, deltas_sorted, marker='o', label=f'Distribution: {distribution}')

    plt.xlabel(varying_param_name.title())
    plt.ylabel('Delta')
    plt.title(f'Delta vs {study.title()} Study')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    studies = ['embedding_dimension', 'sequence_length', 'reduction_rate']
    for study in studies:
        analyze_meta_study(study)