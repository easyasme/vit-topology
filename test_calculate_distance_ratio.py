import torch
import torch.nn.functional as F

def calculate_distance_ratio(x, margin_of_error=0.05, max_sample_size=5000):
    # Total number of data points
    N = x.size(0)
    print(f"Total number of data points (N): {N}")

    # Z-score for 95% confidence level
    Z = 1.960
    print(f"Z-score for 95% confidence level: {Z}")

    p = 0.5 # proportion of variability
    E = margin_of_error

    # Calculate sample size - for normal distribution
    numerator = (Z ** 2) * p * (1 - p)
    denominator = (E ** 2) + ((Z ** 2) * p * (1 - p)) / N
    sample_size = int(numerator / denominator)
    print(f"Calculated sample size: {sample_size}")

    # final sample size
    sample_size = min(sample_size, max_sample_size, N)
    print(f"Final sample size (capped): {sample_size}")

    # Sampling indices
    if N > sample_size:
        probabilities = torch.ones(N, device=x.device)
        indices = torch.multinomial(probabilities, num_samples=sample_size, replacement=False)
        x_sampled = x[indices]
        print(f"Sampled indices: {indices}")
    else:
        x_sampled = x

    # L-infinity norm
    distances = []
    for i in range(x_sampled.size(0)):
        for j in range(i + 1, x_sampled.size(0)):
            dist = torch.max(torch.abs(x_sampled[i] - x_sampled[j]))
            distances.append(dist)
    distances = torch.tensor(distances, device=x_sampled.device)
    max_dist = distances.max()
    min_dist = distances.min()

    # # Pairwise L2 distances
    # dist = F.pdist(x_sampled)
    # max_dist = dist.max()
    # min_dist = dist.min()





    # Calculate distance ratio
    if min_dist == 0:
        ratio = float('inf')
    else:
        ratio = max_dist / min_dist

    print(f"Distance ratio: {ratio}")
    del dist
    return ratio

def test_calculate_distance_ratio():
    print("Running Test 1")
    embedding_dim = 50
    sequence_length = 3000
    batch_size = 10

    x_test_1 = torch.rand(batch_size * sequence_length, embedding_dim, device='cuda')
    ratio_1 = calculate_distance_ratio(x_test_1)
    print(f"Test 1 - Distance Ratio: {ratio_1}\n")

    print("Running Test 2")
    sequence_length = 10000

    x_test_2 = torch.rand(batch_size * sequence_length, embedding_dim, device='cuda')
    ratio_2 = calculate_distance_ratio(x_test_2)
    print(f"Test 2 - Distance Ratio: {ratio_2}\n")

    print("Running Test 3")
    sequence_length = 20000
    x_test_3 = torch.rand(batch_size * sequence_length, embedding_dim, device='cuda')
    ratio_3 = calculate_distance_ratio(x_test_3)
    print(f"Test 3 - Distance Ratio: {ratio_3}\n")


if __name__ == "__main__":
    test_calculate_distance_ratio()
