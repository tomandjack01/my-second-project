import numpy as np
import pandas as pd

# Load the adjacency matrix
adj = pd.read_csv('results/run_20260110_095617/learned_adjacency.csv', header=None).values
N = adj.shape[0]

# Count directions
directions = []
for i in range(N):
    for j in range(i+1, N):  # Only upper triangle (avoid duplicates)
        a_ij = adj[i, j]
        a_ji = adj[j, i]
        diff = abs(a_ij - a_ji)
        
        if a_ij > a_ji:
            directions.append((i, j, a_ij, a_ji, diff, f'{i}->{j}'))
        else:
            directions.append((i, j, a_ij, a_ji, diff, f'{j}->{i}'))

# Sort by difference (strongest causal relationships first)
directions.sort(key=lambda x: -x[4])

print(f'Total node pairs (excluding diagonal): {len(directions)}')
print(f'\nTop 30 strongest causal directions (by |A[i,j] - A[j,i]|):')
print('-' * 70)
print('Pair       A[i,j]     A[j,i]     Diff       Direction')
print('-' * 70)

for p in directions[:30]:
    print(f'({p[0]},{p[1]})'.ljust(10) + f'{p[2]:.4f}'.ljust(11) + f'{p[3]:.4f}'.ljust(11) + f'{p[4]:.4f}'.ljust(11) + p[5])

# Summary statistics
print('\n' + '=' * 70)
print('Summary Statistics:')
diffs = [d[4] for d in directions]
print(f'  Mean difference: {np.mean(diffs):.4f}')
print(f'  Max difference:  {np.max(diffs):.4f}')
print(f'  Min difference:  {np.min(diffs):.4f}')

# Count strong directions (diff > 0.1)
strong = [d for d in directions if d[4] > 0.1]
print(f'\nStrong causal edges (diff > 0.1): {len(strong)} / {len(directions)}')

# Save full results to CSV
results_df = pd.DataFrame(directions, columns=['Node_i', 'Node_j', 'A[i,j]', 'A[j,i]', 'Diff', 'Direction'])
results_df.to_csv('results/run_20260110_095617/causal_directions.csv', index=False)
print(f'\nFull results saved to: results/run_20260110_095617/causal_directions.csv')
