#!/usr/bin/env python3
"""Script to update main_structure_learning.py with noise_guide_adj logic."""

import re

# Read the file
with open('main_structure_learning.py', 'r', encoding='utf-8') as f:
    content = f.read()

# First, update the train call to use noise_guide_adj instead of connected_components
content = content.replace(
    'connected_components=functional_modules,  # Patel modules for noise',
    'noise_guide_adj=noise_guide_adj,  # For neighbor-based noise'
)

# Replace the functional modules section
old_pattern = r'''    # Step 2: Compute functional modules using Patel connectivity
    # These modules are ONLY used for structure-aware noise injection, NOT for initialization
    print\("\\nComputing functional modules using Patel algorithm\.\.\."\)
    functional_modules = get_functional_modules\(
        data_2d\.numpy\(\),  # \[Total_Rows, N\]
        top_k=args\.top_k_edges
    \)
    print\(f"Found \{len\(functional_modules\)\} functional modules"\)
    if len\(functional_modules\) > 0:
        print\(f"Largest module has \{len\(functional_modules\[0\]\)\} nodes: \{functional_modules\[0\]\[:10\]\}\.\.\." 
              if len\(functional_modules\[0\]\) > 10 else 
              f"Largest module has \{len\(functional_modules\[0\]\)\} nodes: \{functional_modules\[0\]\}"\)'''

new_code = '''    # Step 2: Compute Patel connectivity matrix for noise guidance
    print("\\nComputing Patel connectivity matrix for neighbor-based noise...")
    patel_matrix = compute_patel_matrix(data_2d.numpy())  # [N, N]
    patel_matrix = torch.from_numpy(patel_matrix).float()
    print(f"Patel matrix range: [{patel_matrix.min():.4f}, {patel_matrix.max():.4f}]")
    
    # Step 3: Threshold to keep top edges (create sparse structure)
    total_edges = num_nodes * (num_nodes - 1)
    k_edges = min(args.top_k_edges, total_edges)
    
    mask_off_diag = 1.0 - torch.eye(num_nodes)
    patel_off_diag = patel_matrix * mask_off_diag
    flat_values = patel_off_diag.flatten()
    flat_values = flat_values[flat_values > 0]
    
    if len(flat_values) >= k_edges:
        threshold = torch.topk(flat_values, k_edges).values[-1]
    else:
        threshold = 0.0
    
    print(f"Keeping top {k_edges} edges (threshold: {threshold:.4f})")
    
    # Step 4: Binarize and add self-loops
    adj_binary = (patel_matrix >= threshold).float() * mask_off_diag
    adj_with_self = adj_binary + torch.eye(num_nodes)
    
    # Step 5: Row-normalize
    degree = adj_with_self.sum(dim=1, keepdim=True)
    noise_guide_adj = adj_with_self / (degree + 1e-9)
    
    print(f"Noise guide adj: {adj_binary.sum().item():.0f} edges + {num_nodes} self-loops")'''

# Try direct replacement first
if 'functional_modules = get_functional_modules' in content:
    # Find and replace the section
    lines = content.split('\n')
    new_lines = []
    skip_until_blank = False
    i = 0
    while i < len(lines):
        line = lines[i]
        if '# Step 2: Compute functional modules' in line:
            # Found the start, insert new code
            new_lines.append(new_code)
            # Skip until we find the next section
            while i < len(lines) and 'functional_modules[0]}' not in lines[i]:
                i += 1
            i += 1  # Skip the closing line too
            continue
        new_lines.append(line)
        i += 1
    
    content = '\n'.join(new_lines)
    print('Replaced functional modules section')
else:
    print('Section already replaced or not found')

# Write back
with open('main_structure_learning.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('Update complete!')
