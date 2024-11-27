import numpy as np

def AR_ComputeCE(gt_table, gen_table, eps=1e-9):
    """
    Compute the cross-entropy (CE) between the ground truth table (gt_table)
    and the generated table (gen_table).
    
    Parameters:
        gt_table (numpy.ndarray): Ground truth table.
        gen_table (numpy.ndarray): Generated table.
        eps (float): A small constant to avoid log(0).

    Returns:
        float: The computed cross-entropy.
    """
    # Get the total number of rows in the generated table
    gen_total_num = gen_table.shape[0]

    # Find unique rows and their counts in gt_table
    unique_rows, gt_counts = np.unique(gt_table, axis=0, return_counts=True)
    gt_probabilities = gt_counts / len(gt_table)

    # Create a dictionary for generated table row counts
    gen_table_dics = {}
    for row in gen_table:
        row_tuple = tuple(row)
        gen_table_dics[row_tuple] = gen_table_dics.get(row_tuple, 0) + 1

    # Compute cross-entropy
    ce = 0.0
    for unique_row, gt_prob in zip(unique_rows, gt_probabilities):
        row_tuple = tuple(unique_row)
        
        # Get the generated probability for this row
        if row_tuple in gen_table_dics:
            gen_prob = gen_table_dics[row_tuple] / gen_total_num
        else:
            gen_prob = eps
        
        # Update the cross-entropy
        ce -= gt_prob * np.log(gen_prob)

    return ce