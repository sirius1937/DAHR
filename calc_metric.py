

import numpy as np
def calc_metric(detail_matrix, all_round_score):
    final_accuracy = all_round_score[-1]
    overall_accuracy = np.mean(detail_matrix) * 25 / 15
    F = [detail_matrix[i][i] - detail_matrix[4][i] for i in range(4)]
    forgetness = np.mean(F)
    return final_accuracy, overall_accuracy, forgetness, F