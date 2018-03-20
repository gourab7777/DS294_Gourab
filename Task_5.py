import numpy as np

"""
Feature-sign search algorithm
"""

def feat_sign_sear(A, y, sparsity):
    matrix=np.dot(A.T,A)
    cor=np.dot(A.T,y)
    sg = np.dot(y.T, y)
    entry = np.zeros(matrix.shape[0])
    signs = np.zeros(matrix.shape[0], dtype=np.int8)
    active_set = set()
    k = np.inf
    w = 0
    grad = - 2 * cor
    max_grad_zero = np.argmax(np.abs(grad))
    while k > sparsity or np.allclose(w, 0)==False:
        if np.allclose(w, 0):
            cd = np.argmax(np.abs(grad) * (signs == 0))
            if grad[cd] > sparsity:
                signs[cd] = -1.
                entry[cd] = 0.
                active_set.add(cd)
            elif grad[cd] < -sparsity:
                signs[cd] = 1.
                entry[cd] = 0.
                active_set.add(cd)
            if len(active_set) == 0:
                break
        indices = np.array(sorted(active_set))
        matrixn = matrix[np.ix_(indices, indices)]
        corn = cor[indices]
        signs_n = signs[indices]
        rs = corn - sparsity * signs_n / 2
        new_entry = np.linalg.solve(np.atleast_2d(matrixn), rs)
        new_signs = np.sign(new_entry)
        old_entr = entry[indices]
        sign_flips = np.where(abs(new_signs - signs_n) > 1)[0]
        if len(sign_flips) > 0:
            k_n = np.inf
            c_n = new_entry
            k_n = (sg + (np.dot(new_entry,np.dot(matrixn, new_entry))- 2 *np.dot(new_entry, corn)) + sparsity * abs(new_entry).sum())
            for j in sign_flips:
                a = new_entry[j]
                b = old_entr[j]
                prop = b / (b - a)
                cr = old_entr - prop * (old_entr - new_entry)
                cost = sg + (np.dot(cr, np.dot(matrixn, cr))
                              - 2 * np.dot(cr, corn)
                              + sparsity * abs(cr).sum())
                if cost < k_n:
                    k_n = cost
                    best_prop = prop
                    c_n = cr
        else:
            c_n = new_entry;
        entry[indices] = c_n
        zeros = indices[np.abs(entry[indices]) < 1e-15]
        entry[zeros] = 0.
        signs[indices] = np.int8(np.sign(entry[indices]))
        active_set.difference_update(zeros)
        grad = - 2 * cor + 2 * np.dot(matrix, entry)
        k = np.max(abs(grad[signs == 0]))
        w = np.max(abs(grad[signs != 0] + sparsity * signs[signs != 0]))
    return entry

"""
Test data -

A = np.random.random((100,200))
y = np.random.random((100))
sparsity = 0.1
"""