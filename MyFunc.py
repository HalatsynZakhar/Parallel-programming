import numpy as np
N = 752

def setMatr(fill):
    matr = [[fill] * N for i in range(N)]
    return np.matrix(matr)


def setVec(fill):
    vec = [fill for i in range(N)]
    return np.array(vec)



def mltp_matrix(matr1, matr2):

    temp_matr1 = [[0 for j in range(matr1.shape[1])] for i in range(matr1.shape[0])]
    temp_matr2 = [[0 for j in range(matr2.shape[1])] for i in range(matr2.shape[0])]
    for i in range(len(temp_matr1)):
        for j in range(len(temp_matr1[0])):
            temp_matr1[i][j] = matr1[i, j]

    for i in range(len(temp_matr2)):
        for j in range(len(temp_matr2[0])):
            temp_matr2[i][j] = matr2[i, j]

    matr1 = temp_matr1
    matr2 = temp_matr2

    row1 = len(matr1)
    col1 = len(matr1[0])
    row2 = len(matr2)
    col2 = len(matr2[0])

    if col1 != row2:
        raise ArithmeticError("Matrix Multiply ERROR!")

    matrix_result = [[0] * col2 for i in range(row1)]
    for i in range(row1):
        for j in range(col2):
            matrix_result[i][j] = 0
            for k in range(col1):
                matrix_result[i][j] += matr1[i][k] * matr2[k][j]

    return np.array(matrix_result)


def mltp_matrix_vector(matr: np.matrix, vec:np.ndarray):
    if vec.size != matr[0].size:
        raise ArithmeticError("Cannot perform vector-matrix multiplication, dimension invalid.")

    result_vec = list()
    for i in range(matr.shape[0]):
        tmp = 0
        for j in range(vec.size):
            tmp = tmp + matr[i,j] * vec[i]
        result_vec.append(tmp)
    return np.array(result_vec)

def merge_sort(a,b):
    a = list(a)
    b = list(b)

    pa = 0
    pb = 0
    result = []

    while pa < len(a) and pb < len(b):
        if a[pa] <= b[pb]:
            result.append(a[pa])
            pa += 1
        else:
            result.append(b[pb])
            pb += 1

    remained = a[pa:] + b[pb:]
    result.extend(remained)

    return np.array(result)

def sum_vector(vec1:np.ndarray, vec2:np.ndarray):
   if vec1.size != vec2.size:
       raise IndexError("sumVec: Len vec1!=vec2")
   for i in range(vec1.size):
       vec1[i] += vec2[i]
   return vec1
