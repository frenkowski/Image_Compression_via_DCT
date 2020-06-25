# Metodi del Calcolo Scientifico
# Progetto_2
# Mohammad Alì Manan (817205)
# Francesco Porto (816042)
# Stranieri Francesco (816551)

# https://docs.scipy.org/doc/scipy/reference/generated/scipy.fftpack.dct.html
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.fftpack.dctn.html
from scipy.fftpack import dct, idct
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

import seaborn as sns
sns.set()


def dct1_Our(vector):
    N = len(vector)
    coefficients = np.zeros(N)

    for k in range(0, N):
        alpha = 1/np.sqrt(N) if k == 0 else 1/np.sqrt(N/2)
        for i in range(0, N):
            coefficients[k] += vector[i] * np.cos(k * np.pi * (2*i+1) / (2*N))
        coefficients[k] *= alpha

    return coefficients


def idct1_Our(coefficients):
    N = len(coefficients)
    vector = np.zeros(N)

    for i in range(0, N):
        for k in range(0, N):
            alpha = 1/np.sqrt(N) if k == 0 else 1/np.sqrt(N/2)
            vector[i] += coefficients[k] * alpha * \
                np.cos(k * np.pi * (2*i+1) / (2*N))

    return vector


# vectorExample
vectorExample = np.array([231, 32, 233, 161, 24, 71, 140, 245])
print('VectorExample')
print(vectorExample)
print()

# DCT1
print('DCT1 with SciPy')
vectorExampleDCT = dct(vectorExample, type=2, norm='ortho')
print(vectorExampleDCT)

print('DCT1_Our')
vectorExampleDCT1_Our = dct1_Our(vectorExample)
print(vectorExampleDCT1_Our)

print('Are all elements equal?', np.allclose(
    vectorExampleDCT, vectorExampleDCT1_Our))
print()

# IDCT1
print('IDCT1 with SciPy')
vectorExampleIDCT = idct(vectorExampleDCT, type=2, norm='ortho')
print(vectorExampleIDCT)

print('IDCT1_Our')
vectorExampleIDCT1_Our = idct1_Our(vectorExampleDCT)
print(vectorExampleIDCT1_Our)

print('Are all elements equal?', np.allclose(
    vectorExampleIDCT, vectorExampleIDCT1_Our))
print()
print()


def dct2_Our(matrix):
    N = matrix.shape[0]
    M = matrix.shape[1]
    coefficients = np.zeros((N, M))

    for k in range(N):
        coefficients[k, :] = dct1_Our(matrix[k, :])
    for l in range(M):
        coefficients[:, l] = dct1_Our(coefficients[:, l])

    return coefficients


def idct2_Our(coefficients):
    N = coefficients.shape[0]
    M = coefficients.shape[1]
    matrix = np.zeros((N, M))

    for k in range(N):
        matrix[k, :] = idct1_Our(coefficients[k, :])
    for l in range(M):
        matrix[:, l] = idct1_Our(matrix[:, l])

    return matrix


# matrixExample
print('MatrixExample')
matrixExample = np.array([
    [231, 32, 233, 161, 24, 71, 140, 245],
    [247, 40, 248, 245, 124, 204, 36, 107],
    [234, 202, 245, 167, 9, 217, 239, 173],
    [193, 190, 100, 167, 43, 180, 8, 70],
    [11, 24, 210, 177, 81, 243, 8, 112],
    [97, 195, 203, 47, 125, 114, 165, 181],
    [193, 70, 174, 167, 41, 30, 127, 245],
    [87, 149, 57, 192, 65, 129, 178, 228],
])
print(matrixExample)
print()

# https://stackoverflow.com/questions/15978468/using-the-scipy-dct-function-to-create-a-2d-dct-ii
# DCT2
print('DCT2 with SciPy')
matrixExampleDCT2 = dct(dct(matrixExample.transpose(
), type=2, norm='ortho').transpose(), type=2, norm='ortho')
print(matrixExampleDCT2)

print('DCT2_Our')
matrixExampleDCT2_Our = dct2_Our(matrixExample)
print(matrixExampleDCT2_Our)

print('Are all elements equal?', np.allclose(
    matrixExampleDCT2, matrixExampleDCT2_Our))
print()
print()

# IDCT2
print('IDCT2 with SciPy')
matrixExampleIDCT2 = idct(idct(matrixExampleDCT2.transpose(
), type=2, norm='ortho').transpose(), type=2, norm='ortho')
print(matrixExampleIDCT2)

print('IDCT2_Our')
matrixExampleIDCT2_Our = idct2_Our(matrixExampleDCT2)
print(matrixExampleIDCT2_Our)

print('Are all elements equal?', np.allclose(
    matrixExampleIDCT2, matrixExampleIDCT2_Our))
print()
print()

# https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.random.randint.html
low = 0
high = 255

iterations = 100
startIterations = 2
matrixSize = np.zeros(iterations - startIterations)
timeDCT_SciPy = np.zeros(iterations - startIterations)
timeDCT_Our = np.zeros(iterations - startIterations)

for i in range(startIterations, iterations):

    matrix = np.random.randint(low, high, size=(i, i))
    matrixSize[i-startIterations] = i
    # print(matrix)

    start = time.process_time()
    dct(dct(matrix.transpose(), type=2, norm='ortho').transpose(), type=2, norm='ortho')
    end = time.process_time()
    timeDCT_SciPy[i-startIterations] = "%.3f" % ((end - start) * 1000)

    start = time.process_time()
    dct2_Our(matrix)
    end = time.process_time()
    timeDCT_Our[i-startIterations] = "%.3f" % ((end - start) * 1000)


print('DCT2 with SciPy')
for i in range(0, iterations-startIterations):
    print('MatrixSize ' +
          str(int(matrixSize[i])) + 'x' + str(int(matrixSize[i])))
    print('ExecutionTime', timeDCT_SciPy[i])

print()
print('DCT2_Our')
for i in range(0, iterations-startIterations):
    print('MatrixSize ' +
          str(int(matrixSize[i])) + 'x' + str(int(matrixSize[i])))
    print('ExecutionTime', timeDCT_Our[i])


df = pd.DataFrame(list(zip(matrixSize, timeDCT_SciPy, timeDCT_Our)),
                  columns=['MatrixSize', 'ExecutionTime_SciPy', 'ExecutionTime_Our'])
df.MatrixSize = df.MatrixSize.astype(int)
df.ExecutionTime_SciPy = df.ExecutionTime_SciPy.astype(float)
df.ExecutionTime_Our = df.ExecutionTime_Our.astype(float)
# print(df)

df = df.melt('MatrixSize', var_name='ExecutionTime',  value_name='Values')
# print(df)


# ExecutionTime
plt.figure(figsize=(15,8))
sns.lineplot(x='MatrixSize', y='Values', hue='ExecutionTime',
             style='ExecutionTime', markers=['o', 's'], palette='husl', data=df)
plt.yscale('log')
plt.xlabel('MatrixSize', fontsize=12)
plt.ylabel('ExecutionTime', fontsize=12)
plt.title('Comparison between DCT2 (SciPy vs Our)', fontsize=15)
plt.legend(title='Legend', bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
