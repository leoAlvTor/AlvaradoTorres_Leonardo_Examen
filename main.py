from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.rank

data = list()
matrix_data = list()
tam = 1000
a = np.random.randint(100, size=(tam, tam))

c = np.random.randint(100, size=tam)

nuevoVector = np.zeros(len(c))


comm.Barrier()
t_start = MPI.Wtime()


def multiplicarFila_X_Vector(matrix_fila, vector):
    contador = 0
    for i in matrix_fila:
        nuevoVector[contador] = np.inner(i, vector)
        contador += 1
comm.Barrier()
t_diff = MPI.Wtime() - t_start
print(t_diff)


def dividir(comm_size, matrix):
    global data
    rst = np.array_split(matrix, comm_size)
    for r in rst:
        matrix_data.append(r)
    """
    matrix_size = len(matrix)
    chunk_size = int(matrix_size/comm_size)
    resto = 0
    if matrix_size % comm_size != 0:
        resto = matrix_size - (comm_size*chunk_size)
    if resto != 0:
        ultimo_valor = 0
        vals = 0

        for i in range(comm_size):
            vals = vals + chunk_size
            if i+1 == comm_size:
                for j in range(ultimo_valor, vals+resto):
                    data.append(matrix[j])
                ultimo_valor += chunk_size
            else:
                for j in range(ultimo_valor, vals):
                    data.append(matrix[j])
                ultimo_valor += chunk_size
            matrix_data.append(data)
            data = list()
    else:
        ultimo_valor = 0
        vals = 0
        for i in range(comm_size):
            vals = vals + chunk_size
            for j in range(ultimo_valor, vals):
                data.append(matrix[j])
            ultimo_valor += chunk_size
            matrix_data.append(data)
            data = list()
    """


def dividirMatriz():
    tam = comm.Get_size()

    dividir(tam, a)
    multiplicarFila_X_Vector(matrix_data[rank], c)


senddata = nuevoVector
recvdata = np.zeros(len(c), dtype=np.float)
comm.Reduce(senddata, recvdata, op=MPI.SUM)

dividirMatriz()
print(senddata)
