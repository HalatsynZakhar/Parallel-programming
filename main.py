# Задача: A= sort(Z)+ D*(MX*MS))


from mpi4py import MPI
from MyFunc import *
from time import time, sleep
# mpiexec -n 4 python main.py
P = 4
H = int(N / P)
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
# print("Rank", rank)
# print("Size", size)

if rank == 0:
    print("RGR started\nT1 started")  # ------------------------------------------------------T1

    # 1. 	Прийняти від задачі T2 дані: MX (для T1), Z_h(для T1), D(для T1)

    MX = np.zeros(shape=(N, N), dtype=int)
    comm.Recv(MX, source=1, tag=3)
    D = np.zeros(shape=N, dtype=int)
    comm.Recv(D, source=1, tag=4)
    Zh = np.zeros(shape=H, dtype=int)
    comm.Recv(Zh, source=1, tag=5)

    # 2.	Прийняти від задачі T4 дані: MS(для T1)

    MSh = np.zeros(shape=(H, N), dtype=int)
    comm.Recv(MSh, source=3, tag=12)

    #	3.  Sh = sort(Z_h)

    Sh = np.array(sorted(Zh))

    # 4.	Прийняти від задачі T2 дані: Sh

    Sh_task2 = np.zeros(shape=H, dtype=int)
    comm.Recv(Sh_task2, source=1, tag=1)

    # 5.	〖S2〗_h=mergesort(S_h,S_h)

    S2h = merge_sort(Sh, Sh_task2)

    # 6. Прийняти від задачі T2 дані: S2h

    S2h_task3 = np.zeros(shape=H * 2, dtype=int)
    comm.Recv(S2h_task3, source=1, tag=16)

    # 7. 	S = mergesort(S2_h,S2_h)

    S_ressort = merge_sort(S2h, S2h_task3)

    # 8.	Надіслати T2 данні: Sh (для T2, T3)

    comm.Send(S_ressort[H:3 * H], dest=1, tag=17)

    # 9.	Надіслати T4 данні: Sh (для T4)

    comm.Send(S_ressort[3 * H:4 * H], dest=3, tag=18)
    Sh = S_ressort[0:H]

    #	10. A_h=S_h+D*(MX*MS_h)
    Ah = sum_vector(Sh, mltp_matrix_vector(mltp_matrix(MSh, MX), D))

    #   11.	Надіслати T2 данні: Ah
    comm.Send(Ah, dest=1, tag=20)

    print("T1 finished")

elif rank == 1:
    start_time = time()
    print("T2 started")  # ------------------------------------------------------T2

    # 1.	Введення MX.

    MX = setMatr(1)

    #	2. Прийняти від T3 дані: Z_2h(для T1, T2), D (для T1, T2)
    D = np.zeros(shape=N, dtype=int)
    comm.Recv(D, source=2, tag=1)
    Z = np.zeros(shape=H * 2, dtype=int)
    comm.Recv(Z, source=2, tag=2)

    #	3. Передати T1 дані: MX (для T1), Z_h(для T1), D(для T1)
    comm.Send(MX, dest=0, tag=3)
    comm.Send(D, dest=0, tag=4)
    comm.Send(Z[:H], dest=0, tag=5)

    Zh = Z[H:2 * H]

    #   4.	Передати T3 дані: MX (для T3, T4)

    comm.Send(MX, dest=2, tag=6)

    # 5.	Прийняти від T3 дані: MSh (для T2)

    MSh = np.zeros(shape=(H, N), dtype=int)
    comm.Recv(MSh, source=2, tag=11)

    #	6. S_h=sort(Z_h)

    Sh = np.array(sorted(Zh))

    #   7.	Передати T1 дані: Sh
    comm.Send(Sh, dest=0, tag=1)

    #   8.	Прийняти від T3 данні: S2h

    S2h_task3 = np.zeros(shape=H * 2, dtype=int)
    comm.Recv(S2h_task3, source=2, tag=15)

    #   9.	Передати T1 данні:S2h

    comm.Send(S2h_task3, dest=0, tag=16)

    #   10.	Прийняти від T1 данні: Sh (для T2, T3)

    Sh_ressort = np.zeros(shape=H * 2, dtype=int)
    comm.Recv(Sh_ressort, source=0, tag=17)

    Sh = Sh_ressort[0:H]

    #   11.	Передати для T3 данні: Sh (для T3)

    comm.Send(Sh_ressort[H:2 * H], dest=2, tag=19)

    #	12. A_h=S_h+D*(MX*MS_h)
    Ah = sum_vector(Sh, mltp_matrix_vector(mltp_matrix(MSh, MX), D))

    A = np.zeros(shape=N, dtype=int)

    # 1.	Прийняти від T1 дані: Ah
    comm.Recv(A[:H], source=0, tag=20)

    # 14.	Прийняти від T3 дані: A2h
    A[H:2 * H] = Ah

    comm.Recv(A[2 * H:4 * H], source=2, tag=22)

    # 15.	Виведення результату
    print(A)

    print("T2 finished")
    print(time()-start_time)
elif rank == 2:
    print("T3 started")  # ------------------------------------------------------T3

    # 1.	Введення Z, D

    Z = setVec(1)
    D = setVec(1)

    # 2.	Передати T2 дані: Z_2h(для T1,T2), D (для T1, T2)
    comm.Send(D, dest=1, tag=1)
    comm.Send(Z[:2 * H], dest=1, tag=2)

    Zh = Z[2 * H:3 * H]

    #   3.	Отримати від T2 дані: MX(для T3, T4)

    MX = np.zeros(shape=(N, N), dtype=int)
    comm.Recv(MX, source=1, tag=6)

    #   4.  Передати T4 дані: Z_h(для T4), D(для T4), MX (для T4)

    comm.Send(Z[3 * H:4 * H], dest=3, tag=7)
    comm.Send(D, dest=3, tag=8)
    comm.Send(MX, dest=3, tag=9)

    #	5. Отримати від T4 дані: MS_h(для T2, T3)
    MS2h = np.zeros(shape=(2 * H, N), dtype=int)
    comm.Recv(MS2h, source=3, tag=10)

    # 	6. Передати T2 дані: MS_h(для T2)

    comm.Send(MS2h[0:H], dest=1, tag=11)

    MSh = MS2h[H:2 * H]

    #   7. S_h=sort(Z_h)

    Sh = np.array(sorted(Zh))

    #   8.	Отримати від T4 дані: Sh

    Sh_task4 = np.zeros(shape=H, dtype=int)
    comm.Recv(Sh_task4, source=3, tag=14)

    #	9. 	〖S2〗_h=mergesort(S_h,S_h)
    S2h = merge_sort(Sh, Sh_task4)

    #   10.	Передати T2 данні: S2h

    comm.Send(S2h, dest=1, tag=15)

    #   11.	Отримати від T2 данні: Sh (для T3)
    comm.Recv(Sh, source=1, tag=19)

    #	12. A_h = S_h + D * (MX * MS_h)
    Ah = sum_vector(Sh, mltp_matrix_vector(mltp_matrix(MSh, MX), D))

    #   1.	Прийняти від T4 дані: Ah
    Ah_task3_4 = np.zeros(shape=H * 2, dtype=int)
    Ah_task3_4[0:H] = Ah

    comm.Recv(Ah_task3_4[H:2 * H], source=3, tag=21)

    #   14.	Надіслати T2 дані: A2h
    comm.Send(Ah_task3_4, dest=1, tag=22)

    print("T3 finished")
elif rank == 3:
    print("T4 started")  # ------------------------------------------------------T4

    # 1.  Введення MS

    MS = setMatr(1)

    # 2.	Отримати від T3 дані: Z_h(для T4), D(для T4).

    Zh = np.zeros(shape=H, dtype=int)
    comm.Recv(Zh, source=2, tag=7)

    D = np.zeros(shape=N, dtype=int)
    comm.Recv(D, source=2, tag=8)

    MX = np.zeros(shape=(N, N), dtype=int)
    comm.Recv(MX, source=2, tag=9)

    #	3. Передати T3 дані: MS_h(для T2, T3)

    comm.Send(MS[H:3 * H], dest=2, tag=10)

    #   4. 	Передати T1 дані: MS_h(для T1)

    comm.Send(MS[:H], dest=0, tag=12)

    MSh = MS[3 * H:4 * H]

    #   5. S_h=sort(Z_h)

    Sh = np.array(sorted(Zh))

    # 6.	Передати T3 данні: Sh

    comm.Send(Sh, dest=2, tag=14)

    #   7.	Отримати від T1 данні: Sh(для T4)
    comm.Recv(Sh, source=0, tag=18)

    #	8. A_h=S_h+D*(MX*MS_h)

    Ah = sum_vector(Sh, mltp_matrix_vector(mltp_matrix(MSh, MX), D))

    #   9.	Надіслати T3 дані: Ah

    comm.Send(Ah, dest=2, tag=21)

    print("T4 finished")
