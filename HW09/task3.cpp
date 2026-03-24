#include <mpi.h>
#include <iostream>
#include <vector>

int main(int argc, char* argv[]) {

    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    size_t n = std::stoul(argv[1]);

    std::vector<float> sendbuf(n, 1.0f);
    std::vector<float> recvbuf(n, 0.0f);

    double start, end;

    if (rank == 0) {
        start = MPI_Wtime();

        MPI_Send(sendbuf.data(), n, MPI_FLOAT, 1, 0, MPI_COMM_WORLD);
        MPI_Recv(recvbuf.data(), n, MPI_FLOAT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        end = MPI_Wtime();

        double total_time = (end - start) * 1000.0;
        std::cout << total_time << std::endl;
    }
    else if (rank == 1) {
        start = MPI_Wtime();

        MPI_Recv(recvbuf.data(), n, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Send(sendbuf.data(), n, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);

        end = MPI_Wtime();
    }

    MPI_Finalize();
    return 0;
}
