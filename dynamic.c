#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define WIDTH 640
#define HEIGHT 480
#define MAX_ITER 255
#define WORK_CHUNK_SIZE 10
#define NO_MORE_WORK -1
#define NUM_TRIALS 10

struct complex {
    double real;
    double imag;
};

int cal_pixel(struct complex c) {
    int count = 0;
    double z_real = 0.0, z_imag = 0.0;
    while ((z_real * z_real + z_imag * z_imag <= 4.0) && (count < MAX_ITER)) {
        double temp = z_real * z_real - z_imag * z_imag + c.real;
        z_imag = 2 * z_real * z_imag + c.imag;
        z_real = temp;
        count++;
    }
    return count;
}

void save_pgm(const char *filename, int image[HEIGHT][WIDTH]) {
    FILE *fp = fopen(filename, "w");
    fprintf(fp, "P2\n%d %d\n%d\n", WIDTH, HEIGHT, MAX_ITER);
    for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < WIDTH; j++) {
            fprintf(fp, "%d ", image[i][j]);
        }
        fprintf(fp, "\n");
    }
    fclose(fp);
}

int main(int argc, char** argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int image[HEIGHT][WIDTH];
    MPI_Status status;
    double total_execution_time = 0.0, total_communication_time = 0.0;

    for (int trial = 0; trial < NUM_TRIALS; ++trial) {
        double start_time = MPI_Wtime();
        double communication_time = 0.0;

        if (rank == 0) {
            // Master process
            int rows_sent = 0;
            // Initial distribution of work
            for (int i = 1; i < size; ++i) {
                double comm_start = MPI_Wtime();
                MPI_Send(&rows_sent, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
                communication_time += MPI_Wtime() - comm_start;
                rows_sent += WORK_CHUNK_SIZE;
            }

            
            int processed_rows = 0;
            while (processed_rows < HEIGHT) {
                int worker_row_start;
                int buffer[WORK_CHUNK_SIZE][WIDTH];

                double comm_start = MPI_Wtime();
                MPI_Recv(&buffer, WORK_CHUNK_SIZE * WIDTH, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
                communication_time += MPI_Wtime() - comm_start;
                worker_row_start = status.MPI_TAG;
                processed_rows += WORK_CHUNK_SIZE;

             
                for (int i = 0; i < WORK_CHUNK_SIZE && (worker_row_start + i) < HEIGHT; i++) {
                    for (int j = 0; j < WIDTH; j++) {
                        image[worker_row_start + i][j] = buffer[i][j];
                    }
                }

                if (rows_sent < HEIGHT) {
                    comm_start = MPI_Wtime();
                    MPI_Send(&rows_sent, 1, MPI_INT, status.MPI_SOURCE, 0, MPI_COMM_WORLD);
                    communication_time += MPI_Wtime() - comm_start;
                    rows_sent += WORK_CHUNK_SIZE;
                } else {
                    int terminate_signal = NO_MORE_WORK;
                    comm_start = MPI_Wtime();
                    MPI_Send(&terminate_signal, 1, MPI_INT, status.MPI_SOURCE, 0, MPI_COMM_WORLD);
                    communication_time += MPI_Wtime() - comm_start;
                }
            }

            save_pgm("mandelbrot.pgm", image);
        } else {
         
            while (1) {
                int row_start;
                double comm_start = MPI_Wtime();
                MPI_Recv(&row_start, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
                communication_time += MPI_Wtime() - comm_start;
                if (row_start == NO_MORE_WORK) break;

                int buffer[WORK_CHUNK_SIZE][WIDTH];
                for (int i = 0; i < WORK_CHUNK_SIZE && (row_start + i) < HEIGHT; i++) {
                    for (int j = 0; j < WIDTH; j++) {
                        struct complex c = {
                         .real = (j - WIDTH / 2.0) * 4.0 / WIDTH,
.imag = (i + row_start - HEIGHT / 2.0) * 4.0 / HEIGHT
};
buffer[i][j] = cal_pixel(c);
}
}

MPI_Send(&buffer, WORK_CHUNK_SIZE * WIDTH, MPI_INT, 0, row_start, MPI_COMM_WORLD);
communication_time += MPI_Wtime() - comm_start;
}

}

double end_time = MPI_Wtime();
double execution_time = end_time - start_time;
double local_communication_time = communication_time;


MPI_Reduce(&local_communication_time, &total_communication_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

if (rank == 0) {
    total_execution_time += execution_time;
    printf("Trial %d: Execution time: %f seconds\n", trial + 1, execution_time);
    printf("Trial %d: Communication time: %f seconds\n", trial + 1, communication_time);
}

MPI_Barrier(MPI_COMM_WORLD);
}

if (rank == 0) {
    double average_execution_time = total_execution_time / NUM_TRIALS;
    double average_communication_time = total_communication_time / NUM_TRIALS / (size - 1); 
    printf("Average Execution Time: %f seconds\n", average_execution_time);
    printf("Average Communication Time: %f seconds\n", average_communication_time);
}

MPI_Finalize();
return 0;
}

                           
