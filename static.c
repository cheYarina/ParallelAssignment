#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <string.h>

#define WIDTH 640
#define HEIGHT 480
#define MAX_ITER 1000

int main(int argc, char** argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double start_time = MPI_Wtime();

    double real_min = -2.0, real_max = 1.0, imag_min = -1.0, imag_max = 1.0;
    int rows_per_process = HEIGHT / size, start_row = rows_per_process * rank, end_row = start_row + rows_per_process;
    if (rank == size - 1) end_row = HEIGHT;

    int* data = (int*)malloc(WIDTH * rows_per_process * sizeof(int));

    for (int y = start_row; y < end_row; y++) {
        for (int x = 0; x < WIDTH; x++) {
            double c_real = real_min + (real_max - real_min) * x / WIDTH;
            double c_imag = imag_min + (imag_max - imag_min) * y / HEIGHT;
            double z_real = 0.0, z_imag = 0.0;
            int iter = 0;
            while (z_real * z_real + z_imag * z_imag < 4.0 && iter < MAX_ITER) {
                double temp = z_real * z_real - z_imag * z_imag + c_real;
                z_imag = 2.0 * z_real * z_imag + c_imag;
                z_real = temp;
                iter++;
            }
            data[(y - start_row) * WIDTH + x] = (iter == MAX_ITER) ? 0 : iter % 256;
        }
    }

    int* final_data = NULL;
    if (rank == 0) final_data = (int*)malloc(WIDTH * HEIGHT * sizeof(int));

    MPI_Gather(data, WIDTH * rows_per_process, MPI_INT, final_data, WIDTH * rows_per_process, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        FILE* fp = fopen("mandelbrotstatic.pgm", "w"); 
        fprintf(fp, "P2\n%d %d\n255\n", WIDTH, HEIGHT);
        for (int i = 0; i < HEIGHT; i++) {
            for (int j = 0; j < WIDTH; j++) {
               
                fprintf(fp, "%d ", final_data[i * WIDTH + j]);
            }
            fprintf(fp, "\n");
        }
        fclose(fp);
        free(final_data); 
    }

    free(data);

    double end_time = MPI_Wtime();
    if (rank == 0) {
        printf("Total execution time: %f seconds\n", end_time - start_time);
    }

    MPI_Finalize();
    return 0;
}
