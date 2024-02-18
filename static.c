#include <stdio.h>
#include <time.h>
#include <mpi.h>
#include <string.h>

#define WIDTH 640
#define HEIGHT 480
#define MAX_ITER 255

struct complex{
  double real;
  double imag;
};


int cal_pixel(struct complex c) {
    

            double z_real = 0;
            double z_imag = 0;

            double z_real2, z_imag2, lengthsq;

            int iter = 0;
            do {
                z_real2 = z_real * z_real;
                z_imag2 = z_imag * z_imag;

                z_imag = 2 * z_real * z_imag + c.imag;
                z_real = z_real2 - z_imag2 + c.real;
                lengthsq =  z_real2 + z_imag2;
                iter++;
            }
            while ((iter < MAX_ITER) && (lengthsq < 4.0));

            return iter;

}

void save_pgm(const char *filename, int image[HEIGHT][WIDTH]) {
    FILE* pgmimg; 
    int temp;
    pgmimg = fopen(filename, "wb"); 
    fprintf(pgmimg, "P2\n"); // Writing Magic Number to the File   
    fprintf(pgmimg, "%d %d\n", WIDTH, HEIGHT);  // Writing Width and Height
    fprintf(pgmimg, "255\n");  // Writing the maximum gray value 
    int count = 0; 
    
    for (int i = 0; i < HEIGHT; i++) { 
        for (int j = 0; j < WIDTH; j++) { 
            temp = image[i][j]; 
            fprintf(pgmimg, "%d ", temp); // Writing the gray values in the 2D array to the file 
        } 
        fprintf(pgmimg, "\n"); 
    } 
    fclose(pgmimg); 
} 


int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
    const int num_trials = 10; 
    double total_runtime = 0.0, total_communication_time = 0.0;

    for (int trial = 0; trial < num_trials; ++trial) {
        double trial_start = MPI_Wtime();

        int partition_height = HEIGHT / world_size;
        int partition_start = partition_height * world_rank;
        int partition_end = partition_start + partition_height;
        
        if (world_rank == world_size - 1) {
            partition_end = HEIGHT;
        }

        int partition[partition_height][WIDTH]; 

        for (int i = partition_start; i < partition_end; ++i) {
            for (int j = 0; j < WIDTH; ++j) {
                struct complex c = {
                    .real = (j - WIDTH / 2.0) * 4.0 / WIDTH,
                    .imag = (i - HEIGHT / 2.0) * 4.0 / HEIGHT
                };
                partition[i - partition_start][j] = cal_pixel(c) % 256;
            }
        }

        double comm_start = MPI_Wtime();
      
        if (world_rank == 0) {
            int full_image[HEIGHT][WIDTH];
            MPI_Gather(partition, partition_height * WIDTH, MPI_INT, full_image, partition_height * WIDTH, MPI_INT, 0, MPI_COMM_WORLD);
            save_pgm("staticmandelbrot.pgm", full_image);
        } else {
            MPI_Gather(partition, partition_height * WIDTH, MPI_INT, NULL, 0, MPI_INT, 0, MPI_COMM_WORLD);
        }
        double comm_end = MPI_Wtime();

        double trial_end = MPI_Wtime();
        total_runtime += trial_end - trial_start;
        total_communication_time += comm_end - comm_start;
    }

    if (world_rank == 0) {
        printf("Average execution time over %d trials: %f ms\n", num_trials, (total_runtime / num_trials) * 1000);
        printf("Average communication time over %d trials: %f ms\n", num_trials, (total_communication_time / num_trials) * 1000);
    }

    MPI_Finalize();
    return 0;
}
