#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define IMG_WIDTH 640
#define IMG_HEIGHT 480
#define MAX_ITERATIONS 255

typedef struct {
    double real;
    double imaginary;
} ComplexNumber;

int calculatePixel(ComplexNumber c) {
    int iterations = 0;
    double zReal = 0.0, zImag = 0.0, zRealSq, zImagSq, magSq;
    do {
        zRealSq = zReal * zReal;
        zImagSq = zImag * zImag;
        zImag = 2 * zReal * zImag + c.imaginary;
        zReal = zRealSq - zImagSq + c.real;
        magSq = zRealSq + zImagSq;
        iterations++;
    } while (iterations < MAX_ITERATIONS && magSq < 4.0);
    return iterations;
}

void writePGM(const char *filename, int data[IMG_HEIGHT][IMG_WIDTH]) {
    FILE *file = fopen(filename, "wb");
    fprintf(file, "P2\n%d %d\n255\n", IMG_WIDTH, IMG_HEIGHT);
    for (int i = 0; i < IMG_HEIGHT; i++) {
        for (int j = 0; j < IMG_WIDTH; j++) {
            fprintf(file, "%d ", data[i][j]);
        }
        fprintf(file, "\n");
    }
    fclose(file);
}

int main(int argc, char **argv) {
    int processRank, numProcesses;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &processRank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);

    int image[IMG_HEIGHT][IMG_WIDTH] = {0};
    int numRows = IMG_HEIGHT / numProcesses;
    int extraRows = IMG_HEIGHT % numProcesses;
    int startRow = processRank * numRows + (processRank < extraRows ? processRank : extraRows);
    int endRow = startRow + numRows + (processRank < extraRows ? 1 : 0);

    for (int y = startRow; y < endRow; y++) {
        for (int x = 0; x < IMG_WIDTH; x++) {
            ComplexNumber c = {
                .real = (x - IMG_WIDTH / 2.0) * 4.0 / IMG_WIDTH,
                .imaginary = (y - IMG_HEIGHT / 2.0) * 4.0 / IMG_HEIGHT
            };
            image[y][x] = calculatePixel(c);
        }
    }

    if (processRank == 0) {
        for (int i = 1; i < numProcesses; i++) {
            int theirStartRow = i * numRows + (i < extraRows ? i : extraRows);
            int theirEndRow = theirStartRow + numRows + (i < extraRows ? 1 : 0);
            for (int y = theirStartRow; y < theirEndRow; y++) {
                MPI_Recv(image[y], IMG_WIDTH, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
        writePGM("dynamicmandelbrot.pgm", image);
    } else {
        for (int y = startRow; y < endRow; y++) {
            MPI_Send(image[y], IMG_WIDTH, MPI_INT, 0, 0, MPI_COMM_WORLD);
        }
    }

    MPI_Finalize();
    return 0;
}
