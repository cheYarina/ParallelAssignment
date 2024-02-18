#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define IMG_WIDTH 640
#define IMG_HEIGHT 480
#define MAX_ITERATIONS 255

typedef struct {
    double real;
    double imag;
} ComplexNum;

int calculatePixel(ComplexNum c) {
    int iterations = 0;
    double zReal = 0.0, zImag = 0.0;
    double zRealSq, zImagSq, magnitudeSq;
    do {
        zRealSq = zReal * zReal;
        zImagSq = zImag * zImag;
        zImag = 2 * zReal * zImag + c.imag;
        zReal = zRealSq - zImagSq + c.real;
        magnitudeSq = zRealSq + zImagSq;
        iterations++;
    } while (iterations < MAX_ITERATIONS && magnitudeSq < 4.0);
    return iterations;
}

void writeImage(const char *filename, int img[IMG_HEIGHT][IMG_WIDTH]) {
    FILE *file = fopen(filename, "wb");
    fprintf(file, "P2\n%d %d\n255\n", IMG_WIDTH, IMG_HEIGHT);
    for (int i = 0; i < IMG_HEIGHT; i++) {
        for (int j = 0; j < IMG_WIDTH; j++) {
            fprintf(file, "%d ", img[i][j]);
        }
        fprintf(file, "\n");
    }
    fclose(file);
}

int main(int argc, char **argv) {
    int rank, numProcs, trials = 10;
    double totalTime = 0.0, totalCommTime = 0.0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

    for (int trial = 0; trial < trials; trial++) {
        double startTime = MPI_Wtime(), commTime = 0.0;

        if (rank == 0) {
            int img[IMG_HEIGHT][IMG_WIDTH] = {{0}};
            int numRows = IMG_HEIGHT / numProcs;
            int extraRows = IMG_HEIGHT % numProcs;
            int startRow, endRow, numRowsToSend;

            for (int proc = 1; proc < numProcs; proc++) {
                startRow = proc * numRows + (proc <= extraRows ? proc : extraRows);
                numRowsToSend = numRows + (proc <= extraRows);
                endRow = startRow + numRowsToSend - 1;
                MPI_Send(&startRow, 1, MPI_INT, proc, 0, MPI_COMM_WORLD);
                MPI_Send(&numRowsToSend, 1, MPI_INT, proc, 0, MPI_COMM_WORLD);
            }

            for (int proc = 1; proc < numProcs; proc++) {
                MPI_Recv(img[startRow], numRowsToSend * IMG_WIDTH, MPI_INT, proc, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }

            writeImage("dynamic_output.pgm", img);
        } else {
            int startRow, numRows;
            MPI_Recv(&startRow, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&numRows, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            int imgPart[numRows][IMG_WIDTH];
            for (int i = startRow; i < startRow + numRows; i++) {
                for (int j = 0; j < IMG_WIDTH; j++) {
                    ComplexNum c = {.real = (j - IMG_WIDTH / 2.0) * 4.0 / IMG_WIDTH, .imag = (i - IMG_HEIGHT / 2.0) * 4.0 / IMG_HEIGHT};
                    imgPart[i - startRow][j] = calculatePixel(c);
                }
            }
            MPI_Send(imgPart, numRows * IMG_WIDTH, MPI_INT, 0, 0, MPI_COMM_WORLD);
        }

        double endTime = MPI_Wtime();
        double elapsedTime = endTime - startTime;
        MPI_Reduce(&commTime, &totalCommTime, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        totalTime += elapsedTime;
    }

    if (rank == 0) {
        printf("Average Time over 10 trials: %f ms\n", (totalTime / trials) * 1000);
        printf("Average Communication Time over 10 trials: %f ms\n", (totalCommTime / trials) * 1000);
    }

    MPI_Finalize();
    return 0;
}
