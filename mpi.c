#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <stdbool.h>

typedef struct knnresult{
  int    ** nidx;    //!< Indices (0-based) of nearest neighbors [m-by-k]
  double ** ndist;   //!< Distance of nearest neighbors          [m-by-k]
  int      d;       //!< Number of dimensions
  int      n;       //!< Number of total points
  int      m;       //!< Number of query points                 [scalar]
  int      k;       //!< Number of nearest neighbors            [scalar]
} knnresult;


void push_array(double* distances, int* indices, int new_index, int current_index, int current_value, int k){
    int temp, temp_index;
    for(int i = current_index; i < k; i++){
        //push the distances
        temp =distances[i];
        distances[i]=current_value;
        current_value = temp;

        //push the indices
        temp_index = indices[i];
        indices[i] = new_index;
        new_index = temp_index;
    }
}

int find_distance(int* X, int* Y, int d){
    int distance = 0;
    for(int i = 0; i < d; i++){
        distance+= X[i]*X[i] + Y[i]*Y[i] -2*X[i]*Y[i];
    }
    return distance;
}



void find_starting_neighbors(knnresult *result, int X[][2], int Y[][2]){
    int k = result->k;
    int n = result->n;
    int d = result->d;
    for(int j = 0; j < n; j++){
        //printf("%d \n\n",j);
        int distance;
        int* indices = calloc(k,sizeof(int));
        double* distances = calloc(k,sizeof(double));
        bool flag;
        //Insert the first k neighbors and sort them
        for(int start = 0; start < k; start++){
            distance = find_distance(X[j],Y[start],d);
            flag = true;
            for(int i = 0; i < k; i++){
                if(distances[i]>distance){
                    push_array(distances,indices,start,i,distance,k);
                    flag = false;
                    break;
                }
            }
            if(flag){
                indices[start] = start;
                distances[start] = distance;
            }
        }
        for(int i = k; i < n; i++){
            distance = find_distance(X[j],Y[i],d);
            //The array distances is sorted so the max value is the last value
            if(distance<distances[k-1]){
                for(int j = 0; j < k; j++){
                    if(distances[j]>distance){
                        push_array(distances,indices,i,j,distance,k);
                        break;
                    }
                }
            }
        }
        
        // result->ndist[j]=distances;
        // result->nidx[j]=indices;
        // printf("\n");
    }
}

void copy_array(int size, int d, int** Z, int** new_Z){
    for (int i = 0; i < size; i++)
        {   for(int j = 0; j < d; j++){
                
                Z[i][j] = new_Z[i][j];
            }
        }   
}

int new_size(int rank, int procs, int size){
    int rank_next = (rank+1) % procs;
    int rank_prev = rank == 0 ? procs - 1 : rank - 1;
    int new_size;
    MPI_Status status;
    if(rank%2==0){
        MPI_Send(&size,1,MPI_INT,rank_next,0,MPI_COMM_WORLD);

        MPI_Recv(&new_size,1,MPI_INT,rank_prev,0,MPI_COMM_WORLD,&status);
    }
    else{
        MPI_Recv(&new_size,1,MPI_INT,rank_prev,0,MPI_COMM_WORLD,&status);

        MPI_Send(&size,1,MPI_INT,rank_next,0,MPI_COMM_WORLD);
    }
    return new_size;
}


void ring_transfer(int rank, int procs, int size, int previous_size, int d, int** Z, int** new_Z){
    int rank_next = (rank+1) % procs;
    int rank_prev = rank == 0 ? procs - 1 : rank - 1;
    MPI_Status status;
    if(rank%2==0){
        for(int i = 0; i < previous_size; i++){
            MPI_Send((void*)Z[i],d,MPI_INT,rank_next,0,MPI_COMM_WORLD);
            
        }
        for(int i = 0; i < size; i++){
            
            MPI_Recv((void*)new_Z[i],d,MPI_INT,rank_prev,0,MPI_COMM_WORLD,&status);
            
        }
    }
    else{
        for(int i = 0; i < size; i++){
            
            MPI_Recv((void*)new_Z[i],d,MPI_INT,rank_prev,0,MPI_COMM_WORLD,&status);
            
        }
        for(int i = 0; i < previous_size; i++){

            MPI_Send((void*)Z[i],d,MPI_INT,rank_next,0,MPI_COMM_WORLD);
            
        }
    }

}

void data_receive(int procs, int n, int d,int** Z, int tag){
    MPI_Status status;
    for(int i = 0; i < n/procs; i++){   
        
        MPI_Recv((void*)Z[i],d,MPI_INT,0,tag,MPI_COMM_WORLD,&status);
    }
}

void data_send(int procs, int X[9][2], int n, int d, int tag){
    for(int i = 1; i < procs; i++){
        for(int j = (i-1)*n/procs; j < n/procs*i; j++){
                MPI_Send((void*)X[j],d,MPI_INT,i,tag,MPI_COMM_WORLD);    
        }
    }
}

int main(int argc, char** argv) {
    knnresult result;
    result.k = 3;
    result.d = 2;
    result.n = 9;
    int m,n,d,k;
    n = 9;
    d = 2;
    result.ndist = malloc(sizeof(double*)*n);
    result.nidx = malloc(sizeof(int*)*n);


    int procs,rank,indices;
    int X[9][2] ={
         {1,4},
         {2,7},
         {3,6},
         {4,2},
         {5,4},
         {6,8},
         {7,5},
         {8,2},
         {9,1}
    };
    int Y[9][2] ={
         {1,4},
         {2,7},
         {3,6},
         {4,2},
         {5,4},
         {6,8},
         {7,5},
         {8,2},
         {9,1}
    };

    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD, &procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Status status;

    int size = n/procs+n%procs;

    int** Z = malloc(sizeof(int*)*size);
    int** X_piece = malloc(sizeof(int*)*size);
    for(int i = 0; i < size; i++){
        Z[i] = malloc(sizeof(int)*d);
        X_piece[i] = malloc(sizeof(int)*d);
    }
    
    if(rank==0){
        
        data_send(procs, X, n, d, 0);
        data_send(procs, Y, n, d, 1);

        for (int i = 0; i < n/procs+n%procs; i++)
        {   for(int j = 0; j < d; j++){
                
                Z[i][j] = X[(procs-1)*n/procs+i][j];
            }
        }
        size = n/procs+n%procs;
    }
    else{

        data_receive(procs,n,d,X_piece,0);
        data_receive(procs,n,d,Z,1);
        size = n/procs;
    }
    
    int previous_size = size;
    size = new_size(rank,procs,size);
    
    int** new_Z = malloc(sizeof(int*)*(n/procs+n%procs));
    for(int i = 0; i < n/procs+n%procs; i++){
        new_Z[i] = malloc(sizeof(int)*d);
    }

    //find_neighbors(&result,X_piece,Z);
    ring_transfer(rank,procs,size,previous_size,d,Z,new_Z);
    copy_array(size,d,Z,new_Z);
    
    for(int i = 0; i < size; i++){
            printf("%d : %d %d \n", rank,Z[i][0], Z[i][1]);
        }

    for(int i = 1; i < procs; i++){

        


        ring_transfer(rank,procs,size,previous_size,d,Z,new_Z);
        copy_array(size,d,Z,new_Z);
        // for(int i = 0; i < size; i++){
        //     printf("%d : %d %d \n", rank,Z[i][0], Z[i][1]);
        // }
        
        previous_size = size;
        size = new_size(rank,procs,size);
        //printf("%d : %d %d \n",rank,previous_size,size);

    }
    MPI_Finalize();
}