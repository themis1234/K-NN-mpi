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



void find_starting_neighbors(int n, int k , int d, int index, int** X, int** Y,int** indices, double** distances, int rank){
    
    for(int j = 0; j < n; j++){
        //printf("%d \n\n",j);
        int distance;
        
        bool flag;
        //Insert the first k neighbors and sort them
        for(int start = 0; start < k; start++){
            //printf("%d\n", rank);
            distance = find_distance(X[j],Y[start],d);
            //printf("%d\n", rank);
            flag = true;
            for(int i = 0; i < k; i++){
                if(distances[j][i]>distance){
                    push_array(distances[j],indices[j],start+index,i,distance,k);
                    flag = false;
                    break;
                }
            }
            if(flag){
                indices[j][start] = start + index;
                distances[j][start] = distance;
            }
        }
        for(int i = k; i < n; i++){
            distance = find_distance(X[j],Y[i],d);
            //printf("%d\n", rank);

            //The array distances is sorted so the max value is the last value
            if(distance<distances[j][k-1]){
                for(int l = 0; l < k; l++){
                    //printf("%d\n", rank);
                    if(distances[j][l]>distance){
                        push_array(distances[j],indices[j],index+i,l,distance,k);
                        break;
                    }
                }
            }
        }
        

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

void data_receive(int procs, int n, int d,int** Z){
    MPI_Status status;
    for(int i = 0; i < n/procs; i++){   
        
        MPI_Recv((void*)Z[i],d,MPI_INT,0,1,MPI_COMM_WORLD,&status);
    }
}

void data_send(int procs, int X[9][2], int n, int d){
    for(int i = 1; i < procs; i++){
        for(int j = (i-1)*n/procs; j < n/procs*i; j++){
                MPI_Send((void*)X[j],d,MPI_INT,i,1,MPI_COMM_WORLD);    
        }
    }
}

int main(int argc, char** argv) {
    knnresult result;
    result.k = 3;
    result.d = 2;
    result.n = 13;
    int m,n,d,k;
    k = 3;
    n = 13;
    d = 2;
    result.ndist = malloc(sizeof(double*)*n);
    result.nidx = malloc(sizeof(int*)*n);


    int procs,rank,index,X_size;
    int X[13][2] ={
         {1,4},
         {2,7},
         {3,6},
         {4,2},
         {5,4},
         {6,8},
         {7,5},
         {8,2},
         {9,1},
         {4,4},
         {8,3},
         {1,1},
         {2,2}
     
    };
    int Y[13][2] ={
         {1,4},
         {2,7},
         {3,6},
         {4,2},
         {5,4},
         {6,8},
         {7,5},
         {8,2},
         {9,1},
         {4,4},
         {8,3},
         {1,1},
         {2,2}
        
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
        
        data_send(procs, Y, n, d);

        for (int i = 0; i < n/procs+n%procs; i++)
        {   for(int j = 0; j < d; j++){
                
                Z[i][j] = X[(procs-1)*n/procs+i][j];
            }
        }
        copy_array(size,d,X_piece,Z);
        size = n/procs+n%procs;
        X_size = size;
        index = (procs-1)*n/procs;
    }
    else{

        data_receive(procs, n, d, Z);
        size = n/procs;
        X_size = size;
        index = (rank - 1)*n/procs;
        copy_array(size,d,X_piece,Z);
    }
    int** indices = calloc(sizeof(int*),X_size);
    double** distances = calloc(sizeof(double*),X_size);
    for(int i = 0; i < X_size; i++){
        distances[i] = calloc(sizeof(double),k);
        indices[i] = calloc(sizeof(int),k);
    }


    find_starting_neighbors(X_size,k,d,index,X_piece,Z,indices,distances,rank);
    int previous_size = size;
    size = new_size(rank,procs,size);
    index -= size;
    if(index<0){
        index+=n;
    }
    
    int** new_Z = malloc(sizeof(int*)*(n/procs+n%procs));
    for(int i = 0; i < n/procs+n%procs; i++){
        new_Z[i] = malloc(sizeof(int)*d);
    }

    ring_transfer(rank,procs,size,previous_size,d,Z,new_Z);
    copy_array(size,d,Z,new_Z);
    
    int distance;
    for(int p = 1; p < procs; p++){
        for(int j = 0; j < X_size; j++){
            for(int i = 0; i < size; i++){
                distance = find_distance(X_piece[j],Z[i],d);

                //The array distances is sorted so the max value is the last value
                if(distance<distances[j][k-1]){
                    for(int l = 0; l < k; l++){
                        //printf("%d\n", rank);
                        if(distances[j][l]>distance){
                            push_array(distances[j],indices[j],index+i,l,distance,k);
                            break;
                        }
                    }
                }
            }
        }
        
        previous_size = size;
        size = new_size(rank,procs,size);
        index -= size;
        if(index<0){
        index+=n;
        }
        ring_transfer(rank,procs,size,previous_size,d,Z,new_Z);
        copy_array(size,d,Z,new_Z);

    }
    for(int i = 0; i < X_size; i ++){
        for(int j = 0; j < k; j++){
            printf("%d %f ", indices[i][j], distances[i][j]);
        }
        printf("\n");
    }

    MPI_Finalize();
}