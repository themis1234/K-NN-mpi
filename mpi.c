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


//The distances are sorted so we need to put the point in its place and then push the rest points to the right
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

//find the distance between the 2 points
double find_distance(double* X, double* Y, int d){
    double distance = 0;
    for(int i = 0; i < d; i++){
        distance+= X[i]*X[i] + Y[i]*Y[i] -2*X[i]*Y[i];
    }
    return distance;
}



void find_starting_neighbors(int n, int k , int d, int index, double** X, double** Y,int** indices, double** distances, int rank){
    
    for(int j = 0; j < n; j++){
        double distance;
        bool flag;
        //Insert the first k neighbors in distances and sort them
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

void copy_array(int size, int d, double** Z, double** new_Z){
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


void ring_transfer(int rank, int procs, int size, int previous_size, int d, double** Z, double** new_Z){
    int rank_next = (rank+1) % procs;
    int rank_prev = rank == 0 ? procs - 1 : rank - 1;
    MPI_Status status;
    if(rank%2==0){
        for(int i = 0; i < previous_size; i++){
            MPI_Send((void*)Z[i],d,MPI_DOUBLE,rank_next,0,MPI_COMM_WORLD);
            
        }
        for(int i = 0; i < size; i++){
            
            MPI_Recv((void*)new_Z[i],d,MPI_DOUBLE,rank_prev,0,MPI_COMM_WORLD,&status);
            
        }
    }
    else{
        for(int i = 0; i < size; i++){
            
            MPI_Recv((void*)new_Z[i],d,MPI_DOUBLE,rank_prev,0,MPI_COMM_WORLD,&status);
            
        }
        for(int i = 0; i < previous_size; i++){

            MPI_Send((void*)Z[i],d,MPI_DOUBLE,rank_next,0,MPI_COMM_WORLD);
            
        }
    }

}

void data_receive(int procs, int n, int d,double** Z, int tag){
    MPI_Status status;
    for(int i = 0; i < n/procs; i++){   
        
        MPI_Recv((void*)Z[i],d,MPI_DOUBLE,0,tag,MPI_COMM_WORLD,&status);
    }
}

void data_send(int procs, double** X, int n, int d, int tag){
    for(int i = 1; i < procs; i++){
        for(int j = (i-1)*n/procs; j < n/procs*i; j++){
                MPI_Send((void*)X[j],d,MPI_DOUBLE,i,tag,MPI_COMM_WORLD); 
                free(X[j]);   
        }
    }
}

int main(int argc, char** argv) {
    load_mnist();
    knnresult result;
    int m,n,d,k;
    k = 27;
    n = 1000;
    d = 3;
    result.k = k;
    result.d = d;
    result.n = n;
    
    int grid_size = 10;
    
    m = n;
    result.ndist = malloc(sizeof(double*)*m);
    result.nidx = malloc(sizeof(int*)*m);

    int procs,rank,index,X_size;
    double** X = malloc(sizeof(double*)*n);
    double** Y = malloc(sizeof(double*)*n);
    int id = 0;
    //create regular grid
    for(int i = 0; i< grid_size; i++){
        for(int j =0; j < grid_size; j++){
            for(int k = 0; k < grid_size; k++){
                X[id] = malloc(sizeof(double)*d);
                Y[id] = malloc(sizeof(double)*d);
                X[id][0] = i;
                X[id][1] = j;
                X[id][2] = k;
                Y[id][0] = i;
                Y[id][1] = j;
                Y[id][2] = k;
                id++;
            }
        }
    }

    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD, &procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Status status;

    int size = n/procs+n%procs;

    double** Z = malloc(sizeof(double*)*size);
    double** X_piece = malloc(sizeof(double*)*size);
    for(int i = 0; i < size; i++){
        Z[i] = malloc(sizeof(double)*d);
        X_piece[i] = malloc(sizeof(double)*d);
    }
    
    if(rank==0){
        
        data_send(procs, Y, n, d, 1);
        data_send(procs, X, n , d, 0);
        for (int i = 0; i < n/procs+n%procs; i++)
        {   for(int j = 0; j < d; j++){
                
                Z[i][j] = Y[(procs-1)*n/procs+i][j];
                X_piece[i][j] = X[(procs-1)*n/procs+i][j];
                
            }
        }
        size = n/procs+n%procs;
        X_size = size;
        index = (procs-1)*n/procs;
    }
    else{

        data_receive(procs, n, d, Z, 1);
        data_receive(procs, n, d, X_piece, 0);
        size = n/procs;
        X_size = size;
        index = (rank - 1)*n/procs;
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
    
    double** new_Z = malloc(sizeof(double*)*(n/procs+n%procs));

    for(int i = 0; i < n/procs+n%procs; i++){
        new_Z[i] = malloc(sizeof(double)*d);
    }

    ring_transfer(rank,procs,size,previous_size,d,Z,new_Z);
    copy_array(size,d,Z,new_Z);
    
    double distance;
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
 
    if(rank==0){
        for(int i =1; i < procs; i++){
            for(int j = (i-1)*n/procs; j < n/procs*i; j++){
                
                result.ndist[j] = malloc(sizeof(double)*k);
                MPI_Recv(result.ndist[j],k,MPI_DOUBLE,i,0,MPI_COMM_WORLD,&status);
                result.nidx[j] = malloc(sizeof(int)*k);
                MPI_Recv(result.nidx[j],k,MPI_INT,i,1,MPI_COMM_WORLD,&status);
               
            }
        }
        for(int i = 0; i < X_size; i++){
            
            result.ndist[(procs-1)*n/procs+i] = distances[i];
            result.nidx[(procs-1)*n/procs+i] = indices[i];
            
        }
    }
    else{
        for(int i = 0; i < X_size; i++){
            MPI_Send(distances[i],k,MPI_DOUBLE,0,0,MPI_COMM_WORLD);
            MPI_Send(indices[i],k,MPI_INT,0,1,MPI_COMM_WORLD);
            free(indices[i]);
            free(distances[i]);

        }
        free(indices);
        free(distances);
    }
    //print distances for regular grid 
    // if(rank == 0){
    //     for(int i = 0; i < n; i ++){
    //         for(int j = 0; j < k; j++){
    //             printf("%d ", (int)result.ndist[i][j]);
    //         }
    //     printf("\n");
    //     }

    // }
    

    MPI_Finalize();

    
}
