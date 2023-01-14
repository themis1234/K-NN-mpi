#include <stdio.h>
#include <stdlib.h>
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

double find_distance(double* X, double* Y, double d){
    double distance = 0;
    for(int i = 0; i < d; i++){
        distance+= X[i]*X[i] + Y[i]*Y[i] -2*X[i]*Y[i];
    }
    return distance;
}



void find_neighbors(knnresult *result, double** X, double** Y){
    int k = result->k;
    int n = result->n;
    int d = result->d;
    int m = result->m;
    for(int j = 0; j < m; j++){
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
        
        result->ndist[j]=distances;
        result->nidx[j]=indices;
    }
}

int main(int argc, char const *argv[])
{
    knnresult result;
    int grid_size = 10;
    int m,n,d,k;
    n = 1000;
    k =27;
    d = 3;
    m = n;
    result.k = k;
    result.d = d;
    result.n = n;
    result.m = m;
    
    result.ndist = malloc(sizeof(double*)*n);
    result.nidx = malloc(sizeof(int*)*n);
    

    double** X = malloc(sizeof(double*)*n);
    double** Y = malloc(sizeof(double*)*n);
    int id = 0;
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
    load_mnist();
   
    //print distances for regular grid
    // find_neighbors(&result,X,Y);
    // for(int i = 0; i < n; i ++){
    //     for(int j = 0; j < k; j++){
    //         printf("%d ", (int)result.ndist[i][j]);
    //     }
    //     printf("\n");
    // }
    return 0;
}
