#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

//test
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



void find_neighbors(knnresult *result, int X[9][2], int Y[9][2]){
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
        for(int i = 0; i < k; i++){
            printf("%d ", indices[i]);
        }
        result->ndist[j]=distances;
        result->nidx[j]=indices;
        printf("\n");
    }
}

int main(int argc, char const *argv[])
{
    knnresult result;
    int **Y;
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
    int m,n,d,k;
    result.k = 3;
    result.d = 2;
    result.n = 9;
    result.ndist = malloc(sizeof(double*)*9);
    result.nidx = malloc(sizeof(int*)*9);
    m = 9;
    
        find_neighbors(&result,X,X);
    printf("\n");
    for(int i = 0; i < 9; i ++){
        for(int j = 0; j < 3; j++){
            printf("%d %f ", result.nidx[i][j], result.ndist[i][j]);
        }
        printf("\n");
    }
    return 0;
}
