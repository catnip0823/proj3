////////////////
// 
// File: kmeans.cpp
//
//  Main body of K-Means simulaton. Reads in the original data points from
//  `ori.txt`, performs K-Means clustering on randomly-picked initial
//  centers, and writes the results into `res.txt` with the same format.
//
//  * You may (and should) include some extra headers for optimizations.
//
//  * You should and ONLY should modify the function body of `kmeans()`.
//    DO NOT change any other exitsing part of the program.
//
//  * You may add your own auxiliary functions if you wish. Extra declarations
//    can go in `kmeans.h`.
//
// Jose @ ShanghaiTech University
//
////////////////

#include <fstream>
#include <limits>
#include <math.h>
#include <chrono>
#include "kmeans.h"


/*********************************************************
        Your extra headers and static declarations
 *********************************************************/
#include <pthread.h>
#include <emmintrin.h>
#include <immintrin.h>
#include <assert.h>
#include <omp.h>

/*********************************************************
                           End
 *********************************************************/


/*
 * Entrance point. Time ticking will be performed, so it will be better if
 *   you have cleared the cache for precise profiling.
 *
 */
int
main (int argc, char *argv[])
{
    if (argc != 3) {
        std::cout << "Usage: " << argv[0] << " <input.txt> <output.txt>"
                  << std::endl;
        return -1;
    }
    if (!(bool)std::ifstream(argv[1])) {
        std::cerr << "ERROR: Data file " << argv[1] << " does not exist!"
                  << std::endl;
        return -1;
    }
    if ((bool)std::ifstream(argv[2])) {
        std::cerr << "ERROR: Destination " << argv[2] << " already exists!"
                  << std::endl;
        return -1;
    }
    FILE *fi = fopen(argv[1], "r"), *fo = fopen(argv[2], "w");
    
    /* From `ori.txt`, acquire dataset size, number of colors (i.e. K in
       K-Means),and read in all data points into static array `data`. */
    int pn, cn;

    assert(fscanf(fi, "%d / %d\n", &pn, &cn) == 2);

    point_t * const data = new point_t[pn];
    color_t * const coloring = new color_t[pn];

    for (int i = 0; i < pn; ++i)
        coloring[i] = 0;

    int i = 0, c;
    double x, y;

    while (fscanf(fi, "%lf, %lf, %d\n", &x, &y, &c) == 3) {
        data[i++].setXY(x, y);
        if (c < 0 || c >= cn) {
            std::cerr << "ERROR: Invalid color code encoutered!"
                      << std::endl;
            return -1;
        }
    }
    if (i != pn) {
        std::cerr << "ERROR: Number of data points inconsistent!"
                  << std::endl;
        return -1;
    }

    /* Generate a random set of initial center points. */
    point_t * const mean = new point_t[cn];

    srand(5201314);
    for (int i = 0; i < cn; ++i) {
        int idx = rand() % pn;
        mean[i].setXY(data[idx].getX(), data[idx].getY());
    }

    /* Invode K-Means algorithm on the original dataset. It should cluster
       the data points in `data` and assign their color codes to the
       corresponding entry in `coloring`, using `mean` to store the center
       points. */
    std::cout << "Doing K-Means clustering on " << pn
              << " points with K = " << cn << "..." << std::flush;
    auto ts = std::chrono::high_resolution_clock::now();
    kmeans(data, mean, coloring, pn, cn);
    auto te = std::chrono::high_resolution_clock::now();
    std::cout << "done." << std::endl;
    std::cout << " Total time elapsed: "
              << std::chrono::duration_cast<std::chrono::milliseconds> \
                 (te - ts).count()
              << " milliseconds." << std::endl; 

    /* Write the final results to `res.txt`, in the same format as input. */
    fprintf(fo, "%d / %d\n", pn, cn);
    for (i = 0; i < pn; ++i)
        fprintf(fo, "%.8lf, %.8lf, %d\n", data[i].getX(), data[i].getY(),
                coloring[i]);

    /* Free the resources and return. */
    delete[](data);
    delete[](coloring);
    delete[](mean);
    fclose(fi);
    fclose(fo);
    return 0;
}


/*********************************************************
           Feel free to modify the things below
 *********************************************************/

/*
 * K-Means algorithm clustering. Originally implemented in a traditional
 *   sequential way. You should optimize and parallelize it for a better
 *   performance. Techniques you can use include but not limited to:
 *
 *     1. OpenMP shared-memory parallelization.
 *     2. SSE SIMD instructions.
 *     3. Cache optimizations.
 *     4. Manually using pthread.
 *     5. ...
 *
 */
void
kmeans (point_t * const data, point_t * const mean, color_t * const coloring,
        const int pn, const int cn)
{
    bool converge = true;
    int converge_int = 0 ;
    /* Loop through the following two stages until no point changes its color
       during an iteration. */
    do {
    //for (int i = 0; i < 100; i++){
        converge = true;
        converge_int = 0;
        /* Compute the color of each point. A point gets assigned to the
           cluster with the nearest center point. */
#pragma omp parallel reduction(+:converge_int)
        {
            #pragma omp for
            for (int i = 0; i < pn; ++i) {
                color_t new_color = cn;
                double min_dist = std::numeric_limits<double>::infinity();

//                double x[4]={data[i].getX(), data[i].getX(),
//                                  data[i].getX(), data[i].getX()};
//                double y[4] = {data[i].getY() ,data[i].getY(),
//                                    data[i].getY() ,data[i].getY() };
//                __m256d mmx = _mm256_load_pd(x);
//                __m256d mmy = _mm256_load_pd(y);
//
//                for (color_t c = 0; c < cn/4*4; c+=4) {
//                    double cx[4]={mean[c].getX(), mean[c+1].getX(),
//                                 mean[c+2].getX(), mean[c+3].getX()};
//                    double cy[4]={mean[c].getY(), mean[c+1].getY(),
//                                  mean[c+2].getY(), mean[c+3].getY()};
//                    __m256d mmcx = _mm256_load_pd(cx);
//                    __m256d mmcy = _mm256_load_pd(cy);
//                    __m256d mmx_c = _mm256_sub_pd(mmx,mmcx);
//                    __m256d mmy_c = _mm256_sub_pd(mmy,mmcy);
//                    double dists[4];
//                    _mm256_store_pd(dists,_mm256_sqrt_pd(_mm256_mul_pd(mmx_c,mmx_c)+
//                                                         _mm256_mul_pd(mmy_c,mmy_c)));
//
//                    for (int j = 0; j<4;j++){
//                        if (dists[j] < min_dist) {
//                            min_dist = dists[j];
//                            new_color = c+j;
//                        }
//
//                    }
//                }
//                for (color_t c = cn/4*4; c < cn; c+=1) {
//                    double dist = sqrt(pow(data[i].getX()-mean[c].getX(),2)+pow(data[i].getY()-mean[c].getY(),2));
//                    if (dist<min_dist){
//                        min_dist=dist;
//                        new_color=c;
//                    }
//                }


                for(color_t c=0;c<cn;c++){
                    double dist = sqrt(pow(data[i].getX()-mean[c].getX(),2)+pow(data[i].getY()-mean[c].getY(),2));
                    if (dist<min_dist){
                        min_dist=dist;
                        new_color=c;
                    }
                }

                if (coloring[i] != new_color) {
                    coloring[i] = new_color;
                    converge_int = 1;
                }
            }


        }

        if (converge_int) converge=false;
        /* Calculate the new mean for each cluster to be the current average
           of point positions in the cluster. */


        cluster_array *my_cluster = (cluster_array*)malloc(sizeof(cluster_array)*cn);
        for (color_t c = 0; c < cn; ++c) {
            my_cluster[c].sum_x = 0;
            my_cluster[c].sum_y = 0;
            my_cluster[c].count = 0;
        }

        for (int i = 0; i < pn; ++i){
            my_cluster[coloring[i]].sum_x += data[i].getX();
            my_cluster[coloring[i]].sum_y += data[i].getY();
            my_cluster[coloring[i]].count++;
        }
        for (color_t c = 0; c < cn; ++c) {
            mean[c].setXY(my_cluster[c].sum_x/my_cluster[c].count, my_cluster[c].sum_y/my_cluster[c].count);
        }
        free(my_cluster);


    } while (!converge);
}

/*********************************************************
                           End
 *********************************************************/
