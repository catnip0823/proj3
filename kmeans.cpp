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
    cluster_array *my_cluster = (cluster_array*)malloc(sizeof(cluster_array)*cn);
    do {
    //for (int i = 0; i < 100; i++){
        converge = true;
        converge_int = 0;
        /* Compute the color of each point. A point gets assigned to the
           cluster with the nearest center point. */
#pragma omp parallel reduction(+:converge_int)
        {
            #pragma omp for
            for (int i = 0; i < pn/4*4; i+=4) {
                color_t new_color1 = cn;
                color_t new_color2 = cn;
                color_t new_color3 = cn;
                color_t new_color4 = cn;
                double min_dist1 = std::numeric_limits<double>::infinity();
                double min_dist2 = std::numeric_limits<double>::infinity();
                double min_dist3 = std::numeric_limits<double>::infinity();
                double min_dist4 = std::numeric_limits<double>::infinity();


                for(color_t c=0;c<cn/4*4;c+=4){
                    double dist11 = (pow(data[i].getX()-mean[c].getX(),2)+pow(data[i].getY()-mean[c].getY(),2));
                    double dist12 = (pow(data[i].getX()-mean[c+1].getX(),2)+pow(data[i].getY()-mean[c+1].getY(),2));
                    double dist13 = (pow(data[i].getX()-mean[c+2].getX(),2)+pow(data[i].getY()-mean[c+2].getY(),2));
                    double dist14 = (pow(data[i].getX()-mean[c+3].getX(),2)+pow(data[i].getY()-mean[c+3].getY(),2));
                    if (dist11<min_dist1){
                        min_dist1=dist11;
                        new_color1=c;
                    }
                    if (dist12<min_dist1){
                        min_dist1=dist12;
                        new_color1=c+1;
                    }
                    if (dist13<min_dist1){
                        min_dist1=dist13;
                        new_color1=c+2;
                    }
                    if (dist14<min_dist1){
                        min_dist1=dist14;
                        new_color1=c+3;
                    }

                    double dist21 = (pow(data[i+1].getX()-mean[c].getX(),2)+pow(data[i+1].getY()-mean[c].getY(),2));
                    double dist22 = (pow(data[i+1].getX()-mean[c+1].getX(),2)+pow(data[i+1].getY()-mean[c+1].getY(),2));
                    double dist23 = (pow(data[i+1].getX()-mean[c+2].getX(),2)+pow(data[i+1].getY()-mean[c+2].getY(),2));
                    double dist24 = (pow(data[i+1].getX()-mean[c+3].getX(),2)+pow(data[i+1].getY()-mean[c+3].getY(),2));
                    if (dist21<min_dist2){
                        min_dist2=dist21;
                        new_color2=c;
                    }
                    if (dist22<min_dist2){
                        min_dist2=dist22;
                        new_color2=c+1;
                    }
                    if (dist23<min_dist2){
                        min_dist2=dist23;
                        new_color2=c+2;
                    }
                    if (dist24<min_dist2){
                        min_dist2=dist24;
                        new_color2=c+3;
                    }

                    double dist31 = (pow(data[i+2].getX()-mean[c].getX(),2)+pow(data[i+2].getY()-mean[c].getY(),2));
                    double dist32 = (pow(data[i+2].getX()-mean[c+1].getX(),2)+pow(data[i+2].getY()-mean[c+1].getY(),2));
                    double dist33 = (pow(data[i+2].getX()-mean[c+2].getX(),2)+pow(data[i+2].getY()-mean[c+2].getY(),2));
                    double dist34 = (pow(data[i+2].getX()-mean[c+3].getX(),2)+pow(data[i+2].getY()-mean[c+3].getY(),2));
                    if (dist31<min_dist3){
                        min_dist3=dist31;
                        new_color3=c;
                    }
                    if (dist32<min_dist3){
                        min_dist3=dist32;
                        new_color3=c+1;
                    }
                    if (dist33<min_dist3){
                        min_dist3=dist33;
                        new_color3=c+2;
                    }
                    if (dist34<min_dist3){
                        min_dist3=dist34;
                        new_color3=c+3;
                    }

                    double dist41 = (pow(data[i+3].getX()-mean[c].getX(),2)+pow(data[i+3].getY()-mean[c].getY(),2));
                    double dist42 = (pow(data[i+3].getX()-mean[c+1].getX(),2)+pow(data[i+3].getY()-mean[c+1].getY(),2));
                    double dist43 = (pow(data[i+3].getX()-mean[c+2].getX(),2)+pow(data[i+3].getY()-mean[c+2].getY(),2));
                    double dist44 = (pow(data[i+3].getX()-mean[c+3].getX(),2)+pow(data[i+3].getY()-mean[c+3].getY(),2));
                    if (dist41<min_dist4){
                        min_dist4=dist41;
                        new_color4=c;
                    }
                    if (dist42<min_dist4){
                        min_dist4=dist42;
                        new_color4=c+1;
                    }
                    if (dist43<min_dist4){
                        min_dist4=dist43;
                        new_color4=c+2;
                    }
                    if (dist44<min_dist4){
                        min_dist4=dist44;
                        new_color4=c+3;
                    }

                }


                for(color_t c=cn/4*4;c<cn;c++){
                    double dist1 = (pow(data[i].getX()-mean[c].getX(),2)+pow(data[i].getY()-mean[c].getY(),2));
                    double dist2 = (pow(data[i+1].getX()-mean[c].getX(),2)+pow(data[i+1].getY()-mean[c].getY(),2));
                    double dist3 = (pow(data[i+2].getX()-mean[c].getX(),2)+pow(data[i+2].getY()-mean[c].getY(),2));
                    double dist4 = (pow(data[i+3].getX()-mean[c].getX(),2)+pow(data[i+3].getY()-mean[c].getY(),2));
                    if (dist1<min_dist1){
                        min_dist1=dist1;
                        new_color1=c;
                    }
                    if (dist2<min_dist2){
                        min_dist2=dist2;
                        new_color2=c;
                    }
                    if (dist3<min_dist3){
                        min_dist3=dist3;
                        new_color3=c;
                    }
                    if (dist4<min_dist4){
                        min_dist4=dist4;
                        new_color4=c;
                    }
                }

                if (coloring[i] != new_color1) {
                    coloring[i] = new_color1;
                    converge_int = 1;
                }
                if (coloring[i+1] != new_color2) {
                    coloring[i+1] = new_color2;
                    converge_int = 1;
                }
                if (coloring[i+2] != new_color3) {
                    coloring[i+2] = new_color3;
                    converge_int = 1;
                }
                if (coloring[i+3] != new_color4) {
                    coloring[i+3] = new_color4;
                    converge_int = 1;
                }

            }




        }


         for (int i = pn/4*4; i < pn; i+=1) {
                color_t new_color1 = cn;
                double min_dist1 = std::numeric_limits<double>::infinity();



                for(color_t c=0;c<cn/4*4;c+=4){
                    double dist11 = (pow(data[i].getX()-mean[c].getX(),2)+pow(data[i].getY()-mean[c].getY(),2));
                    double dist12 = (pow(data[i].getX()-mean[c+1].getX(),2)+pow(data[i].getY()-mean[c+1].getY(),2));
                    double dist13 = (pow(data[i].getX()-mean[c+2].getX(),2)+pow(data[i].getY()-mean[c+2].getY(),2));
                    double dist14 = (pow(data[i].getX()-mean[c+3].getX(),2)+pow(data[i].getY()-mean[c+3].getY(),2));
                    if (dist11<min_dist1){
                        min_dist1=dist11;
                        new_color1=c;
                    }
                    if (dist12<min_dist1){
                        min_dist1=dist12;
                        new_color1=c+1;
                    }
                    if (dist13<min_dist1){
                        min_dist1=dist13;
                        new_color1=c+2;
                    }
                    if (dist14<min_dist1){
                        min_dist1=dist14;
                        new_color1=c+3;
                    }

                    

                }


                for(color_t c=cn/4*4;c<cn;c++){
                    double dist1 = (pow(data[i].getX()-mean[c].getX(),2)+pow(data[i].getY()-mean[c].getY(),2));

                    if (dist1<min_dist1){
                        min_dist1=dist1;
                        new_color1=c;
                    }
                }

                if (coloring[i] != new_color1) {
                    coloring[i] = new_color1;
                    converge_int += 1;
                }

            }

        if (converge_int) converge=false;
        /* Calculate the new mean for each cluster to be the current average
           of point positions in the cluster. */


        
        for (color_t c = 0; c < cn/5*5; c+=5) {
            my_cluster[c].sum_x = 0;
            my_cluster[c].sum_y = 0;
            my_cluster[c].count = 0;
            my_cluster[c+1].sum_x = 0;
            my_cluster[c+1].sum_y = 0;
            my_cluster[c+1].count = 0;
            my_cluster[c+2].sum_x = 0;
            my_cluster[c+2].sum_y = 0;
            my_cluster[c+2].count = 0;
            my_cluster[c+3].sum_x = 0;
            my_cluster[c+3].sum_y = 0;
            my_cluster[c+3].count = 0;
            my_cluster[c+4].sum_x = 0;
            my_cluster[c+4].sum_y = 0;
            my_cluster[c+4].count = 0;
        }
        for (color_t c = cn/5*5; c < cn;c++) {
            my_cluster[c].sum_x = 0;
            my_cluster[c].sum_y = 0;
            my_cluster[c].count = 0;
        }

        for (int i = 0; i < pn / 5 * 5; i += 5) {
            my_cluster[coloring[i]].sum_x += data[i].getX();
            my_cluster[coloring[i]].sum_y += data[i].getY();
            my_cluster[coloring[i]].count++;
            my_cluster[coloring[i + 1]].sum_x += data[i + 1].getX();
            my_cluster[coloring[i + 1]].sum_y += data[i + 1].getY();
            my_cluster[coloring[i + 1]].count++;
            my_cluster[coloring[i + 2]].sum_x += data[i + 2].getX();
            my_cluster[coloring[i + 2]].sum_y += data[i + 2].getY();
            my_cluster[coloring[i + 2]].count++;
            my_cluster[coloring[i + 3]].sum_x += data[i + 3].getX();
            my_cluster[coloring[i + 3]].sum_y += data[i + 3].getY();
            my_cluster[coloring[i + 3]].count++;
            my_cluster[coloring[i + 4]].sum_x += data[i + 4].getX();
            my_cluster[coloring[i + 4]].sum_y += data[i + 4].getY();
            my_cluster[coloring[i + 4]].count++;
        }

        for (int i = pn/5*5; i < pn; i+=1) {
            my_cluster[coloring[i]].sum_x += data[i].getX();
            my_cluster[coloring[i]].sum_y += data[i].getY();
            my_cluster[coloring[i]].count++;
        }
        for (color_t c = 0; c < cn; c++) {
            mean[c].setXY(my_cluster[c].sum_x/my_cluster[c].count, my_cluster[c].sum_y/my_cluster[c].count);
        }
        


    } while (!converge);
    free(my_cluster);
}

/*********************************************************
                           End
 *********************************************************/
