#include "hnswlib/hnswlib.h"
#include <chrono>
#include <fstream>
#include <vector>
#include <algorithm> //for std::find
#include <cstdio> //for printf
#include <string>
#include <omp.h>

const std::string base_path  = "datasets/sift/sift_base.fvecs";
const std::string query_path = "datasets/sift/sift_query.fvecs";
const std::string truth_path = "datasets/sift/sift_groundtruth.ivecs";
const std::string index_path = "datasets/sift/sift.hnsw";
const int k = 10;

//reads .fvecs files, which are binary files and returns a list of  vecotrs
std::vector<std::vector<float>> load_fvecs(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    std::vector<std::vector<float>> vecs;

    int dim;
    while (f.read((char*)&dim, 4)) {

        //the first 4 bytes of each vector is the dimension, then the rest is the vector data
        std::vector<float> v(dim);

        // #dim floats, each 4 bytes, so dim*4 bytes to read
        f.read((char*)v.data(), dim * 4);
        vecs.push_back(v);
    }
    return vecs;
}

//same as load_fvecs but for .ivecs files which hold integer vectors, used for ground truth in this case
std::vector<std::vector<int>> load_ivecs(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    std::vector<std::vector<int>> vecs;

    int dim;
    while (f.read((char*)&dim, 4)) {
        std::vector<int> v(dim);
        f.read((char*)v.data(), dim * 4);
        vecs.push_back(v);
    }
    return vecs;
}

// build mode: build index from base vectors and save to disk
void run_build() {
    printf("Loading base vectors...\n");
    std::vector<std::vector<float>> base = load_fvecs(base_path);
    printf("Base: %zu vectors, Dim: %zu\n", base.size(), base[0].size());

    int dim = (int)base[0].size();

    printf("Building index...\n");
    hnswlib::L2Space space(dim);
    hnswlib::HierarchicalNSW<float> index(&space, base.size(), 16, 200);

    for (size_t i = 0; i < base.size(); i++)
        index.addPoint(base[i].data(), i);
    printf("Index built.\n");

    printf("Saving index to %s...\n", index_path.c_str());
    index.saveIndex(index_path);
    printf("Done.\n");
}

// query mode: load index from disk and run queries
void run_query(int num_threads) {
    printf("Loading queries and ground truth...\n");
    std::vector<std::vector<float>> queries = load_fvecs(query_path);
    std::vector<std::vector<int>>   truth   = load_ivecs(truth_path);
    printf("Queries: %zu\n", queries.size());

    printf("Loading index from %s...\n", index_path.c_str());
    hnswlib::L2Space space(128);
    hnswlib::HierarchicalNSW<float> index(&space, index_path);
    index.setEf(50);
    printf("Index loaded.\n");

    //rre-seed the VisitedListPool so all threads find a ready VisitedList on their first searchKnn call instead of racing to allocate one
    //the pool starts with initmaxpools=1; this grows it to num_threads, this is a one-time cost that happens before the timed section, so it doesn't affect the benchmark results
    for (int t = 0; t < num_threads; t++)
        index.searchKnn(queries[0].data(), k);

    const size_t num_queries = queries.size();
    
    // 8x oversplit: enough tasks for good work-stealing load balance
    // without drowning the scheduler in tiny tasks.
    const int grainsize = (int)std::max(size_t(1), num_queries / (size_t)(num_threads * 8));

    printf("Running %zu queries with %d threads (grainsize=%d)...\n",
           num_queries, num_threads, grainsize);
    auto start = std::chrono::high_resolution_clock::now();

    int correct = 0;
    // parallel -> single -> taskloop: one thread creates all tasks upfront,
    // the whole team steals and executes them via work-stealing deques.
    // reduction(+:correct) gives each task a private copy; the runtime
    // merges them after the taskwait — zero synchronization during search.
    #pragma omp parallel
    #pragma omp single
    #pragma omp taskloop grainsize(grainsize) reduction(+:correct)
    for (size_t i = 0; i < num_queries; i++) {
        auto result = index.searchKnn(queries[i].data(), k);
        std::vector<int> truth_set(truth[i].begin(), truth[i].begin() + k);
        while (!result.empty()) {
            int label = (int)result.top().second;
            result.pop();
            if (std::find(truth_set.begin(), truth_set.end(), label) != truth_set.end())
                correct++;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();

    double qps    = num_queries / elapsed;
    double recall = (double)correct / (num_queries * k);

    printf("---------------------------\n");
    printf("Threads:   %d\n", num_threads);
    printf("Grainsize: %d\n", grainsize);
    printf("QPS:       %.1f\n", qps);
    printf("Recall@10: %.4f\n", recall);
    printf("Time:      %.3f seconds\n", elapsed);
}

// default mode: build + query in one shot (original behaviour, used for benchmarking)
void run_default(int num_threads) {
    printf("Loading dataset...\n");
    std::vector<std::vector<float>> base    = load_fvecs(base_path);
    std::vector<std::vector<float>> queries = load_fvecs(query_path);
    std::vector<std::vector<int>>   truth   = load_ivecs(truth_path);
    printf("Base: %zu vectors, Queries: %zu, Dim: %zu\n", base.size(), queries.size(), base[0].size());

    int dim = (int)base[0].size();

    printf("Building index...\n");
    hnswlib::L2Space space(dim);
    hnswlib::HierarchicalNSW<float> index(&space, base.size(), 16, 200);

    for (size_t i = 0; i < base.size(); i++)
        index.addPoint(base[i].data(), i);
    printf("Index built.\n");

    index.setEf(50);

    for (int t = 0; t < num_threads; t++)
        index.searchKnn(queries[0].data(), k);

    const size_t num_queries = queries.size();
    const int grainsize = (int)std::max(size_t(1), num_queries / (size_t)(num_threads * 8));

    printf("Running %zu queries with %d threads (grainsize=%d)...\n",
           num_queries, num_threads, grainsize);
    auto start = std::chrono::high_resolution_clock::now();

    int correct = 0;
    #pragma omp parallel
    #pragma omp single
    #pragma omp taskloop grainsize(grainsize) reduction(+:correct)
    for (size_t i = 0; i < num_queries; i++) {
        auto result = index.searchKnn(queries[i].data(), k);
        std::vector<int> truth_set(truth[i].begin(), truth[i].begin() + k);
        while (!result.empty()) {
            int label = (int)result.top().second;
            result.pop();
            if (std::find(truth_set.begin(), truth_set.end(), label) != truth_set.end())
                correct++;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();

    double qps    = num_queries / elapsed;
    double recall = (double)correct / (num_queries * k);

    printf("---------------------------\n");
    printf("Threads:   %d\n", num_threads);
    printf("Grainsize: %d\n", grainsize);
    printf("QPS:       %.1f\n", qps);
    printf("Recall@10: %.4f\n", recall);
    printf("Time:      %.3f seconds\n", elapsed);
}

int main(int argc, char* argv[]) {
    // Default: omp_get_max_threads() respects OMP_NUM_THREADS env var,
    // falling back to the hardware thread count if unset.
    int num_threads = omp_get_max_threads();
    std::string mode = "default";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--build") {
            mode = "build";
        } else if (arg == "--query") {
            mode = "query";
        } else if ((arg == "-t" || arg == "--threads") && i + 1 < argc) {
            num_threads = std::atoi(argv[++i]);
        }
    }

    omp_set_num_threads(num_threads);
    // Thread affinity must be set via env vars before launch (no portable C API):
    //   OMP_PLACES=cores       — bind to physical cores, not SMT siblings
    //   OMP_PROC_BIND=close    — fill cores from 0 outward, no migration

    if (mode == "build")       run_build();
    else if (mode == "query")  run_query(num_threads);
    else                       run_default(num_threads);
    return 0;
}
