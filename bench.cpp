#include "hnswlib/hnswlib.h"
#include <chrono>
#include <fstream>
#include <vector>
#include <algorithm> //for std::find
#include <cstdio> //for printf
#include <string>

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
void run_query() {
    printf("Loading queries and ground truth...\n");
    std::vector<std::vector<float>> queries = load_fvecs(query_path);
    std::vector<std::vector<int>>   truth   = load_ivecs(truth_path);
    printf("Queries: %zu\n", queries.size());

    printf("Loading index from %s...\n", index_path.c_str());
    hnswlib::L2Space space(128);
    hnswlib::HierarchicalNSW<float> index(&space, index_path);
    index.setEf(50);
    printf("Index loaded.\n");

    printf("Running %zu queries...\n", queries.size());
    auto start = std::chrono::high_resolution_clock::now();

    int correct = 0;
    for (size_t i = 0; i < queries.size(); i++) {
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

    double qps    = queries.size() / elapsed;
    double recall = (double)correct / (queries.size() * k);

    printf("---------------------------\n");
    printf("QPS:       %.1f\n", qps);
    printf("Recall@10: %.4f\n", recall);
    printf("Time:      %.3f seconds\n", elapsed);
}

// default mode: build + query in one shot (original behaviour, used for benchmarking)
void run_default() {
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

    printf("Running %zu queries...\n", queries.size());
    auto start = std::chrono::high_resolution_clock::now();

    int correct = 0;
    for (size_t i = 0; i < queries.size(); i++) {
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

    double qps    = queries.size() / elapsed;
    double recall = (double)correct / (queries.size() * k);

    printf("---------------------------\n");
    printf("QPS:       %.1f\n", qps);
    printf("Recall@10: %.4f\n", recall);
    printf("Time:      %.3f seconds\n", elapsed);
}

int main(int argc, char* argv[]) {
    if (argc >= 2 && std::string(argv[1]) == "--build") {
        run_build();
    } else if (argc >= 2 && std::string(argv[1]) == "--query") {
        run_query();
    } else {
        run_default();
    }
    return 0;
}
