#include <stdio.h>
#include <stdlib.h>

// Structure to represent an edge
struct Edge {
    int u, v, weight;
};

// Structure for Union-Find
struct UnionFind {
    int *parent;
    int *rank;
};

// Function to create Union-Find structure
struct UnionFind* createUnionFind(int size) {
    struct UnionFind* uf = (struct UnionFind*)malloc(sizeof(struct UnionFind));
    uf->parent = (int*)malloc(size * sizeof(int));
    uf->rank = (int*)malloc(size * sizeof(int));
    for (int i = 0; i < size; i++) {
        uf->parent[i] = i;
        uf->rank[i] = 0;
    }
    return uf;
}

// Find function with path compression
int find(struct UnionFind* uf, int x) {
    if (uf->parent[x] != x) {
        uf->parent[x] = find(uf, uf->parent[x]);
    }
    return uf->parent[x];
}

// Union function with union by rank
void unionSets(struct UnionFind* uf, int x, int y) {
    int rootX = find(uf, x);
    int rootY = find(uf, y);
    if (rootX != rootY) {
        if (uf->rank[rootX] > uf->rank[rootY]) {
            uf->parent[rootY] = rootX;
        } else if (uf->rank[rootX] < uf->rank[rootY]) {
            uf->parent[rootX] = rootY;
        } else {
            uf->parent[rootY] = rootX;
            uf->rank[rootX]++;
        }
    }
}

// Comparison function for qsort
int compare(const void* a, const void* b) {
    struct Edge* edgeA = (struct Edge*)a;
    struct Edge* edgeB = (struct Edge*)b;
    return edgeA->weight - edgeB->weight;
}

int main() {
    int V, E;
    printf("Enter number of vertices and edges: ");
    scanf("%d %d", &V, &E);

    struct Edge* edges = (struct Edge*)malloc(E * sizeof(struct Edge));
    printf("Enter edges (u v weight):\n");
    for (int i = 0; i < E; i++) {
        scanf("%d %d %d", &edges[i].u, &edges[i].v, &edges[i].weight);
    }

    // Sort edges by weight
    qsort(edges, E, sizeof(struct Edge), compare);

    // Create Union-Find
    struct UnionFind* uf = createUnionFind(V);

    printf("Minimum Spanning Tree Edges:\n");
    int mst_weight = 0;
    int edge_count = 0;

    for (int i = 0; i < E && edge_count < V - 1; i++) {
        int u = edges[i].u;
        int v = edges[i].v;
        int weight = edges[i].weight;

        if (find(uf, u) != find(uf, v)) {
            unionSets(uf, u, v);
            printf("%d -- %d == %d\n", u, v, weight);
            mst_weight += weight;
            edge_count++;
        }
    }

    printf("Total weight of MST: %d\n", mst_weight);

    // Free memory
    free(edges);
    free(uf->parent);
    free(uf->rank);
    free(uf);

    return 0;
}
