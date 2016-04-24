#ifndef BACKTRANSFORMATION_H
#define BACKTRANSFORMATION_H

/*
 * In this file, we describe the structure, how we store the eigenvector matrices of all the nodes
 * in the tree for the backtransformation steps
 */

/**
 * @brief The EVRepNodeStruct struct A representation of the eigenvector matrix on a node in the tree
 *
 * If the node is a leaf node, then the whole eigenvector matrix is stored in Q.
 * The member "taskid" describes which task contains (or can supply from its children)
 * the information contained in the structure (which might be null, if taskid != current task.
 */
struct EVRepNodeStruct {
    int taskid;
    /**
     * @brief n Size of vectors resp. matrix in this node
     */
    int n;
    /**
     * @brief Q Eigenvector matrix if leaf node, else NULL
     */
    double* Q;
    /**
     * @brief L store eigenvalues in here
     */
    double* L;
    /**
     * @brief D diagonal elements in rank-one update
     */
    double* D;
    /**
     * @brief z z vector from rank-one update
     */
    double* z;
    /**
     * @brief N store normalization factors in this vector, which are used to normalize the eigenvectors
     */
    double* N;
    /**
     * @brief G Given's rotation
     */
    int* G;
    /**
     * @brief c Angles of Given's rotations (stored as cos(theta)^2
     */
    double* c;
    /**
     * @brief P Order of Given's rotations
     */
    int* P;
    /**
     * @brief numGR Number of applied Given's Rotations
     */
    int numGR;

    double beta;
    double theta;

    // internal information, to easily move along the tree
    /**
     * @brief parent Parent node in tree
     */
    struct EVRepNodeStruct* parent;
    /**
     * @brief left Left child of node
     */
    struct EVRepNodeStruct* left;
    /**
     * @brief right Right child of node
     */
    struct EVRepNodeStruct* right;
    /**
     * @brief o Offset in current stage: the sum of node sizes for all nodes left to this one (this information is sometimes valuable, since nodes can have different sizes)
     */
    int o;
    /**
     * @brief numLeaves The number of leaf nodes that are children of the current node (e.g. the root has numtasks children)
     */
    int numLeaves;
};
typedef struct EVRepNodeStruct EVRepNode;

/**
 * @brief initEVRepNode Constructor for a EVRepNode struct
 * @param r The struct that should be initialized with default values
 */
void initEVRepNode(EVRepNode* r);

/**
 * @brief The EVRepStageStruct struct This struct represents all the Eigenvalue matrices within a stage of the tree
 */
struct EVRepStageStruct {
    /**
     * @brief n Number of nodes in current stage (size of s)
     */
    int n;
    /**
     * @brief s Store the eigenvalue matrix of each node in the current stage
     */
    EVRepNode* s;
};
typedef struct EVRepStageStruct EVRepStage;

/**
 * @brief The EVRepTreeStruct struct represents all the eigenvalue matrices for each node in the tree.
 */
struct EVRepTreeStruct {
    /**
     * @brief d Depth of tree
     */
    int d;
    /**
     * @brief t Tree: Array of size d with pointer to arrays that have a size equal to the number of nodes in the corresponding stage of the tree
     */
    EVRepStage* t;
};
typedef struct EVRepTreeStruct EVRepTree;

/**
 * @brief initEVRepTree Initialize a binary tree with given depth and leaf nodes
 * @param depth Depth of tree
 * @param numtasks Number of leaf nodes
 * @param n Size of original tridiagonal system
 * @return The initialized struct, with all the allocated memory.
 */
EVRepTree initEVRepTree(int depth, int numtasks, int n);

/**
 * @brief freeEVRepTree Free allocated memory in given tree (also the contained EV matrices
 * @param t Tree
 */
void freeEVRepTree(EVRepTree* t);

/**
 * @brief accessNode Access a node in the tree in a given stage that belongs to a given taskid
 * @param t Tree
 * @param stage
 * @param taskid
 * @return Pointer to node
 *
 * Taskid's on stage s must have zero remainder modulo 2^(depth-1-s)
 */
EVRepNode* accessNode(EVRepTree* t, int stage, int taskid);

#endif // BACKTRANSFORMATION_H
