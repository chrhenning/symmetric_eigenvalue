#include "backtransformation.h"
#include <stdlib.h>
#include <assert.h>
#include <omp.h>

void initEVRepNode(EVRepNode* r) {
    r->taskid = -1;
    r->n = 0;
    r->Q = NULL;
    r->L = NULL;
    r->D = NULL;
    r->z = NULL;
    r->N = NULL;
    r->G = NULL;
    r->C = NULL;
    r->S = NULL;
    r->P = NULL;
    r->beta = 0;
    r->theta = 0;
    r->parent = NULL;
    r->left = NULL;
    r->right = NULL;
    r->o = 0;
    r->numLeaves = 0;
    r->numGR = 0;
}

EVRepTree initEVRepTree(int depth, int numtasks, int n) {
    EVRepTree t;
    t.d = depth;
    t.t = malloc(t.d * sizeof(EVRepStage));

    int s; // stage in tree
    int p = 1; // power of two in current stage (1,2,4,8,...,numtasks)

    for(s = 0; s < t.d; ++s) {
        /*
         * Number of nodes in current stage.
         * This should be usually p, but if numtaks is not a power of two, then there might be
         * nodes in the tree that have only one child, not two. We decided, that all nodes are
         * as left as possible in the tree.
         *
         * Note, that the nodes in stage s belong to the nodes with a taskid which has no remainder
         * when divided by 2^(depth-1-s)
         */
        int n;
        // 2^(depth-1-s)
        int h = 1 << (depth-1-s);
        // max taskid = numtasks-1
        n = (numtasks-1) / h + 1;

        assert(n > 0);
        if (s == t.d-1)
            assert(numtasks <= p && n == numtasks);

        t.t[s].n = n;
        t.t[s].s = malloc(n * sizeof(EVRepNode));

        // init all nodes
        int j;
        for (j = 0; j < n; ++j) {
            EVRepNode* curr = &(t.t[s].s[j]);
            initEVRepNode(curr);
            curr->taskid = j * h;
            // set parent and child pointers
            if (s > 0) {
                curr->parent = &(t.t[s-1].s[j/2]);
                if (j % 2 == 0) {
                    curr->parent->left = curr;                    
                    if (j == n-1) // no split, single path
                        curr->parent->right = curr;
                } else {
                    curr->parent->right = curr;
                }
            }
        }

        p *= 2;
    }

    /*
     * initialize node sizes in tree (and offsets, number of leaf nodes)
     */
    // Note, our goal is to have equally sized leaves
    int leafSize, sizeRemainder;
    leafSize = n / numtasks;
    sizeRemainder = n % numtasks;

    int offset = 0;
    int i,j;
    for (i = 0; i < numtasks; ++i) {
        t.t[t.d-1].s[i].n = leafSize + (i < sizeRemainder ? 1 : 0);
        t.t[t.d-1].s[i].o = offset;
        offset += t.t[t.d-1].s[i].n;
        t.t[t.d-1].s[i].numLeaves = 1;
    }
    for (i = t.d-2; i >= 0; --i){
        offset = 0;
        for (j = 0; j < t.t[i].n; ++j) {
            if (t.t[i].s[j].left == t.t[i].s[j].right) {
                t.t[i].s[j].n = t.t[i].s[j].left->n;
                t.t[i].s[j].numLeaves = t.t[i].s[j].left->numLeaves;
            } else {
                t.t[i].s[j].n = t.t[i].s[j].left->n + t.t[i].s[j].right->n;
                t.t[i].s[j].numLeaves = t.t[i].s[j].left->numLeaves + t.t[i].s[j].right->numLeaves;
            }
            t.t[i].s[j].o = offset;
            offset += t.t[i].s[j].n;
        }
    }
    assert(t.t[0].s[0].numLeaves == numtasks);

    return t;
}


void freeEVRepTree(EVRepTree* t) {
    int s;
    #pragma omp parallel for default(shared) private(s) schedule(static)
    for(s = 0; s < t->d; ++s) {

        // free EV matrices of all nodes in current stage
        int j = 0;
        for (j = 0; j < t->t[s].n; ++j) {
            EVRepNode* n = &(t->t[s].s[j]);
            if (n->taskid == -1) { // nothing to free
                assert(n->Q == NULL && n->L == NULL && n->D == NULL && n->z == NULL && n->N == NULL && n->G == NULL && n->P == NULL);
            } else if (n->Q != NULL) { // node is leaf node
                assert(n->L != NULL && n->D == NULL && n->z == NULL && n->N == NULL && n->G == NULL && n->P == NULL);
                free(n->Q);
                free(n->L);
            } else {
                assert(n->Q == NULL);
                free(n->L);
                free(n->D);
                free(n->z);
                free(n->N);
                free(n->G);
                free(n->C);
                free(n->S);
                free(n->P);
            }
        }

        free(t->t[s].s);
    }

    t->d = 0;
    free(t->t);
    t->t = NULL;
}

EVRepNode* accessNode(EVRepTree* t, int stage, int taskid) {
    int h = 1 << (t->d-1-stage);
    assert(taskid % h == 0);
    assert(t->t[stage].s[taskid/h].taskid == taskid);
    return &(t->t[stage].s[taskid/h]);
}
