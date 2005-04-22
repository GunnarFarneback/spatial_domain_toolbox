#define MAX_PARAMS 8

#define DEFAULT_MINSIZE 200
#define DEFAULT_MAXSIZE 400
#define DEFAULT_KERNELDIST 4
#define DEFAULT_LAMBDA 15.0 /* Factor to make it harder for new regions. */
#define DEFAULT_COVERAGE 1.0 /* How large part of the image to segment. */
/* Whether to recompute the parameters for a region at the end. */
#define DEFAULT_PARAMETER_RECOMPUTATION 0
/* How many times the initial blocks are regrown. */
#define DEFAULT_NUMBER_OF_INITIAL_ITERATIONS 2
/* Which motion models to use. Default is affine only. */
#define DEFAULT_MOTION_MODELS 2
#define DEFAULT_VERBOSE 1

/* Motion models */
#define CONSTANT_MOTION 1
#define AFFINE_MOTION 2
#define EIGHT_PARAMETER_MOTION 4

struct region
{
    double param[MAX_PARAMS];
    int motion_model;
    int startx;
    int starty;
    int size;
};

struct candidate
{
    double param[MAX_PARAMS];
    int motion_model;
    double cost;
    int x;
    int y;
    int last_rebuild;
};

