#include "mex.h"

/*
 * r = POLYEXP_SOLVE_SYSTEM(BASIS, CONVRES_F, CONVRES_C)
 * 
 * Helper function for polyexp.m, solving equations of the form 3.9
 * when the basis functions are monomials. See polyexp.m for the
 * meaning of the parameters.
 * 
 * Author: Gunnar Farnebäck
 *         Computer Vision Laboratory
 *         Linköping University, Sweden
 *         gf@isy.liu.se
 */

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    int i, j, k;
    int N, M;
    const int *basisdims;
    double *basis;
    const mxArray *convres_f_array;
    const mxArray *convres_c_array;
    double **G;
    double **h;
    int indices[3];
    int index;
    int num_outdims;
    int outdims[4];
    mxArray *r_array;
    double *r;
    int num_elements;
    mxArray *Qmatrix;
    double *Q;
    mxArray *qvector;
    double *q;
    double *p;
    mxArray *input[2];
    mxArray *output[1];
    int num_in, num_out;    
    
    /* Check the number of input and output arguments. */
    if (nrhs < 3)
	mexErrMsgTxt("Too few input arguments.");
    if (nrhs > 3)
	mexErrMsgTxt("Too many input arguments.");
    if (nlhs > 1)
	mexErrMsgTxt("Too many output arguments.");

    /* Check the formats of the input arguments. */
    if (!mxIsNumeric(prhs[0]) || mxIsComplex(prhs[0])
	|| mxIsSparse(prhs[0]) || !mxIsDouble(prhs[0]))
    {
	mexErrMsgTxt("Unexpected format for basis.");
    }

    if (!mxIsCell(prhs[1]))
	mexErrMsgTxt("convres_f is expected to be a cell array.");
    
    if (!mxIsCell(prhs[2]))
	mexErrMsgTxt("convres_c is expected to be a cell array.");
    
    if (mxGetNumberOfDimensions(prhs[0]) != 2)
	mexErrMsgTxt("basis must be a matrix.");

    basisdims = mxGetDimensions(prhs[0]);
    N = basisdims[0]; /* Number of signal dimensions. */
    M = basisdims[1]; /* Number of basis functions. */
    basis = mxGetPr(prhs[0]);
    
    convres_f_array = prhs[1];
    convres_c_array = prhs[2];

    /* We want to set up a matrix and a vector with the pointers to
     * the start of the elements in the equation system.
     */
    G = mxCalloc(M*M, sizeof(*G));
    h = mxCalloc(M, sizeof(*h));

    for (i = 0; i < M; i++)
    {
	for (k = 0; k < N; k++)
	    indices[k] = (int) basis[N * i + k];

	index = mxCalcSingleSubscript(convres_f_array, N, indices);
	h[i] = mxGetPr(mxGetCell(convres_f_array, index));
	
	for (j = 0; j < M; j++)
	{
	    for (k = 0; k < N; k++)
		indices[k] = (int) (basis[N * i + k] + basis[N * j + k]);

	    index = mxCalcSingleSubscript(convres_c_array, N, indices);
	    mxGetCell(convres_c_array, index);
	    mxGetPr(mxGetCell(convres_c_array, index));
	    G[i + j * M] = mxGetPr(mxGetCell(convres_c_array, index));
	}
    }

    num_outdims = mxGetNumberOfDimensions(mxGetCell(convres_c_array, index));
    num_elements = 1;
    for (k = 0; k < num_outdims; k++)
    {
	outdims[k] = mxGetDimensions(mxGetCell(convres_c_array, index))[k];
	num_elements *= outdims[k];
    }
    if (M > 1)
    {
	outdims[num_outdims] = M;
	num_outdims++;
    }
    
    r_array = mxCreateNumericArray(num_outdims, outdims,
				   mxDOUBLE_CLASS, mxREAL);
    r = mxGetPr(r_array);
    
    Qmatrix = mxCreateDoubleMatrix(M, M, mxREAL);
    Q = mxGetPr(Qmatrix);
    input[0] = Qmatrix;
    qvector = mxCreateDoubleMatrix(M, 1, mxREAL);
    q = mxGetPr(qvector);
    input[1] = qvector;
    num_in = 2;
    num_out = 1;
    
    for (k = 0; k < num_elements; k++)
    {
	for (i = 0; i < M; i++)
	{
	    for (j = 0; j < M; j++)
		Q[i + j * M] = G[i + j * M][k];
	    
	    q[i] = h[i][k];
	}
	mexCallMATLAB(num_out, output, num_in, input, "\\");
	p = mxGetPr(output[0]);
	for (i = 0; i < M; i++)
	    r[k + i * num_elements] = p[i];
	mxDestroyArray(output[0]);
    }
    
    /* Output the computed result. */
    plhs[0] = r_array;
}
