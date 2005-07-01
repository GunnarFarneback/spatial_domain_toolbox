#include "mex.h"
#include <string.h>

#define RECURSION_SIZE_LIMIT 8

/*
 * ANTIGRADIENT
 * 
 * See the help text in antigradient.m for a description.
 * 
 * Author: Gunnar Farnebäck
 *         Medical Informatics
 *         Linköping University, Sweden
 *         gunnar@imt.liu.se
 */


/*************** 2D ****************/

void
solve_directly2D(double *f, double *rhs,
		 double *f_out,
		 int M, int N)
{
    int s = M * N;
    int dims[2];
    mxArray *A_array;
    mxArray *b_array;
    double *A;
    double *b;
    int k;
    int i, j;
    mxArray *x_array;
    mxArray *input_arrays[2];

    dims[0] = s;
    dims[1] = s;
    A_array = mxCreateNumericArray(2, dims, mxDOUBLE_CLASS, mxREAL);
    A = mxGetPr(A_array);

    dims[1] = 1;
    b_array = mxCreateNumericArray(2, dims, mxDOUBLE_CLASS, mxREAL);
    b = mxGetPr(b_array);

    k = 0;
    for (j = 0; j < N; j++)
	for (i = 0; i < M; i++)
	{
            A[k + s * (j*M+i)] = -4;
            if (i > 0)
                A[k + s * (j*M+i-1)] = 1 + (i == M-1);
	    if (i < M-1)
                A[k + s * (j*M+i+1)] = 1 + (i == 0);
            if (j > 0)
                A[k + s * ((j-1)*M+i)] = 1 + (j == N-1);
            if (j < N-1)
                A[k + s * ((j+1)*M+i)] = 1 + (j == 0);
            
            b[k] = rhs[i+j*M];
            
            k++;
	}

    for (i = 0; i < s*s; i++)
	A[i] += 1.0 / (s*s);

    input_arrays[0] = A_array;
    input_arrays[1] = b_array;
    mexCallMATLAB(1, &x_array, 2, input_arrays, "\\");
    memcpy(f_out, mxGetPr(x_array), s * sizeof(*f));
    mxDestroyArray(x_array);
    mxDestroyArray(A_array);
    mxDestroyArray(b_array);
}


/* Gauss-Seidel smoothing iteration. */
void gauss_seidel2D(double *f, double *d, int M, int N)
{
    int pass;
    int i, j;
    int index;
    
    for (pass = 0; pass <= 1; pass++)
    {
	for (j = 0; j < N; j++)
	    for (i = 0; i < M; i++)
	    {
		double new_f;
		if ((i + j) % 2 != pass)
		    continue;

		index = i + j * M;
		new_f = -d[index];
		if (i == 0)
		    new_f += f[index + 1];
		else
		    new_f += f[index - 1];
		if (i == M-1)
		    new_f += f[index - 1];
		else
		    new_f += f[index + 1];
		
		if (j == 0)
		    new_f += f[index + M];
		else
		    new_f += f[index - M];
		if (j == N-1)
		    new_f += f[index - M];
		else
		    new_f += f[index + M];

		f[index] = 0.25 * new_f;
	    }
    }
}


/* Gauss-Seidel smoothing only applied to odd boundary lines. */
void gauss_seidel_odd_border2D(double *f, double *d, int M, int N)
{
    int pass;
    int i, j;
    int index;
    
    for (pass = 0; pass <= 1; pass++)
    {
	if (M % 2 == 1)
	{
	    i = M - 1;
	    for (j = 0; j < N; j++)
	    {
		double new_f;
		if (j % 2 != pass)
		    continue;
		
		index = i + j * M;
		new_f = -d[index];
		new_f += 2.0 * f[index - 1];
		
		if (j == 0)
		    new_f += f[index + M];
		else
		    new_f += f[index - M];
		if (j == N-1)
		    new_f += f[index - M];
		else
		    new_f += f[index + M];

		f[index] = 0.25 * new_f;
	    }
	}
	
	if (N % 2 == 1)
	{
	    j = N - 1;
	    for (i = 0; i < M; i++)
	    {
		double new_f;
		if (i % 2 != pass)
		    continue;

		index = i + j * M;
		new_f = -d[index];
		if (i == 0)
		    new_f += f[index + 1];
		else
		    new_f += f[index - 1];
		if (i == M-1)
		    new_f += f[index - 1];
		else
		    new_f += f[index + 1];
		
		new_f += 2.0 * f[index - M];

		f[index] = 0.25 * new_f;
	    }
	}
    }
}


void
downsample2D(double *rhs, int M, int N,
	     double *rhs_downsampled, int Mhalf, int Nhalf)
{
    int i, j;
    for (j = 0; j < Nhalf; j++)
	for (i = 0; i < Mhalf; i++)
	{
	    int index1 = (j * Mhalf + i);
	    int index2 = (2 * j * M + 2 * i);

	    if (2 * i == M - 1)
	    {
		if (2 * j == N - 1)
		    rhs_downsampled[index1] = (rhs[index2]
					       + rhs[index2 - M]
					       + rhs[index2 - 1]
					       + rhs[index2 - M - 1]);
		else
		    rhs_downsampled[index1] = (rhs[index2]
					       + rhs[index2 + M]
					       + rhs[index2 - 1]
					       + rhs[index2 + M - 1]);
	    }
	    else
	    {
		if (2 * j == N - 1)
		    rhs_downsampled[index1] = (rhs[index2]
					       + rhs[index2 - M]
					       + rhs[index2 + 1]
					       + rhs[index2 - M + 1]);
		else
		    rhs_downsampled[index1] = (rhs[index2]
					       + rhs[index2 + M]
					       + rhs[index2 + 1]
					       + rhs[index2 + M + 1]);
	    }
	}    
}


void
upsample2D(double *rhs, int M, int N,
	   double *v, int Mhalf, int Nhalf,
	   double *f_out)
{
    int i, j;
    
    /* Upsample and apply correction. Bilinear interpolation. */
    for (j = 0; j < Nhalf; j++)
	for (i = 0; i < Mhalf; i++)
	{
	    int index1 = (j * Mhalf + i);
	    int index2 = (2 * j * M + 2 * i);

	    /* Fine pixels northwest of coarse pixel center. */
	    if (i == 0)
	    {
		if (j == 0) /* NW corner. */
		    f_out[index2] += v[index1] - 0.25 * rhs[index2];
		else /* North edge. */
		    f_out[index2] += (0.75 * v[index1]
				      + 0.25 * v[index1 - Mhalf]
				      - 0.25 * rhs[index2]);
	    }
	    else
	    {
		if (j == 0) /* West edge. */
		    f_out[index2] += (0.75 * v[index1]
				      + 0.25 * v[index1 - 1]
				      - 0.25 * rhs[index2]);
		else /* Inner point. */
		    f_out[index2] += (0.5625 * v[index1]
				      + 0.1875 * v[index1 - 1]
				      + 0.1875 * v[index1 - Mhalf]
				      + 0.0625 * v[index1 - Mhalf - 1]);
	    }

	    /* Fine pixels southwest of coarse pixel center.
	     * These will only appear on the south edge if the
	     * fine height is even.
	     */
	    if (2*i+1 == M-1)
	    {
		if (j == 0) /* SW corner. */
		    f_out[index2 + 1] += v[index1] - 0.25 * rhs[index2 + 1];
		else /* South edge. */
		    f_out[index2 + 1] += (0.75 * v[index1]
					  + 0.25 * v[index1 - Mhalf]
					  - 0.25 * rhs[index2 + 1]);
	    }
	    else if (2*i+1 < M-1)
	    {
		if (j == 0) /* West edge. */
		    f_out[index2 + 1] += (0.75 * v[index1]
					  + 0.25 * v[index1 + 1]
					  - 0.25 * rhs[index2 + 1]);
		else /* Inner point. */
		    f_out[index2 + 1] += (0.5625 * v[index1]
					  + 0.1875 * v[index1 + 1]
					  + 0.1875 * v[index1 - Mhalf]
					  + 0.0625 * v[index1 - Mhalf + 1]);
	    }
	    
	    /* Fine pixels northeast of coarse pixel center.
	     * These will only appear on the east edge if the
	     * fine width is even.
	     */
	    if (i == 0)
	    {
		if (2*j+1 == N-1) /* NE corner. */
		    f_out[index2 + M] += v[index1] - 0.25 * rhs[index2 + M];
		else if (2*j+1 < N-1) /* North edge. */
		    f_out[index2 + M] += (0.75 * v[index1]
					  + 0.25 * v[index1 + Mhalf]
					  - 0.25 * rhs[index2 + M]);
	    }
	    else
	    {
		if (2*j+1 == N-1) /* East edge. */
		    f_out[index2 + M] += (0.75 * v[index1]
					  + 0.25 * v[index1 - 1]
					  - 0.25 * rhs[index2 + M]);
		else if (2*j+1 < N-1) /* Inner point. */
		    f_out[index2 + M] += (0.5625 * v[index1]
					  + 0.1875 * v[index1 - 1]
					  + 0.1875 * v[index1 + Mhalf]
					  + 0.0625 * v[index1 + Mhalf - 1]);
	    }
	    
	    /* Fine pixels southeast of coarse pixel center.
	     * These will only appear on the south edge if the fine
	     * height is even and on the east edge if the fine width
	     * is even.
	     */
	    if (2*i+1 == M-1)
	    {
		if (2*j+1 == N-1) /* SE corner. */
		    f_out[index2 + M + 1] += (v[index1]
					      - 0.25 * rhs[index2 + M + 1]);
		else if (2*j+1 < N-1) /* South edge.*/
		    f_out[index2 + M + 1] += (0.75 * v[index1]
					      + 0.25 * v[index1 + Mhalf]
					      - 0.25 * rhs[index2 + M + 1]);
	    }
	    else if (2*i+1 < M-1)
	    {
		if (2*j+1 == N-1) /* East edge. */
		    f_out[index2 + M + 1] += (0.75 * v[index1]
					      + 0.25 * v[index1 + 1]
					      - 0.25 * rhs[index2 + M + 1]);
		else if (2*j+1 < N-1) /* Inner point. */
		    f_out[index2 + M + 1] += (0.5625 * v[index1]
					      + 0.1875 * v[index1 + 1]
					      + 0.1875 * v[index1 + Mhalf]
					      + 0.0625 * v[index1 + Mhalf + 1]);
	    }
	}
}


/* Recursive multigrid function.*/
void
poisson_multigrid2D(double *f, double *d,
		    int n1, int n2, int nm,
		    double *f_out,
		    int M, int N, int *directly_solved)
{
    int i, j;
    int k;
    double *r;
    double *r_downsampled;
    double *v;
    int Mhalf;
    int Nhalf;

    /* Solve a sufficiently small problem directly. */
    if (M < RECURSION_SIZE_LIMIT || N < RECURSION_SIZE_LIMIT)
    {
	solve_directly2D(f, d, f_out, M, N);
	*directly_solved = 1;
	return;
    }
    *directly_solved = 0;

    /* Initialize solution. */
    memcpy(f_out, f, M * N * sizeof(*f_out));

    /* Pre-smoothing. */
    for (k = 0; k < n1; k++)
	gauss_seidel2D(f_out, d, M, N);
    
    /* Compute residual. */
    r = mxCalloc(M * N, sizeof(*r));
    for (j = 0; j < N; j++)
	for (i = 0; i < M; i++)
	{
	    int index = j * M + i;
	    double residual = d[index] + 4 * f_out[index];
	    if (i == 0)
		residual -= f_out[index + 1];
	    else
		residual -= f_out[index - 1];

	    if (i == M - 1)
		residual -= f_out[index - 1];
	    else
		residual -= f_out[index + 1];
		
	    if (j == 0)
		residual -= f_out[index + M];
	    else
		residual -= f_out[index - M];

	    if (j == N - 1)
		residual -= f_out[index - M];
	    else
		residual -= f_out[index + M];

	    r[index] = residual;
	}

    /* Downsample residual. */
    Mhalf = (M + 1) / 2;
    Nhalf = (N + 1) / 2;
    r_downsampled = mxCalloc(Mhalf * Nhalf, sizeof(*r_downsampled));
    downsample2D(r, M, N, r_downsampled, Mhalf, Nhalf);

    /* Recurse to compute a correction. */
    v = mxCalloc(Mhalf * Nhalf, sizeof(*v));
    for (k = 0; k < nm; k++)
    {
	int directly_solved;
	poisson_multigrid2D(v, r_downsampled, n1, n2, nm, v, Mhalf, Nhalf,
			    &directly_solved);
	if (directly_solved)
	    break;
    }

    upsample2D(r, M, N, v, Mhalf, Nhalf, f_out);

    /* Post-smoothing for odd border lines. */
    if (M % 2 == 1 || N % 2 == 1)
    {
	for (k = 0; k < n2 + 2; k++)
	    gauss_seidel_odd_border2D(f_out, d, M, N);
    }
    
    /* Post-smoothing. */
    for (k = 0; k < n2; k++)
	gauss_seidel2D(f_out, d, M, N);

    mxFree(r);
    mxFree(r_downsampled);
    mxFree(v);
}


/* It is assumed that f_out is initialized to zero when called. */
void
poisson_full_multigrid2D(double *rhs, int number_of_iterations,
			 int M, int N, double *f_out)
{
    double *rhs_downsampled;
    double *f_coarse;
    int k;
    
    /* Unless already coarsest scale, first recurse to coarser scale. */
    if (M >= RECURSION_SIZE_LIMIT && N >= RECURSION_SIZE_LIMIT)
    {
	/* Downsample right hand side. */
	int Mhalf = (M + 1) / 2;
	int Nhalf = (N + 1) / 2;
	rhs_downsampled = mxCalloc(Mhalf * Nhalf, sizeof(*rhs_downsampled));
	downsample2D(rhs, M, N, rhs_downsampled, Mhalf, Nhalf);
	
	f_coarse = mxCalloc(Mhalf * Nhalf, sizeof(*f_coarse));
	poisson_full_multigrid2D(rhs_downsampled, number_of_iterations,
				 Mhalf, Nhalf, f_coarse);

	/* Upsample the coarse result. */
	upsample2D(rhs, M, N, f_coarse, Mhalf, Nhalf, f_out);
    }

    /* Perform number_of_iterations standard multigrid cycles. */
    for (k = 0; k < number_of_iterations; k++)
    {
	int directly_solved;
	poisson_multigrid2D(f_out, rhs, 2, 2, 2, f_out, M, N,
			    &directly_solved);
	if (directly_solved)
	    break;
    }
}


void
antigradient2D(double *g, double mu, int number_of_iterations,
	       int M, int N, double *f_out)
{
    double *rhs;
    double sum;
    double mean;
    int i, j;
    
    /* Compute right hand side of Poisson problem with Neumann
     * boundary conditions, discretized by finite differences.
     */
    rhs = mxCalloc(M * N, sizeof(*rhs));
    for (j = 0; j < N; j++)
	for (i = 0; i < M; i++)
	{
	    int index1 = j * M + i;
	    int index2 = index1 + M * N;
	    double d = 0.0;

	    if (i == 0)
		d = g[index1 + 1] + g[index1];
	    else if (i == M - 1)
		d = - g[index1] - g[index1 - 1];
	    else
		d = 0.5 * (g[index1 + 1] - g[index1 - 1]);
	    
	    if (j == 0)
		d += g[index2 + M] + g[index2];
	    else if (j == N - 1)
		d += - g[index2] - g[index2 - M];
	    else
		d += 0.5 * (g[index2 + M] - g[index2 - M]);
	    rhs[index1] = d;
	}
    
    /* Solve the equation system with the full multigrid algorithm.
     * Use W cycles and 2 presmoothing and 2 postsmoothing
     * Gauss-Seidel iterations.
     */
    poisson_full_multigrid2D(rhs, number_of_iterations, M, N, f_out);
    
    /* Fix the mean value. */
    sum = 0.0;
    for (i = 0; i < M * N; i++)
	sum += f_out[i];
    
    mean = sum / (M * N);
    for (i = 0; i < M * N; i++)
    {
	f_out[i] -= mean;
	f_out[i] += mu;
    }
}


/*************** 3D ****************/

void
solve_directly3D(double *f, double *rhs,
		 double *f_out,
		 int M, int N, int P)
{
    int s = M * N * P;
    int dims[2];
    mxArray *A_array;
    mxArray *b_array;
    double *A;
    double *b;
    int k;
    int i, j, p;
    mxArray *x_array;
    mxArray *input_arrays[2];

    dims[0] = s;
    dims[1] = s;
    A_array = mxCreateNumericArray(2, dims, mxDOUBLE_CLASS, mxREAL);
    A = mxGetPr(A_array);

    dims[1] = 1;
    b_array = mxCreateNumericArray(2, dims, mxDOUBLE_CLASS, mxREAL);
    b = mxGetPr(b_array);

    k = 0;
    for (p = 0; p < P; p++)
	for (j = 0; j < N; j++)
	    for (i = 0; i < M; i++)
	    {
		A[k + s * ((p*N+j)*M+i)] = -6;
		if (i > 0)
		    A[k + s * ((p*N+j)*M+i-1)] = 1 + (i == M-1);
		if (i < M-1)
		    A[k + s * ((p*N+j)*M+i+1)] = 1 + (i == 0);
		if (j > 0)
		    A[k + s * ((p*N+(j-1))*M+i)] = 1 + (j == N-1);
		if (j < N-1)
		    A[k + s * ((p*N+(j+1))*M+i)] = 1 + (j == 0);
		if (p > 0)
		    A[k + s * (((p-1)*N+j)*M+i)] = 1 + (p == P-1);
		if (p < P-1)
		    A[k + s * (((p+1)*N+j)*M+i)] = 1 + (p == 0);
		
		b[k] = rhs[(p*N+j)*M+i];
            
		k++;
	    }
    
    for (i = 0; i < s*s; i++)
	A[i] += 1.0 / (s*s);

    input_arrays[0] = A_array;
    input_arrays[1] = b_array;
    mexCallMATLAB(1, &x_array, 2, input_arrays, "\\");
    memcpy(f_out, mxGetPr(x_array), s * sizeof(*f));
    mxDestroyArray(x_array);
    mxDestroyArray(A_array);
    mxDestroyArray(b_array);
}


/* Gauss-Seidel smoothing iteration. */
void gauss_seidel3D(double *f, double *d, int M, int N, int P)
{
    int pass;
    int i, j, p;
    int index;
    int MN = M * N;
    
    for (pass = 0; pass <= 1; pass++)
    {
	for (p = 0; p < P; p++)
	    for (j = 0; j < N; j++)
		for (i = 0; i < M; i++)
		{
		    double new_f;
		    if ((i + j + p) % 2 != pass)
			continue;
		    
		    index = (p * N + j) * M + i;
		    new_f = -d[index];
		    if (i == 0)
			new_f += f[index + 1];
		    else
			new_f += f[index - 1];
		    if (i == M-1)
			new_f += f[index - 1];
		    else
			new_f += f[index + 1];
		    
		    if (j == 0)
			new_f += f[index + M];
		    else
			new_f += f[index - M];
		    if (j == N-1)
			new_f += f[index - M];
		    else
			new_f += f[index + M];
		    
		    if (p == 0)
			new_f += f[index + MN];
		    else
			new_f += f[index - MN];
		    if (p == P-1)
			new_f += f[index - MN];
		    else
			new_f += f[index + MN];

		    f[index] = (1 / 6.0) * new_f;
		}
    }
}


/* Gauss-Seidel smoothing only applied to odd boundary surfaces. */
void gauss_seidel_odd_border3D(double *f, double *d, int M, int N, int P)
{
    int pass;
    int i, j, p;
    int index;
    int MN = M * N;
    
    for (pass = 0; pass <= 1; pass++)
    {
	if (M % 2 == 1)
	{
	    i = M - 1;
	    for (p = 0; p < P; p++)
		for (j = 0; j < N; j++)
		{
		    double new_f;
		    if ((j + p) % 2 != pass)
			continue;
		
		    index = (p * N + j) * M + i;
		    new_f = -d[index];
		    new_f += 2.0 * f[index - 1];
		    
		    if (j == 0)
			new_f += f[index + M];
		    else
			new_f += f[index - M];
		    if (j == N-1)
			new_f += f[index - M];
		    else
			new_f += f[index + M];
		    
		    if (p == 0)
			new_f += f[index + MN];
		    else
			new_f += f[index - MN];
		    if (p == P-1)
			new_f += f[index - MN];
		    else
			new_f += f[index + MN];

		    f[index] = (1 / 6.0) * new_f;
		}
	}
	
	if (N % 2 == 1)
	{
	    j = N - 1;
	    for (p = 0; p < P; p++)
		for (i = 0; i < M; i++)
		{
		    double new_f;
		    if ((i + p) % 2 != pass)
			continue;

		    index = (p * N + j) * M + i;
		    new_f = -d[index];
		    if (i == 0)
			new_f += f[index + 1];
		    else
			new_f += f[index - 1];
		    if (i == M-1)
			new_f += f[index - 1];
		    else
			new_f += f[index + 1];
		    
		    new_f += 2.0 * f[index - M];

		    if (p == 0)
			new_f += f[index + MN];
		    else
			new_f += f[index - MN];
		    if (p == P-1)
			new_f += f[index - MN];
		    else
			new_f += f[index + MN];

		    f[index] = (1 / 6.0) * new_f;
	    }
	}
	
	if (P % 2 == 1)
	{
	    p = P - 1;
	    for (j = 0; j < N; j++)
		for (i = 0; i < M; i++)
		{
		    double new_f;
		    if ((i + j) % 2 != pass)
			continue;

		    index = (p * N + j) * M + i;
		    new_f = -d[index];
		    if (i == 0)
			new_f += f[index + 1];
		    else
			new_f += f[index - 1];
		    if (i == M-1)
			new_f += f[index - 1];
		    else
			new_f += f[index + 1];
		    
		    if (j == 0)
			new_f += f[index + M];
		    else
			new_f += f[index - M];
		    if (j == N-1)
			new_f += f[index - M];
		    else
			new_f += f[index + M];

		    new_f += 2.0 * f[index - MN];

		    f[index] = (1 / 6.0) * new_f;
	    }
	}
    }
}


void
downsample3D(double *rhs, int M, int N, int P,
	     double *rhs_coarse, int Mhalf, int Nhalf, int Phalf)
{
    int i, j, p;
    int MN = M * N;
    
    for (p = 0; p < Phalf; p++)
	for (j = 0; j < Nhalf; j++)
	    for (i = 0; i < Mhalf; i++)
	    {
		int index1 = ((p * Nhalf + j) * Mhalf + i);
		int index2 = ((2 * p * N + 2 * j) * M + 2 * i);
		
		if (2 * i == M - 1)
		{
		    if (2 * j == N - 1)
		    {
			if (2 * p == P - 1)
			    rhs_coarse[index1] = (rhs[index2]
						  + rhs[index2 - M]
						  + rhs[index2 - 1]
						  + rhs[index2 - M - 1]
						  + rhs[index2 - MN]
						  + rhs[index2 - M - MN]
						  + rhs[index2 - 1 - MN]
						  + rhs[index2 - M - 1 - MN]);
			else
			    rhs_coarse[index1] = (rhs[index2]
						  + rhs[index2 - M]
						  + rhs[index2 - 1]
						  + rhs[index2 - M - 1]
						  + rhs[index2 + MN]
						  + rhs[index2 - M + MN]
						  + rhs[index2 - 1 + MN]
						  + rhs[index2 - M - 1 + MN]);
		    }
		    else
		    {
			if (2 * p == P - 1)
			    rhs_coarse[index1] = (rhs[index2]
						  + rhs[index2 + M]
						  + rhs[index2 - 1]
						  + rhs[index2 + M - 1]
						  + rhs[index2 - MN]
						  + rhs[index2 + M - MN]
						  + rhs[index2 - 1 - MN]
						  + rhs[index2 + M - 1 + MN]);
			else
			    rhs_coarse[index1] = (rhs[index2]
						  + rhs[index2 + M]
						  + rhs[index2 - 1]
						  + rhs[index2 + M - 1]
						  + rhs[index2 + MN]
						  + rhs[index2 + M + MN]
						  + rhs[index2 - 1 + MN]
						  + rhs[index2 + M - 1 + MN]);
		    }
		}
		else
		{
		    if (2 * j == N - 1)
		    {
			if (2 * p == P - 1)
			    rhs_coarse[index1] = (rhs[index2]
						  + rhs[index2 - M]
						  + rhs[index2 + 1]
						  + rhs[index2 - M + 1]
						  + rhs[index2 - MN]
						  + rhs[index2 - M - MN]
						  + rhs[index2 + 1 - MN]
						  + rhs[index2 - M + 1 - MN]);
			else
			    rhs_coarse[index1] = (rhs[index2]
						  + rhs[index2 - M]
						  + rhs[index2 + 1]
						  + rhs[index2 - M + 1]
						  + rhs[index2 + MN]
						  + rhs[index2 - M + MN]
						  + rhs[index2 + 1 + MN]
						  + rhs[index2 - M + 1 + MN]);
		    }
		    else
		    {
			if (2 * p == P - 1)
			    rhs_coarse[index1] = (rhs[index2]
						  + rhs[index2 + M]
						  + rhs[index2 + 1]
						  + rhs[index2 + M + 1]
						  + rhs[index2 - MN]
						  + rhs[index2 + M - MN]
						  + rhs[index2 + 1 - MN]
						  + rhs[index2 + M + 1 - MN]);
			else
			    rhs_coarse[index1] = (rhs[index2]
						  + rhs[index2 + M]
						  + rhs[index2 + 1]
						  + rhs[index2 + M + 1]
						  + rhs[index2 + MN]
						  + rhs[index2 + M + MN]
						  + rhs[index2 + 1 + MN]
						  + rhs[index2 + M + 1 + MN]);
		    }
		}
		rhs_coarse[index1] *= 0.5;
	    } 
}


void
upsample3D(double *rhs, int M, int N, int P,
	   double *v, int Mhalf, int Nhalf, int Phalf,
	   double *f_out)
{
    int i, j, p;
    int MN = M * N;
    int MNhalf = Mhalf * Nhalf;

    /* Upsample and apply correction. Bilinear interpolation. */
    for (p = 0; p < Phalf; p++)
	for (j = 0; j < Nhalf; j++)
	    for (i = 0; i < Mhalf; i++)
	    {
		int index1 = ((p * Nhalf + j) * Mhalf + i);
		int index2 = ((2 * p * N + 2 * j) * M + 2 * i);
		
		/* Fine pixels down-northwest of coarse pixel center. */
		if (i == 0)
		{
		    if (j == 0)
		    {
			if (p == 0) /* DNW corner */
			    f_out[index2] += v[index1] - 0.25 * rhs[index2];
			else /* NW edge */
			    f_out[index2] += (0.75 * v[index1]
					      + 0.25 * v[index1 - MNhalf]
					      - 0.25 * rhs[index2]);
		    }
		    else
		    {
			if (p == 0) /* DN edge */
			    f_out[index2] += (0.75 * v[index1]
					      + 0.25 * v[index1 - Mhalf]
					      - 0.25 * rhs[index2]);
			else /* North surface */
			    f_out[index2] += (0.5625 * v[index1]
					      + 0.1875 * v[index1 - Mhalf]
					      + 0.1875 * v[index1 - MNhalf]
					      + 0.0625 * v[index1 - Mhalf - MNhalf]
					      - 0.25 * rhs[index2]);
		    }
		}
		else
		{
		    if (j == 0)
		    {
			if (p == 0) /* DW edge */
			    f_out[index2] += (0.75 * v[index1]
					      + 0.25 * v[index1 - 1]
					      - 0.25 * rhs[index2]);
			else /* West surface */
			    f_out[index2] += (0.5625 * v[index1]
					      + 0.1875 * v[index1 - 1]
					      + 0.1875 * v[index1 - MNhalf]
					      + 0.0625 * v[index1 - 1 - MNhalf]
					      - 0.25 * rhs[index2]);
		    }
		    else
		    {
			if (p == 0) /* Down surface */
			    f_out[index2] += (0.5625 * v[index1]
					      + 0.1875 * v[index1 - 1]
					      + 0.1875 * v[index1 - Mhalf]
					      + 0.0625 * v[index1 - 1 - Mhalf]
					      - 0.25 * rhs[index2]);
			else /* inner point */
			    f_out[index2] += (0.421875 * v[index1]
					      + 0.140625 * v[index1 - 1]
					      + 0.140625 * v[index1 - Mhalf]
					      + 0.046875 * v[index1 - 1 - Mhalf]
					      + 0.140625 * v[index1 - MNhalf]
					      + 0.046875 * v[index1 - 1 - MNhalf]
					      + 0.046875 * v[index1 - Mhalf - MNhalf]
					      + 0.015625 * v[index1 - 1 - Mhalf - MNhalf]);
		    }
		}
		
		/* Fine pixels down-southwest of coarse pixel center.
		 * These will only appear on the south surface if the fine
		 * height is even.
		 */
		if (2*i+1 == M-1)
		{
		    if (j == 0)
		    {
			if (p == 0) /* DNW corner */
			    f_out[index2 + 1] += v[index1] - 0.25 * rhs[index2 + 1];
			else /* NW edge */
			    f_out[index2 + 1] += (0.75 * v[index1]
						  + 0.25 * v[index1 - MNhalf]
						  - 0.25 * rhs[index2 + 1]);
		    }
		    else
		    {
			if (p == 0) /* DN edge */
			    f_out[index2 + 1] += (0.75 * v[index1]
						  + 0.25 * v[index1 - Mhalf]
						  - 0.25 * rhs[index2 + 1]);
			else /* North surface */
			    f_out[index2 + 1] += (0.5625 * v[index1]
						  + 0.1875 * v[index1 - Mhalf]
						  + 0.1875 * v[index1 - MNhalf]
						  + 0.0625 * v[index1 - Mhalf - MNhalf]
						  - 0.25 * rhs[index2 + 1]);
		    }
		}
		else if (2*i+1 < M-1)
		{
		    if (j == 0)
		    {
			if (p == 0) /* DW edge */
			    f_out[index2 + 1] += (0.75 * v[index1]
						  + 0.25 * v[index1 + 1]
						  - 0.25 * rhs[index2 + 1]);
			else /* West surface */
			    f_out[index2 + 1] += (0.5625 * v[index1]
						  + 0.1875 * v[index1 + 1]
						  + 0.1875 * v[index1 - MNhalf]
						  + 0.0625 * v[index1 + 1 - MNhalf]
						  - 0.25 * rhs[index2 + 1]);
		    }
		    else
		    {
			if (p == 0) /* Down surface */
			    f_out[index2 + 1] += (0.5625 * v[index1]
						  + 0.1875 * v[index1 + 1]
						  + 0.1875 * v[index1 - Mhalf]
						  + 0.0625 * v[index1 + 1 - Mhalf]
						  - 0.25 * rhs[index2 + 1]);
			else /* inner point */
			    f_out[index2 + 1] += (0.421875 * v[index1]
						  + 0.140625 * v[index1 + 1]
						  + 0.140625 * v[index1 - Mhalf]
						  + 0.046875 * v[index1 + 1 - Mhalf]
						  + 0.140625 * v[index1 - MNhalf]
						  + 0.046875 * v[index1 + 1 - MNhalf]
						  + 0.046875 * v[index1 - Mhalf - MNhalf]
						  + 0.015625 * v[index1 + 1 - Mhalf - MNhalf]);
		    }
		}
		
		/* Fine pixels down-northeast of coarse pixel center.
		 * These will only appear on the east surface if the fine
		 * width is even.
		 */
		if (i == 0)
		{
		    if (2*j+1 == N-1)
		    {
			if (p == 0) /* DNE corner */
			    f_out[index2 + M] += (v[index1]
						  - 0.25 * rhs[index2 + M]);
			else /* NE edge */
			    f_out[index2 + M] += (0.75 * v[index1]
						  + 0.25 * v[index1 - MNhalf]
						  - 0.25 * rhs[index2 + M]);
		    }
		    else if (2*j+1 < N-1)
		    {
			if (p == 0) /* DN edge */
			    f_out[index2 + M] += (0.75 * v[index1]
						  + 0.25 * v[index1 + Mhalf]
						  - 0.25 * rhs[index2 + M]);
			else /* North surface */
			    f_out[index2 + M] += (0.5625 * v[index1]
						  + 0.1875 * v[index1 + Mhalf]
						  + 0.1875 * v[index1 - MNhalf]
						  + 0.0625 * v[index1 + Mhalf - MNhalf]
						  - 0.25 * rhs[index2 + M]);
		    }
		}
		else
		{
		    if (2*j+1 == N-1)
		    {
			if (p == 0) /* DE edge */
			    f_out[index2 + M] += (0.75 * v[index1]
						  + 0.25 * v[index1 - 1]
						  - 0.25 * rhs[index2 + M]);
			else /* East surface */
			    f_out[index2 + M] += (0.5625 * v[index1]
						  + 0.1875 * v[index1 - 1]
						  + 0.1875 * v[index1 - MNhalf]
						  + 0.0625 * v[index1 - 1 - MNhalf]
						  - 0.25 * rhs[index2 + M]);
		    }
		    else if (2*j+1 < N-1)
		    {
			if (p == 0) /* Down surface */
			    f_out[index2 + M] += (0.5625 * v[index1]
						  + 0.1875 * v[index1 - 1]
						  + 0.1875 * v[index1 + Mhalf]
						  + 0.0625 * v[index1 - 1 + Mhalf]
						  - 0.25 * rhs[index2 + M]);
			else /* inner point */
			    f_out[index2 + M] += (0.421875 * v[index1]
						  + 0.140625 * v[index1 - 1]
						  + 0.140625 * v[index1 + Mhalf]
						  + 0.046875 * v[index1 - 1 + Mhalf]
						  + 0.140625 * v[index1 - MNhalf]
						  + 0.046875 * v[index1 - 1 - MNhalf]
						  + 0.046875 * v[index1 + Mhalf - MNhalf]
						  + 0.015625 * v[index1 - 1 + Mhalf - MNhalf]);
		    }
		}
		
		/* Fine pixels down-southeast of coarse pixel center.
		 * These will only appear on the south surface if the fine
		 * height is even and on the east surface if the fine width
		 * is even.
		 */
		if (2*i+1 == M-1)
		{
		    if (2*j+1 == N-1)
		    {
			if (p == 0) /* DNE corner */
			    f_out[index2 + 1 + M] += v[index1] - 0.25 * rhs[index2 + 1 + M];
			else /* NE edge */
			    f_out[index2 + 1 + M] += (0.75 * v[index1]
						      + 0.25 * v[index1 - MNhalf]
						      - 0.25 * rhs[index2 + 1 + M]);
		    }
		    else if (2*j+1 < N-1)
		    {
			if (p == 0) /* DN edge */
			    f_out[index2 + 1 + M] += (0.75 * v[index1]
						      + 0.25 * v[index1 + Mhalf]
						      - 0.25 * rhs[index2 + 1 + M]);
			else /* North surface */
			    f_out[index2 + 1 + M] += (0.5625 * v[index1]
						      + 0.1875 * v[index1 + Mhalf]
						      + 0.1875 * v[index1 - MNhalf]
						      + 0.0625 * v[index1 + Mhalf - MNhalf]
						      - 0.25 * rhs[index2 + 1 + M]);
		    }
		}
		else if (2*i+1 < M-1)
		{
		    if (2*j+1 == N-1)
		    {
			if (p == 0) /* DE edge */
			    f_out[index2 + 1 + M] += (0.75 * v[index1]
						      + 0.25 * v[index1 + 1]
						      - 0.25 * rhs[index2 + 1 + M]);
			else /* East surface */
			    f_out[index2 + 1 + M] += (0.5625 * v[index1]
						      + 0.1875 * v[index1 + 1]
						      + 0.1875 * v[index1 - MNhalf]
						      + 0.0625 * v[index1 + 1 - MNhalf]
						      - 0.25 * rhs[index2 + 1 + M]);
		    }
		    else if (2*j+1 < N-1)
		    {
			if (p == 0) /* Down surface */
			    f_out[index2 + 1 + M] += (0.5625 * v[index1]
						      + 0.1875 * v[index1 + 1]
						      + 0.1875 * v[index1 + Mhalf]
						      + 0.0625 * v[index1 + 1 + Mhalf]
						      - 0.25 * rhs[index2 + 1 + M]);
			else /* inner point */
			    f_out[index2 + 1 + M] += (0.421875 * v[index1]
						      + 0.140625 * v[index1 + 1]
						      + 0.140625 * v[index1 + Mhalf]
						      + 0.046875 * v[index1 + 1 + Mhalf]
						      + 0.140625 * v[index1 - MNhalf]
						      + 0.046875 * v[index1 + 1 - MNhalf]
						      + 0.046875 * v[index1 + Mhalf - MNhalf]
						      + 0.015625 * v[index1 + 1 + Mhalf - MNhalf]);
		    }
		}

		/* Fine pixels up-northwest of coarse pixel center.
		 * These will only appear on the up surface if the fine
		 * tallness is even.
		 */
		if (i == 0)
		{
		    if (j == 0)
		    {
			if (2*p+1 == P-1) /* UNW corner */
			    f_out[index2 + MN] += v[index1] - 0.25 * rhs[index2 + MN];
			else if (2*p+1 < P-1) /* NW edge */
			    f_out[index2 + MN] += (0.75 * v[index1]
					      + 0.25 * v[index1 + MNhalf]
					      - 0.25 * rhs[index2 + MN]);
		    }
		    else
		    {
			if (2*p+1 == P-1) /* UN edge */
			    f_out[index2 + MN] += (0.75 * v[index1]
					      + 0.25 * v[index1 - Mhalf]
					      - 0.25 * rhs[index2 + MN]);
			else if (2*p+1 < P-1) /* North surface */
			    f_out[index2 + MN] += (0.5625 * v[index1]
					      + 0.1875 * v[index1 - Mhalf]
					      + 0.1875 * v[index1 + MNhalf]
					      + 0.0625 * v[index1 - Mhalf + MNhalf]
					      - 0.25 * rhs[index2 + MN]);
		    }
		}
		else
		{
		    if (j == 0)
		    {
			if (2*p+1 == P-1) /* UW edge */
			    f_out[index2 + MN] += (0.75 * v[index1]
					      + 0.25 * v[index1 - 1]
					      - 0.25 * rhs[index2 + MN]);
			else if (2*p+1 < P-1) /* West surface */
			    f_out[index2 + MN] += (0.5625 * v[index1]
					      + 0.1875 * v[index1 - 1]
					      + 0.1875 * v[index1 + MNhalf]
					      + 0.0625 * v[index1 - 1 + MNhalf]
					      - 0.25 * rhs[index2 + MN]);
		    }
		    else
		    {
			if (2*p+1 == P-1) /* Up surface */
			    f_out[index2 + MN] += (0.5625 * v[index1]
					      + 0.1875 * v[index1 - 1]
					      + 0.1875 * v[index1 - Mhalf]
					      + 0.0625 * v[index1 - 1 - Mhalf]
					      - 0.25 * rhs[index2 + MN]);
			else if (2*p+1 < P-1) /* inner point */
			    f_out[index2 + MN] += (0.421875 * v[index1]
					      + 0.140625 * v[index1 - 1]
					      + 0.140625 * v[index1 - Mhalf]
					      + 0.046875 * v[index1 - 1 - Mhalf]
					      + 0.140625 * v[index1 + MNhalf]
					      + 0.046875 * v[index1 - 1 + MNhalf]
					      + 0.046875 * v[index1 - Mhalf + MNhalf]
					      + 0.015625 * v[index1 - 1 - Mhalf + MNhalf]);
		    }
		}
		
		/* Fine pixels up-southwest of coarse pixel center.
		 * These will only appear on the south surface if the
		 * fine height is even and on the up surface if the
		 * fine tallness is even.
		 */
		if (2*i+1 == M-1)
		{
		    if (j == 0)
		    {
			if (2*p+1 == P-1) /* UNW corner */
			    f_out[index2 + 1 + MN] += v[index1] - 0.25 * rhs[index2 + 1 + MN];
			else if (2*p+1 < P-1) /* NW edge */
			    f_out[index2 + 1 + MN] += (0.75 * v[index1]
						  + 0.25 * v[index1 + MNhalf]
						  - 0.25 * rhs[index2 + 1 + MN]);
		    }
		    else
		    {
			if (2*p+1 == P-1) /* UN edge */
			    f_out[index2 + 1 + MN] += (0.75 * v[index1]
						  + 0.25 * v[index1 - Mhalf]
						  - 0.25 * rhs[index2 + 1 + MN]);
			else if (2*p+1 < P-1) /* North surface */
			    f_out[index2 + 1 + MN] += (0.5625 * v[index1]
						  + 0.1875 * v[index1 - Mhalf]
						  + 0.1875 * v[index1 + MNhalf]
						  + 0.0625 * v[index1 - Mhalf + MNhalf]
						  - 0.25 * rhs[index2 + 1 + MN]);
		    }
		}
		else if (2*i+1 < M-1)
		{
		    if (j == 0)
		    {
			if (2*p+1 == P-1) /* UW edge */
			    f_out[index2 + 1 + MN] += (0.75 * v[index1]
						  + 0.25 * v[index1 + 1]
						  - 0.25 * rhs[index2 + 1 + MN]);
			else if (2*p+1 < P-1) /* West surface */
			    f_out[index2 + 1 + MN] += (0.5625 * v[index1]
						  + 0.1875 * v[index1 + 1]
						  + 0.1875 * v[index1 + MNhalf]
						  + 0.0625 * v[index1 + 1 + MNhalf]
						  - 0.25 * rhs[index2 + 1 + MN]);
		    }
		    else
		    {
			if (2*p+1 == P-1) /* Up surface */
			    f_out[index2 + 1 + MN] += (0.5625 * v[index1]
						  + 0.1875 * v[index1 + 1]
						  + 0.1875 * v[index1 - Mhalf]
						  + 0.0625 * v[index1 + 1 - Mhalf]
						  - 0.25 * rhs[index2 + 1 + MN]);
			else if (2*p+1 < P-1) /* inner point */
			    f_out[index2 + 1 + MN] += (0.421875 * v[index1]
						  + 0.140625 * v[index1 + 1]
						  + 0.140625 * v[index1 - Mhalf]
						  + 0.046875 * v[index1 + 1 - Mhalf]
						  + 0.140625 * v[index1 + MNhalf]
						  + 0.046875 * v[index1 + 1 + MNhalf]
						  + 0.046875 * v[index1 - Mhalf + MNhalf]
						  + 0.015625 * v[index1 + 1 - Mhalf + MNhalf]);
		    }
		}
		
		/* Fine pixels up-northeast of coarse pixel center.
		 * These will only appear on the east surface if the
		 * fine width is even and on the up surface if the
		 * fine tallness is even.
		 */
		if (i == 0)
		{
		    if (2*j+1 == N-1)
		    {
			if (2*p+1 == P-1) /* UNE corner */
			    f_out[index2 + M + MN] += (v[index1]
						  - 0.25 * rhs[index2 + M + MN]);
			else if (2*p+1 < P-1) /* NE edge */
			    f_out[index2 + M + MN] += (0.75 * v[index1]
						  + 0.25 * v[index1 + MNhalf]
						  - 0.25 * rhs[index2 + M + MN]);
		    }
		    else if (2*j+1 < N-1)
		    {
			if (2*p+1 == P-1) /* UN edge */
			    f_out[index2 + M + MN] += (0.75 * v[index1]
						  + 0.25 * v[index1 + Mhalf]
						  - 0.25 * rhs[index2 + M + MN]);
			else if (2*p+1 < P-1) /* North surface */
			    f_out[index2 + M + MN] += (0.5625 * v[index1]
						  + 0.1875 * v[index1 + Mhalf]
						  + 0.1875 * v[index1 + MNhalf]
						  + 0.0625 * v[index1 + Mhalf + MNhalf]
						  - 0.25 * rhs[index2 + M + MN]);
		    }
		}
		else
		{
		    if (2*j+1 == N-1)
		    {
			if (2*p+1 == P-1) /* UE edge */
			    f_out[index2 + M + MN] += (0.75 * v[index1]
						  + 0.25 * v[index1 - 1]
						  - 0.25 * rhs[index2 + M + MN]);
			else if (2*p+1 < P-1) /* East surface */
			    f_out[index2 + M + MN] += (0.5625 * v[index1]
						  + 0.1875 * v[index1 - 1]
						  + 0.1875 * v[index1 + MNhalf]
						  + 0.0625 * v[index1 - 1 + MNhalf]
						  - 0.25 * rhs[index2 + M + MN]);
		    }
		    else if (2*j+1 < N-1)
		    {
			if (2*p+1 == P-1) /* Up surface */
			    f_out[index2 + M + MN] += (0.5625 * v[index1]
						  + 0.1875 * v[index1 - 1]
						  + 0.1875 * v[index1 + Mhalf]
						  + 0.0625 * v[index1 - 1 + Mhalf]
						  - 0.25 * rhs[index2 + M + MN]);
			else if (2*p+1 < P-1) /* inner point */
			    f_out[index2 + M + MN] += (0.421875 * v[index1]
						  + 0.140625 * v[index1 - 1]
						  + 0.140625 * v[index1 + Mhalf]
						  + 0.046875 * v[index1 - 1 + Mhalf]
						  + 0.140625 * v[index1 + MNhalf]
						  + 0.046875 * v[index1 - 1 + MNhalf]
						  + 0.046875 * v[index1 + Mhalf + MNhalf]
						  + 0.015625 * v[index1 - 1 + Mhalf + MNhalf]);
		    }
		}
		
		/* Fine pixels up-southeast of coarse pixel center.
		 * These will only appear on the south surface if the
		 * fine height is even, on the east surface if the
		 * fine width is even, and on the up surface if the
		 * fine tallness is even.
		 */
		if (2*i+1 == M-1)
		{
		    if (2*j+1 == N-1)
		    {
			if (2*p+1 == P-1) /* UNE corner */
			    f_out[index2 + 1 + M + MN] += v[index1] - 0.25 * rhs[index2 + 1 + M + MN];
			else if (2*p+1 < P-1) /* NE edge */
			    f_out[index2 + 1 + M + MN] += (0.75 * v[index1]
						      + 0.25 * v[index1 + MNhalf]
						      - 0.25 * rhs[index2 + 1 + M + MN]);
		    }
		    else if (2*j+1 < N-1)
		    {
			if (2*p+1 == P-1) /* UN edge */
			    f_out[index2 + 1 + M + MN] += (0.75 * v[index1]
						      + 0.25 * v[index1 + Mhalf]
						      - 0.25 * rhs[index2 + 1 + M + MN]);
			else if (2*p+1 < P-1) /* North surface */
			    f_out[index2 + 1 + M + MN] += (0.5625 * v[index1]
						      + 0.1875 * v[index1 + Mhalf]
						      + 0.1875 * v[index1 + MNhalf]
						      + 0.0625 * v[index1 + Mhalf + MNhalf]
						      - 0.25 * rhs[index2 + 1 + M + MN]);
		    }
		}
		else if (2*i+1 < M-1)
		{
		    if (2*j+1 == N-1)
		    {
			if (2*p+1 == P-1) /* UE edge */
			    f_out[index2 + 1 + M + MN] += (0.75 * v[index1]
						      + 0.25 * v[index1 + 1]
						      - 0.25 * rhs[index2 + 1 + M + MN]);
			else if (2*p+1 < P-1) /* East surface */
			    f_out[index2 + 1 + M + MN] += (0.5625 * v[index1]
						      + 0.1875 * v[index1 + 1]
						      + 0.1875 * v[index1 + MNhalf]
						      + 0.0625 * v[index1 + 1 + MNhalf]
						      - 0.25 * rhs[index2 + 1 + M + MN]);
		    }
		    else if (2*j+1 < N-1)
		    {
			if (2*p+1 == P-1) /* Up surface */
			    f_out[index2 + 1 + M + MN] += (0.5625 * v[index1]
						      + 0.1875 * v[index1 + 1]
						      + 0.1875 * v[index1 + Mhalf]
						      + 0.0625 * v[index1 + 1 + Mhalf]
						      - 0.25 * rhs[index2 + 1 + M + MN]);
			else if (2*p+1 < P-1) /* inner point */
			    f_out[index2 + 1 + M + MN] += (0.421875 * v[index1]
						      + 0.140625 * v[index1 + 1]
						      + 0.140625 * v[index1 + Mhalf]
						      + 0.046875 * v[index1 + 1 + Mhalf]
						      + 0.140625 * v[index1 + MNhalf]
						      + 0.046875 * v[index1 + 1 + MNhalf]
						      + 0.046875 * v[index1 + Mhalf + MNhalf]
						      + 0.015625 * v[index1 + 1 + Mhalf + MNhalf]);
		    }
		}
	    }
}


/* Recursive multigrid function.*/
void
poisson_multigrid3D(double *f, double *d,
		    int n1, int n2, int nm,
		    double *f_out,
		    int M, int N, int P, int *directly_solved)
{
    int i, j, p;
    int k;
    double *r;
    double *r_downsampled;
    double *v;
    int Mhalf;
    int Nhalf;
    int Phalf;
    int MN = M * N;

    /* Solve a sufficiently small problem directly. */
    if (M < RECURSION_SIZE_LIMIT
	|| N < RECURSION_SIZE_LIMIT
	|| P < RECURSION_SIZE_LIMIT)
    {
	solve_directly3D(f, d, f_out, M, N, P);
	*directly_solved = 1;
	return;
    }
    *directly_solved = 0;

    /* Initialize solution. */
    memcpy(f_out, f, M * N * P * sizeof(*f_out));

    /* Pre-smoothing. */
    for (k = 0; k < n1; k++)
	gauss_seidel3D(f_out, d, M, N, P);

    /* Compute residual. */
    r = mxCalloc(M * N * P, sizeof(*r));
    for (p = 0; p < P; p++)
	for (j = 0; j < N; j++)
	    for (i = 0; i < M; i++)
	    {
		int index = (p * N + j) * M + i;
		double residual = d[index] + 6 * f_out[index];
		if (i == 0)
		    residual -= f_out[index + 1];
		else
		    residual -= f_out[index - 1];
		
		if (i == M - 1)
		    residual -= f_out[index - 1];
		else
		    residual -= f_out[index + 1];
		
		if (j == 0)
		    residual -= f_out[index + M];
		else
		    residual -= f_out[index - M];
		
		if (j == N - 1)
		    residual -= f_out[index - M];
		else
		    residual -= f_out[index + M];
		
		if (p == 0)
		    residual -= f_out[index + MN];
		else
		    residual -= f_out[index - MN];
		
		if (p == P - 1)
		    residual -= f_out[index - MN];
		else
		    residual -= f_out[index + MN];
		
		r[index] = residual;
	    }

    /* Downsample residual. */
    Mhalf = (M + 1) / 2;
    Nhalf = (N + 1) / 2;
    Phalf = (P + 1) / 2;
    r_downsampled = mxCalloc(Mhalf * Nhalf * Phalf, sizeof(*r_downsampled));
    downsample3D(r, M, N, P, r_downsampled, Mhalf, Nhalf, Phalf);

    /* Recurse to compute a correction. */
    v = mxCalloc(Mhalf * Nhalf * Phalf, sizeof(*v));
    for (k = 0; k < nm; k++)
    {
	int directly_solved;
	poisson_multigrid3D(v, r_downsampled, n1, n2, nm, v,
			    Mhalf, Nhalf, Phalf, &directly_solved);
	if (directly_solved)
	    break;
    }

    upsample3D(r, M, N, P, v, Mhalf, Nhalf, Phalf, f_out);

    /* Post-smoothing for odd border lines. */
    if (M % 2 == 1 || N % 2 == 1 || P % 2 == 1)
    {
	for (k = 0; k < n2 + 2; k++)
	    gauss_seidel_odd_border3D(f_out, d, M, N, P);
    }
    
    /* Post-smoothing. */
    for (k = 0; k < n2; k++)
	gauss_seidel3D(f_out, d, M, N, P);

    mxFree(r);
    mxFree(r_downsampled);
    mxFree(v);
}


/* It is assumed that f_out is initialized to zero when called. */
void
poisson_full_multigrid3D(double *rhs, int number_of_iterations,
			 int M, int N, int P, double *f_out)
{
    double *rhs_downsampled;
    double *f_coarse;
    int k;
    
    /* Unless already coarsest scale, first recurse to coarser scale. */
    if (M >= RECURSION_SIZE_LIMIT
	&& N >= RECURSION_SIZE_LIMIT
	&& P >= RECURSION_SIZE_LIMIT)
    {
	/* Downsample right hand side. */
	int Mhalf = (M + 1) / 2;
	int Nhalf = (N + 1) / 2;
	int Phalf = (P + 1) / 2;
	rhs_downsampled = mxCalloc(Mhalf * Nhalf * Phalf,
				   sizeof(*rhs_downsampled));
	downsample3D(rhs, M, N, P, rhs_downsampled, Mhalf, Nhalf, Phalf);
	
	f_coarse = mxCalloc(Mhalf * Nhalf * Phalf, sizeof(*f_coarse));
	poisson_full_multigrid3D(rhs_downsampled, number_of_iterations,
				 Mhalf, Nhalf, Phalf, f_coarse);
	/* Upsample the coarse result. */
	upsample3D(rhs, M, N, P, f_coarse, Mhalf, Nhalf, Phalf, f_out);
    }

    /* Perform number_of_iterations standard multigrid cycles. */
    for (k = 0; k < number_of_iterations; k++)
    {
	int directly_solved;
	poisson_multigrid3D(f_out, rhs, 2, 2, 2, f_out, M, N, P,
			    &directly_solved);
	if (directly_solved)
	    break;
    }
}


void
antigradient3D(double *g, double mu, int number_of_iterations,
	       int M, int N, int P, double *f_out)
{
    double *rhs;
    double sum;
    double mean;
    int i, j, p;
    int MN = M * N;
    
    /* Compute right hand side of Poisson problem with Neumann
     * boundary conditions, discretized by finite differences.
     */
    rhs = mxCalloc(M * N * P, sizeof(*rhs));
    for (p = 0; p < P; p++)
	for (j = 0; j < N; j++)
	    for (i = 0; i < M; i++)
	    {
		int index1 = (p * N + j) * M + i;
		int index2 = index1 + M * N * P;
		int index3 = index1 + 2 * M * N * P;
		double d = 0.0;
		
		if (i == 0)
		    d = g[index1 + 1] + g[index1];
		else if (i == M - 1)
		    d = - g[index1] - g[index1 - 1];
		else
		    d = 0.5 * (g[index1 + 1] - g[index1 - 1]);
		
		if (j == 0)
		    d += g[index2 + M] + g[index2];
		else if (j == N - 1)
		    d += - g[index2] - g[index2 - M];
		else
		    d += 0.5 * (g[index2 + M] - g[index2 - M]);
		
		if (p == 0)
		    d += g[index3 + MN] + g[index3];
		else if (p == P - 1)
		    d += - g[index3] - g[index3 - MN];
		else
		    d += 0.5 * (g[index3 + MN] - g[index3 - MN]);
		
		rhs[index1] = d;
	    }
    
    /* Solve the equation system with the full multigrid algorithm.
     * Use W cycles and 2 presmoothing and 2 postsmoothing
     * Gauss-Seidel iterations.
     */
    poisson_full_multigrid3D(rhs, number_of_iterations, M, N, P, f_out);
    
    /* Fix the mean value. */
    sum = 0.0;
    for (i = 0; i < M * N * P; i++)
	sum += f_out[i];
    
    mean = sum / (M * N * P);
    for (i = 0; i < M * N * P; i++)
    {
	f_out[i] -= mean;
	f_out[i] += mu;
    }
}


void
mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    int M, N, P;
    double *g;
    double *f_out;
    double mu;
    int number_of_iterations;
    int dim;
    
    /* Check the input and output arguments. */
    
    /* First we expect an MxNx2 or MxNxPx3 array. */
    dim = mxGetNumberOfDimensions(prhs[0]) - 1;
    if (!mxIsNumeric(prhs[0]) || mxIsComplex(prhs[0])
	|| mxIsSparse(prhs[0]) || !mxIsDouble(prhs[0])
	|| dim < 2 || dim > 3
	|| mxGetDimensions(prhs[0])[dim] != dim)
    {
	mexErrMsgTxt("g is expected to be an MxNx2 or MxNxPx3 array.");
    }

    M = mxGetDimensions(prhs[0])[0];
    N = mxGetDimensions(prhs[0])[1];
    if (dim > 2)
	P = mxGetDimensions(prhs[0])[2];
    else
	P = 0;

    g = mxGetPr(prhs[0]);

    /* Next two scalars. */
    if (nrhs < 2)
	mu = 0.0;
    else
    {
	if (!mxIsNumeric(prhs[1]) || mxIsComplex(prhs[1])
	    || mxIsSparse(prhs[1]) || !mxIsDouble(prhs[1])
	    || mxGetNumberOfElements(prhs[1]) != 1)
	{
	    mexErrMsgTxt("mu is expected to be a scalar.");
	}
	mu = mxGetScalar(prhs[1]);
    }

    if (nrhs < 3)
    {
	number_of_iterations = 2;
	if (M % 2 == 1 || N % 2 == 1 || P % 2 == 1)
	    number_of_iterations += 2;
    }
    else
    {
	if (!mxIsNumeric(prhs[2]) || mxIsComplex(prhs[2])
	    || mxIsSparse(prhs[2]) || !mxIsDouble(prhs[2])
	    || mxGetNumberOfElements(prhs[2]) != 1)
	{
	    mexErrMsgTxt("N is expected to be a scalar.");
	}
	number_of_iterations = (int) mxGetScalar(prhs[2]);
	if (number_of_iterations < 0
	    || (double) number_of_iterations != mxGetScalar(prhs[2]))
	{
	    mexErrMsgTxt("N is expected to be a positive integer.");
	}
    }

    plhs[0] = mxCreateNumericArray(dim, mxGetDimensions(prhs[0]),
				   mxDOUBLE_CLASS, mxREAL);
    f_out = mxGetPr(plhs[0]);

    if (dim == 2)
	antigradient2D(g, mu, number_of_iterations, M, N, f_out);
    else
	antigradient3D(g, mu, number_of_iterations, M, N, P, f_out);
}

