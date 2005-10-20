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


static void
logmatrix(double *M, int m, int n, char *name, char *logfunction)
{
  mxArray *M_array;
  int dims[2];
  int i;
  mxArray *input_arrays[2];
  dims[0] = m;
  dims[1] = n;
  M_array = mxCreateNumericArray(2, dims, mxDOUBLE_CLASS, mxREAL);
  for (i = 0; i < m * n; i++)
    mxGetPr(M_array)[i] = M[i];
  
  input_arrays[0] = M_array;
  input_arrays[1] = mxCreateString(name);
  mexCallMATLAB(0, NULL, 2, input_arrays, logfunction);
}

/*************** 2D ****************/

static void
solve_directly2D(double *f, double *lhs, double *rhs, double *f_out,
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
      int index = j*M+i;
      A[k + s * (j*M+i)] = lhs[9 * index] + (lhs[9 * index] == 0.0);
      if (i > 0)
	A[k + s * (j*M+i-1)] = lhs[9 * index + 1];
      if (i < M-1)
	A[k + s * (j*M+i+1)] = lhs[9 * index + 2];
      if (j > 0)
	A[k + s * ((j-1)*M+i)] = lhs[9 * index + 3];
      if (j < N-1)
	A[k + s * ((j+1)*M+i)] = lhs[9 * index + 4];
      if (i > 0 && j > 0)
	A[k + s * ((j-1)*M+i-1)] = lhs[9 * index + 5];
      if (i > 0 && j < N-1)
	A[k + s * ((j+1)*M+i-1)] = lhs[9 * index + 6];
      if (i < M-1 && j > 0)
	A[k + s * ((j-1)*M+i+1)] = lhs[9 * index + 7];
      if (i < M-1 && j < N-1)
	A[k + s * ((j+1)*M+i+1)] = lhs[9 * index + 8];
      
      b[k] = rhs[index];
      
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


/* Gauss-Seidel smoothing iteration. Red-black ordering. */
static void
gauss_seidel2D(double *f, double *A, double *d, int M, int N)
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
	if (A[9 * index] == 0.0)
	  continue;
	
	new_f = d[index];
	if (i > 0)
	  new_f -= A[9 * index + 1] * f[index - 1];

	if (i < M-1)
	  new_f -= A[9 * index + 2] * f[index + 1];

	if (j > 0)
	  new_f -= A[9 * index + 3] * f[index - M];

	if (j < N-1)
	  new_f -= A[9 * index + 4] * f[index + M];
	
	if (i > 0 && j > 0)
	  new_f -= A[9 * index + 5] * f[index - 1 - M];

	if (i > 0 && j < N-1)
	  new_f -= A[9 * index + 6] * f[index - 1 + M];

	if (i < M-1 && j > 0)
	  new_f -= A[9 * index + 7] * f[index + 1 - M];

	if (i < M-1 && j < N-1)
	  new_f -= A[9 * index + 8] * f[index + 1 + M];

	f[index] = new_f / A[9 * index];
      }
  }
}


static void
downsample2D(double *rhs, int M, int N,
	     double *rhs_coarse, int Mhalf, int Nhalf)
{
  int i, j;
  int index1;
  int index2;
  
  if (M % 2 == 0 && N % 2 == 0)
  {
    for (j = 0; j < Nhalf; j++)
      for (i = 0; i < Mhalf; i++)
      {
	index1 = (j * Mhalf + i);
	index2 = (2 * j * M + 2 * i);
	
	rhs_coarse[index1] = (rhs[index2]
			      + rhs[index2 + M]
			      + rhs[index2 + 1]
			      + rhs[index2 + M + 1]);
      }
  }
  
  if (M % 2 == 1 && N % 2 == 0)
  {
    for (j = 0; j < Nhalf; j++)
    {
      index1 = (j * Mhalf);
      index2 = (2 * j * M);
      rhs_coarse[index1] = (rhs[index2]
			    + rhs[index2 + M]
			    + rhs[index2 + 1]
			    + rhs[index2 + M + 1]);
      for (i = 1; i < Mhalf - 1; i++)
      {
	index1++;
	index2 += 2;
	rhs_coarse[index1] = (rhs[index2]
			      + rhs[index2 + M]
			      + 0.5 * (rhs[index2 - 1]
				       + rhs[index2 + 1]
				       + rhs[index2 + M - 1]
				       + rhs[index2 + M + 1]));
      }
      index1++;
      index2 += 2;
      rhs_coarse[index1] = (rhs[index2]
			    + rhs[index2 + M]
			    + rhs[index2 - 1]
			    + rhs[index2 + M - 1]);
    }
  }
  
  if (M % 2 == 0 && N % 2 == 1)
  {
    for (i = 0; i < Mhalf; i++)
    {
      index1 = i;
      index2 = 2 * i;
      rhs_coarse[index1] = (rhs[index2]
			    + rhs[index2 + 1]
			    + rhs[index2 + M]
			    + rhs[index2 + M + 1]);
    }
    
    for (j = 1; j < Nhalf - 1; j++)
    {
      for (i = 0; i < Mhalf; i++)
      {
	index1 = (j * Mhalf + i);
	index2 = (2 * j * M + 2 * i);
	rhs_coarse[index1] = (rhs[index2]
			      + rhs[index2 + 1]
			      + 0.5 * (rhs[index2 - M]
				       + rhs[index2 + M]
				       + rhs[index2 - M + 1]
				       + rhs[index2 + M + 1]));
      }
    }
    
    for (i = 0; i < Mhalf; i++)
    {
      index1 = (j * Mhalf + i);
      index2 = (2 * j * M + 2 * i);
      rhs_coarse[index1] = (rhs[index2]
			    + rhs[index2 + 1]
			    + rhs[index2 - M]
			    + rhs[index2 - M + 1]);
    }
  }
  
  if (M % 2 == 1 && N % 2 == 1)
  {
    for (j = 1; j < Nhalf - 1; j++)
    {
      for (i = 1; i < Mhalf - 1; i++)
      {
	index1 = (j * Mhalf + i);
	index2 = (2 * j * M + 2 * i);
	rhs_coarse[index1] = (rhs[index2]
			      + 0.5 * (rhs[index2 - 1]
				       + rhs[index2 + 1]
				       + rhs[index2 - M]
				       + rhs[index2 + M])
			      + 0.25* (rhs[index2 - M - 1]
				       + rhs[index2 - M + 1]
				       + rhs[index2 + M - 1]
				       + rhs[index2 + M + 1]));
      }
    }
    
    for (i = 1; i < Mhalf - 1; i++)
    {
      index1 = i;
      index2 = 2 * i;
      rhs_coarse[index1] = (rhs[index2]
			    + rhs[index2 + M]
			    + 0.5 * (rhs[index2 - 1]
				     + rhs[index2 + 1]
				     + rhs[index2 + M - 1]
				     + rhs[index2 + M + 1]));
      index1 = ((Nhalf - 1) * Mhalf + i);
      index2 = (2 * (Nhalf - 1) * M + 2 * i);
      rhs_coarse[index1] = (rhs[index2]
			    + rhs[index2 - M]
			    + 0.5 * (rhs[index2 - 1]
				     + rhs[index2 + 1]
				     + rhs[index2 - M - 1]
				     + rhs[index2 - M + 1]));
    }
    
    for (j = 1; j < Nhalf - 1; j++)
    {
      index1 = (j * Mhalf);
      index2 = (2 * j * M);
      rhs_coarse[index1] = (rhs[index2]
			    + rhs[index2 + 1]
			    + 0.5 * (rhs[index2 - M]
				     + rhs[index2 + M]
				     + rhs[index2 - M + 1]
				     + rhs[index2 + M + 1]));
      index1 = (j * Mhalf + Mhalf - 1);
      index2 = (2 * j * M + 2 * (Mhalf - 1));
      rhs_coarse[index1] = (rhs[index2]
			    + rhs[index2 - 1]
			    + 0.5 * (rhs[index2 - M]
				     + rhs[index2 + M]
				     + rhs[index2 - M - 1]
				     + rhs[index2 + M - 1]));
    }
    
    index1 = 0;
    index2 = 0;
    rhs_coarse[index1] = (rhs[index2]
			  + rhs[index2 + 1]
			  + rhs[index2 + M]
			  + rhs[index2 + M + 1]);
    
    index1 = (Nhalf - 1) * Mhalf;
    index2 = 2 * (Nhalf - 1) * M;
    rhs_coarse[index1] = (rhs[index2]
			  + rhs[index2 + 1]
			  + rhs[index2 - M]
			  + rhs[index2 - M + 1]);
    
    index1 = (Mhalf - 1);
    index2 = (2 * (Mhalf - 1));
    rhs_coarse[index1] = (rhs[index2]
			  + rhs[index2 - 1]
			  + rhs[index2 + M]
			  + rhs[index2 + M - 1]);
    
    index1 = ((Nhalf - 1) * Mhalf + Mhalf - 1);
    index2 = (2 * (Nhalf - 1) * M + 2 * (Mhalf - 1));
    rhs_coarse[index1] = (rhs[index2]
			  + rhs[index2 - 1]
			  + rhs[index2 - M]
			  + rhs[index2 - M - 1]);
  }
}


static void
galerkin2D(double *lhs, int M, int N,
	   double *lhs_coarse, int Mhalf, int Nhalf)
{
  int i, j;

  for (j = 0; j < Nhalf; j++)
    for (i = 0; i < Mhalf; i++)
    {
      int index1 = (j * Mhalf + i);
      int index2 = (2 * j * M + 2 * i);
      double stencil1[3][3];
      double stencil2[5][5];
      double stencil3[3][3];
      int u, v;

      for (u = 0; u < 5; u++)
	for (v = 0; v < 5; v++)
	{
	  stencil2[u][v] = 0;
	  if (u < 3 && v < 3)
	  {
	    stencil1[u][v] = 0;
	    stencil3[u][v] = 0;
	  }
	}

      if (M % 2 == 0 && N % 2 == 0)
      {
	stencil1[1][1] = 1;
	stencil1[1][2] = 1;
	stencil1[2][1] = 1;
	stencil1[2][2] = 1;
      }
      
      if (M % 2 == 1 && N % 2 == 0)
      {
	stencil1[1][1] = 1;
	stencil1[1][2] = 1;
	if (i < Mhalf - 1)
	{
	  stencil1[2][1] = 0.5 * (1 + (i == 0));
	  stencil1[2][2] = 0.5 * (1 + (i == 0));
	}
	if (i > 0)
	{
	  stencil1[0][1] = 0.5 * (1 + (i == Mhalf - 1));
	  stencil1[0][2] = 0.5 * (1 + (i == Mhalf - 1));
	}
      }

      if (M % 2 == 0 && N % 2 == 1)
      {
	stencil1[1][1] = 1;
	stencil1[2][1] = 1;
	if (j < Nhalf - 1)
	{
	  stencil1[1][2] = 0.5 * (1 + (j == 0));
	  stencil1[2][2] = 0.5 * (1 + (j == 0));
	}
	if (j > 0)
	{
	  stencil1[1][0] = 0.5 * (1 + (j == Nhalf - 1));
	  stencil1[2][0] = 0.5 * (1 + (j == Nhalf - 1));
	}
      }
      
      if (M % 2 == 1 && N % 2 == 1)
      {
	stencil1[1][1] = 1;
	stencil1[2][1] = 0.5;
	stencil1[0][1] = 0.5;
	stencil1[1][2] = 0.5;
	stencil1[1][0] = 0.5;
	stencil1[2][2] = 0.25;
	stencil1[0][2] = 0.25;
	stencil1[2][0] = 0.25;
	stencil1[0][0] = 0.25;

	if (i == 0)
	  for (u = 0; u < 3; u++)
	  {
	    stencil1[2][u] += stencil1[0][u];
	    stencil1[0][u] = 0;
	  }

	if (i == Mhalf - 1)
	  for (u = 0; u < 3; u++)
	  {
	    stencil1[0][u] += stencil1[2][u];
	    stencil1[2][u] = 0;
	  }

	if (j == 0)
	  for (u = 0; u < 3; u++)
	  {
	    stencil1[u][2] += stencil1[u][0];
	    stencil1[u][0] = 0;
	  }

	if (j == Nhalf - 1)
	  for (u = 0; u < 3; u++)
	  {
	    stencil1[u][0] += stencil1[u][2];
	    stencil1[u][2] = 0;
	  }
      }

#if 0
      mexPrintf("i=%d j=%d stencil1:\n", i, j);
      for (u = 0; u < 3; u++) {
	for (v = 0; v < 3; v++)
	  mexPrintf("%f ", stencil1[u][v]);
	mexPrintf("\n");
      }
      mexPrintf("\n");
#endif
      
      for (u = 0; u < 3; u++)
	for (v = 0; v < 3; v++)
	{
	  if (stencil1[u][v] != 0)
	  {
	    int index = 9 * (index2 + (u-1) + M*(v-1));
	    if (lhs[index] != 0.0)
	    {
	      stencil2[u+1][v+1] += stencil1[u][v] * lhs[index];
	      stencil2[u  ][v+1] += stencil1[u][v] * lhs[index + 1];
	      stencil2[u+2][v+1] += stencil1[u][v] * lhs[index + 2];
	      stencil2[u+1][v  ] += stencil1[u][v] * lhs[index + 3];
	      stencil2[u+1][v+2] += stencil1[u][v] * lhs[index + 4];
	      stencil2[u  ][v  ] += stencil1[u][v] * lhs[index + 5];
	      stencil2[u  ][v+2] += stencil1[u][v] * lhs[index + 6];
	      stencil2[u+2][v  ] += stencil1[u][v] * lhs[index + 7];
	      stencil2[u+2][v+2] += stencil1[u][v] * lhs[index + 8];
	    }
	    else
	    {
	      stencil2[u+1][v+1] += stencil1[u][v] * (-4);
	      stencil2[u  ][v+1] += stencil1[u][v];
	      stencil2[u+2][v+1] += stencil1[u][v];
	      stencil2[u+1][v  ] += stencil1[u][v];
	      stencil2[u+1][v+2] += stencil1[u][v];
	    }
	  }
	}

#if 0
      mexPrintf("i=%d j=%d stencil2:\n", i, j);
      for (u = 0; u < 5; u++) {
	for (v = 0; v < 5; v++)
	  mexPrintf("%f ", stencil2[u][v]);
	mexPrintf("\n");
      }
      mexPrintf("\n");
#endif
      
      if (M % 2 == 0 && N % 2 == 0)
      {
	for (u = 1; u < 5; u++)
	  for (v = 1; v < 5; v++)
	  {
	    double alpha1, alpha2;
	    if (i == 0 && u < 3)
	      alpha1 = 0;
	    else if (i == Mhalf - 1 && u >= 3)
	      alpha1 = 1;
	    else if (u % 2 == 1)
	      alpha1 = 0.75;
	    else
	      alpha1 = 0.25;

	    if (j == 0 && v < 3)
	      alpha2 = 0;
	    else if (j == Nhalf - 1 && v >= 3)
	      alpha2 = 1;
	    else if (v % 2 == 1)
	      alpha2 = 0.75;
	    else
	      alpha2 = 0.25;

	    stencil3[(u-1)/2  ][(v-1)/2  ] += alpha1 * alpha2 * stencil2[u][v];
	    stencil3[(u-1)/2  ][(v-1)/2+1] += alpha1 * (1-alpha2) * stencil2[u][v];
	    stencil3[(u-1)/2+1][(v-1)/2  ] += (1-alpha1) * alpha2 * stencil2[u][v];
	    stencil3[(u-1)/2+1][(v-1)/2+1] += (1-alpha1) * (1-alpha2) * stencil2[u][v];
	  }
      }

      if (M % 2 == 1 && N % 2 == 0)
      {
	for (u = 0; u < 5; u++)
	  for (v = 1; v < 5; v++)
	  {
	    double alpha1, alpha2;
	    if (u % 2 == 0)
	      alpha1 = 1;
	    else
	      alpha1 = 0.5;
	    
	    if (j == 0 && v < 3)
	      alpha2 = 0;
	    else if (j == Nhalf - 1 && v >= 3)
	      alpha2 = 1;
	    else if (v % 2 == 1)
	      alpha2 = 0.75;
	    else
	      alpha2 = 0.25;

	    stencil3[u/2  ][(v-1)/2  ] += alpha1 * alpha2 * stencil2[u][v];
	    stencil3[u/2  ][(v-1)/2+1] += alpha1 * (1-alpha2) * stencil2[u][v];
	    if (u < 4)
	    {
	      stencil3[u/2+1][(v-1)/2  ] += (1-alpha1) * alpha2 * stencil2[u][v];
	      stencil3[u/2+1][(v-1)/2+1] += (1-alpha1) * (1-alpha2) * stencil2[u][v];
	    }
	  }
      }

      if (M % 2 == 0 && N % 2 == 1)
      {
	for (u = 1; u < 5; u++)
	  for (v = 0; v < 5; v++)
	  {
	    double alpha1, alpha2;
	    if (i == 0 && u < 3)
	      alpha1 = 0;
	    else if (i == Mhalf - 1 && u >= 3)
	      alpha1 = 1;
	    else if (u % 2 == 1)
	      alpha1 = 0.75;
	    else
	      alpha1 = 0.25;

	    if (v % 2 == 0)
	      alpha2 = 1;
	    else
	      alpha2 = 0.5;
	
	    stencil3[(u-1)/2  ][v/2  ] += alpha1 * alpha2 * stencil2[u][v];
	    if (v < 4)
	      stencil3[(u-1)/2  ][v/2+1] += alpha1 * (1-alpha2) * stencil2[u][v];
	    stencil3[(u-1)/2+1][v/2  ] += (1-alpha1) * alpha2 * stencil2[u][v];
	    if (v < 4)
	      stencil3[(u-1)/2+1][v/2+1] += (1-alpha1) * (1-alpha2) * stencil2[u][v];
	  }
      }

      if (M % 2 == 1 && N % 2 == 1)
      {
	for (u = 0; u < 5; u++)
	  for (v = 0; v < 5; v++)
	  {
	    double alpha1, alpha2;
	    if (u % 2 == 0)
	      alpha1 = 1;
	    else
	      alpha1 = 0.5;
	    
	    if (v % 2 == 0)
	      alpha2 = 1;
	    else
	      alpha2 = 0.5;

	    stencil3[u/2  ][v/2  ] += alpha1 * alpha2 * stencil2[u][v];
	    if (v < 4)
	      stencil3[u/2  ][v/2+1] += alpha1 * (1-alpha2) * stencil2[u][v];
	    if (u < 4)
	      stencil3[u/2+1][v/2  ] += (1-alpha1) * alpha2 * stencil2[u][v];
	    if (u < 4 && v < 4)
	      stencil3[u/2+1][v/2+1] += (1-alpha1) * (1-alpha2) * stencil2[u][v];
	  }
      }

#if 0
      mexPrintf("i=%d j=%d stencil3:\n", i, j);
      for (u = 0; u < 3; u++) {
	for (v = 0; v < 3; v++)
	  mexPrintf("%f ", stencil3[u][v]);
	mexPrintf("\n");
      }
      mexPrintf("\n");
#endif
      
      lhs_coarse[9 * index1] = stencil3[1][1];
      lhs_coarse[9 * index1 + 1] = stencil3[0][1];
      lhs_coarse[9 * index1 + 2] = stencil3[2][1];
      lhs_coarse[9 * index1 + 3] = stencil3[1][0];
      lhs_coarse[9 * index1 + 4] = stencil3[1][2];
      lhs_coarse[9 * index1 + 5] = stencil3[0][0];
      lhs_coarse[9 * index1 + 6] = stencil3[0][2];
      lhs_coarse[9 * index1 + 7] = stencil3[2][0];
      lhs_coarse[9 * index1 + 8] = stencil3[2][2];
    }
}


/* Upsample and apply correction. Bilinear interpolation. */
static void
upsample2D(double *rhs, int M, int N,
	   double *v, int Mhalf, int Nhalf,
	   double *f_out)
{
  int i, j;
  int index1, index2;
  double ce, no, so, we, ea, nw, sw, ne, se;
  int CE, NO, SO, WE, EA, SW, NE, SE;
  
  if (M % 2 == 0 && N % 2 == 0)
  {
    for (j = 0; j < Nhalf; j++)
      for (i = 0; i < Mhalf; i++)
      {
	index1 = (j * Mhalf + i);
	index2 = (2 * j * M + 2 * i);
	ce = v[index1];
	no = v[index1 - 1];
	so = v[index1 + 1];
	we = v[index1 - Mhalf];
	ea = v[index1 + Mhalf];
	nw = v[index1 - Mhalf - 1];
	sw = v[index1 - Mhalf + 1];
	ne = v[index1 + Mhalf - 1];
	se = v[index1 + Mhalf + 1];
	CE = index2;
	SO = index2 + 1;
	EA = index2 + M;
	SE = index2 + M + 1;
	
	/* Fine pixels northwest of coarse pixel center. */
	if (i == 0)
	{
	  if (j == 0) /* NW corner. */
	    f_out[CE] += ce - 0.25 * rhs[CE];
	  else        /* North edge. */
	    f_out[CE] += 0.75 * ce + 0.25 * (we - rhs[CE]);
	}
	else
	{
	  if (j == 0) /* West edge. */
	    f_out[CE] += 0.75 * ce + 0.25 * (no - rhs[CE]);
	  else        /* Inner point. */
	    f_out[CE] += (0.5625 * ce + 0.1875 * (no + we) + 0.0625 * nw);
	}
	
	/* Fine pixels southwest of coarse pixel center. */
	if (i == Mhalf - 1)
	{
	  if (j == 0) /* SW corner. */
	    f_out[SO] += ce - 0.25 * rhs[SO];
	  else        /* South edge. */
	    f_out[SO] += 0.75 * ce  + 0.25 * (we - rhs[SO]);
	}
	else
	{
	  if (j == 0) /* West edge. */
	    f_out[SO] += 0.75 * ce + 0.25 * (so - rhs[SO]);
	  else        /* Inner point. */
	    f_out[SO] += 0.5625 * ce + 0.1875 * (so + we) + 0.0625 * sw;
	}
	
	/* Fine pixels northeast of coarse pixel center. */
	if (i == 0)
	{
	  if (j == Nhalf - 1) /* NE corner. */
	    f_out[EA] += ce - 0.25 * rhs[EA];
	  else              /* North edge. */
	    f_out[EA] += 0.75 * ce + 0.25 * (ea - rhs[EA]);
	}
	else
	{
	  if (j == Nhalf - 1) /* East edge. */
	    f_out[EA] += 0.75 * ce + 0.25 * (no - rhs[EA]);
	  else              /* Inner point. */
	    f_out[EA] += 0.5625 * ce + 0.1875 * (no + ea) + 0.0625 * ne;
	}
	
	/* Fine pixels southeast of coarse pixel center. */
	if (i == Mhalf - 1)
	{
	  if (j == Nhalf - 1) /* SE corner. */
	    f_out[SE] += ce - 0.25 * rhs[SE];
	  else              /* South edge.*/
	    f_out[SE] += 0.75 * ce + 0.25 * (ea - rhs[SE]);
	}
	else
	{
	  if (j == Nhalf - 1) /* East edge. */
	    f_out[SE] += 0.75 * ce + 0.25 * (so - rhs[SE]);
	  else              /* Inner point. */
	    f_out[SE] += 0.5625 * ce + 0.1875 * (so + ea) + 0.0625 * se;
	}
      }
  }
  
  if (M % 2 == 1 && N % 2 == 0)
  {
    for (j = 0; j < Nhalf; j++)
      for (i = 0; i < Mhalf; i++)
      {
	index1 = (j * Mhalf + i);
	index2 = (2 * j * M + 2 * i);
	ce = v[index1];
	so = v[index1 + 1];
	we = v[index1 - Mhalf];
	ea = v[index1 + Mhalf];
	sw = v[index1 - Mhalf + 1];
	se = v[index1 + Mhalf + 1];
	CE = index2;
	SO = index2 + 1;
	EA = index2 + M;
	SE = index2 + M + 1;
	NO = index2 - 1;
	NE = index2 + M - 1;
	
	/* Fine pixels west of coarse pixel center. */
	if (j == 0) /* West edge. */
	{
	  if (i == 0) /* NW corner */
	    f_out[CE] += ce - 0.125 * (rhs[SO] + rhs[CE] - rhs[EA]);
	  else if (i == Mhalf - 1) /* SW corner */
	    f_out[CE] += ce - 0.125 * (rhs[NO] + rhs[CE] - rhs[EA]);
	  else
	    f_out[CE] += ce - 0.25 * rhs[CE];
	}
	else        /* Inner point. */
	  f_out[CE] += 0.75 * ce + 0.25 * we;
	
	/* Fine pixels southsouthwest of coarse pixel center. */
	if (i < Mhalf - 1)
	{
	  if (j == 0) /* West edge. */
	    f_out[SO] += 0.5 * (ce + so) - 0.25 * rhs[SO];
	  else        /* Inner point. */
	    f_out[SO] += 0.375 * (ce + so) + 0.125 * (we + sw);
	}
	
	/* Fine pixels east of coarse pixel center. */
	if (j == Nhalf - 1) /* East edge. */
	{
	  if (i == 0) /* NE corner */
	    f_out[EA] += ce - 0.125 * (rhs[SE] + rhs[EA] - rhs[CE]);
	  else if (i == Mhalf - 1) /* SE corner */
	    f_out[EA] += ce - 0.125 * (rhs[NE] + rhs[EA] - rhs[CE]);
	  else
	    f_out[EA] += ce - 0.25 * rhs[EA];
	}
	else          /* Inner point. */
	  f_out[EA] += 0.75 * ce + 0.25 * ea;
	
	/* Fine pixels southsoutheast of coarse pixel center. */
	if (i < Mhalf - 1)
	{
	  if (j == Nhalf - 1) /* East edge. */
	    f_out[SE] += 0.5 * (ce + so) - 0.25 * rhs[SE];
	  else                /* Inner point. */
	    f_out[SE] += 0.375 * (ce + so) + 0.125 * (ea + se);
	}
      }
  }
  
  if (M % 2 == 0 && N % 2 == 1)
  {
    for (j = 0; j < Nhalf; j++)
      for (i = 0; i < Mhalf; i++)
      {
	index1 = (j * Mhalf + i);
	index2 = (2 * j * M + 2 * i);
	ce = v[index1];
	so = v[index1 + 1];
	no = v[index1 - 1];
	ea = v[index1 + Mhalf];
	se = v[index1 + Mhalf + 1];
	ne = v[index1 + Mhalf - 1];
	CE = index2;
	SO = index2 + 1;
	EA = index2 + M;
	SE = index2 + M + 1;
	WE = index2 - M;
	SW = index2 - M + 1;
	
	/* Fine pixels north of coarse pixel center. */
	if (i == 0) /* North edge. */
	{
	  if (j == 0) /* NW corner */
	    f_out[CE] += ce - 0.125 * (rhs[EA] + rhs[CE] - rhs[SO]);
	  else if (j == Nhalf - 1) /* NE corner */
	    f_out[CE] += ce - 0.125 * (rhs[WE] + rhs[CE] - rhs[SO]);
	  else
	    f_out[CE] += ce - 0.25 * rhs[CE];
	}
	else        /* Inner point. */
	  f_out[CE] += 0.75 * ce + 0.25 * no;
	
	/* Fine pixels eastnortheast of coarse pixel center. */
	if (j < Nhalf - 1)
	{
	  if (i == 0) /* North edge. */
	    f_out[EA] += 0.5 * (ce + ea) - 0.25 * rhs[EA];
	  else        /* Inner point. */
	    f_out[EA] += 0.375 * (ce + ea) + 0.125 * (no + ne);
	}
	
	/* Fine pixels south of coarse pixel center. */
	if (i == Mhalf - 1) /* South edge. */
	{
	  if (j == 0) /* SW corner */
	    f_out[SO] += ce - 0.125 * (rhs[SE] + rhs[SO] - rhs[CE]);
	  else if (j == Nhalf - 1) /* SE corner */
	    f_out[SO] += ce - 0.125 * (rhs[SW] + rhs[SO] - rhs[CE]);
	  else
	    f_out[SO] += ce - 0.25 * rhs[SO];
	}
	else          /* Inner point. */
	  f_out[SO] += 0.75 * ce + 0.25 * so;
	
	/* Fine pixels eastsoutheast of coarse pixel center. */
	if (j < Nhalf - 1)
	{
	  if (i == Mhalf - 1) /* South edge. */
	    f_out[SE] += 0.5 * (ce + ea) - 0.25 * rhs[SE];
	  else                /* Inner point. */
	    f_out[SE] += 0.375 * (ce + ea) + 0.125 * (so + se);
	}
      }
  }
  
  if (M % 2 == 1 && N % 2 == 1)
  {
    for (j = 0; j < Nhalf; j++)
      for (i = 0; i < Mhalf; i++)
      {
	index1 = (j * Mhalf + i);
	index2 = (2 * j * M + 2 * i);
	ce = v[index1];
	so = v[index1 + 1];
	ea = v[index1 + Mhalf];
	se = v[index1 + Mhalf + 1];
	CE = index2;
	SO = index2 + 1;
	EA = index2 + M;
	SE = index2 + M + 1;
	
	/* Fine pixels on coarse pixel center. */
	f_out[CE] += ce;
	
	/* Fine pixels south of coarse pixel center. */
	if (i < Mhalf - 1)
	  f_out[SO] += 0.5 * (ce + so);
	
	/* Fine pixels east of coarse pixel center. */
	if (j < Nhalf - 1) /* South edge. */
	  f_out[EA] += 0.5 * (ce + ea);
	
	/* Fine pixels southeast of coarse pixel center. */
	if (i < Mhalf - 1 && j < Nhalf - 1)
	  f_out[SE] += 0.25 * (ce + ea + so + se);
      }
  }
}


/* Recursive multigrid function.*/
static void
poisson_multigrid2D(double *f, double *A, double *d,
		    int n1, int n2, int nm,
		    double *f_out,
		    int M, int N, int *directly_solved)
{
  int i, j;
  int k;
  double *r;
  double *r_downsampled;
  double *A_downsampled;
  double *v;
  int Mhalf;
  int Nhalf;
  
  /* Solve a sufficiently small problem directly. */
  if (M < RECURSION_SIZE_LIMIT || N < RECURSION_SIZE_LIMIT)
  {
    solve_directly2D(f, A, d, f_out, M, N);
    *directly_solved = 1;
    return;
  }
  *directly_solved = 0;
  
  /* Initialize solution. */
  memcpy(f_out, f, M * N * sizeof(*f_out));
  
  /* Pre-smoothing. */
  for (k = 0; k < n1; k++)
    gauss_seidel2D(f_out, A, d, M, N);
  
  /* Compute residual. */
  r = mxCalloc(M * N, sizeof(*r));
  for (j = 0; j < N; j++)
    for (i = 0; i < M; i++)
    {
      int index = j * M + i;
      double residual = 0.0;
      if (A[9 * index] != 0.0)
      {
	residual = d[index] - A[9 * index] * f_out[index];
	if (i > 0)
	  residual -= A[9 * index + 1] * f_out[index - 1];
	
	if (i < M-1)
	  residual -= A[9 * index + 2] * f_out[index + 1];
	
	if (j > 0)
	  residual -= A[9 * index + 3] * f_out[index - M];
	
	if (j < N-1)
	  residual -= A[9 * index + 4] * f_out[index + M];
	
	if (i > 0 && j > 0)
	  residual -= A[9 * index + 5] * f_out[index - 1 - M];
	
	if (i > 0 && j < N-1)
	  residual -= A[9 * index + 6] * f_out[index - 1 + M];
	
	if (i < M-1 && j > 0)
	  residual -= A[9 * index + 7] * f_out[index + 1 - M];
	
	if (i < M-1 && j < N-1)
	  residual -= A[9 * index + 8] * f_out[index + 1 + M];
      }
      
      r[index] = residual;
    }

  logmatrix(A, 9, M*N, "A before residual", "foo");
  logmatrix(d, M, N, "d before residual", "foo");
  logmatrix(f_out, M, N, "f_out before residual", "foo");
  logmatrix(r, M, N, "residual", "foo");
#if 0
  mexPrintf("Residual:\n");
  for (i = 0; i < M; i++) {
    for (j = 0; j < M; j++)
      mexPrintf("%9.3g ", r[j*M+i]);
    mexPrintf("\n");
  }
  mexPrintf("\n");
#endif
  
  /* Downsample residual. */
  Mhalf = (M + 1) / 2;
  Nhalf = (N + 1) / 2;
  r_downsampled = mxCalloc(Mhalf * Nhalf, sizeof(*r_downsampled));
  downsample2D(r, M, N, r_downsampled, Mhalf, Nhalf);
  A_downsampled = mxCalloc(9 * Mhalf * Nhalf, sizeof(*A_downsampled));
  galerkin2D(A, M, N, A_downsampled, Mhalf, Nhalf);
  
  /* Recurse to compute a correction. */
  v = mxCalloc(Mhalf * Nhalf, sizeof(*v));
  for (k = 0; k < nm; k++)
  {
    int directly_solved;
    poisson_multigrid2D(v, A_downsampled, r_downsampled, n1, n2, nm, v,
			Mhalf, Nhalf, &directly_solved);
    if (directly_solved)
      break;
  }
  
  upsample2D(r, M, N, v, Mhalf, Nhalf, f_out);
  
  /* Post-smoothing. */
  for (k = 0; k < n2; k++)
    gauss_seidel2D(f_out, A, d, M, N);
  
  mxFree(r);
  mxFree(r_downsampled);
  mxFree(A_downsampled);
  mxFree(v);
}


/* It is assumed that f_out is initialized to zero when called. */
static void
poisson_full_multigrid2D(double *lhs, double *rhs, int number_of_iterations,
			 int M, int N, double *f_out)
{
  double *lhs_downsampled;
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
    lhs_downsampled = mxCalloc(9 * Mhalf * Nhalf, sizeof(*lhs_downsampled));
    galerkin2D(lhs, M, N, lhs_downsampled, Mhalf, Nhalf);
    logmatrix(rhs_downsampled, Mhalf, Nhalf, "rhs_downsampled", "foo");
    logmatrix(lhs_downsampled, Mhalf, Nhalf, "lhs_downsampled", "foo");
    
    f_coarse = mxCalloc(Mhalf * Nhalf, sizeof(*f_coarse));
    poisson_full_multigrid2D(lhs_downsampled, rhs_downsampled,
			     number_of_iterations,
			     Mhalf, Nhalf, f_coarse);
    logmatrix(f_coarse, Mhalf, Nhalf, "f_coarse", "foo");
    
    /* Upsample the coarse result. */
    upsample2D(rhs, M, N, f_coarse, Mhalf, Nhalf, f_out);
    logmatrix(f_out, M, N, "f_fine", "foo");

    mxFree(f_coarse);
    mxFree(lhs_downsampled);
    mxFree(rhs_downsampled);
  }
  
  /* Perform number_of_iterations standard multigrid cycles. */
  for (k = 0; k < number_of_iterations; k++)
  {
    int directly_solved;
    poisson_multigrid2D(f_out, lhs, rhs, 2, 2, 2, f_out, M, N,
			&directly_solved);
    if (directly_solved)
      break;
  }
}


static void
antigradient2D(double *g, double *mask, double mu, int number_of_iterations,
	       int M, int N, double *f_out)
{
  double *rhs;
  double *lhs;
  double sum;
  double mean;
  int i, j;
  int num_samples_in_mask;
  
  /* Compute left and right hand sides of Poisson problem with Neumann
   * boundary conditions, discretized by finite differences.
   */
  rhs = mxCalloc(M * N, sizeof(*rhs));
  lhs = mxCalloc(9 * M * N, sizeof(*lhs));
  for (j = 0; j < N; j++)
    for (i = 0; i < M; i++)
    {
      int index1 = j * M + i;
      int index2 = index1 + M * N;
      double d = 0.0;
      int N_missing = 0;
      int S_missing = 0;
      int W_missing = 0;
      int E_missing = 0;

      if (mask && mask[index1] == 0)
	continue;
      
      if (i == 0 || (mask && mask[index1 - 1] == 0))
	N_missing = 1;

      if (i == M - 1 || (mask && mask[index1 + 1] == 0))
	S_missing = 1;
      
      if (j == 0 || (mask && mask[index1 - M] == 0))
	W_missing = 1;

      if (j == N - 1 || (mask && mask[index1 + M] == 0))
	E_missing = 1;
      
      if (N_missing && !S_missing)
	d = g[index1 + 1] + g[index1];
      else if (!N_missing && S_missing)
	d = - g[index1] - g[index1 - 1];
      else if (!N_missing && !S_missing)
	d = 0.5 * (g[index1 + 1] - g[index1 - 1]);
      
      if (W_missing && !E_missing)
	d += g[index2 + M] + g[index2];
      else if (!W_missing && E_missing)
	d += - g[index2] - g[index2 - M];
      else if (!W_missing && !E_missing)
	d += 0.5 * (g[index2 + M] - g[index2 - M]);
      
      rhs[index1] = d;

      lhs[9 * index1] = -2 * ((!N_missing || !S_missing) + (!W_missing || !E_missing));
      if (!N_missing)
	lhs[9 * index1 + 1] = 1 + (S_missing);
      if (!S_missing)
	lhs[9 * index1 + 2] = 1 + (N_missing);
      if (!W_missing)
	lhs[9 * index1 + 3] = 1 + (E_missing);
      if (!E_missing)
	lhs[9 * index1 + 4] = 1 + (W_missing);
    }
  
  logmatrix(lhs, 9, M*N, "A", "foo");
  logmatrix(rhs, M*N, 1, "b", "foo");
  /* Solve the equation system with the full multigrid algorithm.
   * Use W cycles and 2 presmoothing and 2 postsmoothing
   * Gauss-Seidel iterations.
   */
  poisson_full_multigrid2D(lhs, rhs, number_of_iterations, M, N, f_out);
  
  /* Fix the mean value. */
  sum = 0.0;
  num_samples_in_mask = 0;
  for (i = 0; i < M * N; i++)
    if (!mask || mask[i])
    {
      sum += f_out[i];
      num_samples_in_mask++;
    }
  
  mean = sum / num_samples_in_mask;
  for (i = 0; i < M * N; i++)
  {
    f_out[i] -= mean;
    f_out[i] += mu;
  }

  mxFree(rhs);
  mxFree(lhs);
}


/*************** 3D ****************/

static void
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


/* Gauss-Seidel smoothing iteration. Red-black ordering. */
static void
gauss_seidel3D(double *f, double *d, int M, int N, int P)
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

static void
downsample3D(double *rhs, int M, int N, int P,
             double *rhs_coarse, int Mhalf, int Nhalf, int Phalf)
{
  int i, j, p;
  int index1;
  int index2;
  int MN = M * N;
  
  if (M % 2 == 0 && N % 2 == 0 && P % 2 == 0)
  {
    for (p = 0; p < Phalf; p++)
    {
      for (j = 0; j < Nhalf; j++)
      {
        for (i = 0; i < Mhalf; i++)
        {
          index1 = (p * Nhalf + j) * Mhalf + i;
          index2 = (2 * p * N + 2 * j) * M + 2 * i;
          rhs_coarse[index1] = (0.5 * (rhs[index2]
                                       + rhs[index2 + 1]
                                       + rhs[index2 + M]
                                       + rhs[index2 + M + 1]
                                       + rhs[index2 + MN]
                                       + rhs[index2 + MN + 1]
                                       + rhs[index2 + MN + M]
                                       + rhs[index2 + MN + M + 1]));
        }
      }
    }
  }
  
  if (M % 2 == 1 && N % 2 == 0 && P % 2 == 0)
  {
    for (p = 0; p < Phalf; p++)
    {
      for (j = 0; j < Nhalf; j++)
      {
        index1 = (p * Nhalf + j) * Mhalf + 0;
        index2 = (2 * p * N + 2 * j) * M + 2 * 0;
        rhs_coarse[index1] = (0.5 * (rhs[index2]
                                     + rhs[index2 + 1]
                                     + rhs[index2 + M]
                                     + rhs[index2 + M + 1]
                                     + rhs[index2 + MN]
                                     + rhs[index2 + MN + 1]
                                     + rhs[index2 + MN + M]
                                     + rhs[index2 + MN + M + 1]));
        
        for (i = 1; i < Mhalf - 1; i++)
        {
          index1 = (p * Nhalf + j) * Mhalf + i;
          index2 = (2 * p * N + 2 * j) * M + 2 * i;
          rhs_coarse[index1] = (0.5 * (rhs[index2]
                                       + rhs[index2 + M]
                                       + rhs[index2 + MN]
                                       + rhs[index2 + MN + M])
                                + 0.25 * (rhs[index2 + 1]
                                          + rhs[index2 - 1]
                                          + rhs[index2 + M + 1]
                                          + rhs[index2 + M - 1]
                                          + rhs[index2 + MN + 1]
                                          + rhs[index2 + MN - 1]
                                          + rhs[index2 + MN + M + 1]
                                          + rhs[index2 + MN + M - 1]));
        }
        
        index1 = (p * Nhalf + j) * Mhalf + (Mhalf - 1);
        index2 = (2 * p * N + 2 * j) * M + 2 * (Mhalf - 1);
        rhs_coarse[index1] = (0.5 * (rhs[index2]
                                     + rhs[index2 - 1]
                                     + rhs[index2 + M]
                                     + rhs[index2 + M - 1]
                                     + rhs[index2 + MN]
                                     + rhs[index2 + MN - 1]
                                     + rhs[index2 + MN + M]
                                     + rhs[index2 + MN + M - 1]));
      }
    }
  }
  
  if (M % 2 == 0 && N % 2 == 1 && P % 2 == 0)
  {
    for (p = 0; p < Phalf; p++)
    {
      for (i = 0; i < Mhalf; i++)
      {
        index1 = (p * Nhalf + 0) * Mhalf + i;
        index2 = (2 * p * N + 2 * 0) * M + 2 * i;
        rhs_coarse[index1] = (0.5 * (rhs[index2]
                                     + rhs[index2 + 1]
                                     + rhs[index2 + M]
                                     + rhs[index2 + M + 1]
                                     + rhs[index2 + MN]
                                     + rhs[index2 + MN + 1]
                                     + rhs[index2 + MN + M]
                                     + rhs[index2 + MN + M + 1]));
      }
      
      for (j = 1; j < Nhalf - 1; j++)
      {
        for (i = 0; i < Mhalf; i++)
        {
          index1 = (p * Nhalf + j) * Mhalf + i;
          index2 = (2 * p * N + 2 * j) * M + 2 * i;
          rhs_coarse[index1] = (0.5 * (rhs[index2]
                                       + rhs[index2 + 1]
                                       + rhs[index2 + MN]
                                       + rhs[index2 + MN + 1])
                                + 0.25 * (rhs[index2 + M]
                                          + rhs[index2 + M + 1]
                                          + rhs[index2 - M]
                                          + rhs[index2 - M + 1]
                                          + rhs[index2 + MN + M]
                                          + rhs[index2 + MN + M + 1]
                                          + rhs[index2 + MN - M]
                                          + rhs[index2 + MN - M + 1]));
        }
      }
      
      for (i = 0; i < Mhalf; i++)
      {
        index1 = (p * Nhalf + (Nhalf - 1)) * Mhalf + i;
        index2 = (2 * p * N + 2 * (Nhalf - 1)) * M + 2 * i;
        rhs_coarse[index1] = (0.5 * (rhs[index2]
                                     + rhs[index2 + 1]
                                     + rhs[index2 - M]
                                     + rhs[index2 - M + 1]
                                     + rhs[index2 + MN]
                                     + rhs[index2 + MN + 1]
                                     + rhs[index2 + MN - M]
                                     + rhs[index2 + MN - M + 1]));
      }
    }
  }
  
  if (M % 2 == 1 && N % 2 == 1 && P % 2 == 0)
  {
    for (p = 0; p < Phalf; p++)
    {
      index1 = (p * Nhalf + 0) * Mhalf + 0;
      index2 = (2 * p * N + 2 * 0) * M + 2 * 0;
      rhs_coarse[index1] = (0.5 * (rhs[index2]
                                   + rhs[index2 + 1]
                                   + rhs[index2 + M]
                                   + rhs[index2 + M + 1]
                                   + rhs[index2 + MN]
                                   + rhs[index2 + MN + 1]
                                   + rhs[index2 + MN + M]
                                   + rhs[index2 + MN + M + 1]));
      
      for (i = 1; i < Mhalf - 1; i++)
      {
        index1 = (p * Nhalf + 0) * Mhalf + i;
        index2 = (2 * p * N + 2 * 0) * M + 2 * i;
        rhs_coarse[index1] = (0.5 * (rhs[index2]
                                     + rhs[index2 + M]
                                     + rhs[index2 + MN]
                                     + rhs[index2 + MN + M])
                              + 0.25 * (rhs[index2 + 1]
                                        + rhs[index2 - 1]
                                        + rhs[index2 + M + 1]
                                        + rhs[index2 + M - 1]
                                        + rhs[index2 + MN + 1]
                                        + rhs[index2 + MN - 1]
                                        + rhs[index2 + MN + M + 1]
                                        + rhs[index2 + MN + M - 1]));
      }
      
      index1 = (p * Nhalf + 0) * Mhalf + (Mhalf - 1);
      index2 = (2 * p * N + 2 * 0) * M + 2 * (Mhalf - 1);
      rhs_coarse[index1] = (0.5 * (rhs[index2]
                                   + rhs[index2 - 1]
                                   + rhs[index2 + M]
                                   + rhs[index2 + M - 1]
                                   + rhs[index2 + MN]
                                   + rhs[index2 + MN - 1]
                                   + rhs[index2 + MN + M]
                                   + rhs[index2 + MN + M - 1]));
      
      for (j = 1; j < Nhalf - 1; j++)
      {
        index1 = (p * Nhalf + j) * Mhalf + 0;
        index2 = (2 * p * N + 2 * j) * M + 2 * 0;
        rhs_coarse[index1] = (0.5 * (rhs[index2]
                                     + rhs[index2 + 1]
                                     + rhs[index2 + MN]
                                     + rhs[index2 + MN + 1])
                              + 0.25 * (rhs[index2 + M]
                                        + rhs[index2 + M + 1]
                                        + rhs[index2 - M]
                                        + rhs[index2 - M + 1]
                                        + rhs[index2 + MN + M]
                                        + rhs[index2 + MN + M + 1]
                                        + rhs[index2 + MN - M]
                                        + rhs[index2 + MN - M + 1]));
        
        for (i = 1; i < Mhalf - 1; i++)
        {
          index1 = (p * Nhalf + j) * Mhalf + i;
          index2 = (2 * p * N + 2 * j) * M + 2 * i;
          rhs_coarse[index1] = (0.5 * (rhs[index2]
                                       + rhs[index2 + MN])
                                + 0.25 * (rhs[index2 + 1]
                                          + rhs[index2 - 1]
                                          + rhs[index2 + M]
                                          + rhs[index2 - M]
                                          + rhs[index2 + MN + 1]
                                          + rhs[index2 + MN - 1]
                                          + rhs[index2 + MN + M]
                                          + rhs[index2 + MN - M])
                                + 0.125 * (rhs[index2 + M + 1]
                                           + rhs[index2 + M - 1]
                                           + rhs[index2 - M + 1]
                                           + rhs[index2 - M - 1]
                                           + rhs[index2 + MN + M + 1]
                                           + rhs[index2 + MN + M - 1]
                                           + rhs[index2 + MN - M + 1]
                                           + rhs[index2 + MN - M - 1]));
        }
        
        index1 = (p * Nhalf + j) * Mhalf + (Mhalf - 1);
        index2 = (2 * p * N + 2 * j) * M + 2 * (Mhalf - 1);
        rhs_coarse[index1] = (0.5 * (rhs[index2]
                                     + rhs[index2 - 1]
                                     + rhs[index2 + MN]
                                     + rhs[index2 + MN - 1])
                              + 0.25 * (rhs[index2 + M]
                                        + rhs[index2 + M - 1]
                                        + rhs[index2 - M]
                                        + rhs[index2 - M - 1]
                                        + rhs[index2 + MN + M]
                                        + rhs[index2 + MN + M - 1]
                                        + rhs[index2 + MN - M]
                                        + rhs[index2 + MN - M - 1]));
      }
      
      index1 = (p * Nhalf + (Nhalf - 1)) * Mhalf + 0;
      index2 = (2 * p * N + 2 * (Nhalf - 1)) * M + 2 * 0;
      rhs_coarse[index1] = (0.5 * (rhs[index2]
                                   + rhs[index2 + 1]
                                   + rhs[index2 - M]
                                   + rhs[index2 - M + 1]
                                   + rhs[index2 + MN]
                                   + rhs[index2 + MN + 1]
                                   + rhs[index2 + MN - M]
                                   + rhs[index2 + MN - M + 1]));
      
      for (i = 1; i < Mhalf - 1; i++)
      {
        index1 = (p * Nhalf + (Nhalf - 1)) * Mhalf + i;
        index2 = (2 * p * N + 2 * (Nhalf - 1)) * M + 2 * i;
        rhs_coarse[index1] = (0.5 * (rhs[index2]
                                     + rhs[index2 - M]
                                     + rhs[index2 + MN]
                                     + rhs[index2 + MN - M])
                              + 0.25 * (rhs[index2 + 1]
                                        + rhs[index2 - 1]
                                        + rhs[index2 - M + 1]
                                        + rhs[index2 - M - 1]
                                        + rhs[index2 + MN + 1]
                                        + rhs[index2 + MN - 1]
                                        + rhs[index2 + MN - M + 1]
                                        + rhs[index2 + MN - M - 1]));
      }
      
      index1 = (p * Nhalf + (Nhalf - 1)) * Mhalf + (Mhalf - 1);
      index2 = (2 * p * N + 2 * (Nhalf - 1)) * M + 2 * (Mhalf - 1);
      rhs_coarse[index1] = (0.5 * (rhs[index2]
                                   + rhs[index2 - 1]
                                   + rhs[index2 - M]
                                   + rhs[index2 - M - 1]
                                   + rhs[index2 + MN]
                                   + rhs[index2 + MN - 1]
                                   + rhs[index2 + MN - M]
                                   + rhs[index2 + MN - M - 1]));
    }
  }
  
  if (M % 2 == 0 && N % 2 == 0 && P % 2 == 1)
  {
    for (j = 0; j < Nhalf; j++)
    {
      for (i = 0; i < Mhalf; i++)
      {
        index1 = (0 * Nhalf + j) * Mhalf + i;
        index2 = (2 * 0 * N + 2 * j) * M + 2 * i;
        rhs_coarse[index1] = (0.5 * (rhs[index2]
                                     + rhs[index2 + 1]
                                     + rhs[index2 + M]
                                     + rhs[index2 + M + 1]
                                     + rhs[index2 + MN]
                                     + rhs[index2 + MN + 1]
                                     + rhs[index2 + MN + M]
                                     + rhs[index2 + MN + M + 1]));
      }
    }
    
    for (p = 1; p < Phalf - 1; p++)
    {
      for (j = 0; j < Nhalf; j++)
      {
        for (i = 0; i < Mhalf; i++)
        {
          index1 = (p * Nhalf + j) * Mhalf + i;
          index2 = (2 * p * N + 2 * j) * M + 2 * i;
          rhs_coarse[index1] = (0.5 * (rhs[index2]
                                       + rhs[index2 + 1]
                                       + rhs[index2 + M]
                                       + rhs[index2 + M + 1])
                                + 0.25 * (rhs[index2 + MN]
                                          + rhs[index2 + MN + 1]
                                          + rhs[index2 + MN + M]
                                          + rhs[index2 + MN + M + 1]
                                          + rhs[index2 - MN]
                                          + rhs[index2 - MN + 1]
                                          + rhs[index2 - MN + M]
                                          + rhs[index2 - MN + M + 1]));
        }
      }
    }
    
    for (j = 0; j < Nhalf; j++)
    {
      for (i = 0; i < Mhalf; i++)
      {
        index1 = ((Phalf - 1) * Nhalf + j) * Mhalf + i;
        index2 = (2 * (Phalf - 1) * N + 2 * j) * M + 2 * i;
        rhs_coarse[index1] = (0.5 * (rhs[index2]
                                     + rhs[index2 + 1]
                                     + rhs[index2 + M]
                                     + rhs[index2 + M + 1]
                                     + rhs[index2 - MN]
                                     + rhs[index2 - MN + 1]
                                     + rhs[index2 - MN + M]
                                     + rhs[index2 - MN + M + 1]));
      }
    }
  }
  
  if (M % 2 == 1 && N % 2 == 0 && P % 2 == 1)
  {
    for (j = 0; j < Nhalf; j++)
    {
      index1 = (0 * Nhalf + j) * Mhalf + 0;
      index2 = (2 * 0 * N + 2 * j) * M + 2 * 0;
      rhs_coarse[index1] = (0.5 * (rhs[index2]
                                   + rhs[index2 + 1]
                                   + rhs[index2 + M]
                                   + rhs[index2 + M + 1]
                                   + rhs[index2 + MN]
                                   + rhs[index2 + MN + 1]
                                   + rhs[index2 + MN + M]
                                   + rhs[index2 + MN + M + 1]));
      
      for (i = 1; i < Mhalf - 1; i++)
      {
        index1 = (0 * Nhalf + j) * Mhalf + i;
        index2 = (2 * 0 * N + 2 * j) * M + 2 * i;
        rhs_coarse[index1] = (0.5 * (rhs[index2]
                                     + rhs[index2 + M]
                                     + rhs[index2 + MN]
                                     + rhs[index2 + MN + M])
                              + 0.25 * (rhs[index2 + 1]
                                        + rhs[index2 - 1]
                                        + rhs[index2 + M + 1]
                                        + rhs[index2 + M - 1]
                                        + rhs[index2 + MN + 1]
                                        + rhs[index2 + MN - 1]
                                        + rhs[index2 + MN + M + 1]
                                        + rhs[index2 + MN + M - 1]));
      }
      
      index1 = (0 * Nhalf + j) * Mhalf + (Mhalf - 1);
      index2 = (2 * 0 * N + 2 * j) * M + 2 * (Mhalf - 1);
      rhs_coarse[index1] = (0.5 * (rhs[index2]
                                   + rhs[index2 - 1]
                                   + rhs[index2 + M]
                                   + rhs[index2 + M - 1]
                                   + rhs[index2 + MN]
                                   + rhs[index2 + MN - 1]
                                   + rhs[index2 + MN + M]
                                   + rhs[index2 + MN + M - 1]));
    }
    
    for (p = 1; p < Phalf - 1; p++)
    {
      for (j = 0; j < Nhalf; j++)
      {
        index1 = (p * Nhalf + j) * Mhalf + 0;
        index2 = (2 * p * N + 2 * j) * M + 2 * 0;
        rhs_coarse[index1] = (0.5 * (rhs[index2]
                                     + rhs[index2 + 1]
                                     + rhs[index2 + M]
                                     + rhs[index2 + M + 1])
                              + 0.25 * (rhs[index2 + MN]
                                        + rhs[index2 + MN + 1]
                                        + rhs[index2 + MN + M]
                                        + rhs[index2 + MN + M + 1]
                                        + rhs[index2 - MN]
                                        + rhs[index2 - MN + 1]
                                        + rhs[index2 - MN + M]
                                        + rhs[index2 - MN + M + 1]));
        
        for (i = 1; i < Mhalf - 1; i++)
        {
          index1 = (p * Nhalf + j) * Mhalf + i;
          index2 = (2 * p * N + 2 * j) * M + 2 * i;
          rhs_coarse[index1] = (0.5 * (rhs[index2]
                                       + rhs[index2 + M])
                                + 0.25 * (rhs[index2 + 1]
                                          + rhs[index2 - 1]
                                          + rhs[index2 + M + 1]
                                          + rhs[index2 + M - 1]
                                          + rhs[index2 + MN]
                                          + rhs[index2 + MN + M]
                                          + rhs[index2 - MN]
                                          + rhs[index2 - MN + M])
                                + 0.125 * (rhs[index2 + MN + 1]
                                           + rhs[index2 + MN - 1]
                                           + rhs[index2 + MN + M + 1]
                                           + rhs[index2 + MN + M - 1]
                                           + rhs[index2 - MN + 1]
                                           + rhs[index2 - MN - 1]
                                           + rhs[index2 - MN + M + 1]
                                           + rhs[index2 - MN + M - 1]));
        }
        
        index1 = (p * Nhalf + j) * Mhalf + (Mhalf - 1);
        index2 = (2 * p * N + 2 * j) * M + 2 * (Mhalf - 1);
        rhs_coarse[index1] = (0.5 * (rhs[index2]
                                     + rhs[index2 - 1]
                                     + rhs[index2 + M]
                                     + rhs[index2 + M - 1])
                              + 0.25 * (rhs[index2 + MN]
                                        + rhs[index2 + MN - 1]
                                        + rhs[index2 + MN + M]
                                        + rhs[index2 + MN + M - 1]
                                        + rhs[index2 - MN]
                                        + rhs[index2 - MN - 1]
                                        + rhs[index2 - MN + M]
                                        + rhs[index2 - MN + M - 1]));
      }
    }
    
    for (j = 0; j < Nhalf; j++)
    {
      index1 = ((Phalf - 1) * Nhalf + j) * Mhalf + 0;
      index2 = (2 * (Phalf - 1) * N + 2 * j) * M + 2 * 0;
      rhs_coarse[index1] = (0.5 * (rhs[index2]
                                   + rhs[index2 + 1]
                                   + rhs[index2 + M]
                                   + rhs[index2 + M + 1]
                                   + rhs[index2 - MN]
                                   + rhs[index2 - MN + 1]
                                   + rhs[index2 - MN + M]
                                   + rhs[index2 - MN + M + 1]));
      
      for (i = 1; i < Mhalf - 1; i++)
      {
        index1 = ((Phalf - 1) * Nhalf + j) * Mhalf + i;
        index2 = (2 * (Phalf - 1) * N + 2 * j) * M + 2 * i;
        rhs_coarse[index1] = (0.5 * (rhs[index2]
                                     + rhs[index2 + M]
                                     + rhs[index2 - MN]
                                     + rhs[index2 - MN + M])
                              + 0.25 * (rhs[index2 + 1]
                                        + rhs[index2 - 1]
                                        + rhs[index2 + M + 1]
                                        + rhs[index2 + M - 1]
                                        + rhs[index2 - MN + 1]
                                        + rhs[index2 - MN - 1]
                                        + rhs[index2 - MN + M + 1]
                                        + rhs[index2 - MN + M - 1]));
      }
      
      index1 = ((Phalf - 1) * Nhalf + j) * Mhalf + (Mhalf - 1);
      index2 = (2 * (Phalf - 1) * N + 2 * j) * M + 2 * (Mhalf - 1);
      rhs_coarse[index1] = (0.5 * (rhs[index2]
                                   + rhs[index2 - 1]
                                   + rhs[index2 + M]
                                   + rhs[index2 + M - 1]
                                   + rhs[index2 - MN]
                                   + rhs[index2 - MN - 1]
                                   + rhs[index2 - MN + M]
                                   + rhs[index2 - MN + M - 1]));
    }
  }
  
  if (M % 2 == 0 && N % 2 == 1 && P % 2 == 1)
  {
    for (i = 0; i < Mhalf; i++)
    {
      index1 = (0 * Nhalf + 0) * Mhalf + i;
      index2 = (2 * 0 * N + 2 * 0) * M + 2 * i;
      rhs_coarse[index1] = (0.5 * (rhs[index2]
                                   + rhs[index2 + 1]
                                   + rhs[index2 + M]
                                   + rhs[index2 + M + 1]
                                   + rhs[index2 + MN]
                                   + rhs[index2 + MN + 1]
                                   + rhs[index2 + MN + M]
                                   + rhs[index2 + MN + M + 1]));
    }
    
    for (j = 1; j < Nhalf - 1; j++)
    {
      for (i = 0; i < Mhalf; i++)
      {
        index1 = (0 * Nhalf + j) * Mhalf + i;
        index2 = (2 * 0 * N + 2 * j) * M + 2 * i;
        rhs_coarse[index1] = (0.5 * (rhs[index2]
                                     + rhs[index2 + 1]
                                     + rhs[index2 + MN]
                                     + rhs[index2 + MN + 1])
                              + 0.25 * (rhs[index2 + M]
                                        + rhs[index2 + M + 1]
                                        + rhs[index2 - M]
                                        + rhs[index2 - M + 1]
                                        + rhs[index2 + MN + M]
                                        + rhs[index2 + MN + M + 1]
                                        + rhs[index2 + MN - M]
                                        + rhs[index2 + MN - M + 1]));
      }
    }
    
    for (i = 0; i < Mhalf; i++)
    {
      index1 = (0 * Nhalf + (Nhalf - 1)) * Mhalf + i;
      index2 = (2 * 0 * N + 2 * (Nhalf - 1)) * M + 2 * i;
      rhs_coarse[index1] = (0.5 * (rhs[index2]
                                   + rhs[index2 + 1]
                                   + rhs[index2 - M]
                                   + rhs[index2 - M + 1]
                                   + rhs[index2 + MN]
                                   + rhs[index2 + MN + 1]
                                   + rhs[index2 + MN - M]
                                   + rhs[index2 + MN - M + 1]));
    }
    
    for (p = 1; p < Phalf - 1; p++)
    {
      for (i = 0; i < Mhalf; i++)
      {
        index1 = (p * Nhalf + 0) * Mhalf + i;
        index2 = (2 * p * N + 2 * 0) * M + 2 * i;
        rhs_coarse[index1] = (0.5 * (rhs[index2]
                                     + rhs[index2 + 1]
                                     + rhs[index2 + M]
                                     + rhs[index2 + M + 1])
                              + 0.25 * (rhs[index2 + MN]
                                        + rhs[index2 + MN + 1]
                                        + rhs[index2 + MN + M]
                                        + rhs[index2 + MN + M + 1]
                                        + rhs[index2 - MN]
                                        + rhs[index2 - MN + 1]
                                        + rhs[index2 - MN + M]
                                        + rhs[index2 - MN + M + 1]));
      }
      
      for (j = 1; j < Nhalf - 1; j++)
      {
        for (i = 0; i < Mhalf; i++)
        {
          index1 = (p * Nhalf + j) * Mhalf + i;
          index2 = (2 * p * N + 2 * j) * M + 2 * i;
          rhs_coarse[index1] = (0.5 * (rhs[index2]
                                       + rhs[index2 + 1])
                                + 0.25 * (rhs[index2 + M]
                                          + rhs[index2 + M + 1]
                                          + rhs[index2 - M]
                                          + rhs[index2 - M + 1]
                                          + rhs[index2 + MN]
                                          + rhs[index2 + MN + 1]
                                          + rhs[index2 - MN]
                                          + rhs[index2 - MN + 1])
                                + 0.125 * (rhs[index2 + MN + M]
                                           + rhs[index2 + MN + M + 1]
                                           + rhs[index2 + MN - M]
                                           + rhs[index2 + MN - M + 1]
                                           + rhs[index2 - MN + M]
                                           + rhs[index2 - MN + M + 1]
                                           + rhs[index2 - MN - M]
                                           + rhs[index2 - MN - M + 1]));
        }
      }
      
      for (i = 0; i < Mhalf; i++)
      {
        index1 = (p * Nhalf + (Nhalf - 1)) * Mhalf + i;
        index2 = (2 * p * N + 2 * (Nhalf - 1)) * M + 2 * i;
        rhs_coarse[index1] = (0.5 * (rhs[index2]
                                     + rhs[index2 + 1]
                                     + rhs[index2 - M]
                                     + rhs[index2 - M + 1])
                              + 0.25 * (rhs[index2 + MN]
                                        + rhs[index2 + MN + 1]
                                        + rhs[index2 + MN - M]
                                        + rhs[index2 + MN - M + 1]
                                        + rhs[index2 - MN]
                                        + rhs[index2 - MN + 1]
                                        + rhs[index2 - MN - M]
                                        + rhs[index2 - MN - M + 1]));
      }
    }
    
    for (i = 0; i < Mhalf; i++)
    {
      index1 = ((Phalf - 1) * Nhalf + 0) * Mhalf + i;
      index2 = (2 * (Phalf - 1) * N + 2 * 0) * M + 2 * i;
      rhs_coarse[index1] = (0.5 * (rhs[index2]
                                   + rhs[index2 + 1]
                                   + rhs[index2 + M]
                                   + rhs[index2 + M + 1]
                                   + rhs[index2 - MN]
                                   + rhs[index2 - MN + 1]
                                   + rhs[index2 - MN + M]
                                   + rhs[index2 - MN + M + 1]));
    }
    
    for (j = 1; j < Nhalf - 1; j++)
    {
      for (i = 0; i < Mhalf; i++)
      {
        index1 = ((Phalf - 1) * Nhalf + j) * Mhalf + i;
        index2 = (2 * (Phalf - 1) * N + 2 * j) * M + 2 * i;
        rhs_coarse[index1] = (0.5 * (rhs[index2]
                                     + rhs[index2 + 1]
                                     + rhs[index2 - MN]
                                     + rhs[index2 - MN + 1])
                              + 0.25 * (rhs[index2 + M]
                                        + rhs[index2 + M + 1]
                                        + rhs[index2 - M]
                                        + rhs[index2 - M + 1]
                                        + rhs[index2 - MN + M]
                                        + rhs[index2 - MN + M + 1]
                                        + rhs[index2 - MN - M]
                                        + rhs[index2 - MN - M + 1]));
      }
    }
    
    for (i = 0; i < Mhalf; i++)
    {
      index1 = ((Phalf - 1) * Nhalf + (Nhalf - 1)) * Mhalf + i;
      index2 = (2 * (Phalf - 1) * N + 2 * (Nhalf - 1)) * M + 2 * i;
      rhs_coarse[index1] = (0.5 * (rhs[index2]
                                   + rhs[index2 + 1]
                                   + rhs[index2 - M]
                                   + rhs[index2 - M + 1]
                                   + rhs[index2 - MN]
                                   + rhs[index2 - MN + 1]
                                   + rhs[index2 - MN - M]
                                   + rhs[index2 - MN - M + 1]));
    }
  }
  
  if (M % 2 == 1 && N % 2 == 1 && P % 2 == 1)
  {
    index1 = (0 * Nhalf + 0) * Mhalf + 0;
    index2 = (2 * 0 * N + 2 * 0) * M + 2 * 0;
    rhs_coarse[index1] = (0.5 * (rhs[index2]
                                 + rhs[index2 + 1]
                                 + rhs[index2 + M]
                                 + rhs[index2 + M + 1]
                                 + rhs[index2 + MN]
                                 + rhs[index2 + MN + 1]
                                 + rhs[index2 + MN + M]
                                 + rhs[index2 + MN + M + 1]));
    
    for (i = 1; i < Mhalf - 1; i++)
    {
      index1 = (0 * Nhalf + 0) * Mhalf + i;
      index2 = (2 * 0 * N + 2 * 0) * M + 2 * i;
      rhs_coarse[index1] = (0.5 * (rhs[index2]
                                   + rhs[index2 + M]
                                   + rhs[index2 + MN]
                                   + rhs[index2 + MN + M])
                            + 0.25 * (rhs[index2 + 1]
                                      + rhs[index2 - 1]
                                      + rhs[index2 + M + 1]
                                      + rhs[index2 + M - 1]
                                      + rhs[index2 + MN + 1]
                                      + rhs[index2 + MN - 1]
                                      + rhs[index2 + MN + M + 1]
                                      + rhs[index2 + MN + M - 1]));
    }
    
    index1 = (0 * Nhalf + 0) * Mhalf + (Mhalf - 1);
    index2 = (2 * 0 * N + 2 * 0) * M + 2 * (Mhalf - 1);
    rhs_coarse[index1] = (0.5 * (rhs[index2]
                                 + rhs[index2 - 1]
                                 + rhs[index2 + M]
                                 + rhs[index2 + M - 1]
                                 + rhs[index2 + MN]
                                 + rhs[index2 + MN - 1]
                                 + rhs[index2 + MN + M]
                                 + rhs[index2 + MN + M - 1]));
    
    for (j = 1; j < Nhalf - 1; j++)
    {
      index1 = (0 * Nhalf + j) * Mhalf + 0;
      index2 = (2 * 0 * N + 2 * j) * M + 2 * 0;
      rhs_coarse[index1] = (0.5 * (rhs[index2]
                                   + rhs[index2 + 1]
                                   + rhs[index2 + MN]
                                   + rhs[index2 + MN + 1])
                            + 0.25 * (rhs[index2 + M]
                                      + rhs[index2 + M + 1]
                                      + rhs[index2 - M]
                                      + rhs[index2 - M + 1]
                                      + rhs[index2 + MN + M]
                                      + rhs[index2 + MN + M + 1]
                                      + rhs[index2 + MN - M]
                                      + rhs[index2 + MN - M + 1]));
      
      for (i = 1; i < Mhalf - 1; i++)
      {
        index1 = (0 * Nhalf + j) * Mhalf + i;
        index2 = (2 * 0 * N + 2 * j) * M + 2 * i;
        rhs_coarse[index1] = (0.5 * (rhs[index2]
                                     + rhs[index2 + MN])
                              + 0.25 * (rhs[index2 + 1]
                                        + rhs[index2 - 1]
                                        + rhs[index2 + M]
                                        + rhs[index2 - M]
                                        + rhs[index2 + MN + 1]
                                        + rhs[index2 + MN - 1]
                                        + rhs[index2 + MN + M]
                                        + rhs[index2 + MN - M])
                              + 0.125 * (rhs[index2 + M + 1]
                                         + rhs[index2 + M - 1]
                                         + rhs[index2 - M + 1]
                                         + rhs[index2 - M - 1]
                                         + rhs[index2 + MN + M + 1]
                                         + rhs[index2 + MN + M - 1]
                                         + rhs[index2 + MN - M + 1]
                                         + rhs[index2 + MN - M - 1]));
      }
      
      index1 = (0 * Nhalf + j) * Mhalf + (Mhalf - 1);
      index2 = (2 * 0 * N + 2 * j) * M + 2 * (Mhalf - 1);
      rhs_coarse[index1] = (0.5 * (rhs[index2]
                                   + rhs[index2 - 1]
                                   + rhs[index2 + MN]
                                   + rhs[index2 + MN - 1])
                            + 0.25 * (rhs[index2 + M]
                                      + rhs[index2 + M - 1]
                                      + rhs[index2 - M]
                                      + rhs[index2 - M - 1]
                                      + rhs[index2 + MN + M]
                                      + rhs[index2 + MN + M - 1]
                                      + rhs[index2 + MN - M]
                                      + rhs[index2 + MN - M - 1]));
    }
    
    index1 = (0 * Nhalf + (Nhalf - 1)) * Mhalf + 0;
    index2 = (2 * 0 * N + 2 * (Nhalf - 1)) * M + 2 * 0;
    rhs_coarse[index1] = (0.5 * (rhs[index2]
                                 + rhs[index2 + 1]
                                 + rhs[index2 - M]
                                 + rhs[index2 - M + 1]
                                 + rhs[index2 + MN]
                                 + rhs[index2 + MN + 1]
                                 + rhs[index2 + MN - M]
                                 + rhs[index2 + MN - M + 1]));
    
    for (i = 1; i < Mhalf - 1; i++)
    {
      index1 = (0 * Nhalf + (Nhalf - 1)) * Mhalf + i;
      index2 = (2 * 0 * N + 2 * (Nhalf - 1)) * M + 2 * i;
      rhs_coarse[index1] = (0.5 * (rhs[index2]
                                   + rhs[index2 - M]
                                   + rhs[index2 + MN]
                                   + rhs[index2 + MN - M])
                            + 0.25 * (rhs[index2 + 1]
                                      + rhs[index2 - 1]
                                      + rhs[index2 - M + 1]
                                      + rhs[index2 - M - 1]
                                      + rhs[index2 + MN + 1]
                                      + rhs[index2 + MN - 1]
                                      + rhs[index2 + MN - M + 1]
                                      + rhs[index2 + MN - M - 1]));
    }
    
    index1 = (0 * Nhalf + (Nhalf - 1)) * Mhalf + (Mhalf - 1);
    index2 = (2 * 0 * N + 2 * (Nhalf - 1)) * M + 2 * (Mhalf - 1);
    rhs_coarse[index1] = (0.5 * (rhs[index2]
                                 + rhs[index2 - 1]
                                 + rhs[index2 - M]
                                 + rhs[index2 - M - 1]
                                 + rhs[index2 + MN]
                                 + rhs[index2 + MN - 1]
                                 + rhs[index2 + MN - M]
                                 + rhs[index2 + MN - M - 1]));
    
    for (p = 1; p < Phalf - 1; p++)
    {
      index1 = (p * Nhalf + 0) * Mhalf + 0;
      index2 = (2 * p * N + 2 * 0) * M + 2 * 0;
      rhs_coarse[index1] = (0.5 * (rhs[index2]
                                   + rhs[index2 + 1]
                                   + rhs[index2 + M]
                                   + rhs[index2 + M + 1])
                            + 0.25 * (rhs[index2 + MN]
                                      + rhs[index2 + MN + 1]
                                      + rhs[index2 + MN + M]
                                      + rhs[index2 + MN + M + 1]
                                      + rhs[index2 - MN]
                                      + rhs[index2 - MN + 1]
                                      + rhs[index2 - MN + M]
                                      + rhs[index2 - MN + M + 1]));
      
      for (i = 1; i < Mhalf - 1; i++)
      {
        index1 = (p * Nhalf + 0) * Mhalf + i;
        index2 = (2 * p * N + 2 * 0) * M + 2 * i;
        rhs_coarse[index1] = (0.5 * (rhs[index2]
                                     + rhs[index2 + M])
                              + 0.25 * (rhs[index2 + 1]
                                        + rhs[index2 - 1]
                                        + rhs[index2 + M + 1]
                                        + rhs[index2 + M - 1]
                                        + rhs[index2 + MN]
                                        + rhs[index2 + MN + M]
                                        + rhs[index2 - MN]
                                        + rhs[index2 - MN + M])
                              + 0.125 * (rhs[index2 + MN + 1]
                                         + rhs[index2 + MN - 1]
                                         + rhs[index2 + MN + M + 1]
                                         + rhs[index2 + MN + M - 1]
                                         + rhs[index2 - MN + 1]
                                         + rhs[index2 - MN - 1]
                                         + rhs[index2 - MN + M + 1]
                                         + rhs[index2 - MN + M - 1]));
      }
      
      index1 = (p * Nhalf + 0) * Mhalf + (Mhalf - 1);
      index2 = (2 * p * N + 2 * 0) * M + 2 * (Mhalf - 1);
      rhs_coarse[index1] = (0.5 * (rhs[index2]
                                   + rhs[index2 - 1]
                                   + rhs[index2 + M]
                                   + rhs[index2 + M - 1])
                            + 0.25 * (rhs[index2 + MN]
                                      + rhs[index2 + MN - 1]
                                      + rhs[index2 + MN + M]
                                      + rhs[index2 + MN + M - 1]
                                      + rhs[index2 - MN]
                                      + rhs[index2 - MN - 1]
                                      + rhs[index2 - MN + M]
                                      + rhs[index2 - MN + M - 1]));
      
      for (j = 1; j < Nhalf - 1; j++)
      {
        index1 = (p * Nhalf + j) * Mhalf + 0;
        index2 = (2 * p * N + 2 * j) * M + 2 * 0;
        rhs_coarse[index1] = (0.5 * (rhs[index2]
                                     + rhs[index2 + 1])
                              + 0.25 * (rhs[index2 + M]
                                        + rhs[index2 + M + 1]
                                        + rhs[index2 - M]
                                        + rhs[index2 - M + 1]
                                        + rhs[index2 + MN]
                                        + rhs[index2 + MN + 1]
                                        + rhs[index2 - MN]
                                        + rhs[index2 - MN + 1])
                              + 0.125 * (rhs[index2 + MN + M]
                                         + rhs[index2 + MN + M + 1]
                                         + rhs[index2 + MN - M]
                                         + rhs[index2 + MN - M + 1]
                                         + rhs[index2 - MN + M]
                                         + rhs[index2 - MN + M + 1]
                                         + rhs[index2 - MN - M]
                                         + rhs[index2 - MN - M + 1]));
        
        for (i = 1; i < Mhalf - 1; i++)
        {
          index1 = (p * Nhalf + j) * Mhalf + i;
          index2 = (2 * p * N + 2 * j) * M + 2 * i;
          rhs_coarse[index1] = (0.5 * (rhs[index2])
                                + 0.25 * (rhs[index2 + 1]
                                          + rhs[index2 - 1]
                                          + rhs[index2 + M]
                                          + rhs[index2 - M]
                                          + rhs[index2 + MN]
                                          + rhs[index2 - MN])
                                + 0.125 * (rhs[index2 + M + 1]
                                           + rhs[index2 + M - 1]
                                           + rhs[index2 - M + 1]
                                           + rhs[index2 - M - 1]
                                           + rhs[index2 + MN + 1]
                                           + rhs[index2 + MN - 1]
                                           + rhs[index2 + MN + M]
                                           + rhs[index2 + MN - M]
                                           + rhs[index2 - MN + 1]
                                           + rhs[index2 - MN - 1]
                                           + rhs[index2 - MN + M]
                                           + rhs[index2 - MN - M])
                                + 0.0625 * (rhs[index2 + MN + M + 1]
                                            + rhs[index2 + MN + M - 1]
                                            + rhs[index2 + MN - M + 1]
                                            + rhs[index2 + MN - M - 1]
                                            + rhs[index2 - MN + M + 1]
                                            + rhs[index2 - MN + M - 1]
                                            + rhs[index2 - MN - M + 1]
                                            + rhs[index2 - MN - M - 1]));
        }
        
        index1 = (p * Nhalf + j) * Mhalf + (Mhalf - 1);
        index2 = (2 * p * N + 2 * j) * M + 2 * (Mhalf - 1);
        rhs_coarse[index1] = (0.5 * (rhs[index2]
                                     + rhs[index2 - 1])
                              + 0.25 * (rhs[index2 + M]
                                        + rhs[index2 + M - 1]
                                        + rhs[index2 - M]
                                        + rhs[index2 - M - 1]
                                        + rhs[index2 + MN]
                                        + rhs[index2 + MN - 1]
                                        + rhs[index2 - MN]
                                        + rhs[index2 - MN - 1])
                              + 0.125 * (rhs[index2 + MN + M]
                                         + rhs[index2 + MN + M - 1]
                                         + rhs[index2 + MN - M]
                                         + rhs[index2 + MN - M - 1]
                                         + rhs[index2 - MN + M]
                                         + rhs[index2 - MN + M - 1]
                                         + rhs[index2 - MN - M]
                                         + rhs[index2 - MN - M - 1]));
      }
      
      index1 = (p * Nhalf + (Nhalf - 1)) * Mhalf + 0;
      index2 = (2 * p * N + 2 * (Nhalf - 1)) * M + 2 * 0;
      rhs_coarse[index1] = (0.5 * (rhs[index2]
                                   + rhs[index2 + 1]
                                   + rhs[index2 - M]
                                   + rhs[index2 - M + 1])
                            + 0.25 * (rhs[index2 + MN]
                                      + rhs[index2 + MN + 1]
                                      + rhs[index2 + MN - M]
                                      + rhs[index2 + MN - M + 1]
                                      + rhs[index2 - MN]
                                      + rhs[index2 - MN + 1]
                                      + rhs[index2 - MN - M]
                                      + rhs[index2 - MN - M + 1]));
      
      for (i = 1; i < Mhalf - 1; i++)
      {
        index1 = (p * Nhalf + (Nhalf - 1)) * Mhalf + i;
        index2 = (2 * p * N + 2 * (Nhalf - 1)) * M + 2 * i;
        rhs_coarse[index1] = (0.5 * (rhs[index2]
                                     + rhs[index2 - M])
                              + 0.25 * (rhs[index2 + 1]
                                        + rhs[index2 - 1]
                                        + rhs[index2 - M + 1]
                                        + rhs[index2 - M - 1]
                                        + rhs[index2 + MN]
                                        + rhs[index2 + MN - M]
                                        + rhs[index2 - MN]
                                        + rhs[index2 - MN - M])
                              + 0.125 * (rhs[index2 + MN + 1]
                                         + rhs[index2 + MN - 1]
                                         + rhs[index2 + MN - M + 1]
                                         + rhs[index2 + MN - M - 1]
                                         + rhs[index2 - MN + 1]
                                         + rhs[index2 - MN - 1]
                                         + rhs[index2 - MN - M + 1]
                                         + rhs[index2 - MN - M - 1]));
      }
      
      index1 = (p * Nhalf + (Nhalf - 1)) * Mhalf + (Mhalf - 1);
      index2 = (2 * p * N + 2 * (Nhalf - 1)) * M + 2 * (Mhalf - 1);
      rhs_coarse[index1] = (0.5 * (rhs[index2]
                                   + rhs[index2 - 1]
                                   + rhs[index2 - M]
                                   + rhs[index2 - M - 1])
                            + 0.25 * (rhs[index2 + MN]
                                      + rhs[index2 + MN - 1]
                                      + rhs[index2 + MN - M]
                                      + rhs[index2 + MN - M - 1]
                                      + rhs[index2 - MN]
                                      + rhs[index2 - MN - 1]
                                      + rhs[index2 - MN - M]
                                      + rhs[index2 - MN - M - 1]));
    }
    
    index1 = ((Phalf - 1) * Nhalf + 0) * Mhalf + 0;
    index2 = (2 * (Phalf - 1) * N + 2 * 0) * M + 2 * 0;
    rhs_coarse[index1] = (0.5 * (rhs[index2]
                                 + rhs[index2 + 1]
                                 + rhs[index2 + M]
                                 + rhs[index2 + M + 1]
                                 + rhs[index2 - MN]
                                 + rhs[index2 - MN + 1]
                                 + rhs[index2 - MN + M]
                                 + rhs[index2 - MN + M + 1]));
    
    for (i = 1; i < Mhalf - 1; i++)
    {
      index1 = ((Phalf - 1) * Nhalf + 0) * Mhalf + i;
      index2 = (2 * (Phalf - 1) * N + 2 * 0) * M + 2 * i;
      rhs_coarse[index1] = (0.5 * (rhs[index2]
                                   + rhs[index2 + M]
                                   + rhs[index2 - MN]
                                   + rhs[index2 - MN + M])
                            + 0.25 * (rhs[index2 + 1]
                                      + rhs[index2 - 1]
                                      + rhs[index2 + M + 1]
                                      + rhs[index2 + M - 1]
                                      + rhs[index2 - MN + 1]
                                      + rhs[index2 - MN - 1]
                                      + rhs[index2 - MN + M + 1]
                                      + rhs[index2 - MN + M - 1]));
    }
    
    index1 = ((Phalf - 1) * Nhalf + 0) * Mhalf + (Mhalf - 1);
    index2 = (2 * (Phalf - 1) * N + 2 * 0) * M + 2 * (Mhalf - 1);
    rhs_coarse[index1] = (0.5 * (rhs[index2]
                                 + rhs[index2 - 1]
                                 + rhs[index2 + M]
                                 + rhs[index2 + M - 1]
                                 + rhs[index2 - MN]
                                 + rhs[index2 - MN - 1]
                                 + rhs[index2 - MN + M]
                                 + rhs[index2 - MN + M - 1]));
    
    for (j = 1; j < Nhalf - 1; j++)
    {
      index1 = ((Phalf - 1) * Nhalf + j) * Mhalf + 0;
      index2 = (2 * (Phalf - 1) * N + 2 * j) * M + 2 * 0;
      rhs_coarse[index1] = (0.5 * (rhs[index2]
                                   + rhs[index2 + 1]
                                   + rhs[index2 - MN]
                                   + rhs[index2 - MN + 1])
                            + 0.25 * (rhs[index2 + M]
                                      + rhs[index2 + M + 1]
                                      + rhs[index2 - M]
                                      + rhs[index2 - M + 1]
                                      + rhs[index2 - MN + M]
                                      + rhs[index2 - MN + M + 1]
                                      + rhs[index2 - MN - M]
                                      + rhs[index2 - MN - M + 1]));
      
      for (i = 1; i < Mhalf - 1; i++)
      {
        index1 = ((Phalf - 1) * Nhalf + j) * Mhalf + i;
        index2 = (2 * (Phalf - 1) * N + 2 * j) * M + 2 * i;
        rhs_coarse[index1] = (0.5 * (rhs[index2]
                                     + rhs[index2 - MN])
                              + 0.25 * (rhs[index2 + 1]
                                        + rhs[index2 - 1]
                                        + rhs[index2 + M]
                                        + rhs[index2 - M]
                                        + rhs[index2 - MN + 1]
                                        + rhs[index2 - MN - 1]
                                        + rhs[index2 - MN + M]
                                        + rhs[index2 - MN - M])
                              + 0.125 * (rhs[index2 + M + 1]
                                         + rhs[index2 + M - 1]
                                         + rhs[index2 - M + 1]
                                         + rhs[index2 - M - 1]
                                         + rhs[index2 - MN + M + 1]
                                         + rhs[index2 - MN + M - 1]
                                         + rhs[index2 - MN - M + 1]
                                         + rhs[index2 - MN - M - 1]));
      }
      
      index1 = ((Phalf - 1) * Nhalf + j) * Mhalf + (Mhalf - 1);
      index2 = (2 * (Phalf - 1) * N + 2 * j) * M + 2 * (Mhalf - 1);
      rhs_coarse[index1] = (0.5 * (rhs[index2]
                                   + rhs[index2 - 1]
                                   + rhs[index2 - MN]
                                   + rhs[index2 - MN - 1])
                            + 0.25 * (rhs[index2 + M]
                                      + rhs[index2 + M - 1]
                                      + rhs[index2 - M]
                                      + rhs[index2 - M - 1]
                                      + rhs[index2 - MN + M]
                                      + rhs[index2 - MN + M - 1]
                                      + rhs[index2 - MN - M]
                                      + rhs[index2 - MN - M - 1]));
    }
    
    index1 = ((Phalf - 1) * Nhalf + (Nhalf - 1)) * Mhalf + 0;
    index2 = (2 * (Phalf - 1) * N + 2 * (Nhalf - 1)) * M + 2 * 0;
    rhs_coarse[index1] = (0.5 * (rhs[index2]
                                 + rhs[index2 + 1]
                                 + rhs[index2 - M]
                                 + rhs[index2 - M + 1]
                                 + rhs[index2 - MN]
                                 + rhs[index2 - MN + 1]
                                 + rhs[index2 - MN - M]
                                 + rhs[index2 - MN - M + 1]));
    
    for (i = 1; i < Mhalf - 1; i++)
    {
      index1 = ((Phalf - 1) * Nhalf + (Nhalf - 1)) * Mhalf + i;
      index2 = (2 * (Phalf - 1) * N + 2 * (Nhalf - 1)) * M + 2 * i;
      rhs_coarse[index1] = (0.5 * (rhs[index2]
                                   + rhs[index2 - M]
                                   + rhs[index2 - MN]
                                   + rhs[index2 - MN - M])
                            + 0.25 * (rhs[index2 + 1]
                                      + rhs[index2 - 1]
                                      + rhs[index2 - M + 1]
                                      + rhs[index2 - M - 1]
                                      + rhs[index2 - MN + 1]
                                      + rhs[index2 - MN - 1]
                                      + rhs[index2 - MN - M + 1]
                                      + rhs[index2 - MN - M - 1]));
    }
    
    index1 = ((Phalf - 1) * Nhalf + (Nhalf - 1)) * Mhalf + (Mhalf - 1);
    index2 = (2 * (Phalf - 1) * N + 2 * (Nhalf - 1)) * M + 2 * (Mhalf - 1);
    rhs_coarse[index1] = (0.5 * (rhs[index2]
                                 + rhs[index2 - 1]
                                 + rhs[index2 - M]
                                 + rhs[index2 - M - 1]
                                 + rhs[index2 - MN]
                                 + rhs[index2 - MN - 1]
                                 + rhs[index2 - MN - M]
                                 + rhs[index2 - MN - M - 1]));
  }
}


static void
upsample3D(double *rhs, int M, int N, int P,
           double *v, int Mhalf, int Nhalf, int Phalf,
           double *f_out)
{
  int i, j, p;
  int index1;
  int index2;
  int MN = M * N;
  int MNhalf = Mhalf * Nhalf;
  
  if (M % 2 == 0 && N % 2 == 0 && P % 2 == 0)
  {
    for (p = 0; p < Phalf; p++)
    {
      for (j = 0; j < Nhalf; j++)
      {
        for (i = 0; i < Mhalf; i++)
        {
          index1 = (p * Nhalf + j) * Mhalf + i;
          index2 = (2 * p * N + 2 * j) * M + 2 * i;
          if (p == 0)
          {
            if (j == 0)
            {
              if (i == 0)
              {
                f_out[index2] += (v[index1]);
              }
              else
              {
                f_out[index2] += (0.75 * v[index1]
                                  + 0.25 * v[index1 - 1]);
              }
            }
            else
            {
              if (i == 0)
              {
                f_out[index2] += (0.75 * v[index1]
                                  + 0.25 * v[index1 - Mhalf]);
              }
              else
              {
                f_out[index2] += (0.5625 * v[index1]
                                  + 0.1875 * (v[index1 - 1]
                                              + v[index1 - Mhalf])
                                  + 0.0625 * v[index1 - Mhalf - 1]);
              }
            }
          }
          else
          {
            if (j == 0)
            {
              if (i == 0)
              {
                f_out[index2] += (0.75 * v[index1]
                                  + 0.25 * v[index1 - MNhalf]);
              }
              else
              {
                f_out[index2] += (0.5625 * v[index1]
                                  + 0.1875 * (v[index1 - 1]
                                              + v[index1 - MNhalf])
                                  + 0.0625 * v[index1 - MNhalf - 1]);
              }
            }
            else
            {
              if (i == 0)
              {
                f_out[index2] += (0.5625 * v[index1]
                                  + 0.1875 * (v[index1 - Mhalf]
                                              + v[index1 - MNhalf])
                                  + 0.0625 * v[index1 - MNhalf - Mhalf]);
              }
              else
              {
                f_out[index2] += (0.421875 * v[index1]
                                  + 0.140625 * (v[index1 - 1]
                                                + v[index1 - Mhalf]
                                                + v[index1 - MNhalf])
                                  + 0.046875 * (v[index1 - Mhalf - 1]
                                                + v[index1 - MNhalf - 1]
                                                + v[index1 - MNhalf - Mhalf])
                                  + 0.015625 * v[index1 - MNhalf - Mhalf - 1]);
              }
            }
          }
          
          if (p == 0)
          {
            if (j == 0)
            {
              if (i < Mhalf - 1)
              {
                f_out[index2 + 1] += (0.75 * v[index1]
                                      + 0.25 * v[index1 + 1]);
              }
              else
              {
                f_out[index2 + 1] += (v[index1]);
              }
            }
            else
            {
              if (i < Mhalf - 1)
              {
                f_out[index2 + 1] += (0.5625 * v[index1]
                                      + 0.1875 * (v[index1 + 1]
                                                  + v[index1 - Mhalf])
                                      + 0.0625 * v[index1 - Mhalf + 1]);
              }
              else
              {
                f_out[index2 + 1] += (0.75 * v[index1]
                                      + 0.25 * v[index1 - Mhalf]);
              }
            }
          }
          else
          {
            if (j == 0)
            {
              if (i < Mhalf - 1)
              {
                f_out[index2 + 1] += (0.5625 * v[index1]
                                      + 0.1875 * (v[index1 + 1]
                                                  + v[index1 - MNhalf])
                                      + 0.0625 * v[index1 - MNhalf + 1]);
              }
              else
              {
                f_out[index2 + 1] += (0.75 * v[index1]
                                      + 0.25 * v[index1 - MNhalf]);
              }
            }
            else
            {
              if (i < Mhalf - 1)
              {
                f_out[index2 + 1] += (0.421875 * v[index1]
                                      + 0.140625 * (v[index1 + 1]
                                                    + v[index1 - Mhalf]
                                                    + v[index1 - MNhalf])
                                      + 0.046875 * (v[index1 - Mhalf + 1]
                                                    + v[index1 - MNhalf + 1]
                                                    + v[index1 - MNhalf - Mhalf])
                                      + 0.015625 * v[index1 - MNhalf - Mhalf + 1]);
              }
              else
              {
                f_out[index2 + 1] += (0.5625 * v[index1]
                                      + 0.1875 * (v[index1 - Mhalf]
                                                  + v[index1 - MNhalf])
                                      + 0.0625 * v[index1 - MNhalf - Mhalf]);
              }
            }
          }
          
          if (p == 0)
          {
            if (j < Nhalf - 1)
            {
              if (i == 0)
              {
                f_out[index2 + M] += (0.75 * v[index1]
                                      + 0.25 * v[index1 + Mhalf]);
              }
              else
              {
                f_out[index2 + M] += (0.5625 * v[index1]
                                      + 0.1875 * (v[index1 - 1]
                                                  + v[index1 + Mhalf])
                                      + 0.0625 * v[index1 + Mhalf - 1]);
              }
            }
            else
            {
              if (i == 0)
              {
                f_out[index2 + M] += (v[index1]);
              }
              else
              {
                f_out[index2 + M] += (0.75 * v[index1]
                                      + 0.25 * v[index1 - 1]);
              }
            }
          }
          else
          {
            if (j < Nhalf - 1)
            {
              if (i == 0)
              {
                f_out[index2 + M] += (0.5625 * v[index1]
                                      + 0.1875 * (v[index1 + Mhalf]
                                                  + v[index1 - MNhalf])
                                      + 0.0625 * v[index1 - MNhalf + Mhalf]);
              }
              else
              {
                f_out[index2 + M] += (0.421875 * v[index1]
                                      + 0.140625 * (v[index1 - 1]
                                                    + v[index1 + Mhalf]
                                                    + v[index1 - MNhalf])
                                      + 0.046875 * (v[index1 + Mhalf - 1]
                                                    + v[index1 - MNhalf - 1]
                                                    + v[index1 - MNhalf + Mhalf])
                                      + 0.015625 * v[index1 - MNhalf + Mhalf - 1]);
              }
            }
            else
            {
              if (i == 0)
              {
                f_out[index2 + M] += (0.75 * v[index1]
                                      + 0.25 * v[index1 - MNhalf]);
              }
              else
              {
                f_out[index2 + M] += (0.5625 * v[index1]
                                      + 0.1875 * (v[index1 - 1]
                                                  + v[index1 - MNhalf])
                                      + 0.0625 * v[index1 - MNhalf - 1]);
              }
            }
          }
          
          if (p == 0)
          {
            if (j < Nhalf - 1)
            {
              if (i < Mhalf - 1)
              {
                f_out[index2 + M + 1] += (0.5625 * v[index1]
                                          + 0.1875 * (v[index1 + 1]
                                                      + v[index1 + Mhalf])
                                          + 0.0625 * v[index1 + Mhalf + 1]);
              }
              else
              {
                f_out[index2 + M + 1] += (0.75 * v[index1]
                                          + 0.25 * v[index1 + Mhalf]);
              }
            }
            else
            {
              if (i < Mhalf - 1)
              {
                f_out[index2 + M + 1] += (0.75 * v[index1]
                                          + 0.25 * v[index1 + 1]);
              }
              else
              {
                f_out[index2 + M + 1] += (v[index1]);
              }
            }
          }
          else
          {
            if (j < Nhalf - 1)
            {
              if (i < Mhalf - 1)
              {
                f_out[index2 + M + 1] += (0.421875 * v[index1]
                                          + 0.140625 * (v[index1 + 1]
                                                        + v[index1 + Mhalf]
                                                        + v[index1 - MNhalf])
                                          + 0.046875 * (v[index1 + Mhalf + 1]
                                                        + v[index1 - MNhalf + 1]
                                                        + v[index1 - MNhalf + Mhalf])
                                          + 0.015625 * v[index1 - MNhalf + Mhalf + 1]);
              }
              else
              {
                f_out[index2 + M + 1] += (0.5625 * v[index1]
                                          + 0.1875 * (v[index1 + Mhalf]
                                                      + v[index1 - MNhalf])
                                          + 0.0625 * v[index1 - MNhalf + Mhalf]);
              }
            }
            else
            {
              if (i < Mhalf - 1)
              {
                f_out[index2 + M + 1] += (0.5625 * v[index1]
                                          + 0.1875 * (v[index1 + 1]
                                                      + v[index1 - MNhalf])
                                          + 0.0625 * v[index1 - MNhalf + 1]);
              }
              else
              {
                f_out[index2 + M + 1] += (0.75 * v[index1]
                                          + 0.25 * v[index1 - MNhalf]);
              }
            }
          }
          
          if (p < Phalf - 1)
          {
            if (j == 0)
            {
              if (i == 0)
              {
                f_out[index2 + MN] += (0.75 * v[index1]
                                       + 0.25 * v[index1 + MNhalf]);
              }
              else
              {
                f_out[index2 + MN] += (0.5625 * v[index1]
                                       + 0.1875 * (v[index1 - 1]
                                                   + v[index1 + MNhalf])
                                       + 0.0625 * v[index1 + MNhalf - 1]);
              }
            }
            else
            {
              if (i == 0)
              {
                f_out[index2 + MN] += (0.5625 * v[index1]
                                       + 0.1875 * (v[index1 - Mhalf]
                                                   + v[index1 + MNhalf])
                                       + 0.0625 * v[index1 + MNhalf - Mhalf]);
              }
              else
              {
                f_out[index2 + MN] += (0.421875 * v[index1]
                                       + 0.140625 * (v[index1 - 1]
                                                     + v[index1 - Mhalf]
                                                     + v[index1 + MNhalf])
                                       + 0.046875 * (v[index1 - Mhalf - 1]
                                                     + v[index1 + MNhalf - 1]
                                                     + v[index1 + MNhalf - Mhalf])
                                       + 0.015625 * v[index1 + MNhalf - Mhalf - 1]);
              }
            }
          }
          else
          {
            if (j == 0)
            {
              if (i == 0)
              {
                f_out[index2 + MN] += (v[index1]);
              }
              else
              {
                f_out[index2 + MN] += (0.75 * v[index1]
                                       + 0.25 * v[index1 - 1]);
              }
            }
            else
            {
              if (i == 0)
              {
                f_out[index2 + MN] += (0.75 * v[index1]
                                       + 0.25 * v[index1 - Mhalf]);
              }
              else
              {
                f_out[index2 + MN] += (0.5625 * v[index1]
                                       + 0.1875 * (v[index1 - 1]
                                                   + v[index1 - Mhalf])
                                       + 0.0625 * v[index1 - Mhalf - 1]);
              }
            }
          }
          
          if (p < Phalf - 1)
          {
            if (j == 0)
            {
              if (i < Mhalf - 1)
              {
                f_out[index2 + MN + 1] += (0.5625 * v[index1]
                                           + 0.1875 * (v[index1 + 1]
                                                       + v[index1 + MNhalf])
                                           + 0.0625 * v[index1 + MNhalf + 1]);
              }
              else
              {
                f_out[index2 + MN + 1] += (0.75 * v[index1]
                                           + 0.25 * v[index1 + MNhalf]);
              }
            }
            else
            {
              if (i < Mhalf - 1)
              {
                f_out[index2 + MN + 1] += (0.421875 * v[index1]
                                           + 0.140625 * (v[index1 + 1]
                                                         + v[index1 - Mhalf]
                                                         + v[index1 + MNhalf])
                                           + 0.046875 * (v[index1 - Mhalf + 1]
                                                         + v[index1 + MNhalf + 1]
                                                         + v[index1 + MNhalf - Mhalf])
                                           + 0.015625 * v[index1 + MNhalf - Mhalf + 1]);
              }
              else
              {
                f_out[index2 + MN + 1] += (0.5625 * v[index1]
                                           + 0.1875 * (v[index1 - Mhalf]
                                                       + v[index1 + MNhalf])
                                           + 0.0625 * v[index1 + MNhalf - Mhalf]);
              }
            }
          }
          else
          {
            if (j == 0)
            {
              if (i < Mhalf - 1)
              {
                f_out[index2 + MN + 1] += (0.75 * v[index1]
                                           + 0.25 * v[index1 + 1]);
              }
              else
              {
                f_out[index2 + MN + 1] += (v[index1]);
              }
            }
            else
            {
              if (i < Mhalf - 1)
              {
                f_out[index2 + MN + 1] += (0.5625 * v[index1]
                                           + 0.1875 * (v[index1 + 1]
                                                       + v[index1 - Mhalf])
                                           + 0.0625 * v[index1 - Mhalf + 1]);
              }
              else
              {
                f_out[index2 + MN + 1] += (0.75 * v[index1]
                                           + 0.25 * v[index1 - Mhalf]);
              }
            }
          }
          
          if (p < Phalf - 1)
          {
            if (j < Nhalf - 1)
            {
              if (i == 0)
              {
                f_out[index2 + MN + M] += (0.5625 * v[index1]
                                           + 0.1875 * (v[index1 + Mhalf]
                                                       + v[index1 + MNhalf])
                                           + 0.0625 * v[index1 + MNhalf + Mhalf]);
              }
              else
              {
                f_out[index2 + MN + M] += (0.421875 * v[index1]
                                           + 0.140625 * (v[index1 - 1]
                                                         + v[index1 + Mhalf]
                                                         + v[index1 + MNhalf])
                                           + 0.046875 * (v[index1 + Mhalf - 1]
                                                         + v[index1 + MNhalf - 1]
                                                         + v[index1 + MNhalf + Mhalf])
                                           + 0.015625 * v[index1 + MNhalf + Mhalf - 1]);
              }
            }
            else
            {
              if (i == 0)
              {
                f_out[index2 + MN + M] += (0.75 * v[index1]
                                           + 0.25 * v[index1 + MNhalf]);
              }
              else
              {
                f_out[index2 + MN + M] += (0.5625 * v[index1]
                                           + 0.1875 * (v[index1 - 1]
                                                       + v[index1 + MNhalf])
                                           + 0.0625 * v[index1 + MNhalf - 1]);
              }
            }
          }
          else
          {
            if (j < Nhalf - 1)
            {
              if (i == 0)
              {
                f_out[index2 + MN + M] += (0.75 * v[index1]
                                           + 0.25 * v[index1 + Mhalf]);
              }
              else
              {
                f_out[index2 + MN + M] += (0.5625 * v[index1]
                                           + 0.1875 * (v[index1 - 1]
                                                       + v[index1 + Mhalf])
                                           + 0.0625 * v[index1 + Mhalf - 1]);
              }
            }
            else
            {
              if (i == 0)
              {
                f_out[index2 + MN + M] += (v[index1]);
              }
              else
              {
                f_out[index2 + MN + M] += (0.75 * v[index1]
                                           + 0.25 * v[index1 - 1]);
              }
            }
          }
          
          if (p < Phalf - 1)
          {
            if (j < Nhalf - 1)
            {
              if (i < Mhalf - 1)
              {
                f_out[index2 + MN + M + 1] += (0.421875 * v[index1]
                                               + 0.140625 * (v[index1 + 1]
                                                             + v[index1 + Mhalf]
                                                             + v[index1 + MNhalf])
                                               + 0.046875 * (v[index1 + Mhalf + 1]
                                                             + v[index1 + MNhalf + 1]
                                                             + v[index1 + MNhalf + Mhalf])
                                               + 0.015625 * v[index1 + MNhalf + Mhalf + 1]);
              }
              else
              {
                f_out[index2 + MN + M + 1] += (0.5625 * v[index1]
                                               + 0.1875 * (v[index1 + Mhalf]
                                                           + v[index1 + MNhalf])
                                               + 0.0625 * v[index1 + MNhalf + Mhalf]);
              }
            }
            else
            {
              if (i < Mhalf - 1)
              {
                f_out[index2 + MN + M + 1] += (0.5625 * v[index1]
                                               + 0.1875 * (v[index1 + 1]
                                                           + v[index1 + MNhalf])
                                               + 0.0625 * v[index1 + MNhalf + 1]);
              }
              else
              {
                f_out[index2 + MN + M + 1] += (0.75 * v[index1]
                                               + 0.25 * v[index1 + MNhalf]);
              }
            }
          }
          else
          {
            if (j < Nhalf - 1)
            {
              if (i < Mhalf - 1)
              {
                f_out[index2 + MN + M + 1] += (0.5625 * v[index1]
                                               + 0.1875 * (v[index1 + 1]
                                                           + v[index1 + Mhalf])
                                               + 0.0625 * v[index1 + Mhalf + 1]);
              }
              else
              {
                f_out[index2 + MN + M + 1] += (0.75 * v[index1]
                                               + 0.25 * v[index1 + Mhalf]);
              }
            }
            else
            {
              if (i < Mhalf - 1)
              {
                f_out[index2 + MN + M + 1] += (0.75 * v[index1]
                                               + 0.25 * v[index1 + 1]);
              }
              else
              {
                f_out[index2 + MN + M + 1] += (v[index1]);
              }
            }
          }
        }
      }
    }
  }
  
  if (M % 2 == 1 && N % 2 == 0 && P % 2 == 0)
  {
    for (p = 0; p < Phalf; p++)
    {
      for (j = 0; j < Nhalf; j++)
      {
        for (i = 0; i < Mhalf; i++)
        {
          index1 = (p * Nhalf + j) * Mhalf + i;
          index2 = (2 * p * N + 2 * j) * M + 2 * i;
          if (p == 0)
          {
            if (j == 0)
            {
              f_out[index2] += (v[index1]);
            }
            else
            {
              f_out[index2] += (0.75 * v[index1]
                                + 0.25 * v[index1 - Mhalf]);
            }
          }
          else
          {
            if (j == 0)
            {
              f_out[index2] += (0.75 * v[index1]
                                + 0.25 * v[index1 - MNhalf]);
            }
            else
            {
              f_out[index2] += (0.5625 * v[index1]
                                + 0.1875 * (v[index1 - Mhalf]
                                            + v[index1 - MNhalf])
                                + 0.0625 * v[index1 - MNhalf - Mhalf]);
            }
          }
          
          if (p == 0)
          {
            if (j == 0)
            {
              if (i < Mhalf - 1)
              {
                f_out[index2 + 1] += (0.5 * (v[index1]
                                             + v[index1 + 1]));
              }
            }
            else
            {
              if (i < Mhalf - 1)
              {
                f_out[index2 + 1] += (0.375 * (v[index1]
                                               + v[index1 + 1])
                                      + 0.125 * (v[index1 - Mhalf]
                                                 + v[index1 - Mhalf + 1]));
              }
            }
          }
          else
          {
            if (j == 0)
            {
              if (i < Mhalf - 1)
              {
                f_out[index2 + 1] += (0.375 * (v[index1]
                                               + v[index1 + 1])
                                      + 0.125 * (v[index1 - MNhalf]
                                                 + v[index1 - MNhalf + 1]));
              }
            }
            else
            {
              if (i < Mhalf - 1)
              {
                f_out[index2 + 1] += (0.28125 * (v[index1]
                                                 + v[index1 + 1])
                                      + 0.09375 * (v[index1 - Mhalf]
                                                   + v[index1 - Mhalf + 1]
                                                   + v[index1 - MNhalf]
                                                   + v[index1 - MNhalf + 1])
                                      + 0.03125 * (v[index1 - MNhalf - Mhalf]
                                                   + v[index1 - MNhalf - Mhalf + 1]));
              }
            }
          }
          
          if (p == 0)
          {
            if (j < Nhalf - 1)
            {
              f_out[index2 + M] += (0.75 * v[index1]
                                    + 0.25 * v[index1 + Mhalf]);
            }
            else
            {
              f_out[index2 + M] += (v[index1]);
            }
          }
          else
          {
            if (j < Nhalf - 1)
            {
              f_out[index2 + M] += (0.5625 * v[index1]
                                    + 0.1875 * (v[index1 + Mhalf]
                                                + v[index1 - MNhalf])
                                    + 0.0625 * v[index1 - MNhalf + Mhalf]);
            }
            else
            {
              f_out[index2 + M] += (0.75 * v[index1]
                                    + 0.25 * v[index1 - MNhalf]);
            }
          }
          
          if (p == 0)
          {
            if (j < Nhalf - 1)
            {
              if (i < Mhalf - 1)
              {
                f_out[index2 + M + 1] += (0.375 * (v[index1]
                                                   + v[index1 + 1])
                                          + 0.125 * (v[index1 + Mhalf]
                                                     + v[index1 + Mhalf + 1]));
              }
            }
            else
            {
              if (i < Mhalf - 1)
              {
                f_out[index2 + M + 1] += (0.5 * (v[index1]
                                                 + v[index1 + 1]));
              }
            }
          }
          else
          {
            if (j < Nhalf - 1)
            {
              if (i < Mhalf - 1)
              {
                f_out[index2 + M + 1] += (0.28125 * (v[index1]
                                                     + v[index1 + 1])
                                          + 0.09375 * (v[index1 + Mhalf]
                                                       + v[index1 + Mhalf + 1]
                                                       + v[index1 - MNhalf]
                                                       + v[index1 - MNhalf + 1])
                                          + 0.03125 * (v[index1 - MNhalf + Mhalf]
                                                       + v[index1 - MNhalf + Mhalf + 1]));
              }
            }
            else
            {
              if (i < Mhalf - 1)
              {
                f_out[index2 + M + 1] += (0.375 * (v[index1]
                                                   + v[index1 + 1])
                                          + 0.125 * (v[index1 - MNhalf]
                                                     + v[index1 - MNhalf + 1]));
              }
            }
          }
          
          if (p < Phalf - 1)
          {
            if (j == 0)
            {
              f_out[index2 + MN] += (0.75 * v[index1]
                                     + 0.25 * v[index1 + MNhalf]);
            }
            else
            {
              f_out[index2 + MN] += (0.5625 * v[index1]
                                     + 0.1875 * (v[index1 - Mhalf]
                                                 + v[index1 + MNhalf])
                                     + 0.0625 * v[index1 + MNhalf - Mhalf]);
            }
          }
          else
          {
            if (j == 0)
            {
              f_out[index2 + MN] += (v[index1]);
            }
            else
            {
              f_out[index2 + MN] += (0.75 * v[index1]
                                     + 0.25 * v[index1 - Mhalf]);
            }
          }
          
          if (p < Phalf - 1)
          {
            if (j == 0)
            {
              if (i < Mhalf - 1)
              {
                f_out[index2 + MN + 1] += (0.375 * (v[index1]
                                                    + v[index1 + 1])
                                           + 0.125 * (v[index1 + MNhalf]
                                                      + v[index1 + MNhalf + 1]));
              }
            }
            else
            {
              if (i < Mhalf - 1)
              {
                f_out[index2 + MN + 1] += (0.28125 * (v[index1]
                                                      + v[index1 + 1])
                                           + 0.09375 * (v[index1 - Mhalf]
                                                        + v[index1 - Mhalf + 1]
                                                        + v[index1 + MNhalf]
                                                        + v[index1 + MNhalf + 1])
                                           + 0.03125 * (v[index1 + MNhalf - Mhalf]
                                                        + v[index1 + MNhalf - Mhalf + 1]));
              }
            }
          }
          else
          {
            if (j == 0)
            {
              if (i < Mhalf - 1)
              {
                f_out[index2 + MN + 1] += (0.5 * (v[index1]
                                                  + v[index1 + 1]));
              }
            }
            else
            {
              if (i < Mhalf - 1)
              {
                f_out[index2 + MN + 1] += (0.375 * (v[index1]
                                                    + v[index1 + 1])
                                           + 0.125 * (v[index1 - Mhalf]
                                                      + v[index1 - Mhalf + 1]));
              }
            }
          }
          
          if (p < Phalf - 1)
          {
            if (j < Nhalf - 1)
            {
              f_out[index2 + MN + M] += (0.5625 * v[index1]
                                         + 0.1875 * (v[index1 + Mhalf]
                                                     + v[index1 + MNhalf])
                                         + 0.0625 * v[index1 + MNhalf + Mhalf]);
            }
            else
            {
              f_out[index2 + MN + M] += (0.75 * v[index1]
                                         + 0.25 * v[index1 + MNhalf]);
            }
          }
          else
          {
            if (j < Nhalf - 1)
            {
              f_out[index2 + MN + M] += (0.75 * v[index1]
                                         + 0.25 * v[index1 + Mhalf]);
            }
            else
            {
              f_out[index2 + MN + M] += (v[index1]);
            }
          }
          
          if (p < Phalf - 1)
          {
            if (j < Nhalf - 1)
            {
              if (i < Mhalf - 1)
              {
                f_out[index2 + MN + M + 1] += (0.28125 * (v[index1]
                                                          + v[index1 + 1])
                                               + 0.09375 * (v[index1 + Mhalf]
                                                            + v[index1 + Mhalf + 1]
                                                            + v[index1 + MNhalf]
                                                            + v[index1 + MNhalf + 1])
                                               + 0.03125 * (v[index1 + MNhalf + Mhalf]
                                                            + v[index1 + MNhalf + Mhalf + 1]));
              }
            }
            else
            {
              if (i < Mhalf - 1)
              {
                f_out[index2 + MN + M + 1] += (0.375 * (v[index1]
                                                        + v[index1 + 1])
                                               + 0.125 * (v[index1 + MNhalf]
                                                          + v[index1 + MNhalf + 1]));
              }
            }
          }
          else
          {
            if (j < Nhalf - 1)
            {
              if (i < Mhalf - 1)
              {
                f_out[index2 + MN + M + 1] += (0.375 * (v[index1]
                                                        + v[index1 + 1])
                                               + 0.125 * (v[index1 + Mhalf]
                                                          + v[index1 + Mhalf + 1]));
              }
            }
            else
            {
              if (i < Mhalf - 1)
              {
                f_out[index2 + MN + M + 1] += (0.5 * (v[index1]
                                                      + v[index1 + 1]));
              }
            }
          }
        }
      }
    }
  }
  
  if (M % 2 == 0 && N % 2 == 1 && P % 2 == 0)
  {
    for (p = 0; p < Phalf; p++)
    {
      for (j = 0; j < Nhalf; j++)
      {
        for (i = 0; i < Mhalf; i++)
        {
          index1 = (p * Nhalf + j) * Mhalf + i;
          index2 = (2 * p * N + 2 * j) * M + 2 * i;
          if (p == 0)
          {
            if (i == 0)
            {
              f_out[index2] += (v[index1]);
            }
            else
            {
              f_out[index2] += (0.75 * v[index1]
                                + 0.25 * v[index1 - 1]);
            }
          }
          else
          {
            if (i == 0)
            {
              f_out[index2] += (0.75 * v[index1]
                                + 0.25 * v[index1 - MNhalf]);
            }
            else
            {
              f_out[index2] += (0.5625 * v[index1]
                                + 0.1875 * (v[index1 - 1]
                                            + v[index1 - MNhalf])
                                + 0.0625 * v[index1 - MNhalf - 1]);
            }
          }
          
          if (p == 0)
          {
            if (i < Mhalf - 1)
            {
              f_out[index2 + 1] += (0.75 * v[index1]
                                    + 0.25 * v[index1 + 1]);
            }
            else
            {
              f_out[index2 + 1] += (v[index1]);
            }
          }
          else
          {
            if (i < Mhalf - 1)
            {
              f_out[index2 + 1] += (0.5625 * v[index1]
                                    + 0.1875 * (v[index1 + 1]
                                                + v[index1 - MNhalf])
                                    + 0.0625 * v[index1 - MNhalf + 1]);
            }
            else
            {
              f_out[index2 + 1] += (0.75 * v[index1]
                                    + 0.25 * v[index1 - MNhalf]);
            }
          }
          
          if (p == 0)
          {
            if (j < Nhalf - 1)
            {
              if (i == 0)
              {
                f_out[index2 + M] += (0.5 * (v[index1]
                                             + v[index1 + Mhalf]));
              }
              else
              {
                f_out[index2 + M] += (0.375 * (v[index1]
                                               + v[index1 + Mhalf])
                                      + 0.125 * (v[index1 - 1]
                                                 + v[index1 + Mhalf - 1]));
              }
            }
          }
          else
          {
            if (j < Nhalf - 1)
            {
              if (i == 0)
              {
                f_out[index2 + M] += (0.375 * (v[index1]
                                               + v[index1 + Mhalf])
                                      + 0.125 * (v[index1 - MNhalf]
                                                 + v[index1 - MNhalf + Mhalf]));
              }
              else
              {
                f_out[index2 + M] += (0.28125 * (v[index1]
                                                 + v[index1 + Mhalf])
                                      + 0.09375 * (v[index1 - 1]
                                                   + v[index1 + Mhalf - 1]
                                                   + v[index1 - MNhalf]
                                                   + v[index1 - MNhalf + Mhalf])
                                      + 0.03125 * (v[index1 - MNhalf - 1]
                                                   + v[index1 - MNhalf + Mhalf - 1]));
              }
            }
          }
          
          if (p == 0)
          {
            if (j < Nhalf - 1)
            {
              if (i < Mhalf - 1)
              {
                f_out[index2 + M + 1] += (0.375 * (v[index1]
                                                   + v[index1 + Mhalf])
                                          + 0.125 * (v[index1 + 1]
                                                     + v[index1 + Mhalf + 1]));
              }
              else
              {
                f_out[index2 + M + 1] += (0.5 * (v[index1]
                                                 + v[index1 + Mhalf]));
              }
            }
          }
          else
          {
            if (j < Nhalf - 1)
            {
              if (i < Mhalf - 1)
              {
                f_out[index2 + M + 1] += (0.28125 * (v[index1]
                                                     + v[index1 + Mhalf])
                                          + 0.09375 * (v[index1 + 1]
                                                       + v[index1 + Mhalf + 1]
                                                       + v[index1 - MNhalf]
                                                       + v[index1 - MNhalf + Mhalf])
                                          + 0.03125 * (v[index1 - MNhalf + 1]
                                                       + v[index1 - MNhalf + Mhalf + 1]));
              }
              else
              {
                f_out[index2 + M + 1] += (0.375 * (v[index1]
                                                   + v[index1 + Mhalf])
                                          + 0.125 * (v[index1 - MNhalf]
                                                     + v[index1 - MNhalf + Mhalf]));
              }
            }
          }
          
          if (p < Phalf - 1)
          {
            if (i == 0)
            {
              f_out[index2 + MN] += (0.75 * v[index1]
                                     + 0.25 * v[index1 + MNhalf]);
            }
            else
            {
              f_out[index2 + MN] += (0.5625 * v[index1]
                                     + 0.1875 * (v[index1 - 1]
                                                 + v[index1 + MNhalf])
                                     + 0.0625 * v[index1 + MNhalf - 1]);
            }
          }
          else
          {
            if (i == 0)
            {
              f_out[index2 + MN] += (v[index1]);
            }
            else
            {
              f_out[index2 + MN] += (0.75 * v[index1]
                                     + 0.25 * v[index1 - 1]);
            }
          }
          
          if (p < Phalf - 1)
          {
            if (i < Mhalf - 1)
            {
              f_out[index2 + MN + 1] += (0.5625 * v[index1]
                                         + 0.1875 * (v[index1 + 1]
                                                     + v[index1 + MNhalf])
                                         + 0.0625 * v[index1 + MNhalf + 1]);
            }
            else
            {
              f_out[index2 + MN + 1] += (0.75 * v[index1]
                                         + 0.25 * v[index1 + MNhalf]);
            }
          }
          else
          {
            if (i < Mhalf - 1)
            {
              f_out[index2 + MN + 1] += (0.75 * v[index1]
                                         + 0.25 * v[index1 + 1]);
            }
            else
            {
              f_out[index2 + MN + 1] += (v[index1]);
            }
          }
          
          if (p < Phalf - 1)
          {
            if (j < Nhalf - 1)
            {
              if (i == 0)
              {
                f_out[index2 + MN + M] += (0.375 * (v[index1]
                                                    + v[index1 + Mhalf])
                                           + 0.125 * (v[index1 + MNhalf]
                                                      + v[index1 + MNhalf + Mhalf]));
              }
              else
              {
                f_out[index2 + MN + M] += (0.28125 * (v[index1]
                                                      + v[index1 + Mhalf])
                                           + 0.09375 * (v[index1 - 1]
                                                        + v[index1 + Mhalf - 1]
                                                        + v[index1 + MNhalf]
                                                        + v[index1 + MNhalf + Mhalf])
                                           + 0.03125 * (v[index1 + MNhalf - 1]
                                                        + v[index1 + MNhalf + Mhalf - 1]));
              }
            }
          }
          else
          {
            if (j < Nhalf - 1)
            {
              if (i == 0)
              {
                f_out[index2 + MN + M] += (0.5 * (v[index1]
                                                  + v[index1 + Mhalf]));
              }
              else
              {
                f_out[index2 + MN + M] += (0.375 * (v[index1]
                                                    + v[index1 + Mhalf])
                                           + 0.125 * (v[index1 - 1]
                                                      + v[index1 + Mhalf - 1]));
              }
            }
          }
          
          if (p < Phalf - 1)
          {
            if (j < Nhalf - 1)
            {
              if (i < Mhalf - 1)
              {
                f_out[index2 + MN + M + 1] += (0.28125 * (v[index1]
                                                          + v[index1 + Mhalf])
                                               + 0.09375 * (v[index1 + 1]
                                                            + v[index1 + Mhalf + 1]
                                                            + v[index1 + MNhalf]
                                                            + v[index1 + MNhalf + Mhalf])
                                               + 0.03125 * (v[index1 + MNhalf + 1]
                                                            + v[index1 + MNhalf + Mhalf + 1]));
              }
              else
              {
                f_out[index2 + MN + M + 1] += (0.375 * (v[index1]
                                                        + v[index1 + Mhalf])
                                               + 0.125 * (v[index1 + MNhalf]
                                                          + v[index1 + MNhalf + Mhalf]));
              }
            }
          }
          else
          {
            if (j < Nhalf - 1)
            {
              if (i < Mhalf - 1)
              {
                f_out[index2 + MN + M + 1] += (0.375 * (v[index1]
                                                        + v[index1 + Mhalf])
                                               + 0.125 * (v[index1 + 1]
                                                          + v[index1 + Mhalf + 1]));
              }
              else
              {
                f_out[index2 + MN + M + 1] += (0.5 * (v[index1]
                                                      + v[index1 + Mhalf]));
              }
            }
          }
        }
      }
    }
  }
  
  if (M % 2 == 1 && N % 2 == 1 && P % 2 == 0)
  {
    for (p = 0; p < Phalf; p++)
    {
      for (j = 0; j < Nhalf; j++)
      {
        for (i = 0; i < Mhalf; i++)
        {
          index1 = (p * Nhalf + j) * Mhalf + i;
          index2 = (2 * p * N + 2 * j) * M + 2 * i;
          if (p == 0)
          {
            f_out[index2] += (v[index1]);
          }
          else
          {
            f_out[index2] += (0.75 * v[index1]
                              + 0.25 * v[index1 - MNhalf]);
          }
          
          if (p == 0)
          {
            if (i < Mhalf - 1)
            {
              f_out[index2 + 1] += (0.5 * (v[index1]
                                           + v[index1 + 1]));
            }
          }
          else
          {
            if (i < Mhalf - 1)
            {
              f_out[index2 + 1] += (0.375 * (v[index1]
                                             + v[index1 + 1])
                                    + 0.125 * (v[index1 - MNhalf]
                                               + v[index1 - MNhalf + 1]));
            }
          }
          
          if (p == 0)
          {
            if (j < Nhalf - 1)
            {
              f_out[index2 + M] += (0.5 * (v[index1]
                                           + v[index1 + Mhalf]));
            }
          }
          else
          {
            if (j < Nhalf - 1)
            {
              f_out[index2 + M] += (0.375 * (v[index1]
                                             + v[index1 + Mhalf])
                                    + 0.125 * (v[index1 - MNhalf]
                                               + v[index1 - MNhalf + Mhalf]));
            }
          }
          
          if (p == 0)
          {
            if (j < Nhalf - 1)
            {
              if (i < Mhalf - 1)
              {
                f_out[index2 + M + 1] += (0.25 * (v[index1]
                                                  + v[index1 + 1]
                                                  + v[index1 + Mhalf]
                                                  + v[index1 + Mhalf + 1]));
              }
            }
          }
          else
          {
            if (j < Nhalf - 1)
            {
              if (i < Mhalf - 1)
              {
                f_out[index2 + M + 1] += (0.1875 * (v[index1]
                                                    + v[index1 + 1]
                                                    + v[index1 + Mhalf]
                                                    + v[index1 + Mhalf + 1])
                                          + 0.0625 * (v[index1 - MNhalf]
                                                      + v[index1 - MNhalf + 1]
                                                      + v[index1 - MNhalf + Mhalf]
                                                      + v[index1 - MNhalf + Mhalf + 1]));
              }
            }
          }
          
          if (p < Phalf - 1)
          {
            f_out[index2 + MN] += (0.75 * v[index1]
                                   + 0.25 * v[index1 + MNhalf]);
          }
          else
          {
            f_out[index2 + MN] += (v[index1]);
          }
          
          if (p < Phalf - 1)
          {
            if (i < Mhalf - 1)
            {
              f_out[index2 + MN + 1] += (0.375 * (v[index1]
                                                  + v[index1 + 1])
                                         + 0.125 * (v[index1 + MNhalf]
                                                    + v[index1 + MNhalf + 1]));
            }
          }
          else
          {
            if (i < Mhalf - 1)
            {
              f_out[index2 + MN + 1] += (0.5 * (v[index1]
                                                + v[index1 + 1]));
            }
          }
          
          if (p < Phalf - 1)
          {
            if (j < Nhalf - 1)
            {
              f_out[index2 + MN + M] += (0.375 * (v[index1]
                                                  + v[index1 + Mhalf])
                                         + 0.125 * (v[index1 + MNhalf]
                                                    + v[index1 + MNhalf + Mhalf]));
            }
          }
          else
          {
            if (j < Nhalf - 1)
            {
              f_out[index2 + MN + M] += (0.5 * (v[index1]
                                                + v[index1 + Mhalf]));
            }
          }
          
          if (p < Phalf - 1)
          {
            if (j < Nhalf - 1)
            {
              if (i < Mhalf - 1)
              {
                f_out[index2 + MN + M + 1] += (0.1875 * (v[index1]
                                                         + v[index1 + 1]
                                                         + v[index1 + Mhalf]
                                                         + v[index1 + Mhalf + 1])
                                               + 0.0625 * (v[index1 + MNhalf]
                                                           + v[index1 + MNhalf + 1]
                                                           + v[index1 + MNhalf + Mhalf]
                                                           + v[index1 + MNhalf + Mhalf + 1]));
              }
            }
          }
          else
          {
            if (j < Nhalf - 1)
            {
              if (i < Mhalf - 1)
              {
                f_out[index2 + MN + M + 1] += (0.25 * (v[index1]
                                                       + v[index1 + 1]
                                                       + v[index1 + Mhalf]
                                                       + v[index1 + Mhalf + 1]));
              }
            }
          }
        }
      }
    }
  }
  
  if (M % 2 == 0 && N % 2 == 0 && P % 2 == 1)
  {
    for (p = 0; p < Phalf; p++)
    {
      for (j = 0; j < Nhalf; j++)
      {
        for (i = 0; i < Mhalf; i++)
        {
          index1 = (p * Nhalf + j) * Mhalf + i;
          index2 = (2 * p * N + 2 * j) * M + 2 * i;
          if (j == 0)
          {
            if (i == 0)
            {
              f_out[index2] += (v[index1]);
            }
            else
            {
              f_out[index2] += (0.75 * v[index1]
                                + 0.25 * v[index1 - 1]);
            }
          }
          else
          {
            if (i == 0)
            {
              f_out[index2] += (0.75 * v[index1]
                                + 0.25 * v[index1 - Mhalf]);
            }
            else
            {
              f_out[index2] += (0.5625 * v[index1]
                                + 0.1875 * (v[index1 - 1]
                                            + v[index1 - Mhalf])
                                + 0.0625 * v[index1 - Mhalf - 1]);
            }
          }
          
          if (j == 0)
          {
            if (i < Mhalf - 1)
            {
              f_out[index2 + 1] += (0.75 * v[index1]
                                    + 0.25 * v[index1 + 1]);
            }
            else
            {
              f_out[index2 + 1] += (v[index1]);
            }
          }
          else
          {
            if (i < Mhalf - 1)
            {
              f_out[index2 + 1] += (0.5625 * v[index1]
                                    + 0.1875 * (v[index1 + 1]
                                                + v[index1 - Mhalf])
                                    + 0.0625 * v[index1 - Mhalf + 1]);
            }
            else
            {
              f_out[index2 + 1] += (0.75 * v[index1]
                                    + 0.25 * v[index1 - Mhalf]);
            }
          }
          
          if (j < Nhalf - 1)
          {
            if (i == 0)
            {
              f_out[index2 + M] += (0.75 * v[index1]
                                    + 0.25 * v[index1 + Mhalf]);
            }
            else
            {
              f_out[index2 + M] += (0.5625 * v[index1]
                                    + 0.1875 * (v[index1 - 1]
                                                + v[index1 + Mhalf])
                                    + 0.0625 * v[index1 + Mhalf - 1]);
            }
          }
          else
          {
            if (i == 0)
            {
              f_out[index2 + M] += (v[index1]);
            }
            else
            {
              f_out[index2 + M] += (0.75 * v[index1]
                                    + 0.25 * v[index1 - 1]);
            }
          }
          
          if (j < Nhalf - 1)
          {
            if (i < Mhalf - 1)
            {
              f_out[index2 + M + 1] += (0.5625 * v[index1]
                                        + 0.1875 * (v[index1 + 1]
                                                    + v[index1 + Mhalf])
                                        + 0.0625 * v[index1 + Mhalf + 1]);
            }
            else
            {
              f_out[index2 + M + 1] += (0.75 * v[index1]
                                        + 0.25 * v[index1 + Mhalf]);
            }
          }
          else
          {
            if (i < Mhalf - 1)
            {
              f_out[index2 + M + 1] += (0.75 * v[index1]
                                        + 0.25 * v[index1 + 1]);
            }
            else
            {
              f_out[index2 + M + 1] += (v[index1]);
            }
          }
          
          if (p < Phalf - 1)
          {
            if (j == 0)
            {
              if (i == 0)
              {
                f_out[index2 + MN] += (0.5 * (v[index1]
                                              + v[index1 + MNhalf]));
              }
              else
              {
                f_out[index2 + MN] += (0.375 * (v[index1]
                                                + v[index1 + MNhalf])
                                       + 0.125 * (v[index1 - 1]
                                                  + v[index1 + MNhalf - 1]));
              }
            }
            else
            {
              if (i == 0)
              {
                f_out[index2 + MN] += (0.375 * (v[index1]
                                                + v[index1 + MNhalf])
                                       + 0.125 * (v[index1 - Mhalf]
                                                  + v[index1 + MNhalf - Mhalf]));
              }
              else
              {
                f_out[index2 + MN] += (0.28125 * (v[index1]
                                                  + v[index1 + MNhalf])
                                       + 0.09375 * (v[index1 - 1]
                                                    + v[index1 - Mhalf]
                                                    + v[index1 + MNhalf - 1]
                                                    + v[index1 + MNhalf - Mhalf])
                                       + 0.03125 * (v[index1 - Mhalf - 1]
                                                    + v[index1 + MNhalf - Mhalf - 1]));
              }
            }
          }
          
          if (p < Phalf - 1)
          {
            if (j == 0)
            {
              if (i < Mhalf - 1)
              {
                f_out[index2 + MN + 1] += (0.375 * (v[index1]
                                                    + v[index1 + MNhalf])
                                           + 0.125 * (v[index1 + 1]
                                                      + v[index1 + MNhalf + 1]));
              }
              else
              {
                f_out[index2 + MN + 1] += (0.5 * (v[index1]
                                                  + v[index1 + MNhalf]));
              }
            }
            else
            {
              if (i < Mhalf - 1)
              {
                f_out[index2 + MN + 1] += (0.28125 * (v[index1]
                                                      + v[index1 + MNhalf])
                                           + 0.09375 * (v[index1 + 1]
                                                        + v[index1 - Mhalf]
                                                        + v[index1 + MNhalf + 1]
                                                        + v[index1 + MNhalf - Mhalf])
                                           + 0.03125 * (v[index1 - Mhalf + 1]
                                                        + v[index1 + MNhalf - Mhalf + 1]));
              }
              else
              {
                f_out[index2 + MN + 1] += (0.375 * (v[index1]
                                                    + v[index1 + MNhalf])
                                           + 0.125 * (v[index1 - Mhalf]
                                                      + v[index1 + MNhalf - Mhalf]));
              }
            }
          }
          
          if (p < Phalf - 1)
          {
            if (j < Nhalf - 1)
            {
              if (i == 0)
              {
                f_out[index2 + MN + M] += (0.375 * (v[index1]
                                                    + v[index1 + MNhalf])
                                           + 0.125 * (v[index1 + Mhalf]
                                                      + v[index1 + MNhalf + Mhalf]));
              }
              else
              {
                f_out[index2 + MN + M] += (0.28125 * (v[index1]
                                                      + v[index1 + MNhalf])
                                           + 0.09375 * (v[index1 - 1]
                                                        + v[index1 + Mhalf]
                                                        + v[index1 + MNhalf - 1]
                                                        + v[index1 + MNhalf + Mhalf])
                                           + 0.03125 * (v[index1 + Mhalf - 1]
                                                        + v[index1 + MNhalf + Mhalf - 1]));
              }
            }
            else
            {
              if (i == 0)
              {
                f_out[index2 + MN + M] += (0.5 * (v[index1]
                                                  + v[index1 + MNhalf]));
              }
              else
              {
                f_out[index2 + MN + M] += (0.375 * (v[index1]
                                                    + v[index1 + MNhalf])
                                           + 0.125 * (v[index1 - 1]
                                                      + v[index1 + MNhalf - 1]));
              }
            }
          }
          
          if (p < Phalf - 1)
          {
            if (j < Nhalf - 1)
            {
              if (i < Mhalf - 1)
              {
                f_out[index2 + MN + M + 1] += (0.28125 * (v[index1]
                                                          + v[index1 + MNhalf])
                                               + 0.09375 * (v[index1 + 1]
                                                            + v[index1 + Mhalf]
                                                            + v[index1 + MNhalf + 1]
                                                            + v[index1 + MNhalf + Mhalf])
                                               + 0.03125 * (v[index1 + Mhalf + 1]
                                                            + v[index1 + MNhalf + Mhalf + 1]));
              }
              else
              {
                f_out[index2 + MN + M + 1] += (0.375 * (v[index1]
                                                        + v[index1 + MNhalf])
                                               + 0.125 * (v[index1 + Mhalf]
                                                          + v[index1 + MNhalf + Mhalf]));
              }
            }
            else
            {
              if (i < Mhalf - 1)
              {
                f_out[index2 + MN + M + 1] += (0.375 * (v[index1]
                                                        + v[index1 + MNhalf])
                                               + 0.125 * (v[index1 + 1]
                                                          + v[index1 + MNhalf + 1]));
              }
              else
              {
                f_out[index2 + MN + M + 1] += (0.5 * (v[index1]
                                                      + v[index1 + MNhalf]));
              }
            }
          }
        }
      }
    }
  }
  
  if (M % 2 == 1 && N % 2 == 0 && P % 2 == 1)
  {
    for (p = 0; p < Phalf; p++)
    {
      for (j = 0; j < Nhalf; j++)
      {
        for (i = 0; i < Mhalf; i++)
        {
          index1 = (p * Nhalf + j) * Mhalf + i;
          index2 = (2 * p * N + 2 * j) * M + 2 * i;
          if (j == 0)
          {
            f_out[index2] += (v[index1]);
          }
          else
          {
            f_out[index2] += (0.75 * v[index1]
                              + 0.25 * v[index1 - Mhalf]);
          }
          
          if (j == 0)
          {
            if (i < Mhalf - 1)
            {
              f_out[index2 + 1] += (0.5 * (v[index1]
                                           + v[index1 + 1]));
            }
          }
          else
          {
            if (i < Mhalf - 1)
            {
              f_out[index2 + 1] += (0.375 * (v[index1]
                                             + v[index1 + 1])
                                    + 0.125 * (v[index1 - Mhalf]
                                               + v[index1 - Mhalf + 1]));
            }
          }
          
          if (j < Nhalf - 1)
          {
            f_out[index2 + M] += (0.75 * v[index1]
                                  + 0.25 * v[index1 + Mhalf]);
          }
          else
          {
            f_out[index2 + M] += (v[index1]);
          }
          
          if (j < Nhalf - 1)
          {
            if (i < Mhalf - 1)
            {
              f_out[index2 + M + 1] += (0.375 * (v[index1]
                                                 + v[index1 + 1])
                                        + 0.125 * (v[index1 + Mhalf]
                                                   + v[index1 + Mhalf + 1]));
            }
          }
          else
          {
            if (i < Mhalf - 1)
            {
              f_out[index2 + M + 1] += (0.5 * (v[index1]
                                               + v[index1 + 1]));
            }
          }
          
          if (p < Phalf - 1)
          {
            if (j == 0)
            {
              f_out[index2 + MN] += (0.5 * (v[index1]
                                            + v[index1 + MNhalf]));
            }
            else
            {
              f_out[index2 + MN] += (0.375 * (v[index1]
                                              + v[index1 + MNhalf])
                                     + 0.125 * (v[index1 - Mhalf]
                                                + v[index1 + MNhalf - Mhalf]));
            }
          }
          
          if (p < Phalf - 1)
          {
            if (j == 0)
            {
              if (i < Mhalf - 1)
              {
                f_out[index2 + MN + 1] += (0.25 * (v[index1]
                                                   + v[index1 + 1]
                                                   + v[index1 + MNhalf]
                                                   + v[index1 + MNhalf + 1]));
              }
            }
            else
            {
              if (i < Mhalf - 1)
              {
                f_out[index2 + MN + 1] += (0.1875 * (v[index1]
                                                     + v[index1 + 1]
                                                     + v[index1 + MNhalf]
                                                     + v[index1 + MNhalf + 1])
                                           + 0.0625 * (v[index1 - Mhalf]
                                                       + v[index1 - Mhalf + 1]
                                                       + v[index1 + MNhalf - Mhalf]
                                                       + v[index1 + MNhalf - Mhalf + 1]));
              }
            }
          }
          
          if (p < Phalf - 1)
          {
            if (j < Nhalf - 1)
            {
              f_out[index2 + MN + M] += (0.375 * (v[index1]
                                                  + v[index1 + MNhalf])
                                         + 0.125 * (v[index1 + Mhalf]
                                                    + v[index1 + MNhalf + Mhalf]));
            }
            else
            {
              f_out[index2 + MN + M] += (0.5 * (v[index1]
                                                + v[index1 + MNhalf]));
            }
          }
          
          if (p < Phalf - 1)
          {
            if (j < Nhalf - 1)
            {
              if (i < Mhalf - 1)
              {
                f_out[index2 + MN + M + 1] += (0.1875 * (v[index1]
                                                         + v[index1 + 1]
                                                         + v[index1 + MNhalf]
                                                         + v[index1 + MNhalf + 1])
                                               + 0.0625 * (v[index1 + Mhalf]
                                                           + v[index1 + Mhalf + 1]
                                                           + v[index1 + MNhalf + Mhalf]
                                                           + v[index1 + MNhalf + Mhalf + 1]));
              }
            }
            else
            {
              if (i < Mhalf - 1)
              {
                f_out[index2 + MN + M + 1] += (0.25 * (v[index1]
                                                       + v[index1 + 1]
                                                       + v[index1 + MNhalf]
                                                       + v[index1 + MNhalf + 1]));
              }
            }
          }
        }
      }
    }
  }
  
  if (M % 2 == 0 && N % 2 == 1 && P % 2 == 1)
  {
    for (p = 0; p < Phalf; p++)
    {
      for (j = 0; j < Nhalf; j++)
      {
        for (i = 0; i < Mhalf; i++)
        {
          index1 = (p * Nhalf + j) * Mhalf + i;
          index2 = (2 * p * N + 2 * j) * M + 2 * i;
          if (i == 0)
          {
            f_out[index2] += (v[index1]);
          }
          else
          {
            f_out[index2] += (0.75 * v[index1]
                              + 0.25 * v[index1 - 1]);
          }
          
          if (i < Mhalf - 1)
          {
            f_out[index2 + 1] += (0.75 * v[index1]
                                  + 0.25 * v[index1 + 1]);
          }
          else
          {
            f_out[index2 + 1] += (v[index1]);
          }
          
          if (j < Nhalf - 1)
          {
            if (i == 0)
            {
              f_out[index2 + M] += (0.5 * (v[index1]
                                           + v[index1 + Mhalf]));
            }
            else
            {
              f_out[index2 + M] += (0.375 * (v[index1]
                                             + v[index1 + Mhalf])
                                    + 0.125 * (v[index1 - 1]
                                               + v[index1 + Mhalf - 1]));
            }
          }
          
          if (j < Nhalf - 1)
          {
            if (i < Mhalf - 1)
            {
              f_out[index2 + M + 1] += (0.375 * (v[index1]
                                                 + v[index1 + Mhalf])
                                        + 0.125 * (v[index1 + 1]
                                                   + v[index1 + Mhalf + 1]));
            }
            else
            {
              f_out[index2 + M + 1] += (0.5 * (v[index1]
                                               + v[index1 + Mhalf]));
            }
          }
          
          if (p < Phalf - 1)
          {
            if (i == 0)
            {
              f_out[index2 + MN] += (0.5 * (v[index1]
                                            + v[index1 + MNhalf]));
            }
            else
            {
              f_out[index2 + MN] += (0.375 * (v[index1]
                                              + v[index1 + MNhalf])
                                     + 0.125 * (v[index1 - 1]
                                                + v[index1 + MNhalf - 1]));
            }
          }
          
          if (p < Phalf - 1)
          {
            if (i < Mhalf - 1)
            {
              f_out[index2 + MN + 1] += (0.375 * (v[index1]
                                                  + v[index1 + MNhalf])
                                         + 0.125 * (v[index1 + 1]
                                                    + v[index1 + MNhalf + 1]));
            }
            else
            {
              f_out[index2 + MN + 1] += (0.5 * (v[index1]
                                                + v[index1 + MNhalf]));
            }
          }
          
          if (p < Phalf - 1)
          {
            if (j < Nhalf - 1)
            {
              if (i == 0)
              {
                f_out[index2 + MN + M] += (0.25 * (v[index1]
                                                   + v[index1 + Mhalf]
                                                   + v[index1 + MNhalf]
                                                   + v[index1 + MNhalf + Mhalf]));
              }
              else
              {
                f_out[index2 + MN + M] += (0.1875 * (v[index1]
                                                     + v[index1 + Mhalf]
                                                     + v[index1 + MNhalf]
                                                     + v[index1 + MNhalf + Mhalf])
                                           + 0.0625 * (v[index1 - 1]
                                                       + v[index1 + Mhalf - 1]
                                                       + v[index1 + MNhalf - 1]
                                                       + v[index1 + MNhalf + Mhalf - 1]));
              }
            }
          }
          
          if (p < Phalf - 1)
          {
            if (j < Nhalf - 1)
            {
              if (i < Mhalf - 1)
              {
                f_out[index2 + MN + M + 1] += (0.1875 * (v[index1]
                                                         + v[index1 + Mhalf]
                                                         + v[index1 + MNhalf]
                                                         + v[index1 + MNhalf + Mhalf])
                                               + 0.0625 * (v[index1 + 1]
                                                           + v[index1 + Mhalf + 1]
                                                           + v[index1 + MNhalf + 1]
                                                           + v[index1 + MNhalf + Mhalf + 1]));
              }
              else
              {
                f_out[index2 + MN + M + 1] += (0.25 * (v[index1]
                                                       + v[index1 + Mhalf]
                                                       + v[index1 + MNhalf]
                                                       + v[index1 + MNhalf + Mhalf]));
              }
            }
          }
        }
      }
    }
  }
  
  if (M % 2 == 1 && N % 2 == 1 && P % 2 == 1)
  {
    for (p = 0; p < Phalf; p++)
    {
      for (j = 0; j < Nhalf; j++)
      {
        for (i = 0; i < Mhalf; i++)
        {
          index1 = (p * Nhalf + j) * Mhalf + i;
          index2 = (2 * p * N + 2 * j) * M + 2 * i;
          f_out[index2] += (v[index1]);
          
          if (i < Mhalf - 1)
          {
            f_out[index2 + 1] += (0.5 * (v[index1]
                                         + v[index1 + 1]));
          }
          
          if (j < Nhalf - 1)
          {
            f_out[index2 + M] += (0.5 * (v[index1]
                                         + v[index1 + Mhalf]));
          }
          
          if (j < Nhalf - 1)
          {
            if (i < Mhalf - 1)
            {
              f_out[index2 + M + 1] += (0.25 * (v[index1]
                                                + v[index1 + 1]
                                                + v[index1 + Mhalf]
                                                + v[index1 + Mhalf + 1]));
            }
          }
          
          if (p < Phalf - 1)
          {
            f_out[index2 + MN] += (0.5 * (v[index1]
                                          + v[index1 + MNhalf]));
          }
          
          if (p < Phalf - 1)
          {
            if (i < Mhalf - 1)
            {
              f_out[index2 + MN + 1] += (0.25 * (v[index1]
                                                 + v[index1 + 1]
                                                 + v[index1 + MNhalf]
                                                 + v[index1 + MNhalf + 1]));
            }
          }
          
          if (p < Phalf - 1)
          {
            if (j < Nhalf - 1)
            {
              f_out[index2 + MN + M] += (0.25 * (v[index1]
                                                 + v[index1 + Mhalf]
                                                 + v[index1 + MNhalf]
                                                 + v[index1 + MNhalf + Mhalf]));
            }
          }
          
          if (p < Phalf - 1)
          {
            if (j < Nhalf - 1)
            {
              if (i < Mhalf - 1)
              {
                f_out[index2 + MN + M + 1] += (0.125 * (v[index1]
                                                        + v[index1 + 1]
                                                        + v[index1 + Mhalf]
                                                        + v[index1 + Mhalf + 1]
                                                        + v[index1 + MNhalf]
                                                        + v[index1 + MNhalf + 1]
                                                        + v[index1 + MNhalf + Mhalf]
                                                        + v[index1 + MNhalf + Mhalf + 1]));
              }
            }
          }
        }
      }
    }
  }
}


/* Recursive multigrid function.*/
static void
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
  
  /* Post-smoothing. */
  for (k = 0; k < n2; k++)
    gauss_seidel3D(f_out, d, M, N, P);
  
  mxFree(r);
  mxFree(r_downsampled);
  mxFree(v);
}


/* It is assumed that f_out is initialized to zero when called. */
static void
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


static void
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
  double *mask = NULL;
  double *f_out;
  double mu;
  int number_of_iterations;
  int dim;
  int argno;
  
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

  /* Then an optional mask. */
  argno = 1;
  if (mxGetNumberOfElements(prhs[1]) > 1)
  {
    if (!mxIsNumeric(prhs [1]) || mxIsComplex(prhs [1])
	|| mxIsSparse(prhs [1]) || !mxIsDouble(prhs [1]))
    {
      mexErrMsgTxt("mask is expected to be a numeric array.");
    }

    if (mxGetNumberOfDimensions(prhs[1]) != dim
	|| mxGetDimensions(prhs[1])[0] != M
	|| mxGetDimensions(prhs[1])[1] != N
	|| (dim > 2 && mxGetDimensions(prhs[1])[2] != P))
    {
      mexErrMsgTxt("g and mask have incompatible sizes.");
    }
    
    mask = mxGetPr(prhs[1]);
    argno++;
  }
  
  /* Next two scalars. */
  if (argno >= nrhs)
    mu = 0.0;
  else
  {
    if (!mxIsNumeric(prhs[argno]) || mxIsComplex(prhs[argno])
	|| mxIsSparse(prhs[argno]) || !mxIsDouble(prhs[argno])
	|| mxGetNumberOfElements(prhs[argno]) != 1)
    {
      mexErrMsgTxt("mu is expected to be a scalar.");
    }
    mu = mxGetScalar(prhs[argno]);
  }
  argno++;

  if (argno >= nrhs)
  {
    number_of_iterations = 2;
  }
  else
  {
    if (!mxIsNumeric(prhs[argno]) || mxIsComplex(prhs[argno])
	|| mxIsSparse(prhs[argno]) || !mxIsDouble(prhs[argno])
	|| mxGetNumberOfElements(prhs[argno]) != 1)
    {
      mexErrMsgTxt("N is expected to be a scalar.");
    }
    number_of_iterations = (int) mxGetScalar(prhs[argno]);
    if (number_of_iterations < 0
	|| (double) number_of_iterations != mxGetScalar(prhs[argno]))
    {
      mexErrMsgTxt("N is expected to be a positive integer.");
    }
  }
  
  plhs[0] = mxCreateNumericArray(dim, mxGetDimensions(prhs[0]),
				 mxDOUBLE_CLASS, mxREAL);
  f_out = mxGetPr(plhs[0]);
  
  if (dim == 2)
    antigradient2D(g, mask, mu, number_of_iterations, M, N, f_out);
  else
    antigradient3D(g, mu, number_of_iterations, M, N, P, f_out);
}

