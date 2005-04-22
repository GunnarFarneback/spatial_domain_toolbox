#include "mex.h"

#define DEBUG 0

/*
 *  normconv.c
 *
 *  Perform normalized convolution. Only real images and filters are
 *  supported, but at arbitrary dimensionality.
 *
 *  Call:
 *  result = normconv(signal, certainty, basis, applicability)
 *  or
 *  result = normconv(signal, certainty, basis, applicability,
 *                    region_of_interest)
 *
 *  signal             - signal values
 *  certainty          - signal certainty (pointwise)
 *  basis              - subspace basis
 *  applicability      - applicability function for the basis
 *  region_of_interest - where to compute the normalized convolution
 *  result             - local basis coefficients
 *
 *  Formats: (Also see note 2 for special cases.)
 *  signal is an n-dimensional array.
 *  certainty is the same size as signal.
 *  basis is an (n+1)-dimensional array. The first n dimensions correspond
 *          to the signal dimensions. The size of the last dimension gives
 *          the number of basis functions.
 *  applicability is an n-dimensional array, with the same size as the first
 *          n dimensions of basis.
 *  region_of_interest is either an nx2 matrix, where each row gives start
 *          and end indices for each dimension, or a sparse matrix of the
 *          same size as signal.
 *  result is an (n+1)-dimensional array. The size of the first n dimensions
 *          are the same as for signal, unless region_of_interest is
 *          specified. The size of the last dimension equals the number of
 *          basis functions.
 *
 *  Note 1: Only double, nonsparse arrays are currently supported.
 *
 *  Note 2: Trailing singleton dimensions cannot exist in matlab 5.
 *          If there is a mismatch in the number of dimensions of the
 *          parameters it will be assumed that there are additional
 *          singleton dimensions, if that turns out to make sense.
 *          Particularly, in the case of a single basis function, basis
 *          will normally be n-dimensional instead of (n+1)-dimensional.
 *          The same goes for result.
 *
 *          In the case of column vectors, the second, singleton, dimension
 *          will be ignored, although it is included by matlab 5.
 *
 *  Note 3: The special cases of 1-3 dimensions and/or a single basis
 *          function have been optimized for speed. To use the general
 *          algorithm also in these cases, add as a final parameter in
 *          the call the string 'general'. This will certainly be slower
 *          but can be used to verify that the optimized code works as
 *          intended. The results should not differ by more than
 *          numerical deviations. The general algorithm does not work with
 *          the sparse form of region_of_interest.
 *
 *  Note 4: The above mentioned sparse region of interest has not been
 *          implemented, as it turned out that matlab only supports
 *          sparse matrices in two dimensions.
 *
 *  Author: Gunnar Farnebäck
 *          Computer Vision Laboratory
 *          Linköping University, Sweden
 *          gf@isy.liu.se
 */

/* One signal dimension, one basis function. */
static void
normconv1_1(const double *signal,
	    const double *certainty,
	    const double *basis,
	    const double *applicability,
	    double *result,
	    const int *signal_dimensions,
	    const int *model_dimensions,
	    const int *result_dimensions,
	    const int *start_indices,
	    const int *stop_indices)
{
    double A;
    double b;
    double x;
    
    int result_index;
    int model_index;
    int signal_index;
    int displacement;
    
    int resultindex;

    double p;

    /* Initialize the 2 result indices and the model indices. */
    result_index = start_indices[0];
    model_index = 0;
    
    /* The center of the basis function is supposed
     * to be local origin. Compute the corresponding offset.
     * Note that this is integer division.
     */
    displacement = (model_dimensions[0] - 1) / 2;

    /* Loop over the signal dimensions */
    for (result_index = start_indices[0];
	 result_index <= stop_indices[0];
	 result_index++)
    {
	
	/* Compute inner products of the basis function with itself
	 * (A) and between basis function and signal (b).
	 */
	A = 0.0;
	b = 0.0;
	
	/* Loop over the dimensions for the basis functions. */
	for (model_index = 0;
	     model_index < model_dimensions[0];
	     model_index++)
	{
	    /* Compute the signal index corresponding to the current
	     * result index and model index.
	     */
	    signal_index = result_index + model_index - displacement;
	    /* Check if we are outside the signal boundary. It is
	     * implied that the certainty is zero then.
	     */
	    if (signal_index < 0 || signal_index >= signal_dimensions[0])
		continue;
	    
	    p = certainty[signal_index];
	    p *= applicability[model_index];
	    p *= basis[model_index];
	    A += p*basis[model_index];
	    b += p*signal[signal_index];
	}
	x = b / A;
	resultindex = result_index - start_indices[0];
	result[resultindex] = x;
    }
}

/* Two signal dimensions, one basis function. */
static void
normconv2_1(const double *signal,
	    const double *certainty,
	    const double *basis,
	    const double *applicability,
	    double *result,
	    const int *signal_dimensions,
	    const int *model_dimensions,
	    const int *result_dimensions,
	    const int *start_indices,
	    const int *stop_indices)
{
    double A;
    double b;
    double x;
    
    int result_indices[2];
    int model_indices[2];
    int signal_indices[2];
    int displacements[2];
    
    int signalindex;
    int modelindex;
    int resultindex;

    double p;

    int i;

    /* Initialize the 2 result indices and the model indices. */
    for (i=0; i<2; i++)
    {
	result_indices[i] = start_indices[i];
	model_indices[i] = 0;
    }
    
    /* The center of the basis function is supposed to be local
     * origin. Compute the corresponding offsets.
     */
    for (i=0; i<2; i++)
    {
	/* Note that this is integer division. */
	displacements[i] = (model_dimensions[i] - 1) / 2;
    }

    /* Loop over the signal dimensions */
    for (result_indices[1] = start_indices[1];
	 result_indices[1] <= stop_indices[1];
	 result_indices[1]++)
    {
	for (result_indices[0] = start_indices[0];
	     result_indices[0] <= stop_indices[0];
	     result_indices[0]++)
	{
	    
	    /* Compute inner products of the basis function with
	     * itself (A) and between basis function and signal (b).
	     */
	    A = 0.0;
	    b = 0.0;
	    
	    /* Loop over the dimensions for the basis functions. */
	    for (model_indices[1] = 0;
		 model_indices[1] < model_dimensions[1];
		 model_indices[1]++)
	    {
		/* Compute the signal index corresponding to the
		 * current result index and model index.
		 */
		signal_indices[1] = (result_indices[1]
				     + model_indices[1]
				     - displacements[1]);
		/* Check if we are outside the signal boundary. It is
		 * implied that the certainty is zero then.
		 */
		if (signal_indices[1] < 0
		    || signal_indices[1] >= signal_dimensions[1])
		    continue;

		for (model_indices[0] = 0;
		     model_indices[0] < model_dimensions[0];
		     model_indices[0]++)
		{
		    signal_indices[0] = (result_indices[0]
					 + model_indices[0]
					 - displacements[0]);
		    if (signal_indices[0] < 0
			|| signal_indices[0] >= signal_dimensions[0])
			continue;
		    
		    signalindex = (signal_indices[1] * signal_dimensions[0]
				   + signal_indices[0]);
		    modelindex = (model_indices[1] * model_dimensions[0]
				  + model_indices[0]);
		    p = certainty[signalindex];
		    p *= applicability[modelindex];
		    p *= basis[modelindex];
		    A += p * basis[modelindex];
		    b += p * signal[signalindex];
		}
	    }
	    x = b / A;

	    resultindex = ((result_indices[1] - start_indices[1])
			   * result_dimensions[0]
			   + (result_indices[0] - start_indices[0]));

	    result[resultindex] = x;
	}
    }
}

/* Three signal dimensions, one basis function. */
static void
normconv3_1(const double *signal,
	    const double *certainty,
	    const double *basis,
	    const double *applicability,
	    double *result,
	    const int *signal_dimensions,
	    const int *model_dimensions,
	    const int *result_dimensions,
	    const int *start_indices,
	    const int *stop_indices)
{
    double A;
    double b;
    double x;
    
    int result_indices[3];
    int model_indices[3];
    int signal_indices[3];
    int displacements[3];
    
    int signalindex;
    int modelindex;
    int resultindex;

    double p;

    int i;

    /* Initialize the 3 result indices and the model indices. */
    for (i=0; i<3; i++)
    {
	result_indices[i] = start_indices[i];
	model_indices[i] = 0;
    }
    
    /* The center of the basis function is supposed to be local
     * origin. Compute the corresponding offsets.
     */
    for (i=0; i<3; i++)
    {
	/* Note that this is integer division. */
	displacements[i] = (model_dimensions[i] - 1) / 2;
    }

    /* Loop over the signal dimensions */
    for (result_indices[2] = start_indices[2];
	 result_indices[2] <= stop_indices[2];
	 result_indices[2]++)
    {
	for (result_indices[1] = start_indices[1];
	     result_indices[1] <= stop_indices[1];
	     result_indices[1]++)
	{
	    for (result_indices[0] = start_indices[0];
		 result_indices[0] <= stop_indices[0];
		 result_indices[0]++)
	    {

		/* Compute inner products of the basis function with
		 * itself (A) and between basis function and signal
		 * (b).
		 */
		A = 0.0;
		b = 0.0;
		
		/* Loop over the dimensions for the basis functions. */
		for (model_indices[2] = 0;
		     model_indices[2] < model_dimensions[2];
		     model_indices[2]++)
		{
		    /* Compute the signal index corresponding to the
		     * current result index and model index.
		     */
		    signal_indices[2] = (result_indices[2]
					 + model_indices[2]
					 - displacements[2]);
		    /* Check if we are outside the signal boundary. It
		     * is implied that the certainty is zero then.
		     */
		    if (signal_indices[2] < 0
			|| signal_indices[2] >= signal_dimensions[2])
			continue;

		    for (model_indices[1] = 0;
			 model_indices[1] < model_dimensions[1];
			 model_indices[1]++)
		    {
			signal_indices[1] = (result_indices[1]
					     + model_indices[1]
					     - displacements[1]);
			if (signal_indices[1] < 0
			    || signal_indices[1] >= signal_dimensions[1])
			    continue;

			for (model_indices[0] = 0;
			     model_indices[0] < model_dimensions[0];
			     model_indices[0]++)
			{
			    signal_indices[0] = (result_indices[0]
						 + model_indices[0]
						 - displacements[0]);
			    if (signal_indices[0] < 0
				|| signal_indices[0] >= signal_dimensions[0])
				continue;
			    
			    signalindex = ((signal_indices[2]
					    * signal_dimensions[1]
					    + signal_indices[1])
					   * signal_dimensions[0]
					   + signal_indices[0]);
			    modelindex = ((model_indices[2]
					   * model_dimensions[1]
					   + model_indices[1])
					  * model_dimensions[0]
					  + model_indices[0]);
			    p = certainty[signalindex];
			    p *= applicability[modelindex];
			    p *= basis[modelindex];
			    A += p*basis[modelindex];
			    b += p*signal[signalindex];
			}
		    }
		}
		x = b / A;

		resultindex = (((result_indices[2] - start_indices[2])
				* result_dimensions[1]
				+ (result_indices[1] - start_indices[1]))
			       * result_dimensions[0]
			       + (result_indices[0] - start_indices[0]));
		
		result[resultindex] = x;
	    }
	}
    }
}

/* One signal dimension, multiple basis functions. */
static void
normconv1(const double *signal,
	  const double *certainty,
	  const double *basis,
	  const double *applicability,
	  double *result,
	  const int *signal_dimensions,
	  const int *model_dimensions,
	  const int *result_dimensions,
	  int basis_size,
	  const int *start_indices,
	  const int *stop_indices)
{
    mxArray *arguments[2];
    mxArray *A_matrix;
    mxArray *b_vector;
    mxArray *x_vector;
    double *A;
    double *b;
    double *x;

    int result_index;
    int model_index;
    int signal_index;
    int displacement;
    
    int resultindex;

    int modeldimprod;           /* Product of the first n dimensions. */
    int resultdimprod;          /* Product of the first n dimensions. */
    
    double p;
    double ip;

    int k, k1, k2;

    A_matrix = mxCreateDoubleMatrix(basis_size, basis_size, mxREAL);
    b_vector = mxCreateDoubleMatrix(basis_size, 1, mxREAL);
    A = mxGetPr(A_matrix);
    b = mxGetPr(b_vector);

    arguments[0] = A_matrix;
    arguments[1] = b_vector;

    /* Initialize the first result index and the model index. */
    result_index = start_indices[0];
    model_index = 0;
    
    /* The centers of the basis functions are supposed to be local
     * origins. Compute the corresponding offset. Note that this is
     * integer division.
     */
    displacement = (model_dimensions[0] - 1) / 2;

    /* Compute the product of the first n dimensions of the model and
     * result arrays respectively.
     */
    modeldimprod = model_dimensions[0];
    resultdimprod = result_dimensions[0];

    /* Loop over the signal dimensions */
    for (result_index = start_indices[0];
	 result_index <= stop_indices[0];
	 result_index++)
    {
	
	/* Double loop over the basis functions to compute inner
	 * products. Inner products between basis functions and signal
	 * are computed when k2 == basis_size.
	 */
	for (k1=0; k1<basis_size; k1++)
	{
	    for (k2=k1; k2<=basis_size; k2++)
	    {
		/* Reset the accumulated inner product. */
		ip = 0.0;
		
		/* Loop over the dimensions for the basis functions. */
		for (model_index = 0;
		     model_index < model_dimensions[0];
		     model_index++)
		{
		    /* Compute the signal index corresponding to the
		     * current result index and model index.
		     */
		    signal_index = (result_index
				    + model_index
				    - displacement);
		    /* Check if we are outside the signal boundary. It
		     * is implied that the certainty is zero then.
		     */
		    if (signal_index < 0
			|| signal_index >= signal_dimensions[0])
			continue;
		    
		    if (k2 == basis_size)
			p = signal[signal_index];
		    else
			p = basis[model_index + modeldimprod * k2];

		    p *= certainty[signal_index];
		    p *= applicability[model_index];
		    p *= basis[model_index + modeldimprod * k1];
		    ip += p;
		}
		
		if (k2 == basis_size)
		    b[k1] = ip;
		else
		{
		    A[k1 + k2 * basis_size] = ip;
		    A[k2 + k1 * basis_size] = ip;
		}
	    }
	}
	
	mexCallMATLAB(1, &x_vector, 2, arguments, "\\");
	x = mxGetPr(x_vector);
	
	resultindex = (result_index - start_indices[0]);
	
	for (k=0; k<basis_size; k++)
	    result[resultindex + resultdimprod * k] = x[k];

	mxDestroyArray(x_vector);
    }
}

/* Two signal dimensions, multiple basis functions. */
static void
normconv2(const double *signal,
	  const double *certainty,
	  const double *basis,
	  const double *applicability,
	  double *result,
	  const int *signal_dimensions,
	  const int *model_dimensions,
	  const int *result_dimensions,
	  int basis_size,
	  const int *start_indices,
	  const int *stop_indices)
{
    mxArray *arguments[2];
    mxArray *A_matrix;
    mxArray *b_vector;
    mxArray *x_vector;
    double *A;
    double *b;
    double *x;

    int result_indices[2];
    int model_indices[2];
    int signal_indices[2];
    int displacements[2];
    
    int signalindex;
    int modelindex;
    int resultindex;

    int modeldimprod;           /* Product of the first n dimensions. */
    int resultdimprod;          /* Product of the first n dimensions. */
    
    double p;
    double ip;

    int i, k, k1, k2;

    A_matrix = mxCreateDoubleMatrix(basis_size, basis_size, mxREAL);
    b_vector = mxCreateDoubleMatrix(basis_size, 1, mxREAL);
    A = mxGetPr(A_matrix);
    b = mxGetPr(b_vector);

    arguments[0] = A_matrix;
    arguments[1] = b_vector;

    /* Initialize the first 2 result indices and the model indices. */
    for (i=0; i<2; i++)
    {
	result_indices[i] = start_indices[i];
	model_indices[i] = 0;
    }
    
    /* The centers of the basis functions are supposed to be local
     * origins. Compute the corresponding offsets.
     */
    for (i=0; i<2; i++)
    {
	/* Note that this is integer division. */
	displacements[i] = (model_dimensions[i] - 1) / 2;
    }

    /* Compute the product of the first n dimensions of the model and
     * result arrays respectively.
     */
    modeldimprod = 1;
    resultdimprod = 1;
    for (i=0; i<2; i++)
    {
	modeldimprod *= model_dimensions[i];
	resultdimprod *= result_dimensions[i];
    }

    for (result_indices[1] = start_indices[1];
	 result_indices[1] <= stop_indices[1];
	 result_indices[1]++)
    {
	for (result_indices[0] = start_indices[0];
	     result_indices[0] <= stop_indices[0];
	     result_indices[0]++)
	{
	    
	    /* Double loop over the basis functions to compute inner
	     * products. Inner products between basis functions and
	     * signal are computed when k2 == basis_size.
	     */
	    for (k1=0; k1<basis_size; k1++)
	    {
		for (k2=k1; k2<=basis_size; k2++)
		{
		    /* Reset the accumulated inner product. */
		    ip = 0.0;
		    
		    /* Loop over the dimensions for the basis functions. */
		    for (model_indices[1] = 0;
			 model_indices[1] < model_dimensions[1];
			 model_indices[1]++)
		    {
			/* Compute the signal index corresponding to
			 * the current result index and model index.
			 */
			signal_indices[1] = (result_indices[1]
					     + model_indices[1]
					     - displacements[1]);
			/* Check if we are outside the signal
			 * boundary. It is implied that the certainty
			 * is zero then.
			 */
			if (signal_indices[1] < 0
			    || signal_indices[1] >= signal_dimensions[1])
			    continue;

			for (model_indices[0] = 0;
			     model_indices[0] < model_dimensions[0];
			     model_indices[0]++)
			{
			    signal_indices[0] = (result_indices[0]
						 + model_indices[0]
						 - displacements[0]);
			    if (signal_indices[0] < 0
				|| signal_indices[0] >= signal_dimensions[0])
				continue;
			    
			    signalindex = (signal_indices[1]
					   * signal_dimensions[0]
					   + signal_indices[0]);
			    modelindex = (model_indices[1]
					  * model_dimensions[0]
					  + model_indices[0]);
			    if (k2 == basis_size)
				p = signal[signalindex];
			    else
				p = basis[modelindex + modeldimprod * k2];

			    p *= certainty[signalindex];
			    p *= applicability[modelindex];
			    p *= basis[modelindex + modeldimprod * k1];
			    ip += p;
			}
		    }
		    
		    if (k2 == basis_size)
			b[k1] = ip;
		    else
		    {
			A[k1 + k2 * basis_size] = ip;
			A[k2 + k1 * basis_size] = ip;
		    }
		}
	    }
	    
	    mexCallMATLAB(1, &x_vector, 2, arguments, "\\");
	    x = mxGetPr(x_vector);
	    
	    resultindex = ((result_indices[1] - start_indices[1])
			   * result_dimensions[0]
			   + (result_indices[0] - start_indices[0]));
	    
	    for (k=0; k<basis_size; k++)
		result[resultindex + resultdimprod * k] = x[k];

	    mxDestroyArray(x_vector);
	}
    }
}

/* Three signal dimensions, multiple basis functions. */
static void
normconv3(const double *signal,
	  const double *certainty,
	  const double *basis,
	  const double *applicability,
	  double *result,
	  const int *signal_dimensions,
	  const int *model_dimensions,
	  const int *result_dimensions,
	  int basis_size,
	  const int *start_indices,
	  const int *stop_indices)
{
    mxArray *arguments[2];
    mxArray *A_matrix;
    mxArray *b_vector;
    mxArray *x_vector;
    double *A;
    double *b;
    double *x;

    int result_indices[3];
    int model_indices[3];
    int signal_indices[3];
    int displacements[3];
    
    int signalindex;
    int modelindex;
    int resultindex;

    int modeldimprod;           /* Product of the first n dimensions. */
    int resultdimprod;          /* Product of the first n dimensions. */
    
    double p;
    double ip;

    int i, k, k1, k2;

    A_matrix = mxCreateDoubleMatrix(basis_size, basis_size, mxREAL);
    b_vector = mxCreateDoubleMatrix(basis_size, 1, mxREAL);
    A = mxGetPr(A_matrix);
    b = mxGetPr(b_vector);

    arguments[0] = A_matrix;
    arguments[1] = b_vector;

    /* Initialize the first 3 result indices and the model indices. */
    for (i=0; i<3; i++)
    {
	result_indices[i] = start_indices[i];
	model_indices[i] = 0;
    }
    
    /* The centers of the basis functions are supposed to be local
     * origins. Compute the corresponding offsets.
     */
    for (i=0; i<3; i++)
    {
	/* Note that this is integer division. */
	displacements[i] = (model_dimensions[i] - 1) / 2;
    }

    /* Compute the product of the first n dimensions of the model and
     * result arrays respectively.
     */
    modeldimprod = 1;
    resultdimprod = 1;
    for (i=0; i<3; i++)
    {
	modeldimprod *= model_dimensions[i];
	resultdimprod *= result_dimensions[i];
    }

    /* Loop over the signal dimensions */
    for (result_indices[2] = start_indices[2];
	 result_indices[2] <= stop_indices[2];
	 result_indices[2]++)
    {
	for (result_indices[1] = start_indices[1];
	     result_indices[1] <= stop_indices[1];
	     result_indices[1]++)
	{
	    for (result_indices[0] = start_indices[0];
		 result_indices[0] <= stop_indices[0];
		 result_indices[0]++)
	    {

		/* Double loop over the basis functions to compute
		 * inner products. Inner products between basis
		 * functions and signal are computed when k2 ==
		 * basis_size.
		 */
		for (k1=0; k1<basis_size; k1++)
		{
		    for (k2=k1; k2<=basis_size; k2++)
		    {
			/* Reset the accumulated inner product. */
			ip = 0.0;

			/* Loop over the dimensions for the basis functions. */
			for (model_indices[2] = 0;
			     model_indices[2] < model_dimensions[2];
			     model_indices[2]++)
			{
			    /* Compute the signal index corresponding
			     * to the current result index and model
			     * index.
			     */
			    signal_indices[2] = (result_indices[2]
						 + model_indices[2]
						 - displacements[2]);
			    /* Check if we are outside the signal
			     * boundary. It is implied that the
			     * certainty is zero then.
			     */
			    if (signal_indices[2] < 0
				|| signal_indices[2] >= signal_dimensions[2])
				continue;

			    for (model_indices[1] = 0;
				 model_indices[1] < model_dimensions[1];
				 model_indices[1]++)
			    {
				signal_indices[1] = (result_indices[1]
						     + model_indices[1]
						     - displacements[1]);
				if (signal_indices[1] < 0
				    || (signal_indices[1]
					>= signal_dimensions[1]))
				    continue;

				for (model_indices[0] = 0;
				     model_indices[0] < model_dimensions[0];
				     model_indices[0]++)
				{
				    signal_indices[0] = (result_indices[0]
							 + model_indices[0]
							 - displacements[0]);
				    if (signal_indices[0] < 0
					|| (signal_indices[0]
					    >= signal_dimensions[0]))
					continue;

				    signalindex = ((signal_indices[2]
						    * signal_dimensions[1]
						    + signal_indices[1])
						   * signal_dimensions[0]
						   + signal_indices[0]);
				    modelindex = ((model_indices[2]
						   * model_dimensions[1]
						   + model_indices[1])
						  * model_dimensions[0]
						  + model_indices[0]);
				    if (k2 == basis_size)
					p = signal[signalindex];
				    else
					p = basis[(modelindex
						   + modeldimprod * k2)];

				    p *= certainty[signalindex];
				    p *= applicability[modelindex];
				    p *= basis[modelindex + modeldimprod * k1];
				    ip += p;
				}
			    }
			}

			if (k2 == basis_size)
			    b[k1] = ip;
			else
			{
			    A[k1 + k2 * basis_size] = ip;
			    A[k2 + k1 * basis_size] = ip;
			}
		    }
		}
		
		mexCallMATLAB(1, &x_vector, 2, arguments, "\\");
		x = mxGetPr(x_vector);

		resultindex = (((result_indices[2] - start_indices[2])
				* result_dimensions[1]
				+ (result_indices[1] - start_indices[1]))
			       * result_dimensions[0]
			       + (result_indices[0] - start_indices[0]));
		
		for (k=0; k<basis_size; k++)
		    result[resultindex + resultdimprod * k] = x[k];

		mxDestroyArray(x_vector);
	    }
	}
    }
}


static int
calc_single_subscript(const int *dimensions, const int *indices, int dim)
{
    int i;
    int sub = 0;
    for (i=dim-1; i>=0; i--)
    {
	sub += indices[i];
	if (i > 0)
	    sub *= dimensions[i-1];
    }
    return sub;
}

static int
calc_single_subscript2(const int *dimensions, const int *indices,
		       const int *start_indices, int dim)
{
    int i;
    int sub = 0;
    for (i=dim-1; i>=0; i--)
    {
	sub += (indices[i] - start_indices[i]);
	if (i > 0)
	    sub *= dimensions[i-1];
    }
    return sub;
}

/* General case, no optimization. */
static void
normconv(const double *signal,
	 const double *certainty,
	 const double *basis,
	 const double *applicability,
	 double *result,
	 int dimensionality,
	 const int *signal_dimensions,
	 const int *model_dimensions,
	 const int *result_dimensions,
	 int basis_size,
	 const int *start_indices,
	 const int *stop_indices)
{
    
    mxArray *arguments[2];
    mxArray *A_matrix;
    mxArray *b_vector;
    mxArray *x_vector;
    double *A;
    double *b;
    double *x;

    int *result_indices;
    int *model_indices;
    int *signal_indices;
    int *displacements;

    int signalindex;
    int modelindex;
    int resultindex;

    int modeldimprod;           /* Product of the first n dimensions. */
    int resultdimprod;          /* Product of the first n dimensions. */
    
    double p;
    double ip;

    int i, k, k1, k2;

    int outside;
    
    A_matrix = mxCreateDoubleMatrix(basis_size, basis_size, mxREAL);
    b_vector = mxCreateDoubleMatrix(basis_size, 1, mxREAL);
    A = mxGetPr(A_matrix);
    b = mxGetPr(b_vector);

    arguments[0] = A_matrix;
    arguments[1] = b_vector;

    result_indices = (int *)mxCalloc(dimensionality, sizeof(int));
    model_indices  = (int *)mxCalloc(dimensionality, sizeof(int));
    signal_indices = (int *)mxCalloc(dimensionality, sizeof(int));
    displacements  = (int *)mxCalloc(dimensionality, sizeof(int));

    /* Initialize the first n result indices. Note that the model
     * indices already are initialized to zero by the call to
     * mxCalloc.
     */
    for (i=0; i<dimensionality; i++)
	result_indices[i] = start_indices[i];

    /* The centers of the basis functions are supposed to be local
     * origins. Compute the corresponding offsets.
     */
    for (i=0; i<dimensionality; i++)
    {
	/* Note that this is integer division. */
	displacements[i] = (model_dimensions[i] - 1) / 2;
    }

    /* Compute the product of the first n dimensions of the model and
     * result arrays respectively.
     */
    modeldimprod = 1;
    resultdimprod = 1;
    for (i=0; i<dimensionality; i++)
    {
	modeldimprod *= model_dimensions[i];
	resultdimprod *= result_dimensions[i];
    }

    /* Loop over the signal dimensions */
    while (1)
    {
	/* (The indices are incremented at the end of the loop.) */
	
#if DEBUG
	for (i=0; i<dimensionality; i++)
	    mexPrintf("%3d ", result_indices[i]);
	mexPrintf("\n");
#endif

	/* Double loop over the basis functions to compute inner
	 * products. Inner products between basis functions and signal
	 * are computed when k2 == basis_size.
	 */
	for (k1=0; k1<basis_size; k1++)
	{
	    for (k2=k1; k2<=basis_size; k2++)
	    {
		/* Reset the accumulated inner product. */		
		ip = 0.0;

		/* Loop over the dimensions for the basis functions. */
		while (1)
		{
#if DEBUG
		    mexPrintf("Model:");
		    for (i=0; i<dimensionality; i++)
			mexPrintf("%3d ", model_indices[i]);
		    mexPrintf("\n");
#endif

		    /* Compute the signal indices corresponding to the
		     * current result indices and model indices.
		     */
		    for (i=0; i<dimensionality; i++)
		    {
			signal_indices[i] = result_indices[i]+
			    model_indices[i] - displacements[i];
		    }

		    /* Check if we are outside the signal boundary. It
		     * is implied that the certainty is zero then.
		     */
		    outside = 0;
		    for (i=0; i<dimensionality; i++)
		    {
			if (signal_indices[i] < 0
			    || signal_indices[i] >= signal_dimensions[i])
			{
			    outside = 1;
			    break;
			}
		    }
		    if (!outside)
		    {
			signalindex = calc_single_subscript(signal_dimensions,
							    signal_indices,
							    dimensionality);
			modelindex = calc_single_subscript(model_dimensions,
							   model_indices,
							   dimensionality);
			if (k2 == basis_size)
			    p = signal[signalindex];
			else
			    p = basis[modelindex + modeldimprod * k2];

			p *= certainty[signalindex];
			p *= applicability[modelindex];
			p *= basis[modelindex + modeldimprod * k1];
			ip += p;
			
		    }

		    /* Increment the indices. */
		    for (i=0; i<dimensionality; i++)
		    {
			model_indices[i]++;
			if (model_indices[i] >= model_dimensions[i])
			    model_indices[i] = 0;
			else
			    break;
		    }
		    if (i == dimensionality)
		    {
			/* Loop finished. By the way, the indices have
			 * all been reset to zero.
			 */
			break;
		    }
		}
		if (k2 == basis_size)
		{
		    b[k1] = ip;
		}
		else
		{
		    A[k1 + k2 * basis_size] = ip;
		    A[k2 + k1 * basis_size] = ip;
		}
	    }
	}
	mexCallMATLAB(1, &x_vector, 2, arguments, "\\");
	x = mxGetPr(x_vector);
	resultindex = calc_single_subscript2(result_dimensions,
					     result_indices,
					     start_indices,
					     dimensionality);
	for (k=0; k<basis_size; k++)
	    result[resultindex + resultdimprod * k] = x[k];

	mxDestroyArray(x_vector);

	/* Increment the indices. */
	for (i=0; i<dimensionality; i++)
	{
	    result_indices[i]++;
	    if (result_indices[i] > stop_indices[i])
		result_indices[i] = start_indices[i];
	    else
		break;
	}
	if (i == dimensionality)
	{
	    /* Loop finished */	
	    break;
	}

    }
    mxFree(result_indices);
    mxFree(model_indices);
    mxFree(signal_indices);
    mxFree(displacements);
}

void
mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    int i;
    int dimensionality;
    int sn, cn, bn, an;
    const int *sdim;
    const int *cdim;
    const int *bdim;
    const int *adim;
    int *signal_dimensions;
    int *model_dimensions;
    int *result_dimensions;
    mxArray *resultarray;
    double *region_of_interest;
    int startindex, stopindex;
    int *start_indices;
    int *stop_indices;
    int basis_size;
    int optimization;

    /* First see if there is a final string parameter */
    if (nrhs > 0 && mxIsChar(prhs[nrhs-1]))
    {
	optimization = 0;
	nrhs--;
    }
    else
	optimization = 1;
    
    /* Check the number of input and output arguments. */
    if (nrhs < 4)
	mexErrMsgTxt("Too few input arguments.");
    if (nrhs > 5)
	mexErrMsgTxt("Too many input arguments.");
    if (nlhs > 1)
	mexErrMsgTxt("Too many output arguments.");

    /* Check the formats of the first four input arguments. */
    for (i=0; i<4; i++)
    {
	if (!mxIsNumeric(prhs[i]) || mxIsComplex(prhs[i])
	    || mxIsSparse(prhs[i]) || !mxIsDouble(prhs[i]))
	{
	    mexErrMsgTxt("The first four input arguments must be real and full numeric arrays, stored as doubles.");
	}
    }

    /* Check the format of region_of_interest, if present. */
    if (nrhs == 5)
    {
	if (!mxIsNumeric(prhs[4]) || mxIsComplex(prhs[4])
	    || !mxIsDouble(prhs[4]))
	{
	    mexErrMsgTxt("The first four input arguments must be real and full numeric arrays, stored as doubles.");
	}
    }
    
    /* We won't deal with empty arrays. */
    for (i=0; i<4; i++)
	if (mxIsEmpty(prhs[i]))
	    mexErrMsgTxt("Begone, empty arrays.");

    /* Get the dimensionalities. Trailing singleton dimensions are
     * ignored.
     */
    
    /* signal */
    sdim = mxGetDimensions(prhs[0]);
    for (sn = mxGetNumberOfDimensions(prhs[0]); sn>0; sn--)
	if (sdim[sn-1] > 1)
	    break;
    
    /* certainty */
    cdim = mxGetDimensions(prhs[1]);
    for (cn = mxGetNumberOfDimensions(prhs[1]); cn>0; cn--)
	if (cdim[cn-1] > 1)
	    break;
    
    /* basis */
    bdim = mxGetDimensions(prhs[2]);
    for (bn = mxGetNumberOfDimensions(prhs[2]); bn>0; bn--)
	if (bdim[bn-1] > 1)
	    break;
    
    /* applicability */
    adim = mxGetDimensions(prhs[3]);
    for (an = mxGetNumberOfDimensions(prhs[3]); an>0; an--)
	if (adim[an-1] > 1)
	    break;

    /* Let the dimensionality of the convolution be the largest of the
     * given parameters.
     */
    dimensionality = sn;
    if (cn > dimensionality)
	dimensionality = cn;
    if (bn-1 > dimensionality)
	dimensionality = bn-1;
    if (an > dimensionality)
	dimensionality = an;

    /* Fix to manage the rather uninteresting special case of all
     * parameters being scalars.
     */
    if (dimensionality == 0)
	dimensionality = 1;
    
    /* Build dimension vectors for signal and model. Also check the
     * consistency of the dimensions.
     */
    signal_dimensions = (int *)mxCalloc(dimensionality, sizeof(int));
    model_dimensions  = (int *)mxCalloc(dimensionality, sizeof(int));
    
    if (cn != sn)
	mexErrMsgTxt("Signal and certainty must have the same size.");
    for (i=0; i<sn; i++)
	if (cdim[i] != sdim[i])
	    mexErrMsgTxt("Signal and certainty must have the same size.");

    for (i=0; i<sn; i++)
	signal_dimensions[i] = sdim[i];
    for (; i<dimensionality; i++)
	signal_dimensions[i] = 1;

    /* The interpretation of the dimensions of basis is made in a very
     * friendly manner. It is required that the non-singleton
     * dimensions of applicability are present at the beginning of
     * basis. If all the following dimensions are singleton it is
     * assumed to be only one basis function. If exactly one of the
     * following dimensions is larger than one, that is assumed to be
     * the number of basis functions. Otherwise an error is declared.
     */
    if (bn < an)
	mexErrMsgTxt("Each basis function must have the same size as applicability.");

    for (i=0; i<an; i++)
	if (bdim[i] != adim[i])
	    mexErrMsgTxt("Each basis function must have the same size as applicability.");

    for (; i<bn; i++)
	if (bdim[i] > 1)
	    break;

    if (i == bn)
	basis_size = 1;
    else
    {
	basis_size = bdim[i];
	for (i++; i<bn; i++)
	    if (bdim[i] > 1)
		break;
	if (i < bn)
	    mexErrMsgTxt("Each basis function must have the same size as applicability.");
    }

    for (i=0; i<an; i++)
	model_dimensions[i] = adim[i];
    for (; i<dimensionality; i++)
	model_dimensions[i] = 1;

    /* Check the validity of the region of interest. */
    if (nrhs == 5)
    {
	if (mxGetNumberOfDimensions(prhs[4]) != 2
	    || mxGetM(prhs[4]) != sn
	    || mxGetN(prhs[4]) != 2)
	{
	    mexErrMsgTxt("Region of interest must be an N by 2 matrix, where N is the dimensionality of the signal.");
	}
	region_of_interest = mxGetPr(prhs[4]);
	for (i=0; i<sn; i++)
	{
	    startindex = region_of_interest[i];
	    stopindex = region_of_interest[i + sn];
	    if (startindex < 1
		|| startindex > stopindex
		|| stopindex > sdim[i])
		mexErrMsgTxt("Invalid region of interest.");
	}
    }

    /* Create the start and stop indices. */
    start_indices = (int *)mxCalloc(dimensionality, sizeof(int));
    stop_indices  = (int *)mxCalloc(dimensionality, sizeof(int));

    if (nrhs == 4)
    {
	for (i=0; i<sn; i++)
	{
	    start_indices[i] = 0;
	    stop_indices[i] = sdim[i] - 1;
	}
    }
    else
    {
	region_of_interest = mxGetPr(prhs[4]);
	for (i=0; i<sn; i++)
	{
	    start_indices[i] = region_of_interest[i] - 1;
	    stop_indices[i] = region_of_interest[i + sn] - 1;
	}
    }
    
    for (i=sn; i<dimensionality; i++)
    {
	start_indices[i] = 0;
	stop_indices[i] = 0;
    }

    /* Compute the output dimensions. */
    result_dimensions = (int *)mxCalloc(dimensionality + 1, sizeof(int));
    for (i=0; i<dimensionality; i++)
	result_dimensions[i] = stop_indices[i] - start_indices[i] + 1;
    result_dimensions[dimensionality] = basis_size;

    /* Create the output array. */
    resultarray = mxCreateNumericArray(dimensionality+1, result_dimensions,
				       mxDOUBLE_CLASS, mxREAL);

    /* Call the appropriate computational routine. */
    if (dimensionality > 3)
	optimization = 0;
    switch ((dimensionality + 3 * (basis_size > 1)) * optimization)
    {
      case 0: /* General case, no optimization */
	normconv(mxGetPr(prhs[0]),
		 mxGetPr(prhs[1]),
		 mxGetPr(prhs[2]),
		 mxGetPr(prhs[3]),
		 mxGetPr(resultarray),
		 dimensionality,
		 signal_dimensions,
		 model_dimensions,
		 result_dimensions,
		 basis_size,
		 start_indices,
		 stop_indices);
	break;
	
      case 1: /* One dimension, one basis function */
	normconv1_1(mxGetPr(prhs[0]),
		    mxGetPr(prhs[1]),
		    mxGetPr(prhs[2]),
		    mxGetPr(prhs[3]),
		    mxGetPr(resultarray),
		    signal_dimensions,
		    model_dimensions,
		    result_dimensions,
		    start_indices,
		    stop_indices);
	break;
	
      case 2: /* Two dimensions, one basis function */
	normconv2_1(mxGetPr(prhs[0]),
		    mxGetPr(prhs[1]),
		    mxGetPr(prhs[2]),
		    mxGetPr(prhs[3]),
		    mxGetPr(resultarray),
		    signal_dimensions,
		    model_dimensions,
		    result_dimensions,
		    start_indices,
		    stop_indices);
	break;
	
      case 3: /* Three dimensions, one basis function */
	normconv3_1(mxGetPr(prhs[0]),
		    mxGetPr(prhs[1]),
		    mxGetPr(prhs[2]),
		    mxGetPr(prhs[3]),
		    mxGetPr(resultarray),
		    signal_dimensions,
		    model_dimensions,
		    result_dimensions,
		    start_indices,
		    stop_indices);
	break;
	
      case 4: /* One dimension, multiple basis functions */
	normconv1(mxGetPr(prhs[0]),
		  mxGetPr(prhs[1]),
		  mxGetPr(prhs[2]),
		  mxGetPr(prhs[3]),
		  mxGetPr(resultarray),
		  signal_dimensions,
		  model_dimensions,
		  result_dimensions,
		  basis_size,
		  start_indices,
		  stop_indices);
	break;
	
      case 5: /* Two dimensions, multiple basis functions */
	normconv2(mxGetPr(prhs[0]),
		  mxGetPr(prhs[1]),
		  mxGetPr(prhs[2]),
		  mxGetPr(prhs[3]),
		  mxGetPr(resultarray),
		  signal_dimensions,
		  model_dimensions,
		  result_dimensions,
		  basis_size,
		  start_indices,
		  stop_indices);
	break;
	
      case 6: /* Three dimensions, multiple basis functions */
	normconv3(mxGetPr(prhs[0]),
		  mxGetPr(prhs[1]),
		  mxGetPr(prhs[2]),
		  mxGetPr(prhs[3]),
		  mxGetPr(resultarray),
		  signal_dimensions,
		  model_dimensions,
		  result_dimensions,
		  basis_size,
		  start_indices,
		  stop_indices);
	break;
	
      default:
	mexErrMsgTxt("Internal error, impossible case.");
	break;
    }
	    
    /* Free allocated memory. */
    mxFree(start_indices);
    mxFree(stop_indices);

    /* Output the computed result. */
    plhs[0] = resultarray;
}
