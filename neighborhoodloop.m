function varargout = neighborhoodloop(varargin)
% NEIGHBORHOODLOOP Implicit loop over neighborhoods.
%
% Example:
% Let f be a matrix and g a quadratic matrix of odd size. Then
% h = neighborhoodloop(2, size(g,1), f, @(x,y) sum(sum(x .* rot90(y, 2))), g);
% implements a convolution h = conv2(f, g, 'same').
%
% This can be generalized to
% [y1, y2, ...] = neighborhoodloop(N, size, x1, x2, ..., func, a1, a2, ...);
%
% where y1, y2, ... is an arbitrary number of output variables,
% the loops are over the first N dimensions in the arbitrary number
% of arrays x1, x2, ..., and a1, a2, ... is an arbitrary number of
% additional parameters. The called function is given by func,
% which is the name of either a builtin function, an m-file function,
% or a mex function. Alternatively func can be a function handle or
% an inline function. It is supposed that the first argument after size
% which is either a string or a scalar is func. The parameter size, which
% must be an odd integer, determines how large neighborhoods are passed
% to func.
%
% LIMITATIONS: Currently size can only be a scalar, meaning that only
%              isotropic neighborhoods can be used. A better
%              implementation would allow a different size in each
%              dimension.
%
% See also ARRAYLOOP.
%
% Author: Gunnar Farnebäck
%         Medical Informatics
%         Linköping University, Sweden
%         gunnar@imt.liu.se

error('NEIGHBORHOODLOOP is implemented as a mex-file. It has not been compiled on this platform.')
