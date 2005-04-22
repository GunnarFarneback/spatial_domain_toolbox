% P = MIN_QUADFORM(Qtot)
% 
% Minimize quadratic forms according to equations 5.20 -- 5.24 in
% "Spatial Domain Methods for Orientation and Velocity Estimation"
% by Gunnar Farnebäck. MIN_QUADFORM is a mex function.
%
% Qtot   - A collection of quadratic forms, having the size
%          HEIGTHxWIDTHxNxN.
% 
% P      - A collection of optimal parameters, having the size
%          HEIGHTxWIDTHx(N-1).
% 
% Author: Gunnar Farnebäck
%         Computer Vision Laboratory
%         Linköping University, Sweden
%         gf@isy.liu.se
