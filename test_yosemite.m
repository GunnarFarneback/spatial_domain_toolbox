function test_yosemite
% TEST_YOSEMITE
%
% This function reproduces the results in the last four rows of table 5.2,
% page 82, of "Spatial Domain Methods for Orientation and Velocity
% Estimation" by Gunnar Farnebäck.
%
% Author: Gunnar Farnebäck
%         Computer Vision Laboratory
%         Linköping University, Sweden
%         gf@isy.liu.se

% Read the Yosemite sequence.
[sequence, mask, correct_flow] = read_yosemite;

disp('Yosemite, fast algorithm, constant motion:')
test_velocity(sequence, mask, correct_flow, 9, 1.4, 1/32, 'constant', 15, 3.5);

disp('Yosemite, fast algorithm, affine motion:')
test_velocity(sequence, mask, correct_flow, 11, 1.6, 1/256, 'affine', 41, 6.5);

return


% Subroutine doing the actual work.
function test_velocity(sequence, mask, correct_flow, kernelsize, sigma, ...
		       gamma, model, kernelsize_avg, sigma_avg)

% Extract the size of the frames.
sides = size(sequence);
middle = (sides(3)+1)/2;
sides = sides(1:2);

% Region of interest for tensor computation. We only compute the tensor
% field for the middle frame.
roi = [[[1;1] sides'];[middle middle]];

% Compute tensors.
options = struct('sigma', sigma, 'gamma', gamma);
T = make_tensors_fast(sequence, kernelsize, roi, options);

% Certainty mask for averaging. Close to the border the tensors will be
% affected by edge effects, so we ignore them.
bwidth = (kernelsize-1)/2;
mask2 = ones(sides);
mask2([1:bwidth,end-bwidth+1:end],:) = 0;
mask2(:,[1:bwidth,end-bwidth+1:end]) = 0;

% Compute velocity from a tensor field using the fast algorithm and the
% specified motion model.
[v, c] = velocity_from_tensors(T, model, kernelsize_avg, sigma_avg, mask2);

% Evaluate the estimated velocity.
evaluate_velocity(v, correct_flow, mask, c);
