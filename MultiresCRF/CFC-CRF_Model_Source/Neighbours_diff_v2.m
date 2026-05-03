function [ diff_above, diff_below, diff_left, diff_right ] = Neighbours_diff_v2( image, sigma, square )
%Neighbours_differences_RBF
%   Takes in input an image and return the differences computed with
%   respect to four connected neightbours. Then it uses a sigma in order to
%   have sharper borders and uniform surfaces. Important to remember that
%   output image will be 1 pixel less on each side.

xsy = size(image,1);
xsx = size(image,2);

%differences in values between activations maps
%we sum the differences across all the layers image have
%Class_ativation map has just 6 layers, then next layers has 24 and so on,
%increasing with layer depth
diff_above = sum((double(image(2:(xsy),1:(xsx),:))...
                 -double(image(1:(xsy-1),1:(xsx),:))).^2,3);
diff_below = sum((double(image(1:(xsy-1),1:(xsx),:))...
                 -double(image(2:(xsy)  ,1:(xsx),:))).^2,3);
diff_right = sum((double(image(1:(xsy),1:(xsx-1),:))...
                 -double(image(1:(xsy),2:xsx    ,:))).^2,3);
diff_left =  sum((double(image(1:(xsy),2:(xsx),:))...
                 -double(image(1:(xsy),1:(xsx-1),:))).^2,3);
%considering the differences that way (before cutting borders) diff_above  
%is exactly equal to diff_below (same for left and right) cause we are
%considering different starting points.
                    
%set similarity as exp^-(diff/sigma) i.e. Radial Basis Function
diff_above = real(exp(-diff_above/sigma));
diff_below = real(exp(-diff_below/sigma));
diff_right = real(exp(-diff_right/sigma));
diff_left =  real(exp(-diff_left/sigma ));

%is flag square is up we consider only the pixels of which we can define
%the full 4-connected neightbourhood, then we remove borders not yet
%considered
if square
    diff_above = diff_above(1:(xsy-2),2:(xsx-1));
    diff_below = diff_below(2:(xsy)-1,2:(xsx-1));
    diff_right = diff_right(2:(xsy-1),1:(xsx-2));
    diff_left =  diff_left( 2:(xsy-1),2:(xsx-1));
    %after that cut differences are referred to "central pixels", then
    %differences are all different one from the other
end

end