function str = FInt2Str(len,int)

% function str = FInt2Str(len,int)
% function for converting integer int to string str with length len

formatstr = ['%0' int2str(len) 'd'];
str = sprintf(formatstr,int);