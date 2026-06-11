function mift_sift_compile(type)
% MIFT_SIFT_COMPILE  Compile MEX files
%   Compiling under Windows requires at least Visual C 6 or LCC. You
%   might try other compilers, but most likely you will need to edit
%   this file.
% Created by: Xiaojie Guo
% Affiliate: School of Computer Science and Technology, Tianjin Univ.,
%            Tianjin, 300072, China
% Reference: [1] MIFT: A Mirror Reflection Invariant Descriptor,        
%                Xiaojie Guo, Xiaochun Cao, Jiawan Zhang and Xuewei Li,
%                ACCV, 2009.
% All Rights Reserved.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This implementation is based on the codes written by Andrea Vadaldi 
% UCLA Vision Lab - Department of Computer Science
% Copyright (c) 2006 The Regents of the University of California.
% All Rights Reserved.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


siftroot = fileparts(which('miftsiftcompile')) ;
opts = { '-O', '-I.', '-g' } ;
%opts = { opts{:}, '-v' } ;

if nargin < 1
    type = 'visualc' ;
end

switch computer
  case 'PCWIN'
    opts = {opts{:}, '-DWINDOWS'} ;
  
  case 'MAC'
    opts = {opts{:}, '-DMACOSX'} ;
    opts = {opts{:}, 'CFLAGS=\$CFLAGS -faltivec'} ;
    
  case 'MACI'
    opts = {opts{:}, '-DMACOSX'} ;
  
  case 'GLNX86'
    opts = {opts{:}, '-DLINUX' } ;
        
  otherwise
    error(['Unsupported architecture ', computer, '. Please edit this M-file to fix the issue.']) ;    
end

mex('imsmooth.c',opts{:}) ;

mex('mift_siftrefinemx.c',opts{:}) ;
mex('mift_siftormx.c',opts{:}) ;
mex('mift_sift_desc.cpp',opts{:}) ;
mex('mift_siftmatch.c',opts{:}) ;
mex('mift_siftlocalmax.c',opts{:}) ;
fprintf('done');

    


