function BraKet=InnerProduct(Psi1,Psi2,h)

% Numerical evaluation of inner product <Psi1 | Psi2>
% With wavefunction provided by matrices in which each column
% corresponds to an l-channel, the inner product would, in fact,
% be provided by the trace of the matrix product between
% Psi1^\dagger and Psi2. This, in turn, may be evaluated
% as sum(sum(conj(Psi1).*Psi2), i.e. the sum of all elements in 
% the Hadamard product.

BraKet= sum(sum(conj(Psi1).*Psi2))*h;