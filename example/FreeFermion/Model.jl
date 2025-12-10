"""
     SpinlessFreeFermion(Latt::AbstractLattice;
          t::Float64 = 1.0,
          μ::Float64 = 0.0) -> ::InteractionTree

Return the interaction tree for the spinless free fermion with nearest-neighbor hopping `t` and chemical potential `μ` on a given lattice `Latt`.
"""
function SpinlessFreeFermion(Latt::AbstractLattice; t::Float64 = 1.0, μ::Float64 = 0.0)
     
     Tree = InteractionTree(size(Latt))
     for (i, j) in neighbor(Latt; level = 1, ordered = true) 
          addIntr!(Tree, U1SpinlessFermion.FdagF, (i, j), (true, true), -t; name = (:Fdag, :F), Z = U1SpinlessFermion.Z)
     end

     for i in 1:size(Latt)
          addIntr!(Tree, U1SpinlessFermion.n, i, -μ; name = :n)
     end
     return merge!(Tree)
end