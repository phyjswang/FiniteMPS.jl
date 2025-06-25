"""
     module SU₂Spin

Prepare the local space of SU₂ spin-1/2.

# Fields
     pspace::VectorSpace
Local `d = 2` Hilbert space.

     SS::NTuple{2, TensorMap}
Two rank-`3` operators of Heisenberg `S⋅S` interaction.
"""
module SU₂Spin

using TensorKit

const pspace = Rep[SU₂](1//2 => 1)
# S⋅S interaction
const SS = let
     aspace = Rep[SU₂](1 => 1)
     SL = TensorMap(ones, Float64, pspace, pspace ⊗ aspace) * sqrt(3) / 2

     SR = permute(SL', ((2, 1), (3,)))
     SL, SR
end

end

"""
     const SU2Spin = SU₂Spin
"""
const SU2Spin = SU₂Spin

"""
     module U₁Spin

Prepare the local space of U₁ spin-1/2.

# Fields
     pspace::VectorSpace
Local `d = 2` Hilbert space.

    Sz::TensorMap
Rank-`2` spin-z operator `Sz = (n↑ - n↓)/2`.

    S₊₋::NTuple{2, TensorMap}
    S₋₊::NTuple{2, TensorMap}
Two rank-`3` operators of `S₊₋` and `S₋₊` interaction. Note Heisenberg `S⋅S = SzSz + (S₊₋ + S₋₊)/2`.
"""
module U₁Spin

using TensorKit

const pspace = Rep[U₁](-1 // 2 => 1, 1 // 2 => 1)

const Sz = let
     Sz = TensorMap(ones, pspace, pspace)
     block(Sz, Irrep[U₁](1 // 2)) .= 1/2
     block(Sz, Irrep[U₁](-1 // 2)) .= -1/2
     Sz
end

# S+ S- interaction
# convention: S⋅S = SzSz + (S₊₋ + S₋₊)/2
const S₊₋ = let
     aspace = Rep[U₁](1 => 1)
     S₊ = TensorMap(ones, pspace, pspace ⊗ aspace)
     S₋ = TensorMap(ones, aspace ⊗ pspace, pspace)
     S₊, S₋
end

const S₋₊ = let
     aspace = Rep[U₁](1 => 1)
     iso = isometry(aspace, flip(aspace))
     @tensor S₋[a; c d] := S₊₋[1]'[a, b, c] * iso'[d, b]
     @tensor S₊[d a; c] := S₊₋[2]'[a, b, c] * iso[b, d]
     S₋, S₊
end

end

const U1Spin = U₁Spin

"""
     module NoSymSpinOneHalf

Prepare the local space of U₁ spin-1/2. Basis convention is `{|↑⟩, |↓⟩}`.

# Fields
     pspace::VectorSpace
Local `d = 2` Hilbert space.

    Sz::TensorMap
    Sx::TensorMap
    Sy::TensorMap
Rank-`2` spin-1/2 operators.

    S₊::TensorMap
Rank-`2` spin-plus operator `S₊ = Sx + iSy`.
    S₋::TensorMap
Rank-`2` spin-minus operator `S₋ = Sx - iSy`.
"""
module NoSymSpinOneHalf

using TensorKit

const pspace = ℂ^2

const Sz = let
     mat = Float64[1/2 0; 0 -1/2]
     TensorMap(mat, pspace, pspace)
end

const S₊ = let
    mat = Float64[0 1; 0 0]
    TensorMap(mat, pspace, pspace)
end

const S₋ = let
    mat = Float64[0 0; 1 0]
    TensorMap(mat, pspace, pspace)
end

const Sx = (S₊ + S₋) / 2.

const Sy = (S₊ - S₋) / (2. * 1im)

end

"""
    module U₁HardcoreBoson

prepare the local space of U₁ hardcore bosons. Basis convention is `{|1⟩, |0⟩}`.

a⁺ = S₊, a = S₋, n = Sᶻ + 1/2

# Fields
    pspace::VectorSpace
Local `d = 2` Hilbert space.

    n::TensorMap
Rank-`2` particle number operator.

    a⁺a::NTuple{2, TensorMap}
    aa⁺::NTuple{2, TensorMap}
Two rank-`3` operators of `a⁺a` and `aa⁺` interaction.
"""
module U₁HardcoreBoson

using TensorKit

const pspace = Rep[U₁](-1/2 => 1, 1/2 => 1)

const n = let
    n = TensorMap(ones, pspace, pspace)
    block(n, Irrep[U₁](1 // 2)) .= 1
    block(n, Irrep[U₁](-1 // 2)) .= 0
    n
end

const a⁺a = let
     aspace = Rep[U₁](1 => 1)
     a⁺ = TensorMap(ones, pspace, pspace ⊗ aspace)
     a = TensorMap(ones, aspace ⊗ pspace, pspace)
     a⁺, a
end


const aa⁺ = let
     aspace = Rep[U₁](1 => 1)
     iso = isometry(aspace, flip(aspace))
     @tensor a[a; c d] := a⁺a[1]'[a, b, c] * iso'[d, b]
     @tensor a⁺[d a; c] := a⁺a[2]'[a, b, c] * iso[b, d]
     a, a⁺
end

end

const U1HardcoreBoson = U₁HardcoreBoson

"""
    module NoSymHardcoreBoson

prepare the local space of hardcore bosons. Basis convention is `{|0⟩, |1⟩}`.

# Fields
    pspace::VectorSpace
Local `d = 2` Hilbert space.

    n::TensorMap
    a⁺::TensorMap
    a::TensorMap
Rank-`2` particle number operator, creation operator, and annihilation operator.
"""

module NoSymHardcoreBoson

using TensorKit

const pspace = ℂ^2

const n = let
    mat = Float64[0 0; 0 1]
    TensorMap(mat, pspace, pspace)
end

const a⁺ = let
    mat = Float64[0 1; 0 0]
    TensorMap(mat, pspace, pspace)
end

const a = let
    mat = Float64[0 0; 1 0]
    TensorMap(mat, pspace, pspace)
end

end
