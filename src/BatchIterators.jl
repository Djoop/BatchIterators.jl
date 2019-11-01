module BatchIterators

export BatchIterator
export choose_batchsize

"""
	BatchIterator(X, bsz[, limit])
Wrapper allowing to iterate over batches of `bsz` columns of `X`. `X` can be of any type supporting `size` and 2d indexing. When `limit` is provided, iteration is restricted to the columns of `X[:, 1:limit]`.
"""
struct BatchIterator{T}
	X::T
	length::Int		# Number of batches
	bsz::Int		# Batch size
	limit::Int
	function BatchIterator(X; batchsize=nothing, limit=size(X,2))
		@assert limit > 0 && limit ≤ size(X,2)
		bsz = (batchsize == nothing) ? choose_batchsize(size(X,1), limit) : batchsize
		nb = ceil(Int, limit/bsz)
		new{typeof(X)}(X, nb, bsz, limit)
	end
end

view_compatible(::Any) = false
view_compatible(::Array) = true
view_compatible(bi::BatchIterator) = view_compatible(bi.X)

#######################################################################
#                              Matrices                               #
#######################################################################

function Base.getindex(it::BatchIterator, i)
	d = i - it.length		# > 0 means overflow, == 0 means last batch
	cbsz = (d == 0) ? mod(it.limit - 1, it.bsz) + 1 : it.bsz		# Size of current batch
	if (i<1 || d > 0)
		@error "Out of bounds."
	else
		view_compatible(it) ? (@view it.X[:, (i-1)*it.bsz+1:(i-1)*it.bsz+cbsz]) : it.X[:, (i-1)*it.bsz+1:(i-1)*it.bsz+cbsz]
	end
end
Base.length(it::BatchIterator)  = it.length
function Base.iterate(it::BatchIterator{T}, st = 0) where T
	st = st + 1				# new state
	d = st - it.length		# > 0 means overflow, == 0 means last batch
	if d > 0
		nothing
	else
		(it[st], st)
	end
end

#######################################################################
#                              Utilities                              #
#######################################################################

"""
	choose_batchsize(d, n; maxmemGB = 1.0, maxbatchsize = 2^14, sizeoneB = d*sizeof(Float64))

Computes the size (nb. of columns) of a batch, so that each column of the batch can be converted to a vector of size `sizeoneB` (in bytes) with a total memory constrained by `maxmemGB` (gigabytes).
"""
function choose_batchsize(d, n; 
				   maxmemGB = 1.0, 
				   maxbatchsize = 2^14,
				   sizeoneB = d*sizeof(Float64),
				   forcepow2 = true)

	fullsizeGB = n * sizeoneB/1024^3		# Size of the sketches of all samples
	batchsize = (fullsizeGB > maxmemGB) ? ceil(Int, n/ceil(Int, fullsizeGB/maxmemGB)) : n
	batchsize = min(batchsize, maxbatchsize)
	(forcepow2 && batchsize != n) ? prevpow(2, batchsize) : batchsize
end

end # module
