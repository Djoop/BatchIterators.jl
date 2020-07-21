# Summary 

A very small package providing `BatchIterator(X; batchsize=…, limit=…)` and `centered_batch_iterator(X; kwargs…)`, which allows iteration on blocks of columns of `X`, for any object `X` supporting 2d indexing and for which `size` (e.g. out-of-core matrix). 

The function `choose_batchsize` helps finding a good batch size while controlling memory usage.
