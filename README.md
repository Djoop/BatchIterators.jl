# Summary 

Licence: MIT.

A very small package providing the constructor `BatchIterator(X; batchsize=…, limit=…)` and the function `centered_batch_iterator(X; kwargs…)`, which allow iteration over blocks of columns of `X`, for any object `X` supporting 2d indexing and for which the function `size` is defined.
The function `choose_batchsize` helps finding a good batch size while controlling memory usage.

The package was originally design to iterate over samples of an out-of-core dataset.

