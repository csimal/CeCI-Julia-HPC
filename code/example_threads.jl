
function threaded_mul!(C,A,B)
    m,n = size(C)
    l = size(A,2)
    Threads.@threads for j in 1:n # NB. We can only apply this to the outer loop
        for k in 1:l, i in 1:m
          @inbounds C[i,j] += A[i,k]*B[k,j]
        end
    end
end

n = 256; A = rand(n,n); B = rand(n,n); C = rand(n,n)
using BenchmarkTools
b_threaded = @benchmark threaded_mul!(C,A,B)
