"""
    Adjustable robust optimization
    Ax <= b
    uncertainty only on right sides vector b
    b - right sides vector
    c - costs vector
    A - nominal values for constraint matrix (variables symbol 'x')
    D - constraint matrix for wait and see variables (symbol 'y')
    Gamma - budget of uncertainty (discrete)
    bU - uncertainties for vector b
    zeroQ - indices of matrix Q which are to be set to 0 in form of tuples (row, column)
    printModel - when true model is printed
    printSolution - when true the solution (decision variables) is printed

    Returns JuMP model, dictionary with decision variables and their values, and optimum cost
"""
function adjustableMinB(c::Union{Vector, SparseVector, SparseMatrixCSC},
    b::Union{Vector, SparseVector, SparseMatrixCSC},
    A::Union{Matrix, Vector, SparseVector, SparseMatrixCSC},
    D::Union{Matrix, Vector, SparseVector, SparseMatrixCSC},
    Gamma::Float64, bU::Union{Vector, SparseVector, SparseMatrixCSC},
    zeroQ::Vector,
    printModel::Bool, printSolution::Bool)

    n = size(c)[1]

    m = size(A)[1] # number of contraints
    if (length(size(A)) > 1 && size(A)[2] != n)
        throw("Matrix A has wrong dimensions")
    end

    if (size(bU)[1] != m)
        throw("Vector bU has wrong dimensions")
    end

    if (size(D)[1] != m)
        throw("Matrix D has wrong dimensions")
    end

    if length(size(D)) > 1
        k = size(D)[2]
    else
        k = 1
    end

    if (size(b)[1] != m)
        throw("Vector b has wrong dimension")
    end


    model = Model(Cbc.Optimizer)
    set_attribute(model, "logLevel", 1)

    @variable(model, x[i=1:n] >= 0)
    @variable(model, z[i in 1:m] >= 0)
    @variable(model, p[i in 1:m, j in 1:m] >= 0)
    @variable(model, y[1:k] >= 0)
    @variable(model, Q[i in 1:k, j in 1:m])

     for i in 1:m
        @constraint(model, sum(A[i, j] * x[j] for j in 1:n) - b[i]
        + sum(D[i, j] * y[j] for j in 1:k) + z[i] * Gamma + sum(p[i, j] for j in 1:m) <= 0)
        for j in 1:m
            if i != j
                @constraint(model, z[i] + p[i, j] >= sum(D[i, l] * Q[l, j] for l in 1:k))
            else
                @constraint(model, z[i] + p[i, j] >= sum(D[i, l] * Q[l, j] for l in 1:k) - bU[i])
            end
        end
    end

    for (a, b) in zeroQ
        @constraint(model, Q[a, b] == 0)
    end

    @objective(model, Min, sum(c[i] * x[i] for i in 1:n))
    if (printModel)
        println(model)
    end
    optimize!(model)
    if printSolution
        printAdjustable(model, n, k, x, y)
    end

    d = Dict(
        k => value.(v) for
        (k, v) in object_dictionary(model))
    return model, d, objective_value(model)

end

"""
Function that prints solution of adjustable model

model - JuMP model
n - number of first stage decison variables
k - number of second stage decision variables
x - JuMP array of first stage decision variables
y - JuMP array of second stage decision variables
"""
function printAdjustable(model, n, k, x, y)
    if termination_status(model) == OPTIMAL
       println("Solution is optimal")
    elseif termination_status(model) == TIME_LIMIT && has_values(model)
       println("Solution is suboptimal due to a time limit, but a primal solution is available")
    else
       error("The model was not solved correctly.")
    end
    println("  objective value = ", objective_value(model))
    if primal_status(model) == FEASIBLE_POINT
        for j in 1:n
            println("  x", j, " = ", value(x[j]))
        end
        for i in 1:k
            println("  y", i, " = ", value(y[i]))
        end
    end
end