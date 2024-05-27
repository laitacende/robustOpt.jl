"""
    Nominal minimum

    Ax <= b
    A - constraint matrix
    b - right sides vector
    c - cost vector
    printModel - when true model is printed
    printSolution - when true the solution (decision variables) is printed

    Returns JuMP model, dictionary with decision variables and their values, and optimum cost
"""
function nominal(c::Union{Vector, SparseVector}, b::Union{Vector, SparseVector},
     A::Union{Matrix, Vector, SparseVector, SparseMatrixCSC}, printModel::Bool, printSolution::Bool)
    n = size(c)[1]

    m = size(A)[1] # number of contraints
    if (size(A)[2] != n)
        throw("Matrix A has wrong dimensions")
    end

    if (size(b)[1] != m)
        throw("Vector b has wrong dimension")
    end

    model = Model(Cbc.Optimizer)
    set_attribute(model, "logLevel", 0)
    @variable(model, x[i=1:n] >= 0)

    for i in 1:m
        @constraint(model, sum(A[i, j] * x[j] for j in 1:n)  <= b[i])
    end
    @objective(model, Min, sum(c[i] * x[i] for i in 1:n))

    if (printModel)
        println(model)
    end
    optimize!(model)
    zOpt = objective_value(model)
    if printSolution
        printNominal(model, n, x)
    end

    d = Dict(
        k => value.(v) for
        (k, v) in object_dictionary(model) if v isa AbstractArray{VariableRef})

    return model, d, zOpt
end

"""
Function that prints solution of nominal model

model - JuMP model
n - number of decison variables
x - JuMP array of decision variables
"""
function printNominal(model, n, x)
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
    end
end