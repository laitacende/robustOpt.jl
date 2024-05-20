"""
    Light robustness minimum
    Ax <= b
    b - right sides vector
    c - costs vector
    A - nominal values for constraint matrix
    Gamma - array of Gammas for rows
    AU - uncertainties of matrix A
    rho - maximum value to distort from optimum (if val == True)
        - maximum percent to distort from optimum (if val == False)
    printModel - when true model is printed
    printSolution - when true the solution (decision variables) is printed

    Returns JuMP model, dictionary with decision variables and their values, and optimum cost
"""
function lightRobustnessMin(c::Union{Vector, SparseVector, SparseMatrixCSC}, b::Union{Vector, SparseVector, SparseMatrixCSC},
     A::Union{Matrix, Vector, SparseVector, SparseMatrixCSC}, Gamma::Union{Vector, SparseVector},
     AU::Union{Matrix, Vector, SparseMatrixCSC, SparseVector},
    rho::Float64, val::Bool, printModel::Bool, printSolution::Bool)

    n = size(c)[1]

    m = size(A)[1] # number of contraints
    if (size(A)[2] != n)
        throw("Matrix A has wrong dimensions")
    end
    if ((length(size(AU)) > 1 && size(AU)[2] > n) || size(AU)[1] > m)
        throw("Matrix AU has wrong dimensions")
    end

    if (size(b)[1] != m)
        throw("Vector b has wrong dimension")
    end

    if (size(Gamma)[1] > m)
        throw("Vector Gamma has wrong dimension")
    end

    # get optimal cost for nominal problem
    modelNom = Model(Cbc.Optimizer)
    set_attribute(modelNom, "logLevel", 0)
    @variable(modelNom, x[i=1:n] >= 0)

    for i in 1:m
        @constraint(modelNom, sum(A[i, j] * x[j] for j in 1:n)  <= b[i])
    end

    @objective(modelNom, Min, sum(c[i] * x[i] for i in 1:n))
    optimize!(modelNom)
    zOpt = objective_value(modelNom)


    # light robustness model
    model = Model(Cbc.Optimizer)
    set_attribute(model, "logLevel", 0)
    @variable(model, x[1:n] >= 0)
    @variable(model, z[1:m] >= 0)
    @variable(model, p[i in 1:m, j in 1:n] >= 0)
    @variable(model, y[1:m] >= 0)

    for i in 1:m
        @constraint(model, sum(A[i, j] * x[j] for j in 1:n) <= b[i])
        @constraint(model, sum(A[i, j] * x[j] for j in 1:n) + z[i] * Gamma[i] + sum(p[i, j] for j in 1:n) - y[i] <= b[i])
    end

    for i in 1:m
        for j in 1:n
            @constraint(model, -x[j] * AU[i, j] + z[i] + p[i, j] >= 0)
        end
    end

    if (val)
        @constraint(model, sum(c[j] * x[j] for j in 1:n) <= zOpt + rho)
    else
        @constraint(model, sum(c[j] * x[j] for j in 1:n) <= zOpt * ( 1 + rho))
    end

    @objective(model, Min, sum(y[i] for i in 1:m))
    if (printModel)
        println(model)
    end
    optimize!(model)
    cost = 0.0
    for j in 1:n
        cost += c[j] * value(x[j])
    end
     if (printSolution)
        printLightRobustness(model, n, x, zOpt)
    end
    d = Dict(
        k => value.(v) for(k, v) in object_dictionary(model) if v isa AbstractArray{VariableRef})

    return model, d, cost
end

"""
    Light robustness maximum
    Ax <= b
    b - right sides vector
    c - costs vector
    A - nominal values for constraint matrix
    Gamma - array of Gammas for rows
    AU - uncertainties of matrix A
    rho - maximum value to distort from optimum (if val == True)
        - maximum percent to distort from optimum (if val == False)
    printModel - when true model is printed
    printSolution - when true the solution (decision variables) is printed

    Returns JuMP model, dictionary with decision variables and their values, and optimum cost
"""
function lightRobustnessMax(c::Union{Vector, SparseVector, SparseMatrixCSC}, b::Union{Vector, SparseVector, SparseMatrixCSC},
     A::Union{Matrix, Vector, SparseVector, SparseMatrixCSC}, Gamma::Union{Vector, SparseVector},
     AU::Union{Matrix, Vector, SparseMatrixCSC, SparseVector},
    rho::Float64, val::Bool, printModel::Bool, printSolution::Bool)

    n = size(c)[1]

    m = size(A)[1] # number of contraints
    if (size(A)[2] != n)
        throw("Matrix A has wrong dimensions")
    end
    if ((length(size(AU)) > 1 && size(AU)[2] > n) || size(AU)[1] > m)
        throw("Matrix AU has wrong dimensions")
    end

    if (size(b)[1] != m)
        throw("Vector b has wrong dimension")
    end

    if (size(Gamma)[1] > m)
        throw("Vector Gamma has wrong dimension")
    end


    # get optimal cost for nominal problem
    modelNom = Model(Cbc.Optimizer)
    set_attribute(modelNom, "logLevel", 1)
    @variable(modelNom, x[i=1:n] >= 0)

    for i in 1:m
        @constraint(modelNom, sum(A[i, j] * x[j] for j in 1:n)  <= b[i])
    end

    @objective(modelNom, Max, sum(c[i] * x[i] for i in 1:n))
    optimize!(modelNom)
    zOpt = objective_value(modelNom)

    # light robustness model
    model = Model(Cbc.Optimizer)
    set_attribute(model, "logLevel", 1)
    @variable(model, x[i=1:n] >= 0)
    @variable(model, z[1:m] >= 0)
    @variable(model, p[i in 1:m, j in 1:n] >= 0)
    @variable(model, y[1:m] >= 0)

    for i in 1:m
        @constraint(model, sum(A[i, j] * x[j] for j in 1:n) <= b[i])
        @constraint(model, sum(A[i, j] * x[j] for j in 1:n) + z[i] * Gamma[i] + sum(p[i, j] for j in 1:n) - y[i] <= b[i])
    end

    for i in 1:m
        for j in 1:n
            @constraint(model, -x[j] * AU[i, j] + z[i] + p[i, j] >= 0)
        end
    end

    if (val)
        @constraint(model, sum(c[j] * x[j] for j in 1:n) >= zOpt - rho)
    else
        @constraint(model, sum(c[j] * x[j] for j in 1:n) >= zOpt * ( 1 - rho))
    end

    @objective(model, Min, sum(y[i] for i in 1:m))
     if (printModel)
        println(model)
    end
    optimize!(model)
    cost = 0.0
    for j in 1:n
        cost += c[j] * value(x[j])
    end
    if (printSolution)
        printLightRobustness(model, n, x, zOpt, c)
    end
    d = Dict(
        k => value.(v) for(k, v) in object_dictionary(model) if v isa AbstractArray{VariableRef})

    return model, d, cost
end

"""
    Light robustness minimum without solving nominal problem
    Ax <= b
    b - right sides vector
    c - costs vector
    A - nominal values for constraint matrix
    Gamma - array of Gammas for rows
    AU - uncertainties of matrix A
    rho - maximum value to distort from optimum (if val == True)
        - maximum percent to distort from optimum (if val == False)
    printModel - when true model is printed
    printSolution - when true the solution (decision variables) is printed
    zOpt - optimal cost function for nominal problem

    Returns JuMP model, dictionary with decision variables and their values, and optimum cost
"""
function lightRobustnessMinOpt(c::Union{Vector, SparseVector, SparseMatrixCSC}, b::Union{Vector, SparseVector, SparseMatrixCSC},
     A::Union{Matrix, Vector, SparseVector, SparseMatrixCSC}, Gamma::Union{Vector, SparseVector},
     AU::Union{Matrix, Vector, SparseMatrixCSC, SparseVector},
    rho::Float64, val::Bool, printModel::Bool, printSolution::Bool, zOpt::Float64)

    n = size(c)[1]

    m = size(A)[1] # number of contraints
    if (size(A)[2] != n)
        throw("Matrix A has wrong dimensions")
    end
    if ((length(size(AU)) > 1 && size(AU)[2] > n) || size(AU)[1] > m)
        throw("Matrix AU has wrong dimensions")
    end

    if (size(b)[1] != m)
        throw("Vector b has wrong dimension")
    end

    if (size(Gamma)[1] > m)
        throw("Vector Gamma has wrong dimension")
    end

    # light robustness model
    model = Model(Cbc.Optimizer)
    set_attribute(model, "logLevel", 0)
    @variable(model, x[1:n] >= 0)
    @variable(model, z[1:m] >= 0)
    @variable(model, p[i in 1:m, j in 1:n] >= 0)
    @variable(model, y[1:m] >= 0)

    for i in 1:m
        @constraint(model, sum(A[i, j] * x[j] for j in 1:n) <= b[i])
        @constraint(model, sum(A[i, j] * x[j] for j in 1:n) + z[i] * Gamma[i] + sum(p[i, j] for j in 1:n) - y[i] <= b[i])
    end

    for i in 1:m
        for j in 1:n
            @constraint(model, -x[j] * AU[i, j] + z[i] + p[i, j] >= 0)
        end
    end

    if (val)
        @constraint(model, sum(c[j] * x[j] for j in 1:n) <= zOpt + rho)
    else
        @constraint(model, sum(c[j] * x[j] for j in 1:n) <= zOpt * ( 1 + rho))
    end

    @objective(model, Min, sum(y[i] for i in 1:m))
    if (printModel)
        println(model)
    end
    optimize!(model)
    cost = 0.0
    for j in 1:n
        cost += c[j] * value(x[j])
    end
     if (printSolution)
        printLightRobustness(model, n, x, zOpt)
    end
    d = Dict(
        k => value.(v) for(k, v) in object_dictionary(model) if v isa AbstractArray{VariableRef})

    return model, d, cost
end

"""
    Light robustness maximum without solving nominal problem
    Ax <= b
    b - right sides vector
    c - costs vector
    A - nominal values for constraint matrix
    Gamma - array of Gammas for rows
    AU - uncertainties of matrix A
    rho - maximum value to distort from optimum (if val == True)
        - maximum percent to distort from optimum (if val == False)
    printModel - when true model is printed
    printSolution - when true the solution (decision variables) is printed
    zOpt - optimal cost function for nominal problem

    Returns JuMP model, dictionary with decision variables and their values, and optimum cost
"""
function lightRobustnessMaxOpt(c::Union{Vector, SparseVector, SparseMatrixCSC}, b::Union{Vector, SparseVector, SparseMatrixCSC},
     A::Union{Matrix, Vector, SparseVector, SparseMatrixCSC}, Gamma::Union{Vector, SparseVector},
     AU::Union{Matrix, Vector, SparseMatrixCSC, SparseVector},
    rho::Float64, val::Bool, printModel::Bool, printSolution::Bool, zOpt::Float64)

    n = size(c)[1]

    m = size(A)[1] # number of contraints
    if (size(A)[2] != n)
        throw("Matrix A has wrong dimensions")
    end
    if ((length(size(AU)) > 1 && size(AU)[2] > n) || size(AU)[1] > m)
        throw("Matrix AU has wrong dimensions")
    end

    if (size(b)[1] != m)
        throw("Vector b has wrong dimension")
    end

    if (size(Gamma)[1] > m)
        throw("Vector Gamma has wrong dimension")
    end

    # light robustness model
    model = Model(Cbc.Optimizer)
    set_attribute(model, "logLevel", 1)
    @variable(model, x[i=1:n] >= 0)
    @variable(model, z[1:m] >= 0)
    @variable(model, p[i in 1:m, j in 1:n] >= 0)
    @variable(model, y[1:m] >= 0)

    for i in 1:m
        @constraint(model, sum(A[i, j] * x[j] for j in 1:n) <= b[i])
        @constraint(model, sum(A[i, j] * x[j] for j in 1:n) + z[i] * Gamma[i] + sum(p[i, j] for j in 1:n) - y[i] <= b[i])
    end

    for i in 1:m
        for j in 1:n
            @constraint(model, -x[j] * AU[i, j] + z[i] + p[i, j] >= 0)
        end
    end

    if (val)
        @constraint(model, sum(c[j] * x[j] for j in 1:n) >= zOpt - rho)
    else
        @constraint(model, sum(c[j] * x[j] for j in 1:n) >= zOpt * ( 1 - rho))
    end

    @objective(model, Min, sum(y[i] for i in 1:m))
     if (printModel)
        println(model)
    end
    optimize!(model)
    cost = 0.0
    for j in 1:n
        cost += c[j] * value(x[j])
    end
    if (printSolution)
        printLightRobustness(model, n, x, zOpt, c)
    end
    d = Dict(
        k => value.(v) for(k, v) in object_dictionary(model) if v isa AbstractArray{VariableRef})

    return model, d, cost
end

"""
Function that prints solution of light robustness model

model - JuMP model
n - number of decison variables
x - JuMP array of decision variables
zOpt - optimal value of nominal problem
c - cost vector
"""
function printLightRobustness(model, n, x, zOpt, c)
    if termination_status(model) == OPTIMAL
       println("Solution is optimal")
    elseif termination_status(model) == TIME_LIMIT && has_values(model)
       println("Solution is suboptimal due to a time limit, but a primal solution is available")
    else
       error("The model was not solved correctly.")
    end
    println("  nominal optimum = ", zOpt)
    println("  objective value (new) = ", objective_value(model))
    cost = 0
    for j in 1:n
        cost += c[j] * value(x[j])
    end
    println("  objective value  = ", cost)
    if primal_status(model) == FEASIBLE_POINT
        for j in 1:n
            println("  x", j, " = ", value(x[j]))
        end
    end
end