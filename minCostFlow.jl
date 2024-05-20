include("./robustOpt.jl")

using .robustOpt

using Random, SparseArrays, Distributions, LinearAlgebra, Base
import Base.stderr

# output file: objective_function violeted_constrains(percents) time
redirect_stdout(open("/dev/null", "w"))

function checkConstraints(A, B, b, dict, n, k, mode)
    bad = 0
    all = size(A)[1]
    for i in 1:(size(A)[1])
        # min max
        if mode == 0 && !(sum(A[i, j] * dict[:x][j] for j in 1:n) <= b[i])
            bad += 1
        # light robustness
        elseif mode == 1 && !(sum(A[i, j] * dict[:x][j] for j in 1:n) <= b[i])
            bad += 1
        # recoverable
        elseif mode == 2 && !(sum(A[i, j] * dict[:x][j] for j in 1:n) == b[i])
            bad += 1
        elseif mode == 3
            res1 = B * d'
            if !(sum(A[i, j] * dict[:x][j] for j in 1:n)
                + sum(B[i, j] * dict[:d][j] for j in 1:k) <= b[i])
                bad += 1
            end
        end
    end
    return bad / all
end

function testMinCostFlow(fileName, steps, Gammas, n, per, KPerc, rhos)
    fMinMax = open("./" * fileName * "_minmax.txt", "a")
    fLight = open("./" * fileName * "_light.txt", "a")
    fRecov = open("./" * fileName * "_recov.txt", "a")
    fRecovInf = open("./" * fileName * "_recov_inf.txt", "a")
    fAdj = open("./" * fileName * "_adj.txt", "a")
    fNom = open("./" * fileName * "_nom.txt", "a")
    fNomWorst = open("./" * fileName * "_nomWorst.txt", "a")

    # generate graph

    # probability of creating and edge
    p = 1
    # incident matrix
    A = spzeros(n, n * n)
    for i in 1:n
        for j in 1:n
            # if edge from i to j exists
            if i != j && A[i, n * (i - 1) + j] == 0 && rand() <= p
                A[i, n * (i - 1) + j] = 1
                A[j, n * (i - 1) + j] = -1
            end
        end
    end

    u = spzeros(n * n)
    for t in 1:length(u)
        u[t] += rand((Uniform(0, 25))) # 25 wczesniej 150
    end
    b = spzeros(n)

    # must sum to 0
    # generate supply
    for j in 1:100
        supply = rand(Uniform(0, 10)) # 10
        # add suply to random node
        b[rand(1:n)] += supply
        # add demand to random node
        b[rand(1:n)] -= supply
    end

    # generate vectors
    c = spzeros(n * n)
    for t in 1:length(c)
        c[t] += rand((Uniform(0, 100)))
    end

    # nominal
    identity0 = spdiagm(0 => ones(n * n))

    model0, dict0, obj0 = robustOpt.nominal(c, [b; -b; u], [A; -A; identity0], false, false)
    time = @elapsed robustOpt.nominal(c, [b; -b; u], [A; -A; identity0], false, false)
    write(fNom, string(obj0) * " " * string(time) * "\n")

    c1 = spzeros(n*n + 1)
    c1[n * n + 1] = 1
    cC = copy(c)
    A1 = [sparse([cC' -1]); A spzeros(n, 1); -A spzeros(n, 1)]
    # all of the costs can be uncertain
    J1 = [Int64[] for i in 1:(2*n + 1)]
    J1[1] = [i for i in 1:(n*n)]
    Gamma1 = [0.0 for i in 1:(2*n + 1)]

    identity = [spdiagm(0 => ones(n * n)) spzeros(n*n, 1)]
    Gamma2 = [0.0 for i in 1:(2*n + 1 + n*n)]


    for i in 1:steps
        println(stderr, string(i))
        cU = spzeros(n * n)
        for r in 1:length(c)
                cU[r] = rand(Uniform(0, c[i] * per))
        end

        cWorst = copy(c)
        for j in 1:length(c)
            cWorst[j] += cU[j]
        end

        model01, dict01, obj01 = robustOpt.nominal(cWorst, [b; -b; u], [A; -A; identity0], false, false)
        time = @elapsed robustOpt.nominal(cWorst, [b; -b; u], [A; -A; identity0], false, false)
        write(fNomWorst, string(obj01) * " " * string(time) * "\n")

         for Gamma in Gammas
            println(stderr, "Gamma " * string(Gamma))
            Gamma1[1] = Gamma * n * n
            cU1 = reshape(cU, 1, length(cU))
            cU1 = spzeros((1, length(cU)))
            for r in 1:length(cU)
                cU1[r] = cU[r]
            end

            c12 = copy(c)
            cU12 = copy(cU1)
            for r in 1:(length(c12))
                c12[r] = c[r] + cU1[r] / 2
                cU12[r] = cU1[r] / 2
            end
            A12 = [sparse([c12' -1]); A spzeros(n, 1); -A spzeros(n, 1)]
            model1, dict1, obj1 = robustOpt.minmax(c1, spzeros(n*n + 1), [u; 100000000000000000],
            sparse([0; b; -b]), A1, Gamma1, J1, cU1, true, false, false)
            time = @elapsed robustOpt.minmax(c1, spzeros(n*n + 1), [u; 100000000000000000],
            sparse([0; b; -b]), A1, Gamma1, J1, cU1, true, false, false)
            constraints = checkConstraints(A, [], b, dict1, n * n, 0, 0)
            write(fMinMax, string(Gamma) * " " * string(obj1) * " " * string(constraints) * " " * string(time) * "\n")

            # light robustness
            Gamma2[1] = Gamma * n * n
            cU2 = [cU1 0; spzeros(2*n + n*n, n*n + 1)]
            for j in 1:length(rhos)
                model2, dict2, obj2 =  robustOpt.lightRobustnessMin(c1, sparse([0; b; -b; u]), sparse([A1; identity]), Gamma2,
                cU2, rhos[j], false, false, false)
                time = @elapsed robustOpt.lightRobustnessMin(c1, sparse([0; b; -b; u]), sparse([A1; identity]), Gamma2,
                cU2, rhos[j], false, false, false)
                constraints = checkConstraints(A, [], b, dict2, n * n, 0, 1)
                if j == length(rhos)
                    write(fLight, string(Gamma) * " " * string(obj2) * " " * string(constraints) * " " * string(time) * "\n")
                else
                   write(fLight, string(Gamma) * " " * string(obj2) * " " * string(constraints) * " " * string(time) * " ")
                end
            end

            # recoverable - continous budget!
            sumUnc = sum(cU)
            Gamma3 = Gamma * sumUnc
            for j in 1:length(KPerc)
                K = KPerc[j] * sum(abs.(b)) * (1.0 + Gamma)
                model3, dict3, obj3 = robustOpt.recoverableMin(sparse([c; spzeros(n*n)]), spzeros(2 * n * n),
                sparse([cU; spzeros(n * n)]), sparse([b; u]), sparse([A spzeros(n, n*n); identity0 identity0]), Gamma3, K, false, false)

                time = @elapsed robustOpt.recoverableMin(sparse([c; spzeros(n*n)]), spzeros(2 * n * n),
                sparse([cU; spzeros(n * n)]), sparse([b; u]), sparse([A spzeros(n, n*n); identity0 identity0]), Gamma3, K, false, false)
                constraints = checkConstraints(A, [], b, dict3, n * n, 0, 2)

                model4, dict4, obj4 = robustOpt.recoverableMinInf(sparse([c; spzeros(n*n)]), spzeros(2 * n * n),
                sparse([cU; spzeros(n * n)]), sparse([b; u]), sparse([A spzeros(n, n*n); identity0 identity0]), Gamma3, K, false, false)

                time = @elapsed robustOpt.recoverableMinInf(sparse([c; spzeros(n*n)]), spzeros(2 * n * n),
                sparse([cU; spzeros(n * n)]), sparse([b; u]), sparse([A spzeros(n, n*n); identity0 identity0]), Gamma3, K, false, false)
                constraints = checkConstraints(A, [], b, dict4, n * n, 0, 2)
                if j == length(KPerc)
                    write(fRecov, string(Gamma) * " " * string(obj3) * " " * string(constraints) * " " * string(time) * "\n")
                     write(fRecovInf, string(Gamma) * " " * string(obj4) * " " * string(constraints) * " " * string(time) * "\n")
                else
                   write(fRecov, string(Gamma) * " " * string(obj3) * " " * string(constraints) * " " * string(time) * " ")
                    write(fRecovInf, string(Gamma) * " " * string(obj4) * " " * string(constraints) * " " * string(time) * " ")
                end
            end
            GC.gc()
        end
    end
    close(fMinMax)
    close(fLight)
    close(fRecov)
    close(fRecovInf)
    close(fAdj)
    close(fNom)
    close(fNomWorst)
end

redirect_stdout(stdout)