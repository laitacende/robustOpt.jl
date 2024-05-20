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
        # adjustable
        elseif mode == 2
             res1 = sum(B[i, j] * dict[:y][j] for j in 1:k)
             if !(sum(A[i, j] * dict[:x][j] for j in 1:n) + res1 <= b[i])
                bad += 1
             end
        end
    end
    return bad / all
end


function testProduction(fileName, steps, Gammas, per, rhos, T, n)

    fMinMax = open("./" * fileName * "_minmax.txt", "a")
    fLight = open("./" * fileName * "_light.txt", "a")
    fAdj = open("./" * fileName * "_adj.txt", "a")
    fNom = open("./" * fileName * "_nom.txt", "a")
    fNomWorst = open("./" * fileName * "_nomWorst.txt", "a")

    # periods - T, no of factories - n

    c = spzeros(n * T)
    for t in 1:length(c)
        c[t] += rand((Uniform(0, 100)))
    end

    VMax = rand(Uniform(1000, 10000))
    VMin = rand(Uniform(0, 100))
    # total production of factory accumulated over time
    Cap = spzeros(n)
    for t in 1:length(Cap)
        Cap[t] += rand((Uniform(500, 10000)))
    end

    # demands
    d = spzeros(T)
    for t in 1:length(d)
        d[t] += rand((Uniform(0, 100)))
    end
    # first period inventory
    v1 = rand(Uniform(0, 50))

    # maximum production at facotry in period t
    P = spzeros(n * T)
    for i in 1:(n*T)
        P[i] = rand(Uniform(50, 900))
    end
    A = zeros(n + T + T + n * T + 2, n * T + 1)
    A[n + T + T + n * T + 1, n * T + 1] = 1
    A[n + T + T + n * T + 2, n * T + 1] = -1

    AA = zeros(n + T + T + n * T + 1 + 2, n + 2)
    # factory1_period1 factory1_period2...
    B = zeros(n + T + T + n * T + 1 + 2, (n) * (T - 1))
    for i in 1:n
        AA[1, i] = c[(i - 1) * T  + 1]
    end
    for i in 1:n
        for j in 1:(T - 1)
            B[1, (i - 1) * (T - 1) + j] = c[(i - 1) * (T) + j + 1]
        end
    end

    AA[1, n + 1] = -1
    AA[n + T + T + n * T + 1 + 1, n + 2] = 1
    AA[n + T + T + n * T + 1 + 2, n + 2] = -1

    for i in 1:n
        for j in 1:T
            A[i, j + T * (i - 1)] = 1
        end
        A[i, n*T+1] = 0
    end
    for i in 2:(n+1)
        AA[i, i - 1] = 1
        for j in 1:(T-1)
            B[i, (i - 2) * (T - 1) + j] = 1
        end
    end

    # vmin
    for i in (n + 2):(n + 1 + T)
        for j in 1:n
            AA[i, j] = -1
        end
        AA[i, n + 2] = VMin - v1
        for k in 1:n
            for j in 1:(i - n - 2)
                B[i, (k - 1) * (T - 1) + j] = -1
            end
        end
    end
    # v max
    for i in (n + 2 + T):(n + 1 + T + T)
        for j in 1:n
            AA[i, j] = 1
        end
        AA[i, n + 2] = v1 - VMax
        for k in 1:n
            for j in 1:(i - n - 2 - T)
                B[i, (k - 1) * (T - 1) + j] = 1
            end
        end
    end

    for i in 1:n
        for j in 1:T
            A[(n + T + T) + (i - 1) * T + j, (i - 1) * T + j] = 1
        end
         A[i, n*T+1] = 0
    end

    for i in 1:n
        AA[(n + T + T + 2) + (i - 1) * T, i] = 1
    end

    for i in 1:n
        for j in 1:(T - 1)
            B[(n + T + T + 2) + (i - 1) * T + j, (i - 1) * (T - 1) + j] = 1
        end
    end

    zeroQ = []
    # v min
    for k in 1:n
        for j in 1:(T - 1)
            for i in (j + 2):T
                append!(zeroQ, [((k - 1) * (T - 1) + j, n + i + 1)])
            end
        end
    end
    # v max
    for k in 1:n
        for j in 1:(T - 1)
            for i in (j + 2):T
               append!(zeroQ, [((k - 1) * (T - 1) + j, n + T + i + 1)])
            end
        end
    end

    vm = [-VMin + v1 for i in 1:T]
    vma = [VMax - v1 for i in 1:T]
    b  = sparse([Cap; vm; vma; P; [1]; [-1]])
    z = zeros(n + T + T + n * T + 2, n * T)
    J = [Int64[] for i in 1:size(A)[1]]
    for i in (n + 1):(n + T + T)
        J[i] = [n*T + 1]
    end

    for s in 1:steps
        println(stderr, s)
        # uncertainty
        dU = spzeros(T)
        for r in 1:length(dU)
            dU[r] = rand(Uniform(d[r] * 0.2 * per, d[r] * per))
        end

        dSum = zeros(T)
        dUSum = zeros(T)
        dSum[1] = d[1]
        dUSum[1] = dU[1]
        for t in 2:T
            dSum[t] = dSum[t - 1]
            dSum[t] += d[t]
            dUSum[t] = dUSum[t - 1]
            dUSum[t] += dU[t]
        end


        for i in (n + 1):(n + T)
            for k in 1:n
                for j in 1:(i - n)
                    A[i, (k - 1) * T + j] = -1
                end
            end
            A[i, n*T+1] = dSum[i - n]
        end

        for i in (n + 1 + T):(n + T + T)
            for k in 1:n
                for j in 1:(i - n - T)
                    A[i, (k - 1) * T + j] = 1
                end
            end
             A[i, n*T+1] = -dSum[i - n - T]
        end

        ANom = A[:, 1:(size(A)[2] - 1)]
        ANom = ANom[1:(size(A)[1] - 2), :]

        if s == 1
            vmNom = [-VMin + v1 - dSum[i] for i in 1:T]
            vmaNom = [VMax - v1 + dSum[i] for i in 1:T]
            # nominal
            model0, dict0, obj0 = robustOpt.nominal(c, [Cap; vmNom; vmaNom; P], ANom, false, false)
            time = @elapsed robustOpt.nominal(c, [Cap; vmNom; vmaNom; P], ANom, false, false)
            write(fNom, string(obj0) * " " * string(time) * "\n")
        end


        # worst
        vmNomW = [-VMin + v1 - dSum[i] - dUSum[i] for i in 1:T]
        vmaNomW = [VMax - v1 + dSum[i] + dUSum[i] for i in 1:T]
        model01, dict01, obj01 = robustOpt.nominal(c, [Cap; vmNomW; vmaNomW; P], ANom, false, false)
        time = @elapsed robustOpt.nominal(c, [Cap; vmNomW; vmaNomW; P], ANom, false, false)
        write(fNomWorst, string(obj01) * " " * string(time) * "\n")

        for g in Gammas
            Gamma = spzeros(size(A)[1])
            for i in (n + 1):(n + T + T)
                Gamma[i] = g
            end

            # min max
            bMU = [spzeros(length(Cap)); dUSum; -dUSum; spzeros(length(P) + 2)]
            model1, dict1, obj1 = robustOpt.minmax(sparse([c; 0]), [], [], b, A, Gamma, J, bMU, false, false, false)
            time = @elapsed robustOpt.minmax(sparse([c; 0]), [], [], b, A, Gamma, J, bMU, false, false, false)
            constraints = checkConstraints(A, [], b, dict1,  n * T + 1, 0, 0)
            write(fMinMax, string(g) * " " * string(obj1) * " " * string(constraints) * " " * string(time) * "\n")

            # light robustness
            for j in 1:length(rhos)
                model2, dict2, obj2 =  robustOpt.lightRobustnessMin(sparse([c; 0]), b, A, Gamma, [z bMU], rhos[j], false, false, false)
                time = @elapsed robustOpt.lightRobustnessMin(sparse([c; 0]), b, A, Gamma, [z bMU], rhos[j], false, false, false)
                constraints = checkConstraints(A, [], b, dict2,  n * T + 1, 0, 1)
                if j == length(rhos)
                    write(fLight, string(g) * " " * string(obj2) * " " * string(constraints) * " " * string(time) * "\n")
                else
                   write(fLight, string(g) * " " * string(obj2) * " " * string(constraints) * " " * string(time) * " ")
                end
            end
            # adjustable
            bA = sparse([0; Cap; -dSum; dSum; P; [1]; [-1]])
            bAU = sparse([spzeros(1 + n); -dUSum; dUSum; spzeros(n*T); 0; 0])
            infos = [-7, -3, 0, 1]
            for p in 1:length(infos)
                zeroQ1 = copy(zeroQ)
                 for k in 1:n
                    for j in 1:(T - 1)
                        for i in 1:(j + infos[p])
                            append!(zeroQ1, [((k - 1) * (T - 1) + j, n + i + 1)])
                        end
                    end
                end
                # v max
                for k in 1:n
                    for j in 1:(T - 1)
                        for i in 1:(j + infos[p]) # j + 2
                           append!(zeroQ1, [((k - 1) * (T - 1) + j, n + T + i + 1)])
                        end
                    end
                end

                model3, dict3, obj3 = robustOpt.adjustableMinB(sparse([zeros(n); [1]; [0]]), bA, AA, B, g, bAU, zeroQ1, false, false)
                time = @elapsed robustOpt.adjustableMinB(sparse([zeros(n); [1]; [0]]), bA, AA, B, g, bAU, zeroQ1, false, false)
                constraints = checkConstraints(AA, B, bA, dict3, n + 2, (n) * (T - 1), 2)
                write(fAdj, string(g) * " " * string(obj3) * " " * string(constraints) * " " * string(time) * " ")
            end
            model3, dict3, obj3 = robustOpt.adjustableMinB(sparse([zeros(n); [1]; [0]]), bA, AA, B, g, bAU, zeroQ, false, false)
            time = @elapsed robustOpt.adjustableMinB(sparse([zeros(n); [1]; [0]]), bA, AA, B, g, bAU, zeroQ, false, false)
            constraints = checkConstraints(AA, B, bA, dict3, n + 2, (n) * (T - 1), 2)
          write(fAdj, string(g) * " " * string(obj3) * " " * string(constraints) * " " * string(time) * " ")

            model3, dict3, obj3 = robustOpt.adjustableMinB(sparse([zeros(n); [1]; [0]]), bA, AA, B, g, bAU, [], false, false)
            time = @elapsed robustOpt.adjustableMinB(sparse([zeros(n); [1]; [0]]), bA, AA, B, g, bAU, [], false, false)
            constraints = checkConstraints(AA, B, bA, dict3, n + 2, (n) * (T - 1), 2)
          write(fAdj, string(g) * " " * string(obj3) * " " * string(constraints) * " " * string(time) * "\n")

       end
    end
    close(fMinMax)
    close(fLight)
    close(fAdj)
    close(fNom)
    close(fNomWorst)
end

redirect_stdout(stdout)
