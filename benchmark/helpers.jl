# Recursively create benchmark groups
function mkgrp!(base, path)
    path = map(string, path)
    curr = base
    for k in path
        if !(k in keys(curr))
            curr[k] = BenchmarkGroup()
        end
        curr = curr[k]
    end
    curr
end

function mkbench!(f, base, path)
    grp = mkgrp!(base, path[1:end-1])
    grp[path[end]] = f()
end
