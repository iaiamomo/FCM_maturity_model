import glob
import numpy as np
from FCM_class import FCM


# class representing the FCM algorithm
class Algorithm():
    
    def __init__(self, particles = []):
        n_fcm = 6   # number of sub-fcms
        lambda_value = 2  # lambda value
        iterations = 25  # number of iterations
        company_type = 1    # al file type of sub-fcms
        model_type = 4  # model type
        fcm_algorithm = "papageorgiou"    # christoforou, papageorgiou, kosko, stylios

        fcm_obj = FCM(n_fcm, iterations, model_type, company_type, particles)
        fcm_obj.run_fcm(lambda_value, fcm_algorithm)
        
        result = fcm_obj.final_activation_level

        return result


# Papageorgiou, Elpiniki I., et al. "Fuzzy cognitive maps learning using particle swarm optimization." Journal of intelligent information systems 25 (2005): 95-121.
class PSO:

    # compute the number of genes in the FCM
    def compute_individual_size():
        individual_size = 0
        files = glob.glob("model/*_wm.csv")
        for file in files:
            with open(file) as f:
                line = f.readline()
                individual_size += len(line.split(","))-1
        n_fcm = len(files)
        individual_size -= n_fcm - 1    # remove the common concepts between FCMs
        individual_size -= 1    # remove the target concept
        return individual_size
    

    def pso(function, swarm_size=60, omega=0.5, c1=0.8, c2=0.8, maxiter=100, 
            minstep=1e-8, minfunc=1e-8, debug=False, processes=1,
            particle_output=False):
        
        # population = swarm
        # individual = particle
        # c1 = cognitive parameter
        # c2 = social parameter

        S = swarm_size                              # number of particles
        D = PSO.compute_individual_size()           # dimensions of each particle

        # lower bounds and upper bounds
        lb = np.array([0] * D)
        ub = np.array([1] * D)
    
        # velocity bounds
        v_high = np.abs(ub - lb)
        v_low = -v_high
            
        # initialize the particles and velocity
        x = np.random.uniform(0, 1, size=(S, D))    # particle positions
        v = np.zeros_like(x)                        # particle velocities
        p = np.zeros_like(x)                        # best particle positions
        g = []                                      # best swarm position

        ch = [] # convergence history
        
        fp = np.ones(S) * np.inf  # best particle function values
        fg = np.inf  # best swarm position starting value

        # initialize the particle's position
        x = lb + x * (ub - lb)  # lowerbounds + x * (upperbounds - lowerbounds)

        fx = np.zeros()                             # current particle function values
        for i in range(S):
            fx[i] = Algorithm(x[i, :])
            #fs[i] = is_feasible(x[i, :])
        
        '''# store particle's best position (if constraints are satisfied)
        if fx < fp:
            p[i_update, :] = x[i_update, :].copy()
            fp[i_update] = fx[i_update]'''

        # pdate swarm's best position
        i_min = np.argmin(fp)
        if fp[i_min] < fg:
            fg = fp[i_min]
            g = p[i_min, :].copy()
        else:
            # At the start, there may not be any feasible starting point, so just
            # give it a temporary "best" point since it's likely to change
            g = x[0, :].copy()
        
        # Initialize the particle's velocity
        v = v_low + np.random.rand(S, D) * (v_high - v_low)
        
        # iterate until termination criterion met
        it = 1
        while it <= maxiter:
            # update velocities
            r1 = np.random.uniform(0, 1, size=(S, D))
            r2 = np.random.uniform(0, 1, size=(S, D))

            # Update the particles velocities
            # V_i(t + 1) = w * V_i(t) + c1 * r1 * (P_i(t) − X_i(t)) + c2 * r2 * (Pg_i(t) - X_i(t))
            v = omega * v + c1 * r1 * (p - x) + c2 * r2 * (g - x)

            # Update the particles' positions
            x = x + v
            # Correct for bound violations
            maskl = x < lb
            masku = x > ub
            x = x*(~np.logical_or(maskl, masku)) + lb*maskl + ub*masku

            # Update objectives and constraints
            for i in range(S):
                fx[i] = Algorithm(x[i, :])
                #fs[i] = is_feasible(x[i, :])

            '''# Store particle's best position (if constraints are satisfied)
            i_update = np.logical_and((fx < fp), fs)
            p[i_update, :] = x[i_update, :].copy()
            fp[i_update] = fx[i_update]'''

            # Compare swarm's best position with global best position
            i_min = np.argmin(fp)
            if fp[i_min] < fg:
                if debug:
                    print(f'New best for swarm at iteration {it}: {p[i_min, :]} {fp[i_min]}')

                p_min = p[i_min, :].copy()
                stepsize = np.sqrt(np.sum((g - p_min)**2))

                if np.abs(fg - fp[i_min]) <= minfunc:
                    print(f'Stopping search: Swarm best objective change less than {minfunc}')
                    if particle_output:
                        return p_min, fp[i_min], ch,  p, fp
                    else:
                        return p_min, fp[i_min], ch
                elif stepsize <= minstep:
                    print(f'Stopping search: Swarm best position change less than {minstep}')
                    if particle_output:
                        return p_min, fp[i_min], p, fp
                    else:
                        return p_min, fp[i_min]
                else:
                    g = p_min.copy()
                    fg = fp[i_min]
        
            if fg<0.02:
                fg=0.0
                
            if debug:
                print(f'Best after iteration {it}: {g} {fg}')
            ch.append(fg)
            print('Iteration {:}: {:}'.format(it, fg))
            it += 1
            if fg < 0.02:
                if particle_output:
                    return g, fg, ch, p, fp
                else:
                    return g, fg, ch

        print('Stopping search: maximum iterations reached --> {:}'.format(maxiter))
        
        #if not is_feasible(g):
        #    print("However, the optimization couldn't find a feasible design. Sorry")
        if particle_output:
            return g, fg, ch, p, fp
        else:
            return g, fg, ch