from probFactors import Prob
from variable import Variable
from typing import Dict, List
from probGraphicalModels import BeliefNetwork
from probStochSim import RejectionSampling
from probVE import VE
from os import path
import json
import timeit
import csv


def perform_exact_inference(model: BeliefNetwork, Q:Variable, E: Dict[Variable, int], ordering: List[Variable]) -> Dict[int, float]:
    """ Computes P(Q | E) on a Bayesian Network using variable elimination
        Arguments:
            model, the Bayesian Network
            Q, the query variable
            E, the evidence
            ordering, the order in which variables are eliminated
        
        Returns
            result, a dict mapping each possible value (q) of Q to the probability P(Q = q | E)
    """
    # Use the VE class to perform variable elimination
    exact_infer = VE(gm=model)
    return exact_infer.query(var=Q, obs=E, elim_order=ordering)


def perform_approximate_inference(model: BeliefNetwork, Q:Variable, E: Dict[Variable, int], n_samples: int) -> Dict[int, float]:
    """
        Arguments:
            model, the Bayesian Network
            Q, the query variable
            E, the evidence
            n_samples, the number of samples used to approximate P(Q | E)
        
        Returns
            result, a dict mapping each possible value (q) of Q to the probability P(Q = q | E)
    """
    # Use the RejectionSampling class to perform approximate inference
    approx_infer = RejectionSampling(bn)
    return approx_infer.query(qvar=Q, obs=E, number_samples=n_samples)

def measure_time(function, *args):
    """Utility to measure execution time of a function."""
    timer = timeit.Timer(lambda: function(*args))
    return timer.timeit(10) / 10



if __name__ == "__main__":
    # 1. Load the bayesian network from the directory named "child"
    # 2. Compute P(Q | E) on the BN using exact inference (variable elimination)
    #   Case 1: use ascending order of the names of the variables as the elimination ordering
    #   Case 2: use a better ordering. You may ask an LLM. 
    # 3. Compute P(Q | E) on the BN using approximiate inference (rejection sampling)
    #   Case 1: use 10 samples
    #   Case 2: Use 100 samples
    #   Case 3: Use a number of samples that guarantees PAC(epsilon = 0.01, delta = 0.05)
    #           Refer https://artint.info/3e/html/ArtInt3e.Ch9.S7.html#SS1.SSSx2
    #   For each case, compute the amount of error (as mean squared error) in the approximate inference results.
    #   Report the average over 10 trials.
    # Find the average time taken by each of the 5 methods.
    #1)
    variables = json.load(open(path.join("child", "variables.json")))

    name_to_node = {}
    nodes = []
    value_to_name = {}
    name_to_value = {}
    for variable in variables:
        variable_name = variable["name"]
        variable_values = variable["values"]
        variable_value_names = variable["value_names"]
        node = Variable(variable_name, variable_values)
        name_to_node[variable_name] = node
        nodes.append(node)
        value_to_name[variable_name] = {value: name for value, name in zip(variable_values, variable_value_names)}
        name_to_value[variable_name] = {name: value for value, name in zip(variable_values, variable_value_names)}

    tables = json.load(open(path.join("child", "tables.json")))
    cpts = []
    for table in tables:
        variable_name = table["variable"]
        node = name_to_node[variable_name]
        parent_names = table["parents"]
        parents = [name_to_node[parent_name] for parent_name in parent_names]
        probability_values = table["values"]
        cpt = Prob(node, parents, probability_values)
        cpts.append(cpt)

    bn = BeliefNetwork("child", nodes, cpts)
    #2)
    #case1)
    alphabetical_list = sorted(nodes, key=lambda x: x.name)
    print("Alphabetical Ordering: \n")
    print(alphabetical_list)
    E = {name_to_node["CO2Report"]: 1, name_to_node["XrayReport"]: 0, name_to_node["Age"]: 0}
    Q = "Disease"
    exact_results = perform_exact_inference(bn, name_to_node[Q], E, alphabetical_list)
    for i in name_to_node[Q].domain:
        p = exact_results[i]
        print(f"P({Q} = {value_to_name[Q][i]}) = {p}")
    #case2)
    print("Topologically Sorted: \n")
    topological_list = bn.topological_sort()
    topological_list.reverse()
    exact_results = perform_exact_inference(bn, name_to_node["Disease"], E, topological_list)
    for i in name_to_node[Q].domain:
        p = exact_results[i]
        print(f"P({Q} = {value_to_name[Q][i]}) = {p}")

    ##Time Exact
    alphabetical_time = measure_time(perform_exact_inference, bn, name_to_node["Disease"], E, alphabetical_list)
    optimal_time = measure_time(perform_exact_inference, bn, name_to_node["Disease"], E, topological_list)

    ##write to csv file
    with open("part1.csv", "w", newline='') as file:
        writer = csv.writer(file)
        writer.writerow([alphabetical_time, optimal_time])
    
    #3)
    print("results for 10 samples")
    mean_squared_error10 = 0
    approx_result10 = perform_approximate_inference(bn, name_to_node[Q], E, 10)
    for i in name_to_node[Q].domain:
        p = approx_result10[i]
        exactval = exact_results[i]
        mean_squared_error10 += abs(pow(exactval-p, 2))
        print(f"P({Q} = {value_to_name[Q][i]}) = {p}")
        print(f"Squared error: {mean_squared_error10}")
    mean_squared_error10 /= 6
    print(f"Mean squared error: {mean_squared_error10}\n")

    print("results for 100 samples")
    mean_squared_error100 = 0
    approx_result100 = perform_approximate_inference(bn, name_to_node[Q], E, 100)
    for i in name_to_node[Q].domain:
        p = approx_result100[i]
        exactval = exact_results[i]
        mean_squared_error100 += abs(pow(exactval-p, 2))
        print(f"P({Q} = {value_to_name[Q][i]}) = {p}")
        print(f"Squared error: {mean_squared_error100}")
    mean_squared_error100 /= 6
    print(f"Mean squared error: {mean_squared_error100}\n")

    print("results for 18445 samples")
    mean_squared_error18445 = 0
    approx_result18445 = perform_approximate_inference(bn, name_to_node[Q], E, 18445)
    for i in name_to_node[Q].domain:
        p = approx_result18445[i]
        exactval = exact_results[i]
        mean_squared_error18445 += abs(pow(exactval-p, 2))
        print(f"P({Q} = {value_to_name[Q][i]}) = {p}")
        print(f"Squared error: {mean_squared_error18445}")
    mean_squared_error18445 /= 6
    print(f"Mean squared error: {mean_squared_error18445}\n")

    ## Time the three Approaches
    n_10 = measure_time(perform_approximate_inference, bn, name_to_node[Q], E, 10)
    n_100 = measure_time(perform_approximate_inference, bn, name_to_node[Q], E, 100)
    n_18445 = measure_time(perform_approximate_inference, bn, name_to_node[Q], E, 18445)

    #Write avgs
    with open("part21.csv", "w", newline='') as file:
        writer = csv.writer(file)
        writer.writerow([n_10, n_100, n_18445])

    # Compute errors
    sample_sizes = [10, 100, 18445]  # The last size adjusted to PAC guarantee
    errors = []

    #Write errors
    with open("part22.csv", "w", newline='') as file:
        writer = csv.writer(file)
        writer.writerow([mean_squared_error10, mean_squared_error100, mean_squared_error18445])
    pass