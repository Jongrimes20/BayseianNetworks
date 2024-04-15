from variable import Variable
from typing import Dict, List
from probGraphicalModels import BeliefNetwork
from probStochSim import RejectionSampling
from probFactors import Variable, Prob
from probVE import VE
from os import path #to connect to child jsons
import json


def perform_exact_inference(model: BeliefNetwork, Q:Variable, E: Dict[Variable, int], ordering: List[Variable]) -> Dict[int, float]:
    # Assume VE is defined and imported
    exact_infer = VE(gm=model)
    VE.max_display_level = -1
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
    pass

def main():
    # 1. Load the bayesian network from the directory named "child"
    variables = json.load(open(path.join("child", "variables.json")))

    name_to_node = {}
    nodes = []
    value_to_name = {}
    name_to_value = {}
    for variable in variables:
        variable_name = variable["name"]
        variable_values = variable["values"]
        variable_value_names = variable["value_names"]
        node =  Variable(variable_name, variable_values)
        name_to_node[variable_name] = node 
        nodes.append(node)

        value_to_name[variable_name] = { value: name for value, name in zip(variable_values, variable_value_names)}
        name_to_value[variable_name] = { name: value for value, name in zip(variable_values, variable_value_names)}

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

    # Prepare evidence and ordering for query
    evidence = {"CO2Report": 1, "XrayReport": 0, "Age": 0}
    ordering = ["Age", "CO2Report", "XrayReport"]

    # Perform exact inference
    # Prints itself
    Exact_Inference = perform_exact_inference(
        bn, 
        name_to_node["Disease"], 
        evidence, 
        ordering
    )

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
    main()




