import numpy as np


def read_orlib_ufl(filename):
    """
    Parse an OR-Library uncapacitated facility location instance file.

    Returns:
        m (int): number of facilities
        n (int): number of customers
        fixed_costs (np.ndarray): shape (m,), facility opening costs
        capacities (np.ndarray): shape (m,), facility capacities (dummy for UFL)
        service_costs (np.ndarray): shape (n, m), customer-facility costs
    """
    with open(filename, "r") as f:
        # remove empty lines and split tokens
        tokens = [t for line in f for t in line.strip().split() if t]

    # read first two numbers
    pos = 0
    m = int(tokens[pos])
    n = int(tokens[pos + 1])
    pos += 2

    # facility fixed costs and capacities
    fixed_costs = np.zeros(m)
    capacities = np.zeros(m)
    for i in range(m):
        # capacities[i] = float(tokens[pos])
        fixed_costs[i] = float(tokens[pos + 1])
        pos += 2

    # service costs: n customers × m facilities
    service_costs = np.zeros((n, m))
    for j in range(n):
        pos += 1
        service_costs[j, :] = [float(tokens[pos + k]) for k in range(m)]
        pos += m

    return m, n, fixed_costs, capacities, service_costs


# Example usage:
if __name__ == "__main__":
    filename = "./uncap_data/cap71.txt"  # or any other .txt instance
    m, n, fcosts, caps, costs = read_orlib_ufl(filename)
    print(f"Facilities: {m}, Customers: {n}")
    print("Fixed costs:", fcosts)
    print("Service cost matrix shape:", costs.shape)
    print("First customer cost vector:", costs[0])
