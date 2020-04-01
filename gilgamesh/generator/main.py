import networkx as nx
import numpy as np
import os
import shutil
import json
from connected_graph import generate_connected_graph

def clear_directory():
    delete = raw_input("Directory already exists. Do you want to delete it? (Y/N): ")
    if (delete == "Y" or delete == "y"):
        shutil.rmtree(out_folder)
        print("Deleted......")
    else:
        save_to_text = False
        print("Abort.........")

def select_graph_type():
    """Process user input to generate appropriate graphs accordingly"""

    invalid = True
    inputs = []

    while (invalid):
        print("Choose a connected graph type : \n (1) Line, (2) Cycle, (3) Complete, (4) Gnp")
        print("Input format: <graph_type> <number_of_nodes> <probability?>")
        try:
            inputs = raw_input(">> ").split()
            if len(inputs) < 2:
                print("Wrong arguments")
                continue
            type = int(inputs[0])
            size = int(inputs[1])
            p = 0
            if type == 4:
                if len(inputs) >= 3:
                    p = float(inputs[2])
                else:
                    print("Invalid format")
                    continue


            correct_type = type > 0 and type < 5
            positve_no_of_nodes = size > 0
            correct_p = p >= 0 and p <= 1
            if correct_p and correct_type and positve_no_of_nodes:
                invalid = False
            else:
                print("Invalid input format, please try again!")
        except ValueError:
            print("Positive integers only!")
    return generate_connected_graph(type, size, p)




# Main functions for IO processing

out_folder = "out/"
dir_name = raw_input("Enter directory name:\n>> ")
out_folder = os.path.join(out_folder, dir_name)
save_to_text = True
output_path = os.path.join(out_folder, "test.json")


# Ask to clear directory if already existed

if os.path.exists(out_folder) or os.path.isdir(out_folder):
    clear_directory()

os.makedirs(out_folder)



# Writes graph to json file if possible

if save_to_text:
    data = {}
    np_matrix = nx.to_numpy_matrix(select_graph_type(), dtype=np.int64)
    data["input"] = np_matrix.tolist()
    data["output"] = "True"
    with open(output_path, "w") as outfile:
     outfile.write(json.dumps(data, outfile))
    print('Success, file written %s ' % outfile)
