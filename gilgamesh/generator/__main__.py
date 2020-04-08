import networkx as nx
import numpy as np
import os
import shutil
import json
import sys
from connected_graph import generate_connected_graph

def clear_directory():
    delete = raw_input("Directory already exists. Do you want to delete it? (Y/N): ")
    if (delete == "Y" or delete == "y"):
        shutil.rmtree(out_folder)
        print("Deleted......")
    else:
        save_to_text = False
        print("Abort.........")



def select_user_inputs():
    """Process user input to generate appropriate graphs accordingly"""

    invalid = True
    inputs = sys.argv[1:]
    sys_input_valid = True
    while (invalid):
        try:
            if not sys_input_valid:
                print("Choose a connected graph type : \n (1) Line, (2) Cycle, (3) Complete, (4) Gnp")
                print("Input format: <training_size> <test_size> <graph_type> <number_of_nodes> <probability?>")
                inputs = raw_input(">> ").split()
            sys_input_valid = False
            if len(inputs) < 4:
                print("Wrong arguments")
                continue
            training_size = int(inputs[0])
            test_size = int(inputs[1])
            type = int(inputs[2])
            graph_size = int(inputs[3])
            p = 0
            if type == 4:
                if len(inputs) >= 4:
                    p = float(inputs[4])
                else:
                    print("Invalid format")
                    continue


            correct_type = type > 0 and type < 5
            positve_no_of_nodes = graph_size > 0 and training_size > 0 and test_size > 0
            correct_p = p >= 0 and p <= 1
            if correct_p and correct_type and positve_no_of_nodes:
                invalid = False
            else:
                print("Invalid input format, please try again!")
        except ValueError:
            print("Positive integers only!")

    return training_size, test_size, type, graph_size, p



def graph_generator():
    """Take 4 or 5 parameters input of users and use it to generate graph"""
    inputs = select_user_inputs()

    train_dir = os.path.join(out_folder, "train")
    os.mkdir(train_dir)
    test_dir = os.path.join(out_folder, "test")
    os.mkdir(test_dir)

    for i in range(0, inputs[0]):
        graph = generate_connected_graph(inputs[2], inputs[3], inputs[4])
        train_data = {}
        train_data["input"] = nx.to_numpy_matrix(graph, dtype=np.int64).tolist()
        train_data["output"] = nx.is_connected(graph)
        with open(os.path.join(train_dir, str(i) + ".json"), "w") as outfile:
            outfile.write(json.dumps(train_data, outfile))


    for i in range(0, inputs[1]):
        graph = generate_connected_graph(inputs[2], inputs[3], inputs[4])
        test_data = {}
        test_data["input"] = nx.to_numpy_matrix(graph, dtype=np.int64).tolist()
        test_data["output"] = nx.is_connected(graph)
        with open(os.path.join(test_dir, str(i) + ".json"), "w") as outfile:
            outfile.write(json.dumps(test_data, outfile))


    print('Success, file written %s ' % out_folder)
    return



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
    graph_generator()
